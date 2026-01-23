"""Textual Inversion trainer for learning new concepts via embeddings."""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .base_trainer import BaseTrainer
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TextualInversionTrainer(BaseTrainer):
    """Trainer for Textual Inversion.

    Learns new concepts by optimizing text embeddings while keeping
    the model frozen.
    """

    def __init__(self, config: DictConfig):
        """Initialize Textual Inversion trainer.

        Args:
            config: Training configuration.
        """
        super().__init__(config)

        # Get config
        ti_config = config.training.textual_inversion
        self.placeholder_token = ti_config.get("placeholder_token", "<concept>")
        self.initializer_token = ti_config.get("initializer_token", "photo")
        self.num_vectors = ti_config.get("num_vectors", 1)

        # Initialize embeddings
        self._setup_embeddings()

    def _setup_model(self) -> nn.Module:
        """Setup model with frozen weights."""
        model = super()._setup_model()

        # Freeze everything
        for param in model.parameters():
            param.requires_grad = False

        logger.info("All model parameters frozen for textual inversion")

        return model

    def _setup_embeddings(self) -> None:
        """Setup learnable text embeddings."""
        if not hasattr(self.model, "text_encoders"):
            raise ValueError("Model must have text_encoders for textual inversion")

        text_encoders = self.model.text_encoders

        # Get tokenizer and embedding layer
        if hasattr(text_encoders, "tokenizer"):
            tokenizer = text_encoders.tokenizer
            text_encoder = text_encoders.text_encoder
        else:
            raise ValueError("Text encoder must have tokenizer")

        # Add placeholder tokens to tokenizer
        placeholder_tokens = [
            f"{self.placeholder_token}_{i}" if self.num_vectors > 1
            else self.placeholder_token
            for i in range(self.num_vectors)
        ]

        num_added = tokenizer.add_tokens(placeholder_tokens)
        if num_added != self.num_vectors:
            logger.warning(f"Expected to add {self.num_vectors} tokens, added {num_added}")

        # Resize embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Get embedding layer
        embeddings = text_encoder.get_input_embeddings()

        # Initialize with initializer token embedding
        initializer_ids = tokenizer.encode(self.initializer_token, add_special_tokens=False)
        initializer_embedding = embeddings.weight[initializer_ids[0]].clone()

        # Set placeholder embeddings
        placeholder_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
        for idx in placeholder_ids:
            embeddings.weight.data[idx] = initializer_embedding.clone()
            # Make trainable
            embeddings.weight.requires_grad = True

        # Store for training
        self.placeholder_token_ids = placeholder_ids
        self.text_encoder_embeddings = embeddings

        logger.info(f"Initialized {self.num_vectors} embedding vectors from '{self.initializer_token}'")

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer for embedding parameters only."""
        # Get only the placeholder embeddings
        embedding_params = []
        embeddings = self.text_encoder_embeddings

        # Create a parameter that only tracks our placeholder tokens
        # We'll optimize the full embedding table but mask gradients
        embedding_params = [embeddings.weight]

        opt_config = self.config.training.optimizer

        optimizer = torch.optim.AdamW(
            embedding_params,
            lr=opt_config.get("lr", self.config.training.learning_rate),
            betas=tuple(opt_config.get("betas", [0.9, 0.999])),
            weight_decay=opt_config.get("weight_decay", 0.0),  # Usually no weight decay for TI
            eps=opt_config.get("eps", 1e-8),
        )

        return optimizer

    def _get_trainable_params(self) -> list[nn.Parameter]:
        """Get embedding parameters."""
        return [self.text_encoder_embeddings.weight]

    def training_step(self, batch: dict) -> torch.Tensor:
        """Textual Inversion training step.

        Args:
            batch: Training batch.

        Returns:
            Loss tensor.
        """
        # Get latents
        if "latents" in batch:
            latents = batch["latents"]
        else:
            images = batch["images"]
            with torch.no_grad():
                latents = self.model.encode_image(images)

        # Get captions (should contain placeholder token)
        captions = batch["captions"]

        # Encode text with trainable embeddings
        text_output = self.model.encode_text(captions, device=self.device)
        encoder_hidden_states = text_output["prompt_embeds"]
        pooled_embeds = text_output.get("pooled_prompt_embeds")

        # Sample timesteps
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.num_train_timesteps,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )

        # Sample noise
        noise = torch.randn_like(latents)

        # Add noise
        noisy_latents = self._add_noise(latents, noise, timesteps)

        # Prepare conditioning
        added_cond_kwargs = None
        if pooled_embeds is not None:
            resolution = self.config.data.resolution
            time_ids = self.model.get_time_ids(
                original_size=(resolution, resolution),
                target_size=(resolution, resolution),
                batch_size=batch_size,
                device=self.device,
                dtype=self.dtype,
            )
            added_cond_kwargs = {
                "text_embeds": pooled_embeds,
                "time_ids": time_ids,
            }

        # Forward pass (model is frozen, gradients flow through embeddings)
        noise_pred = self.model(
            latents=noisy_latents,
            timesteps=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )

        # Get target
        prediction_type = self.config.model.scheduler.get("prediction_type", "epsilon")
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            target = noise

        # Compute loss
        loss = torch.nn.functional.mse_loss(noise_pred, target)

        return loss

    def _mask_embedding_gradients(self) -> None:
        """Mask gradients to only update placeholder embeddings."""
        if self.text_encoder_embeddings.weight.grad is not None:
            # Create mask
            mask = torch.zeros_like(self.text_encoder_embeddings.weight.grad)
            for idx in self.placeholder_token_ids:
                mask[idx] = 1.0

            # Apply mask
            self.text_encoder_embeddings.weight.grad *= mask

    def train(self) -> None:
        """Override train to add gradient masking."""
        # Hook to mask gradients before optimizer step
        original_step = self.optimizer.step

        def masked_step():
            self._mask_embedding_gradients()
            original_step()

        self.optimizer.step = masked_step

        # Run normal training
        super().train()

    def _save_checkpoint(self, final: bool = False) -> None:
        """Save learned embeddings."""
        from pathlib import Path
        from safetensors.torch import save_file

        output_dir = Path(self.config.experiment.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if final:
            checkpoint_dir = output_dir / "final"
        else:
            checkpoint_dir = output_dir / f"checkpoint-{self.global_step}"

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save learned embeddings
        learned_embeddings = {}
        for i, idx in enumerate(self.placeholder_token_ids):
            token_name = f"{self.placeholder_token}_{i}" if self.num_vectors > 1 else self.placeholder_token
            learned_embeddings[token_name] = self.text_encoder_embeddings.weight[idx].cpu()

        save_file(learned_embeddings, checkpoint_dir / "learned_embeds.safetensors")

        # Save metadata
        import json
        metadata = {
            "placeholder_token": self.placeholder_token,
            "initializer_token": self.initializer_token,
            "num_vectors": self.num_vectors,
            "token_ids": self.placeholder_token_ids,
        }
        with open(checkpoint_dir / "ti_config.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Textual inversion embeddings saved to {checkpoint_dir}")
