"""Full fine-tuning trainer."""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .base_trainer import BaseTrainer
from ..utils.logging import get_logger

logger = get_logger(__name__)


class FullFinetuneTrainer(BaseTrainer):
    """Trainer for full model fine-tuning.

    Trains all UNet parameters while keeping VAE and text encoders frozen.
    """

    def __init__(self, config: DictConfig):
        """Initialize full finetune trainer.

        Args:
            config: Training configuration.
        """
        super().__init__(config)

    def _setup_model(self) -> nn.Module:
        """Setup model for full fine-tuning."""
        model = super()._setup_model()

        # Freeze VAE
        if hasattr(model, "vae"):
            for param in model.vae.parameters():
                param.requires_grad = False
            logger.info("VAE frozen")

        # Freeze text encoders
        if hasattr(model, "text_encoders"):
            model.text_encoders.freeze()
            logger.info("Text encoders frozen")

        # UNet stays trainable
        if hasattr(model, "unet"):
            trainable = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)
            logger.info(f"UNet trainable parameters: {trainable:,}")

        return model

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer for UNet parameters."""
        # Get UNet parameters
        if hasattr(self.model, "unet"):
            params = list(self.model.unet.parameters())
        else:
            params = list(self.model.parameters())

        # Filter to trainable only
        params = [p for p in params if p.requires_grad]

        opt_config = self.config.training.optimizer

        optimizer = torch.optim.AdamW(
            params,
            lr=opt_config.get("lr", self.config.training.learning_rate),
            betas=tuple(opt_config.get("betas", [0.9, 0.999])),
            weight_decay=opt_config.get("weight_decay", 0.01),
            eps=opt_config.get("eps", 1e-8),
        )

        logger.info(f"Optimizer created with {len(params)} parameter groups")
        return optimizer

    def training_step(self, batch: dict) -> torch.Tensor:
        """Full finetune training step.

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

        # Get text embeddings
        if "prompt_embeds" in batch:
            encoder_hidden_states = batch["prompt_embeds"]
            pooled_embeds = batch.get("pooled_prompt_embeds")
        else:
            captions = batch["captions"]
            with torch.no_grad():
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

        # Predict noise
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
        loss = torch.nn.functional.mse_loss(noise_pred, target, reduction="none")
        loss = loss.mean(dim=[1, 2, 3])

        # SNR weighting
        if self.config.training.loss.get("snr_gamma"):
            snr_weights = self._compute_snr_weights(timesteps)
            loss = loss * snr_weights

        return loss.mean()
