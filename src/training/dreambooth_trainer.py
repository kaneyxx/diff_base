"""DreamBooth trainer for subject personalization."""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .base_trainer import BaseTrainer
from ..data import create_dreambooth_dataloader
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DreamBoothTrainer(BaseTrainer):
    """Trainer for DreamBooth personalization.

    Trains the model to generate specific subjects/concepts while
    using prior preservation to prevent language drift.
    """

    def __init__(self, config: DictConfig):
        """Initialize DreamBooth trainer.

        Args:
            config: Training configuration.
        """
        super().__init__(config)

        # DreamBooth specific settings
        db_config = config.training.dreambooth
        self.prior_preservation = db_config.get("prior_preservation_weight", 1.0) > 0
        self.prior_weight = db_config.get("prior_preservation_weight", 1.0)

    def _setup_model(self) -> nn.Module:
        """Setup model for DreamBooth training."""
        model = super()._setup_model()

        # Freeze VAE
        if hasattr(model, "vae"):
            for param in model.vae.parameters():
                param.requires_grad = False
            logger.info("VAE frozen")

        # Optionally train text encoder
        db_config = self.config.training.dreambooth
        if not db_config.get("train_text_encoder", False):
            if hasattr(model, "text_encoders"):
                model.text_encoders.freeze()
                logger.info("Text encoders frozen")
        else:
            logger.info("Text encoders will be trained")

        return model

    def _setup_dataloader(self):
        """Setup DreamBooth specific dataloader."""
        return create_dreambooth_dataloader(self.config)

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer for UNet and optionally text encoder."""
        param_groups = []
        opt_config = self.config.training.optimizer

        # UNet parameters
        if hasattr(self.model, "unet"):
            unet_params = list(self.model.unet.parameters())
        else:
            unet_params = [p for p in self.model.parameters() if p.requires_grad]

        unet_params = [p for p in unet_params if p.requires_grad]

        param_groups.append({
            "params": unet_params,
            "lr": opt_config.get("lr", self.config.training.learning_rate),
        })

        # Text encoder parameters (if training)
        db_config = self.config.training.dreambooth
        if db_config.get("train_text_encoder", False):
            if hasattr(self.model, "text_encoders"):
                text_params = list(self.model.text_encoders.parameters())
                text_params = [p for p in text_params if p.requires_grad]

                if text_params:
                    param_groups.append({
                        "params": text_params,
                        "lr": opt_config.get("text_encoder_lr", 5e-6),
                    })

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=tuple(opt_config.get("betas", [0.9, 0.999])),
            weight_decay=opt_config.get("weight_decay", 0.01),
            eps=opt_config.get("eps", 1e-8),
        )

        return optimizer

    def training_step(self, batch: dict) -> torch.Tensor:
        """DreamBooth training step with prior preservation.

        Args:
            batch: Training batch with instance and optionally class data.

        Returns:
            Loss tensor.
        """
        total_loss = torch.tensor(0.0, device=self.device)

        # Instance loss (the subject we're learning)
        if "instance_images" in batch or "instance_latents" in batch:
            instance_loss = self._compute_loss(
                images=batch.get("instance_images"),
                latents=batch.get("instance_latents"),
                captions=batch.get("instance_captions", batch.get("instance_prompt")),
                prompt_embeds=batch.get("instance_prompt_embeds"),
                pooled_embeds=batch.get("instance_pooled_prompt_embeds"),
            )
            total_loss = total_loss + instance_loss

        # Class loss (prior preservation)
        if self.prior_preservation:
            if "class_images" in batch or "class_latents" in batch:
                class_loss = self._compute_loss(
                    images=batch.get("class_images"),
                    latents=batch.get("class_latents"),
                    captions=batch.get("class_captions", batch.get("class_prompt")),
                    prompt_embeds=batch.get("class_prompt_embeds"),
                    pooled_embeds=batch.get("class_pooled_prompt_embeds"),
                )
                total_loss = total_loss + self.prior_weight * class_loss

        return total_loss

    def _compute_loss(
        self,
        images=None,
        latents=None,
        captions=None,
        prompt_embeds=None,
        pooled_embeds=None,
    ) -> torch.Tensor:
        """Compute diffusion loss for a batch.

        Args:
            images: Input images (if not using cached latents).
            latents: Cached latents.
            captions: Text captions.
            prompt_embeds: Cached prompt embeddings.
            pooled_embeds: Cached pooled embeddings.

        Returns:
            Loss tensor.
        """
        # Get latents
        if latents is None and images is not None:
            with torch.no_grad():
                latents = self.model.encode_image(images)

        if latents is None:
            raise ValueError("Either images or latents must be provided")

        # Get text embeddings
        if prompt_embeds is None:
            if captions is None:
                raise ValueError("Either captions or prompt_embeds must be provided")

            # Handle single prompt vs list
            if isinstance(captions, str):
                captions = [captions] * latents.shape[0]

            with torch.no_grad():
                text_output = self.model.encode_text(captions, device=self.device)
            prompt_embeds = text_output["prompt_embeds"]
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
            encoder_hidden_states=prompt_embeds,
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
