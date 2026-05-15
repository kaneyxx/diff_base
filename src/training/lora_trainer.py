"""LoRA trainer for efficient fine-tuning."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from ..utils.logging import get_logger
from .base_trainer import BaseTrainer
from .methods.lora import get_lora_parameters, inject_lora_layers, load_lora_weights, save_lora_weights

logger = get_logger(__name__)


class LoRATrainer(BaseTrainer):
    """Trainer for LoRA fine-tuning.

    Injects LoRA layers into the model and trains only those parameters,
    keeping the base model frozen.
    """

    def __init__(self, config: DictConfig):
        """Initialize LoRA trainer.

        Args:
            config: Training configuration.
        """
        super().__init__(config)

    def _setup_model(self) -> nn.Module:
        """Setup model with LoRA layers injected."""
        # Create base model
        model = super()._setup_model()

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Freeze VAE and text encoders
        if hasattr(model, "vae"):
            for param in model.vae.parameters():
                param.requires_grad = False

        if hasattr(model, "text_encoders"):
            model.text_encoders.freeze()

        # Get UNet for LoRA injection
        if hasattr(model, "unet"):
            target = model.unet
        else:
            target = model

        # Inject LoRA layers
        lora_config = self.config.training.lora
        target = inject_lora_layers(
            target,
            rank=lora_config.get("rank", 16),
            alpha=lora_config.get("alpha", 16.0),
            dropout=lora_config.get("dropout", 0.0),
            target_modules=list(lora_config.get("target_modules", [])),
        )

        # Replace UNet with LoRA version
        if hasattr(model, "unet"):
            model.unet = target

        # Optionally train text encoder
        if lora_config.get("train_text_encoder", False):
            self._setup_text_encoder_lora(model)

        # Log parameter counts
        params = model.get_param_count() if hasattr(model, "get_param_count") else {}
        logger.info(f"Model parameters: {params}")

        return model

    def _setup_text_encoder_lora(self, model: nn.Module) -> None:
        """Inject LoRA into text encoders."""
        if not hasattr(model, "text_encoders"):
            return

        lora_config = self.config.training.lora
        text_encoder_targets = lora_config.get("text_encoder_target_modules", [
            "q_proj", "k_proj", "v_proj", "out_proj"
        ])

        # Inject LoRA into text encoders
        if hasattr(model.text_encoders, "text_encoder"):
            model.text_encoders.text_encoder = inject_lora_layers(
                model.text_encoders.text_encoder,
                rank=lora_config.get("rank", 16) // 2,
                alpha=lora_config.get("alpha", 16.0) // 2,
                target_modules=text_encoder_targets,
            )

        if hasattr(model.text_encoders, "text_encoder_2"):
            model.text_encoders.text_encoder_2 = inject_lora_layers(
                model.text_encoders.text_encoder_2,
                rank=lora_config.get("rank", 16) // 2,
                alpha=lora_config.get("alpha", 16.0) // 2,
                target_modules=text_encoder_targets,
            )

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer for LoRA parameters only."""
        lora_params = get_lora_parameters(self.model)

        if not lora_params:
            raise ValueError("No LoRA parameters found. Check target_modules config.")

        opt_config = self.config.training.optimizer

        optimizer = torch.optim.AdamW(
            lora_params,
            lr=opt_config.get("lr", self.config.training.learning_rate),
            betas=tuple(opt_config.get("betas", [0.9, 0.999])),
            weight_decay=opt_config.get("weight_decay", 0.01),
            eps=opt_config.get("eps", 1e-8),
        )

        logger.info(f"Optimizer created with {len(lora_params)} parameters")
        return optimizer

    def _get_trainable_params(self) -> list[nn.Parameter]:
        """Get LoRA parameters."""
        return get_lora_parameters(self.model)

    def training_step(self, batch: dict) -> torch.Tensor:
        """LoRA training step.

        Args:
            batch: Training batch.

        Returns:
            Loss tensor.
        """
        # Get latents (from cache or encode)
        if "latents" in batch:
            latents = batch["latents"]
        else:
            images = batch["images"]
            latents = self.model.encode_image(images)

        # Get text embeddings (from cache or encode)
        if "prompt_embeds" in batch:
            encoder_hidden_states = batch["prompt_embeds"]
            pooled_embeds = batch.get("pooled_prompt_embeds")
        else:
            captions = batch["captions"]
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

        # Add noise to latents
        noisy_latents = self._add_noise(latents, noise, timesteps)

        # Prepare added conditioning (for SDXL)
        added_cond_kwargs = None
        if pooled_embeds is not None:
            # Create time IDs
            batch_size = latents.shape[0]
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

        # Get target based on prediction type
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

        # Apply SNR weighting if configured
        if self.config.training.loss.get("snr_gamma"):
            snr_weights = self._compute_snr_weights(timesteps)
            loss = loss * snr_weights

        return loss.mean()

    def _save_checkpoint(self, final: bool = False) -> None:
        """Save LoRA checkpoint.

        By default (training.save_full_model=false), saves only the slim
        LoRA-only checkpoint:
          - lora/adapter_model.safetensors  (LoRA params)
          - lora/adapter_config.json        (LoRA hyper-params)
          - training_state.pt               (step, epoch, optimizer, scheduler)
          - config.yaml                     (resolved experiment config)

        When training.save_full_model=true (explicit opt-in), the legacy
        behavior is preserved: super()._save_checkpoint() writes model.safetensors
        containing the full model state (12B+ parameters).

        Args:
            final: Whether this is the final checkpoint.
        """
        output_dir = Path(self.config.experiment.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if final:
            checkpoint_dir = output_dir / "final"
        else:
            checkpoint_dir = output_dir / f"checkpoint-{self.global_step}"

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Check whether the caller opted into the legacy full-model save.
        save_full_model = self.config.training.get("save_full_model", False)

        if save_full_model:
            # Legacy path: writes model.safetensors (full model state).
            super()._save_checkpoint(final)
        else:
            # Slim path: training state + config only (no full model weights).
            training_state: dict[str, Any] = {
                "step": self.global_step,
                "epoch": self.current_epoch,
                "optimizer": self.optimizer.state_dict(),
            }
            if self.lr_scheduler is not None:
                training_state["scheduler"] = self.lr_scheduler.state_dict()

            torch.save(training_state, checkpoint_dir / "training_state.pt")
            OmegaConf.save(self.config, checkpoint_dir / "config.yaml")

        # Always save LoRA adapter weights (the whole point of LoRA training).
        # Unwrap the DDP wrapper when Accelerator is active so that the
        # state_dict keys do not carry the "module." prefix.
        lora_dir = checkpoint_dir / "lora"
        save_lora_weights(self._unwrap_model(), lora_dir, self.config)

        logger.info(f"LoRA checkpoint saved to {checkpoint_dir} (save_full_model={save_full_model})")

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load LoRA checkpoint.

        Handles both the slim format (lora/ subdir with adapter_model.safetensors)
        and the legacy full-model format (model.safetensors at checkpoint root).

        Args:
            checkpoint_path: Path to checkpoint directory.
        """
        checkpoint_dir = Path(checkpoint_path)

        # Restore training state (step, epoch, optimizer, scheduler).
        training_state_path = checkpoint_dir / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(
                training_state_path, map_location=self.device
            )
            self.global_step = training_state.get("step", 0)
            self.current_epoch = training_state.get("epoch", 0)

            if self.optimizer is not None and "optimizer" in training_state:
                self.optimizer.load_state_dict(training_state["optimizer"])
                logger.info("Loaded optimizer state from training_state.pt")

            if self.lr_scheduler is not None and "scheduler" in training_state:
                self.lr_scheduler.load_state_dict(training_state["scheduler"])
                logger.info("Loaded scheduler state from training_state.pt")

        # Load LoRA weights from the lora/ subdir (slim format).
        lora_dir = checkpoint_dir / "lora"
        if lora_dir.exists():
            load_lora_weights(self.model, lora_dir, device=self.device)
            logger.info(f"Loaded LoRA weights from {lora_dir}")
        elif (checkpoint_dir / "model.safetensors").exists():
            # Fall back to full-model checkpoint (legacy format).
            from ..utils.checkpoint import load_checkpoint as _load_ckpt
            _load_ckpt(
                path=checkpoint_dir,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
                device=self.device,
            )
            logger.info(f"Loaded full-model checkpoint from {checkpoint_dir}")
        else:
            raise FileNotFoundError(
                f"No LoRA weights or model.safetensors found in {checkpoint_dir}"
            )

        logger.info(f"Resumed from step {self.global_step}, epoch {self.current_epoch}")
