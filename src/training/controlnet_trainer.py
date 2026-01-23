"""ControlNet trainer for conditional image generation."""

import torch
import torch.nn as nn
from copy import deepcopy
from omegaconf import DictConfig

from .base_trainer import BaseTrainer
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ControlNetModel(nn.Module):
    """ControlNet model - copy of UNet encoder with zero convolutions."""

    def __init__(
        self,
        unet: nn.Module,
        conditioning_channels: int = 3,
    ):
        """Initialize ControlNet from UNet.

        Args:
            unet: Base UNet model to copy encoder from.
            conditioning_channels: Number of conditioning input channels.
        """
        super().__init__()

        # Get model channels from UNet
        model_channels = getattr(unet, "config", {}).get("model_channels", 320)
        if hasattr(unet, "conv_in"):
            model_channels = unet.conv_in.out_channels

        # Conditioning input
        self.controlnet_cond_embedding = nn.Sequential(
            nn.Conv2d(conditioning_channels, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 96, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, model_channels, kernel_size=3, padding=1),
        )

        # Copy input conv
        self.conv_in = deepcopy(unet.conv_in)

        # Copy time embedding
        self.time_proj = deepcopy(unet.time_proj) if hasattr(unet, "time_proj") else None
        self.time_embedding = deepcopy(unet.time_embedding) if hasattr(unet, "time_embedding") else None

        # Copy additional embeddings (SDXL)
        if hasattr(unet, "add_time_proj"):
            self.add_time_proj = deepcopy(unet.add_time_proj)
            self.add_embedding = deepcopy(unet.add_embedding)
        else:
            self.add_time_proj = None
            self.add_embedding = None

        # Copy down blocks
        self.down_blocks = deepcopy(unet.down_blocks)

        # Copy mid block
        self.mid_block = deepcopy(unet.mid_block)

        # Zero convolutions for outputs
        self.zero_convs = nn.ModuleList()
        self._setup_zero_convs(unet)

        self.mid_zero_conv = self._make_zero_conv(
            self._get_mid_channels(unet), self._get_mid_channels(unet)
        )

    def _get_mid_channels(self, unet: nn.Module) -> int:
        """Get mid block channels."""
        if hasattr(unet, "mid_block") and hasattr(unet.mid_block, "resnets"):
            return unet.mid_block.resnets[0].conv1.out_channels
        return 1280  # Default SDXL

    def _setup_zero_convs(self, unet: nn.Module) -> None:
        """Setup zero convolutions for each down block output."""
        for down_block in unet.down_blocks:
            # Get output channels
            if hasattr(down_block, "resnets"):
                out_ch = down_block.resnets[-1].conv2.out_channels
            else:
                out_ch = 320  # Default

            # Add zero conv for each resnet output
            num_outputs = len(down_block.resnets) if hasattr(down_block, "resnets") else 2
            for _ in range(num_outputs):
                self.zero_convs.append(self._make_zero_conv(out_ch, out_ch))

            # Add one for downsampler if present
            if hasattr(down_block, "downsamplers") and down_block.downsamplers:
                self.zero_convs.append(self._make_zero_conv(out_ch, out_ch))

    def _make_zero_conv(self, in_channels: int, out_channels: int) -> nn.Conv2d:
        """Create a zero-initialized convolution."""
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        return conv

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        added_cond_kwargs: dict = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Forward pass.

        Args:
            sample: Noisy latent.
            timestep: Timestep.
            encoder_hidden_states: Text embeddings.
            controlnet_cond: Conditioning image.
            added_cond_kwargs: Additional SDXL conditioning.

        Returns:
            Tuple of (down_block_residuals, mid_block_residual).
        """
        # Time embedding
        if self.time_proj is not None:
            t_emb = self.time_proj(timestep.to(sample.dtype))
            t_emb = self.time_embedding(t_emb)

            # SDXL additional embeddings
            if added_cond_kwargs is not None and self.add_time_proj is not None:
                text_embeds = added_cond_kwargs.get("text_embeds")
                time_ids = added_cond_kwargs.get("time_ids")

                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape(time_ids.shape[0], -1)

                add_embeds = torch.cat([text_embeds, time_embeds], dim=-1)
                add_embeds = self.add_embedding(add_embeds)

                t_emb = t_emb + add_embeds
        else:
            t_emb = timestep

        # Add conditioning
        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        sample = self.conv_in(sample)
        sample = sample + controlnet_cond

        # Down blocks
        down_block_res_samples = []
        zero_idx = 0

        for down_block in self.down_blocks:
            sample, res_samples = down_block(sample, t_emb, encoder_hidden_states)

            for res in res_samples:
                down_block_res_samples.append(self.zero_convs[zero_idx](res))
                zero_idx += 1

        # Mid block
        sample = self.mid_block(sample, t_emb, encoder_hidden_states)
        mid_block_res = self.mid_zero_conv(sample)

        return down_block_res_samples, mid_block_res


class ControlNetTrainer(BaseTrainer):
    """Trainer for ControlNet conditioning.

    Trains a ControlNet to add spatial conditioning to a frozen base model.
    """

    def __init__(self, config: DictConfig):
        """Initialize ControlNet trainer.

        Args:
            config: Training configuration.
        """
        super().__init__(config)

    def _setup_model(self) -> nn.Module:
        """Setup frozen base model + trainable ControlNet."""
        # Create base model
        model = super()._setup_model()

        # Freeze the entire base model
        for param in model.parameters():
            param.requires_grad = False
        logger.info("Base model frozen")

        # Create ControlNet from UNet
        cn_config = self.config.training.controlnet
        conditioning_channels = self._get_conditioning_channels(
            cn_config.get("conditioning_type", "canny")
        )

        unet = model.unet if hasattr(model, "unet") else model
        self.controlnet = ControlNetModel(unet, conditioning_channels)
        self.controlnet = self.controlnet.to(self.device, dtype=self.dtype)

        trainable = sum(p.numel() for p in self.controlnet.parameters())
        logger.info(f"ControlNet created with {trainable:,} parameters")

        return model

    def _get_conditioning_channels(self, conditioning_type: str) -> int:
        """Get input channels for conditioning type."""
        channels = {
            "canny": 1,
            "depth": 1,
            "normal": 3,
            "pose": 3,
            "segmentation": 3,
            "scribble": 1,
            "hed": 1,
        }
        return channels.get(conditioning_type, 3)

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer for ControlNet only."""
        opt_config = self.config.training.optimizer

        optimizer = torch.optim.AdamW(
            self.controlnet.parameters(),
            lr=opt_config.get("lr", self.config.training.learning_rate),
            betas=tuple(opt_config.get("betas", [0.9, 0.999])),
            weight_decay=opt_config.get("weight_decay", 0.01),
            eps=opt_config.get("eps", 1e-8),
        )

        return optimizer

    def _get_trainable_params(self) -> list[nn.Parameter]:
        """Get ControlNet parameters."""
        return list(self.controlnet.parameters())

    def training_step(self, batch: dict) -> torch.Tensor:
        """ControlNet training step.

        Args:
            batch: Training batch with images and conditioning.

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

        # Get conditioning image
        controlnet_cond = batch["conditioning_images"]

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

        # Prepare SDXL conditioning
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

        # Get ControlNet conditioning
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            controlnet_cond,
            added_cond_kwargs,
        )

        # Apply conditioning scale
        conditioning_scale = self.config.training.controlnet.get("conditioning_scale", 1.0)
        down_block_res_samples = [
            sample * conditioning_scale for sample in down_block_res_samples
        ]
        mid_block_res_sample = mid_block_res_sample * conditioning_scale

        # Forward through frozen UNet with ControlNet conditioning
        with torch.no_grad():
            noise_pred = self.model(
                latents=noisy_latents,
                timesteps=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

        # Actually, we need gradients through UNet for ControlNet training
        # Let me fix this - we need to detach the UNet but allow ControlNet gradients
        noise_pred = self.model.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
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

    def _save_checkpoint(self, final: bool = False) -> None:
        """Save ControlNet checkpoint."""
        from pathlib import Path
        from safetensors.torch import save_file

        output_dir = Path(self.config.experiment.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if final:
            checkpoint_dir = output_dir / "final"
        else:
            checkpoint_dir = output_dir / f"checkpoint-{self.global_step}"

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save ControlNet weights
        controlnet_state = self.controlnet.state_dict()
        save_file(controlnet_state, checkpoint_dir / "controlnet.safetensors")

        # Save training state
        torch.save({
            "step": self.global_step,
            "epoch": self.current_epoch,
            "optimizer": self.optimizer.state_dict(),
        }, checkpoint_dir / "training_state.pt")

        logger.info(f"ControlNet checkpoint saved to {checkpoint_dir}")
