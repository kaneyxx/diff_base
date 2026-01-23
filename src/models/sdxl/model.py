"""SDXL Model - combines UNet, VAE, and Text Encoders."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from safetensors.torch import load_file

from ..base import BaseDiffusionModel
from .unet import SDXLUNet
from .vae import SDXLVAE
from .text_encoder import SDXLTextEncoders
from ...utils.logging import get_logger

logger = get_logger(__name__)


class SDXLModel(BaseDiffusionModel):
    """Complete SDXL model with UNet, VAE, and text encoders."""

    def __init__(self, config: DictConfig):
        """Initialize SDXL model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)

    def _build_model(self) -> None:
        """Build all model components."""
        # UNet
        unet_config = self.config.get("unet", {})
        self.unet = SDXLUNet(unet_config)

        # VAE
        vae_config = self.config.get("vae", {})
        self.vae = SDXLVAE(vae_config)

        # Text encoders
        text_config = self.config.get("text_encoder", {})
        self.text_encoders = SDXLTextEncoders(text_config)

        # VAE scaling factor
        self.vae_scale_factor = self.vae.scaling_factor

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through UNet.

        Args:
            latents: Noisy latent tensor [B, 4, H, W].
            timesteps: Timestep values [B].
            encoder_hidden_states: Text embeddings [B, seq_len, 2048].
            added_cond_kwargs: Additional SDXL conditioning.
            **kwargs: Additional arguments.

        Returns:
            Predicted noise [B, 4, H, W].
        """
        return self.unet(
            sample=latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            **kwargs,
        )

    def encode_text(
        self,
        text: str | list[str],
        device: torch.device | str = "cuda",
    ) -> dict[str, torch.Tensor]:
        """Encode text prompts.

        Args:
            text: Single prompt or list of prompts.
            device: Target device.

        Returns:
            Dictionary with prompt_embeds and pooled_prompt_embeds.
        """
        return self.text_encoders.encode(
            prompt=text,
            device=device,
            do_classifier_free_guidance=False,
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space.

        Args:
            image: Image tensor [B, 3, H, W] in range [-1, 1].

        Returns:
            Latent tensor [B, 4, h, w].
        """
        with torch.no_grad():
            posterior = self.vae.encode(image)
            latent = posterior.sample()
            latent = latent * self.vae_scale_factor
        return latent

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image.

        Args:
            latent: Latent tensor [B, 4, h, w].

        Returns:
            Image tensor [B, 3, H, W] in range [-1, 1].
        """
        latent = latent / self.vae_scale_factor
        with torch.no_grad():
            image = self.vae.decode(latent)
        return image

    def _load_diffusers_checkpoint(self, path: Path) -> None:
        """Load from diffusers directory format.

        Args:
            path: Path to diffusers model directory.
        """
        logger.info(f"Loading SDXL from diffusers format: {path}")

        # Load UNet
        unet_path = path / "unet"
        if unet_path.exists():
            self._load_unet_weights(unet_path)
        else:
            logger.warning(f"UNet not found at {unet_path}")

        # Load VAE
        vae_path = path / "vae"
        if vae_path.exists():
            self._load_vae_weights(vae_path)
        else:
            logger.warning(f"VAE not found at {vae_path}")

        # Load text encoders
        if (path / "text_encoder").exists():
            self.text_encoders.load_pretrained(path)
        else:
            logger.warning("Text encoders not found")

    def _load_unet_weights(self, unet_path: Path) -> None:
        """Load UNet weights from diffusers format."""
        # Try safetensors first
        weights_file = unet_path / "diffusion_pytorch_model.safetensors"
        if not weights_file.exists():
            weights_file = unet_path / "diffusion_pytorch_model.bin"

        if weights_file.suffix == ".safetensors":
            state_dict = load_file(weights_file)
        else:
            state_dict = torch.load(weights_file, map_location="cpu")

        # Remap keys from diffusers format
        state_dict = self._remap_unet_keys(state_dict)

        missing, unexpected = self.unet.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"UNet missing keys: {len(missing)}")
        if unexpected:
            logger.warning(f"UNet unexpected keys: {len(unexpected)}")
        logger.info("Loaded UNet weights")

    def _load_vae_weights(self, vae_path: Path) -> None:
        """Load VAE weights from diffusers format."""
        weights_file = vae_path / "diffusion_pytorch_model.safetensors"
        if not weights_file.exists():
            weights_file = vae_path / "diffusion_pytorch_model.bin"

        if weights_file.suffix == ".safetensors":
            state_dict = load_file(weights_file)
        else:
            state_dict = torch.load(weights_file, map_location="cpu")

        # Remap keys
        state_dict = self._remap_vae_keys(state_dict)

        missing, unexpected = self.vae.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"VAE missing keys: {len(missing)}")
        if unexpected:
            logger.warning(f"VAE unexpected keys: {len(unexpected)}")
        logger.info("Loaded VAE weights")

    def _remap_unet_keys(self, state_dict: dict) -> dict:
        """Remap diffusers UNet keys to our format."""
        new_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Common remappings
            new_key = new_key.replace("attentions.", "attentions.")
            new_key = new_key.replace("resnets.", "resnets.")

            new_state_dict[new_key] = value

        return new_state_dict

    def _remap_vae_keys(self, state_dict: dict) -> dict:
        """Remap diffusers VAE keys to our format."""
        new_state_dict = {}

        for key, value in state_dict.items():
            new_key = key
            new_state_dict[new_key] = value

        return new_state_dict

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing on all components."""
        self.unet.enable_gradient_checkpointing()
        logger.info("Enabled gradient checkpointing")

    def freeze_vae(self) -> None:
        """Freeze VAE parameters."""
        for param in self.vae.parameters():
            param.requires_grad = False
        logger.info("VAE frozen")

    def freeze_text_encoders(self) -> None:
        """Freeze text encoder parameters."""
        self.text_encoders.freeze()
        logger.info("Text encoders frozen")

    def get_time_ids(
        self,
        original_size: tuple[int, int],
        crops_coords_top_left: tuple[int, int] = (0, 0),
        target_size: Optional[tuple[int, int]] = None,
        batch_size: int = 1,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Generate SDXL time IDs for conditioning.

        Args:
            original_size: Original image size (H, W).
            crops_coords_top_left: Crop coordinates.
            target_size: Target size (defaults to original_size).
            batch_size: Batch size.
            device: Target device.
            dtype: Data type.

        Returns:
            Time IDs tensor [B, 6].
        """
        if target_size is None:
            target_size = original_size

        time_ids = torch.tensor([
            original_size[0],  # original height
            original_size[1],  # original width
            crops_coords_top_left[0],  # crop top
            crops_coords_top_left[1],  # crop left
            target_size[0],  # target height
            target_size[1],  # target width
        ], device=device, dtype=dtype)

        return time_ids.unsqueeze(0).repeat(batch_size, 1)

    def to(
        self,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Move model to device/dtype."""
        if device is not None or dtype is not None:
            self.unet = self.unet.to(device=device, dtype=dtype)
            self.vae = self.vae.to(device=device, dtype=dtype)
            self.text_encoders = self.text_encoders.to(device=device, dtype=dtype)
        return self
