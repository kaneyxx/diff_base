"""FLUX.1 Model - combines Transformer, VAE, and Text Encoders.

FLUX.1 variants:
- dev: Full model with guidance, 12B parameters
- schnell: Distilled model without guidance, 12B parameters

Image Editing Support:
- Kontext Mode: Reference image editing via encode_reference_images()
- Fill Mode: NOT supported in FLUX.1
"""

from pathlib import Path
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from safetensors.torch import load_file

from ...base import BaseDiffusionModel
from .transformer import Flux1Transformer
from .vae import Flux1VAE
from .text_encoder import Flux1TextEncoders
from .conditioning import prepare_kontext_conditioning
from ....utils.logging import get_logger

logger = get_logger(__name__)


class Flux1Model(BaseDiffusionModel):
    """Complete FLUX.1 model with Transformer, VAE, and text encoders.

    Supports both dev and schnell variants.
    """

    def __init__(self, config: DictConfig, variant: str = "dev"):
        """Initialize FLUX.1 model.

        Args:
            config: Model configuration.
            variant: Model variant ("dev" or "schnell").
        """
        self.variant = variant
        super().__init__(config)

    def _build_model(self) -> None:
        """Build all model components."""
        # Transformer (DiT)
        transformer_config = self.config.get("transformer", {})
        self.transformer = Flux1Transformer(transformer_config, variant=self.variant)

        # VAE
        vae_config = self.config.get("vae", {})
        self.vae = Flux1VAE(vae_config)

        # Text encoders
        text_config = self.config.get("text_encoder", {})
        self.text_encoders = Flux1TextEncoders(text_config)

        # VAE scaling factors
        self.vae_scale_factor = self.vae.scaling_factor
        self.vae_shift_factor = self.vae.shift_factor

        # Whether to use guidance (dev vs schnell)
        self.use_guidance = self.transformer.guidance_embeds

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        img_cond_seq: Optional[torch.Tensor] = None,
        img_cond_seq_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through Transformer.

        Args:
            latents: Latent tensor [B, seq_len, channels] or [B, C, H, W].
            timesteps: Timestep values [B].
            encoder_hidden_states: T5 text embeddings [B, txt_seq, dim].
            pooled_projections: CLIP pooled embeddings [B, pool_dim].
            guidance: Optional guidance scale [B].
            img_cond_seq: Kontext conditioning sequence [B, ref_seq, in_channels].
            img_cond_seq_ids: Position IDs for Kontext conditioning [B, ref_seq, 4].
            **kwargs: Additional arguments.

        Returns:
            Predicted output.
        """
        # Flux expects flattened latents [B, seq_len, channels]
        if latents.dim() == 4:
            B, C, H, W = latents.shape
            latents = latents.view(B, C, H * W).transpose(1, 2)

        return self.transformer(
            hidden_states=latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            guidance=guidance,
            img_cond_seq=img_cond_seq,
            img_cond_seq_ids=img_cond_seq_ids,
        )

    def encode_reference_images(
        self,
        images: torch.Tensor,
        mode: Literal["kontext"] = "kontext",
        patch_size: int = 2,
    ) -> Dict[str, torch.Tensor]:
        """Encode reference images for Kontext conditioning.

        Note: FLUX.1 only supports 'kontext' mode, not 'fill' mode.
        Fill mode (inpainting with channel-wise concatenation) is only
        available in FLUX.2.

        Args:
            images: Reference images [B, 3, H, W] in range [-1, 1].
            mode: Conditioning mode (only 'kontext' is supported).
            patch_size: Patch size for sequence conversion.

        Returns:
            Dictionary with:
            - img_cond_seq: Encoded reference sequence [B, ref_seq, patch_dim]
            - img_cond_seq_ids: Position IDs [B, ref_seq, 4]

        Raises:
            ValueError: If mode is not 'kontext'.
        """
        if mode != "kontext":
            raise ValueError(
                f"FLUX.1 only supports 'kontext' mode, got '{mode}'. "
                "Fill mode is only available in FLUX.2."
            )

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        img_cond_seq, img_cond_seq_ids = prepare_kontext_conditioning(
            reference_images=images,
            vae=self.vae,
            device=device,
            dtype=dtype,
            patch_size=patch_size,
        )

        return {
            "img_cond_seq": img_cond_seq,
            "img_cond_seq_ids": img_cond_seq_ids,
        }

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
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space.

        Args:
            image: Image tensor [B, 3, H, W] in range [-1, 1].

        Returns:
            Latent tensor.
        """
        with torch.no_grad():
            latent = self.vae.encode_to_latent(image)
        return latent

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image.

        Args:
            latent: Latent tensor.

        Returns:
            Image tensor [B, 3, H, W] in range [-1, 1].
        """
        with torch.no_grad():
            image = self.vae.decode_from_latent(latent)
        return image

    def _load_diffusers_checkpoint(self, path: Path) -> None:
        """Load from diffusers directory format.

        Args:
            path: Path to diffusers model directory.
        """
        logger.info(f"Loading FLUX.1 from diffusers format: {path}")

        # Load Transformer
        transformer_path = path / "transformer"
        if transformer_path.exists():
            self._load_transformer_weights(transformer_path)
        else:
            logger.warning(f"Transformer not found at {transformer_path}")

        # Load VAE
        vae_path = path / "vae"
        if vae_path.exists():
            self._load_vae_weights(vae_path)
        else:
            logger.warning(f"VAE not found at {vae_path}")

        # Load text encoders
        self.text_encoders.load_pretrained(path)

    def _load_transformer_weights(self, transformer_path: Path) -> None:
        """Load transformer weights."""
        weights_file = transformer_path / "diffusion_pytorch_model.safetensors"
        if not weights_file.exists():
            weights_file = transformer_path / "diffusion_pytorch_model.bin"

        if weights_file.suffix == ".safetensors":
            state_dict = load_file(weights_file)
        else:
            state_dict = torch.load(weights_file, map_location="cpu")

        missing, unexpected = self.transformer.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Transformer missing keys: {len(missing)}")
        if unexpected:
            logger.warning(f"Transformer unexpected keys: {len(unexpected)}")
        logger.info("Loaded Transformer weights")

    def _load_vae_weights(self, vae_path: Path) -> None:
        """Load VAE weights."""
        weights_file = vae_path / "diffusion_pytorch_model.safetensors"
        if not weights_file.exists():
            weights_file = vae_path / "diffusion_pytorch_model.bin"

        if weights_file.suffix == ".safetensors":
            state_dict = load_file(weights_file)
        else:
            state_dict = torch.load(weights_file, map_location="cpu")

        missing, unexpected = self.vae.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"VAE missing keys: {len(missing)}")
        if unexpected:
            logger.warning(f"VAE unexpected keys: {len(unexpected)}")
        logger.info("Loaded VAE weights")

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing."""
        self.transformer.enable_gradient_checkpointing()
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

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device | str,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Prepare random latents for generation.

        Args:
            batch_size: Batch size.
            height: Image height.
            width: Image width.
            device: Target device.
            dtype: Data type.
            generator: Random generator.

        Returns:
            Random latent tensor.
        """
        # FLUX.1 latent dimensions (16 channels)
        latent_channels = self.vae.latent_channels

        # Calculate latent spatial dimensions
        latent_height = height // 8
        latent_width = width // 8

        shape = (batch_size, latent_channels, latent_height, latent_width)

        latents = torch.randn(
            shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )

        return latents

    def to(
        self,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Move model to device/dtype."""
        if device is not None or dtype is not None:
            self.transformer = self.transformer.to(device=device, dtype=dtype)
            self.vae = self.vae.to(device=device, dtype=dtype)
            self.text_encoders = self.text_encoders.to(device=device, dtype=dtype)
        return self
