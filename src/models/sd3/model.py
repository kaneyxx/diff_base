"""SD3 Model - combines MM-DiT Transformer, VAE, and Triple Text Encoders.

This is the main entry point for SD3.5 models (Large, Large-Turbo, Medium).
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from safetensors.torch import load_file

from ..base import BaseDiffusionModel
from .mmdit import SD3Transformer, SD3_VARIANT_CONFIGS
from .vae import SD3VAE
from .text_encoder import SD3TextEncoders
from ...utils.logging import get_logger

logger = get_logger(__name__)


class SD3Model(BaseDiffusionModel):
    """Complete SD3 model with MM-DiT Transformer, VAE, and text encoders.

    Supports three variants:
    - SD3.5-Large: 38 depth, 2432 hidden, ~8B params
    - SD3.5-Large-Turbo: Same as Large but tuned for faster inference
    - SD3.5-Medium: 24 depth, 1536 hidden, ~2.5B params
    """

    def __init__(self, config: DictConfig):
        """Initialize SD3 model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)

    def _build_model(self) -> None:
        """Build all model components from config."""
        # Determine variant
        self.variant = self.config.get("variant", "large")
        if self.variant not in SD3_VARIANT_CONFIGS:
            logger.warning(
                f"Unknown variant '{self.variant}', defaulting to 'large'"
            )
            self.variant = "large"

        logger.info(f"Building SD3 model with variant: {self.variant}")

        # Build Transformer (MM-DiT)
        transformer_config = self.config.get("transformer", {})
        self.transformer = SD3Transformer(transformer_config, variant=self.variant)

        # Build VAE
        vae_config = self.config.get("vae", {})
        self.vae = SD3VAE(vae_config)

        # Build Text Encoders
        text_config = self.config.get("text_encoder", {})
        self.text_encoders = SD3TextEncoders(text_config)

        # Store scaling factors
        self.vae_scale_factor = self.vae.scaling_factor
        self.vae_shift_factor = self.vae.shift_factor

        # Patch size for latent calculations
        self.patch_size = self.transformer.patch_size

        logger.info(
            f"SD3 model built: transformer depth={self.transformer.depth}, "
            f"hidden_size={self.transformer.hidden_size}"
        )

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through the MM-DiT Transformer.

        Args:
            latents: Latent tensor [B, 16, H, W].
            timesteps: Timestep values [B] (float for flow matching).
            encoder_hidden_states: T5 text embeddings [B, seq_len, 4096].
            pooled_projections: Pooled CLIP embeddings [B, 2048].
            **kwargs: Additional arguments (unused).

        Returns:
            Predicted noise/velocity [B, 16, H, W].
        """
        if pooled_projections is None:
            raise ValueError(
                "SD3 requires pooled_projections from CLIP encoders"
            )

        return self.transformer(
            hidden_states=latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
        )

    def encode_text(
        self,
        text: str | list[str],
        device: torch.device | str = "cuda",
    ) -> dict[str, torch.Tensor]:
        """Encode text prompts using triple text encoders.

        Args:
            text: Single prompt or list of prompts.
            device: Target device.

        Returns:
            Dictionary with:
                - prompt_embeds: T5 embeddings [B, seq_len, 4096]
                - pooled_prompt_embeds: CLIP pooled [B, 2048]
        """
        return self.text_encoders.encode(text, device=device)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to latent space using VAE.

        Args:
            image: Image tensor [B, 3, H, W] in range [-1, 1].

        Returns:
            Latent tensor [B, 16, h, w].
        """
        with torch.no_grad():
            latent = self.vae.encode_to_latent(image)
        return latent

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image using VAE.

        Args:
            latent: Latent tensor [B, 16, h, w].

        Returns:
            Decoded image [B, 3, H, W] in range [-1, 1].
        """
        with torch.no_grad():
            image = self.vae.decode_from_latent(latent)
        return image

    def _load_diffusers_checkpoint(self, path: Path) -> None:
        """Load from diffusers directory format.

        Args:
            path: Path to diffusers model directory.
        """
        logger.info(f"Loading SD3 from diffusers format: {path}")

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
        """Load transformer weights from path.

        Args:
            transformer_path: Path to transformer directory or weights file.
        """
        # Try safetensors first, then bin
        weights_file = transformer_path / "diffusion_pytorch_model.safetensors"
        if not weights_file.exists():
            weights_file = transformer_path / "diffusion_pytorch_model.bin"

        if not weights_file.exists():
            # Check for sharded weights
            weights_file = transformer_path / "diffusion_pytorch_model-00001-of-00002.safetensors"

        if weights_file.suffix == ".safetensors":
            state_dict = load_file(weights_file)
        else:
            state_dict = torch.load(weights_file, map_location="cpu")

        # Map keys if needed
        state_dict = self._map_transformer_keys(state_dict)

        missing, unexpected = self.transformer.load_state_dict(state_dict, strict=False)

        if missing:
            logger.warning(f"Transformer missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logger.warning(f"Transformer unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

        logger.info(f"Loaded transformer weights from {weights_file}")

    def _map_transformer_keys(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Map diffusers transformer keys to local model keys.

        Args:
            state_dict: Original state dict.

        Returns:
            Mapped state dict.
        """
        # Add any key mapping here if needed
        # SD3 diffusers format should be fairly close to our local definition
        mapped = {}

        for key, value in state_dict.items():
            # Example mappings (adjust based on actual diffusers naming)
            new_key = key

            # Handle common diffusers prefixes
            if key.startswith("model."):
                new_key = key[6:]  # Remove "model." prefix

            mapped[new_key] = value

        return mapped

    def _load_vae_weights(self, vae_path: Path) -> None:
        """Load VAE weights from path.

        Args:
            vae_path: Path to VAE directory.
        """
        weights_file = vae_path / "diffusion_pytorch_model.safetensors"
        if not weights_file.exists():
            weights_file = vae_path / "diffusion_pytorch_model.bin"

        if weights_file.suffix == ".safetensors":
            state_dict = load_file(weights_file)
        else:
            state_dict = torch.load(weights_file, map_location="cpu")

        missing, unexpected = self.vae.load_state_dict(state_dict, strict=False)

        if missing:
            logger.warning(f"VAE missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            logger.warning(f"VAE unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

        logger.info(f"Loaded VAE weights from {weights_file}")

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        self.transformer.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled")

    def freeze_vae(self) -> None:
        """Freeze VAE parameters."""
        for param in self.vae.parameters():
            param.requires_grad = False
        logger.info("VAE frozen")

    def freeze_text_encoders(self) -> None:
        """Freeze text encoder parameters."""
        self.text_encoders.freeze()

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
            height: Image height in pixels.
            width: Image width in pixels.
            device: Target device.
            dtype: Data type.
            generator: Random generator for reproducibility.

        Returns:
            Random latent tensor [B, 16, h, w].
        """
        latent_height = height // 8
        latent_width = width // 8

        shape = (batch_size, 16, latent_height, latent_width)

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
        """Move model to device/dtype.

        Args:
            device: Target device.
            dtype: Target dtype.

        Returns:
            Self for chaining.
        """
        if device is not None or dtype is not None:
            self.transformer = self.transformer.to(device=device, dtype=dtype)
            self.vae = self.vae.to(device=device, dtype=dtype)
            self.text_encoders = self.text_encoders.to(device=device, dtype=dtype)
        return self

    def get_param_count(self) -> dict:
        """Get parameter counts for all components.

        Returns:
            Dictionary with parameter counts.
        """
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        vae_params = sum(p.numel() for p in self.vae.parameters())
        text_params = self.text_encoders.get_param_count().get("total", 0)

        return {
            "transformer": transformer_params,
            "vae": vae_params,
            "text_encoders": text_params,
            "total": transformer_params + vae_params + text_params,
            "transformer_millions": transformer_params / 1e6,
            "total_billions": (transformer_params + vae_params + text_params) / 1e9,
        }
