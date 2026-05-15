"""Model architectures for diffusion training."""

from typing import TYPE_CHECKING

from omegaconf import DictConfig

from .base import BaseDiffusionModel

if TYPE_CHECKING:
    from .flux import Flux1Model, Flux2Model, FluxModel
    from .sd3 import SD3Model
    from .sdxl import SDXLModel


def create_model(config: DictConfig) -> BaseDiffusionModel:
    """Factory function to create a model from configuration.

    Args:
        config: Full training configuration with model section.

    Returns:
        Instantiated model.

    Raises:
        ValueError: If model type is unknown.

    Examples:
        # SDXL
        config.model.type = "sdxl"

        # FLUX.1 dev (default)
        config.model.type = "flux"
        config.model.variant = "dev"  # or "flux1-dev"

        # FLUX.1 schnell
        config.model.type = "flux"
        config.model.variant = "schnell"  # or "flux1-schnell"

        # FLUX.2 dev
        config.model.type = "flux"
        config.model.variant = "flux2-dev"

        # FLUX.2 klein-4b
        config.model.type = "flux"
        config.model.variant = "flux2-klein-4b"  # or "klein-4b"

        # SD3.5-Large
        config.model.type = "sd3"
        config.model.variant = "large"

        # SD3.5-Large-Turbo
        config.model.type = "sd3"
        config.model.variant = "large-turbo"

        # SD3.5-Medium
        config.model.type = "sd3"
        config.model.variant = "medium"
    """
    model_type = config.model.type

    if model_type == "sdxl":
        from .sdxl import SDXLModel
        model = SDXLModel(config.model)
    elif model_type in ("flux", "flux2"):
        # Use the Flux factory which handles all variants (v1 and v2)
        from .flux import create_flux_model
        model = create_flux_model(config.model)
    elif model_type == "sd3":
        # Use the SD3 factory which handles all variants
        from .sd3 import create_sd3_model
        model = create_sd3_model(config.model)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: ['sdxl', 'flux', 'sd3']"
        )

    import os
    if os.environ.get("FLUX_TINY_OVERRIDE") == "1" and model_type in ("flux", "flux2"):
        # Swap to a tiny model for smoke tests — bypasses weight loading,
        # installs identity VAE and random-embedding text encoder stubs.
        _apply_tiny_override(model)
    elif hasattr(config.model, "pretrained_path") and config.model.pretrained_path:
        model.load_pretrained(config.model.pretrained_path)

    return model


def _apply_tiny_override(model: "BaseDiffusionModel") -> None:
    """Replace model components with tiny stubs for smoke testing.

    When ``FLUX_TINY_OVERRIDE=1`` is set:
    - Replaces ``model.transformer`` with a tiny 1-block version (hidden=64).
    - Replaces ``model.vae`` with an identity stub (no-op encode/decode).
    - Replaces ``model.text_encoders`` with a random-embedding stub.

    This allows the full CLI and trainer code paths to execute without real
    weights or GPU memory, purely for integration testing (AC10).

    Args:
        model: FLUX model instance to patch in-place.
    """
    import torch
    import torch.nn as nn

    model_type = getattr(model, "variant", "dev")
    is_v2 = hasattr(model, "transformer") and type(model.transformer).__name__.startswith("Flux2")

    # --- Tiny transformer ---
    if hasattr(model, "transformer"):
        orig = model.transformer
        if is_v2:
            from .flux.v2.transformer import Flux2Transformer
            from omegaconf import OmegaConf
            # Identity VAE returns 32 latent channels; patch_size=2 → in_channels=32*4=128.
            # head_dim = 64//1 = 64 = sum([16,16,16,16]) — axes_dims_rope must sum to head_dim.
            tiny_cfg = OmegaConf.create({
                "hidden_size": 64,
                "num_attention_heads": 1,
                "num_layers": 1,
                "num_single_layers": 1,
                "in_channels": 128,  # 32 latent_ch * 4 (2x2 patch), matches identity VAE
                "guidance_embeds": True,
                "axes_dims_rope": [16, 16, 16, 16],
                "rope_theta": 2000,
            })
            model.transformer = Flux2Transformer(tiny_cfg)
        else:
            from .flux.v1.transformer import Flux1Transformer
            from omegaconf import OmegaConf
            # head_dim = hidden_size // num_heads must equal sum(FLUX1_AXES_DIM) = 16+56+56 = 128
            tiny_cfg = OmegaConf.create({
                "hidden_size": 128,
                "num_attention_heads": 1,
                "num_layers": 1,
                "num_single_layers": 1,
                "in_channels": 64,  # 16 latent_ch * 4 (2x2 patch)
                "guidance_embeds": True,
            })
            model.transformer = Flux1Transformer(tiny_cfg, variant="dev")
        model.use_guidance = model.transformer.guidance_embeds

    # --- Identity VAE stub ---
    class _IdentityVAE(nn.Module):
        """No-op VAE: encode returns a tiny fixed-shape latent, decode is identity."""

        def __init__(self, latent_channels: int = 16) -> None:
            super().__init__()
            self.scaling_factor = 1.0
            self.shift_factor = 0.0
            self.latent_channels = latent_channels

        def encode_to_latent(self, image: "torch.Tensor") -> "torch.Tensor":
            B, _, H, W = image.shape
            # Downsample 8x to match real VAE spatial factor
            return torch.zeros(
                B, self.latent_channels, H // 8, W // 8,
                device=image.device, dtype=image.dtype
            )

        def decode_from_latent(self, latent: "torch.Tensor") -> "torch.Tensor":
            B, _, H, W = latent.shape
            return torch.zeros(B, 3, H * 8, W * 8, device=latent.device, dtype=latent.dtype)

    latent_ch = 32 if is_v2 else 16
    model.vae = _IdentityVAE(latent_channels=latent_ch)
    if hasattr(model, "vae_scale_factor"):
        model.vae_scale_factor = 1.0
    if hasattr(model, "vae_shift_factor"):
        model.vae_shift_factor = 0.0

    # --- Random text encoder stub ---
    class _RandomTextEncoders(nn.Module):
        """Returns random embeddings of the correct shape for FLUX.1 (T5+CLIP)."""

        def __init__(self, seq_dim: int = 4096, pool_dim: int = 768) -> None:
            super().__init__()
            self.seq_dim = seq_dim
            self.pool_dim = pool_dim

        def encode(
            self,
            prompt: "str | list[str]",
            device: "torch.device | str" = "cpu",
            **kwargs,
        ) -> "dict[str, torch.Tensor]":
            if isinstance(prompt, str):
                prompt = [prompt]
            B = len(prompt)
            return {
                "prompt_embeds": torch.randn(B, 77, self.seq_dim, device=device),
                "pooled_prompt_embeds": torch.randn(B, self.pool_dim, device=device),
            }

        def freeze(self) -> None:
            pass

    # FLUX.2 uses Mistral/Qwen with 4096-dim pooled; FLUX.1 uses CLIP-L (768)
    pool_dim = 4096 if is_v2 else 768
    model.text_encoders = _RandomTextEncoders(pool_dim=pool_dim)


def get_available_flux_variants() -> dict[str, tuple[str, str]]:
    """Get all available Flux model variants.

    Returns:
        Dictionary mapping variant names to (version, variant_key) tuples.
        - version: "v1" for FLUX.1, "v2" for FLUX.2
        - variant_key: specific variant (e.g., "dev", "schnell", "klein-4b")
    """
    from .flux import get_available_variants
    return get_available_variants()


def get_available_sd3_variants() -> dict[str, dict]:
    """Get all available SD3 model variants.

    Returns:
        Dictionary mapping variant names to configuration dictionaries.
        Available variants: "large", "large-turbo", "medium"
    """
    from .sd3 import get_available_variants
    return get_available_variants()


__all__ = [
    "BaseDiffusionModel",
    "create_model",
    "get_available_flux_variants",
    "get_available_sd3_variants",
]
