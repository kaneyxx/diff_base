"""SD3 (Stable Diffusion 3.5) model implementations.

SD3.5 uses the MM-DiT (Multimodal Diffusion Transformer) architecture, which is
fundamentally different from:
- SDXL: UNet-based
- FLUX: DiT with joint/single blocks

Supported variants:
- SD3.5-Large: 38 depth, 2432 hidden_size, ~8B params
- SD3.5-Large-Turbo: Same architecture as Large, tuned for fast inference
- SD3.5-Medium: 24 depth, 1536 hidden_size, ~2.5B params

Key architectural differences from other models:
- Triple text encoder: CLIP-L + OpenCLIP-G + T5-XXL
- QK normalization for numerical stability
- JointTransformerBlock for multimodal processing
- 16 latent channels (same as FLUX)
- Patch size of 2 (vs 1 for FLUX)
"""

from typing import Dict, Tuple

from omegaconf import DictConfig

from .model import SD3Model
from .mmdit import SD3Transformer, SD3_VARIANT_CONFIGS
from .vae import SD3VAE
from .text_encoder import SD3TextEncoders


# Variant mapping: user-facing names -> (variant_key)
VARIANT_ALIASES: Dict[str, str] = {
    # SD3.5-Large
    "large": "large",
    "sd3-large": "large",
    "sd3.5-large": "large",
    "sd35-large": "large",
    # SD3.5-Large-Turbo
    "large-turbo": "large-turbo",
    "sd3-large-turbo": "large-turbo",
    "sd3.5-large-turbo": "large-turbo",
    "sd35-large-turbo": "large-turbo",
    "turbo": "large-turbo",
    # SD3.5-Medium
    "medium": "medium",
    "sd3-medium": "medium",
    "sd3.5-medium": "medium",
    "sd35-medium": "medium",
}


def create_sd3_model(config: DictConfig) -> SD3Model:
    """Factory function to create SD3 model from config.

    Args:
        config: Model configuration containing:
            - variant: Model variant (large, large-turbo, medium)
            - transformer: Transformer configuration
            - vae: VAE configuration
            - text_encoder: Text encoder configuration
            - pretrained_path: Optional path to pretrained weights

    Returns:
        SD3Model instance.

    Raises:
        ValueError: If variant is not recognized.

    Examples:
        >>> config = load_config("configs/models/sd3_large.yaml")
        >>> model = create_sd3_model(config.model)
    """
    # Resolve variant alias
    variant = config.get("variant", "large")
    variant = VARIANT_ALIASES.get(variant, variant)

    if variant not in SD3_VARIANT_CONFIGS:
        available = list(SD3_VARIANT_CONFIGS.keys())
        raise ValueError(
            f"Unknown SD3 variant: {variant}. Available: {available}"
        )

    # Update config with resolved variant
    config = config.copy()
    config["variant"] = variant

    return SD3Model(config)


def get_available_variants() -> Dict[str, Dict]:
    """Get all available SD3 variants and their configurations.

    Returns:
        Dictionary mapping variant names to their configurations.
    """
    return SD3_VARIANT_CONFIGS.copy()


def get_variant_info(variant: str) -> Dict:
    """Get configuration info for a specific variant.

    Args:
        variant: Variant name or alias.

    Returns:
        Variant configuration dictionary.

    Raises:
        ValueError: If variant is not recognized.
    """
    variant = VARIANT_ALIASES.get(variant, variant)
    if variant not in SD3_VARIANT_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}")
    return SD3_VARIANT_CONFIGS[variant]


__all__ = [
    # Main exports
    "SD3Model",
    "create_sd3_model",
    # Components
    "SD3Transformer",
    "SD3VAE",
    "SD3TextEncoders",
    # Utilities
    "get_available_variants",
    "get_variant_info",
    "SD3_VARIANT_CONFIGS",
    "VARIANT_ALIASES",
]
