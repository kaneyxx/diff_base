"""Flux model architecture with support for FLUX.1 and FLUX.2 variants.

Variant Registry:
- FLUX.1 (v1): dev, schnell
  - T5-XXL + CLIP-L text encoders
  - 16 latent channels
  - 19 joint + 38 single blocks

- FLUX.2 (v2): dev, klein-4b, klein-9b
  - Mistral/Qwen text encoders
  - 32 or 128 latent channels
  - Variable block counts
"""

from typing import TYPE_CHECKING

from omegaconf import DictConfig

if TYPE_CHECKING:
    from .v1 import Flux1Model
    from .v2 import Flux2Model
    from ..base import BaseDiffusionModel

# Registry of all supported Flux variants
# Format: variant_name -> (version, variant_key)
FLUX_VARIANTS = {
    # FLUX.1 variants
    "flux1-dev": ("v1", "dev"),
    "flux1-schnell": ("v1", "schnell"),
    "flux.1-dev": ("v1", "dev"),  # Alias
    "flux.1-schnell": ("v1", "schnell"),  # Alias
    "dev": ("v1", "dev"),  # Short alias
    "schnell": ("v1", "schnell"),  # Short alias

    # FLUX.2 variants
    "flux2-dev": ("v2", "dev"),
    "flux2-klein-4b": ("v2", "klein-4b"),
    "flux2-klein-9b": ("v2", "klein-9b"),
    "flux.2-dev": ("v2", "dev"),  # Alias
    "flux.2-klein-4b": ("v2", "klein-4b"),  # Alias
    "flux.2-klein-9b": ("v2", "klein-9b"),  # Alias
    "klein-4b": ("v2", "klein-4b"),  # Short alias
    "klein-9b": ("v2", "klein-9b"),  # Short alias
}


def create_flux_model(config: DictConfig) -> "BaseDiffusionModel":
    """Factory function to create appropriate Flux model variant.

    Args:
        config: Full model configuration. Must contain:
            - model.variant: Variant name (e.g., "flux1-dev", "flux2-klein-4b")
            OR
            - model.version: "v1" or "v2"
            - model.variant: variant within version (e.g., "dev", "schnell")

    Returns:
        Instantiated Flux model of the appropriate variant.

    Raises:
        ValueError: If variant is not recognized.
    """
    # Get variant from config
    variant = config.get("variant", "dev")
    version = config.get("version", None)

    # Normalize variant name
    variant_lower = variant.lower().replace("_", "-")

    if variant_lower in FLUX_VARIANTS:
        version, variant_key = FLUX_VARIANTS[variant_lower]
    elif version is not None:
        # Version explicitly specified
        variant_key = variant_lower
    else:
        # Default to v1 dev for backwards compatibility
        version = "v1"
        variant_key = "dev"

    if version == "v1":
        from .v1 import Flux1Model
        return Flux1Model(config, variant=variant_key)
    elif version == "v2":
        from .v2 import Flux2Model
        return Flux2Model(config, variant=variant_key)
    else:
        raise ValueError(
            f"Unknown Flux version: {version}. "
            f"Supported: 'v1' (FLUX.1), 'v2' (FLUX.2)"
        )


def create_flux_transformer(
    version: str,
    config: DictConfig,
    variant: str = "dev",
) -> "FluxTransformerBase":
    """Create a FLUX transformer by version.

    This is the recommended way to instantiate transformers for training code.
    Returns a FluxTransformerBase subclass, so training code can use the
    unified forward() signature without caring about the FLUX version.

    Args:
        version: "v1" or "v2".
        config: Transformer configuration (DictConfig or dict-like).
        variant: Model variant ("dev", "schnell", "klein-4b", "klein-9b").

    Returns:
        FluxTransformerBase subclass instance.

    Raises:
        ValueError: If version is not recognized.
    """
    if version == "v1":
        from .v1.transformer import Flux1Transformer
        return Flux1Transformer(config, variant)
    elif version == "v2":
        from .v2.transformer import Flux2Transformer
        return Flux2Transformer(config, variant)
    else:
        raise ValueError(
            f"Unknown FLUX version: {version}. Supported: 'v1' (FLUX.1), 'v2' (FLUX.2)"
        )


def get_available_variants() -> dict[str, tuple[str, str]]:
    """Get all available Flux variants.

    Returns:
        Dictionary mapping variant names to (version, variant_key) tuples.
    """
    return FLUX_VARIANTS.copy()


# Backwards compatibility - expose old names
from .base_transformer import FluxTransformerBase
from .v1 import Flux1Model, Flux1Transformer, Flux1VAE, Flux1TextEncoders
from .v2 import Flux2Model, Flux2Transformer, Flux2VAE, Flux2TextEncoders

# Legacy aliases for backwards compatibility
FluxModel = Flux1Model
FluxTransformer = Flux1Transformer
FluxVAE = Flux1VAE
FluxTextEncoders = Flux1TextEncoders

__all__ = [
    # Factory
    "create_flux_model",
    "create_flux_transformer",
    "get_available_variants",
    "FLUX_VARIANTS",
    # Base interface
    "FluxTransformerBase",
    # FLUX.1 (v1)
    "Flux1Model",
    "Flux1Transformer",
    "Flux1VAE",
    "Flux1TextEncoders",
    # FLUX.2 (v2)
    "Flux2Model",
    "Flux2Transformer",
    "Flux2VAE",
    "Flux2TextEncoders",
    # Legacy aliases
    "FluxModel",
    "FluxTransformer",
    "FluxVAE",
    "FluxTextEncoders",
]
