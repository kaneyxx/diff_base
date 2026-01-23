"""Model architectures for diffusion training."""

from typing import TYPE_CHECKING

from omegaconf import DictConfig

from .base import BaseDiffusionModel

if TYPE_CHECKING:
    from .sdxl import SDXLModel
    from .flux import FluxModel, Flux1Model, Flux2Model


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
    """
    model_type = config.model.type

    if model_type == "sdxl":
        from .sdxl import SDXLModel
        model = SDXLModel(config.model)
    elif model_type == "flux":
        # Use the Flux factory which handles all variants
        from .flux import create_flux_model
        model = create_flux_model(config.model)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: ['sdxl', 'flux']"
        )

    # Load pretrained weights if specified
    if hasattr(config.model, "pretrained_path") and config.model.pretrained_path:
        model.load_pretrained(config.model.pretrained_path)

    return model


def get_available_flux_variants() -> dict[str, tuple[str, str]]:
    """Get all available Flux model variants.

    Returns:
        Dictionary mapping variant names to (version, variant_key) tuples.
        - version: "v1" for FLUX.1, "v2" for FLUX.2
        - variant_key: specific variant (e.g., "dev", "schnell", "klein-4b")
    """
    from .flux import get_available_variants
    return get_available_variants()


__all__ = [
    "BaseDiffusionModel",
    "create_model",
    "get_available_flux_variants",
]
