"""Model architectures for diffusion training."""

from typing import TYPE_CHECKING

from omegaconf import DictConfig

from .base import BaseDiffusionModel

if TYPE_CHECKING:
    from .sdxl import SDXLModel
    from .flux import FluxModel


def create_model(config: DictConfig) -> BaseDiffusionModel:
    """Factory function to create a model from configuration.

    Args:
        config: Full training configuration with model section.

    Returns:
        Instantiated model.

    Raises:
        ValueError: If model type is unknown.
    """
    model_type = config.model.type

    if model_type == "sdxl":
        from .sdxl import SDXLModel
        model = SDXLModel(config.model)
    elif model_type == "flux":
        from .flux import FluxModel
        model = FluxModel(config.model)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: ['sdxl', 'flux']"
        )

    # Load pretrained weights if specified
    if hasattr(config.model, "pretrained_path") and config.model.pretrained_path:
        model.load_pretrained(config.model.pretrained_path)

    return model


__all__ = [
    "BaseDiffusionModel",
    "create_model",
]
