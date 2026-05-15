"""Training infrastructure for diffusion models."""

from typing import TYPE_CHECKING

from omegaconf import DictConfig

from .base_trainer import BaseTrainer

if TYPE_CHECKING:
    from .controlnet_trainer import ControlNetTrainer  # noqa: F401
    from .dreambooth_trainer import DreamBoothTrainer  # noqa: F401
    from .finetune_trainer import FullFinetuneTrainer  # noqa: F401
    from .flux_full_finetune_trainer import FluxFullFinetuneTrainer  # noqa: F401
    from .kontext_trainer import (  # noqa: F401
        KontextFullFinetuneTrainer,
        KontextLoRATrainer,
    )
    from .lora_trainer import LoRATrainer  # noqa: F401


TRAINER_REGISTRY = {
    "lora": "LoRATrainer",
    "full_finetune": "FullFinetuneTrainer",
    "flux_full_finetune": "FluxFullFinetuneTrainer",
    "dreambooth": "DreamBoothTrainer",
    "controlnet": "ControlNetTrainer",
    "textual_inversion": "TextualInversionTrainer",
    "kontext_lora": "KontextLoRATrainer",
    "kontext_finetune": "KontextFullFinetuneTrainer",
}


def create_trainer(config: DictConfig) -> BaseTrainer:
    """Factory function to create trainer from configuration.

    Args:
        config: Full training configuration.

    Returns:
        Instantiated trainer.

    Raises:
        ValueError: If training method is unknown.
    """
    method = config.training.method

    if method not in TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown training method: {method}. "
            f"Available: {list(TRAINER_REGISTRY.keys())}"
        )

    trainer_name = TRAINER_REGISTRY[method]

    # Import trainer class
    if method == "lora":
        from .lora_trainer import LoRATrainer
        return LoRATrainer(config)
    elif method == "full_finetune":
        from .finetune_trainer import FullFinetuneTrainer
        return FullFinetuneTrainer(config)
    elif method == "dreambooth":
        from .dreambooth_trainer import DreamBoothTrainer
        return DreamBoothTrainer(config)
    elif method == "controlnet":
        from .controlnet_trainer import ControlNetTrainer
        return ControlNetTrainer(config)
    elif method == "textual_inversion":
        from .textual_inversion_trainer import TextualInversionTrainer
        return TextualInversionTrainer(config)
    elif method == "flux_full_finetune":
        from .flux_full_finetune_trainer import FluxFullFinetuneTrainer
        return FluxFullFinetuneTrainer(config)
    elif method == "kontext_lora":
        from .kontext_trainer import KontextLoRATrainer
        return KontextLoRATrainer(config)
    elif method == "kontext_finetune":
        from .kontext_trainer import KontextFullFinetuneTrainer
        return KontextFullFinetuneTrainer(config)
    else:
        raise ValueError(f"Trainer not implemented: {trainer_name}")


__all__ = [
    "BaseTrainer",
    "create_trainer",
    "FluxFullFinetuneTrainer",
    "KontextLoRATrainer",
    "KontextFullFinetuneTrainer",
]
