"""Training infrastructure for diffusion models."""

from typing import TYPE_CHECKING

from omegaconf import DictConfig

from .base_trainer import BaseTrainer

if TYPE_CHECKING:
    from .lora_trainer import LoRATrainer
    from .finetune_trainer import FullFinetuneTrainer
    from .dreambooth_trainer import DreamBoothTrainer
    from .controlnet_trainer import ControlNetTrainer


TRAINER_REGISTRY = {
    "lora": "LoRATrainer",
    "full_finetune": "FullFinetuneTrainer",
    "dreambooth": "DreamBoothTrainer",
    "controlnet": "ControlNetTrainer",
    "textual_inversion": "TextualInversionTrainer",
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
    else:
        raise ValueError(f"Trainer not implemented: {trainer_name}")


__all__ = [
    "BaseTrainer",
    "create_trainer",
]
