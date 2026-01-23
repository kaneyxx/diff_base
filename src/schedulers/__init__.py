"""Noise schedulers for diffusion training and inference."""

from typing import TYPE_CHECKING

from omegaconf import DictConfig

from .ddpm import DDPMScheduler
from .euler import EulerDiscreteScheduler
from .flow_matching import FlowMatchingScheduler

if TYPE_CHECKING:
    from .base import BaseScheduler


SCHEDULER_REGISTRY = {
    "ddpm": DDPMScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_discrete": EulerDiscreteScheduler,
    "flow_matching": FlowMatchingScheduler,
}


def create_scheduler(config: DictConfig) -> "BaseScheduler":
    """Create scheduler from configuration.

    Args:
        config: Scheduler configuration.

    Returns:
        Instantiated scheduler.
    """
    scheduler_type = config.get("type", "ddpm")

    if scheduler_type not in SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Available: {list(SCHEDULER_REGISTRY.keys())}"
        )

    scheduler_cls = SCHEDULER_REGISTRY[scheduler_type]
    return scheduler_cls(config)


__all__ = [
    "DDPMScheduler",
    "EulerDiscreteScheduler",
    "FlowMatchingScheduler",
    "create_scheduler",
]
