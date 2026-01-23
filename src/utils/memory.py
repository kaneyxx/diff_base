"""Memory optimization utilities for diffusion model training."""

import gc
from typing import Any

import torch
import torch.nn as nn

from .logging import get_logger

logger = get_logger(__name__)


def get_memory_stats(device: torch.device | str = "cuda") -> dict[str, float]:
    """Get GPU memory statistics.

    Args:
        device: CUDA device to query.

    Returns:
        Dictionary with memory stats in GB.
    """
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}

    return {
        "allocated": torch.cuda.memory_allocated(device) / 1024**3,
        "reserved": torch.cuda.memory_reserved(device) / 1024**3,
        "max_allocated": torch.cuda.max_memory_allocated(device) / 1024**3,
    }


def log_memory_stats(device: torch.device | str = "cuda", prefix: str = "") -> None:
    """Log current GPU memory usage.

    Args:
        device: CUDA device to query.
        prefix: Prefix for log message.
    """
    stats = get_memory_stats(device)
    logger.info(
        f"{prefix}Memory: {stats['allocated']:.2f}GB allocated, "
        f"{stats['reserved']:.2f}GB reserved, "
        f"{stats['max_allocated']:.2f}GB peak"
    )


def clear_memory(device: torch.device | str = "cuda") -> None:
    """Clear GPU memory and garbage collect.

    Args:
        device: CUDA device to clear.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def enable_gradient_checkpointing(model: nn.Module) -> None:
    """Enable gradient checkpointing for memory efficiency.

    Args:
        model: Model to enable gradient checkpointing on.
    """
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
        logger.info("Enabled gradient checkpointing")
    elif hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing (transformers API)")
    else:
        # Try to enable on submodules
        enabled = False
        for name, module in model.named_modules():
            if hasattr(module, "enable_gradient_checkpointing"):
                module.enable_gradient_checkpointing()
                enabled = True
            elif hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()
                enabled = True

        if enabled:
            logger.info("Enabled gradient checkpointing on submodules")
        else:
            logger.warning("Model does not support gradient checkpointing")


def optimize_memory(
    model: nn.Module,
    config: Any | None = None,
    device: torch.device | str = "cuda",
) -> nn.Module:
    """Apply memory optimizations to model.

    Args:
        model: Model to optimize.
        config: Configuration with optimization settings.
        device: Target device.

    Returns:
        Optimized model.
    """
    if config is None:
        config = {}

    hardware_config = config.get("hardware", {})

    # Gradient checkpointing
    if hardware_config.get("gradient_checkpointing", False):
        enable_gradient_checkpointing(model)

    # Mixed precision dtype
    dtype_str = hardware_config.get("mixed_precision", None)
    if dtype_str:
        dtype = get_dtype(dtype_str)
        model = model.to(dtype=dtype)
        logger.info(f"Model converted to {dtype}")

    # torch.compile
    if hardware_config.get("compile", False):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Model compiled with torch.compile")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    # Enable xformers if available
    if hardware_config.get("xformers", False):
        enable_xformers(model)

    # Move to device
    model = model.to(device)

    return model


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch.dtype.

    Args:
        dtype_str: String like "bf16", "fp16", "fp32".

    Returns:
        Corresponding torch.dtype.
    """
    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }

    if dtype_str not in dtype_map:
        raise ValueError(
            f"Unknown dtype: {dtype_str}. "
            f"Supported: {list(dtype_map.keys())}"
        )

    return dtype_map[dtype_str]


def enable_xformers(model: nn.Module) -> bool:
    """Enable xformers memory efficient attention.

    Args:
        model: Model to enable xformers on.

    Returns:
        Whether xformers was enabled.
    """
    try:
        import xformers
        import xformers.ops

        if hasattr(model, "enable_xformers_memory_efficient_attention"):
            model.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention")
            return True
        else:
            # Try on submodules
            enabled = False
            for module in model.modules():
                if hasattr(module, "enable_xformers_memory_efficient_attention"):
                    module.enable_xformers_memory_efficient_attention()
                    enabled = True

            if enabled:
                logger.info("Enabled xformers on submodules")
            return enabled

    except ImportError:
        logger.warning("xformers not installed")
        return False


def cpu_offload(
    model: nn.Module,
    device: torch.device | str = "cuda",
) -> nn.Module:
    """Enable CPU offloading for large models.

    Args:
        model: Model to offload.
        device: Execution device.

    Returns:
        Model with offloading enabled.
    """
    try:
        from accelerate import cpu_offload as accel_cpu_offload
        model = accel_cpu_offload(model, device)
        logger.info("Enabled CPU offloading")
    except ImportError:
        logger.warning("accelerate not installed, skipping CPU offload")

    return model


def sequential_offload(
    model: nn.Module,
    device: torch.device | str = "cuda",
) -> nn.Module:
    """Enable sequential CPU offloading for minimal memory usage.

    Args:
        model: Model to offload.
        device: Execution device.

    Returns:
        Model with sequential offloading enabled.
    """
    try:
        from accelerate import cpu_offload_with_hook

        hook = None
        for name, module in model.named_children():
            _, hook = cpu_offload_with_hook(module, device, prev_module_hook=hook)

        logger.info("Enabled sequential CPU offloading")
    except ImportError:
        logger.warning("accelerate not installed, skipping sequential offload")

    return model


class MemoryTracker:
    """Context manager to track memory usage."""

    def __init__(self, device: torch.device | str = "cuda", name: str = ""):
        self.device = device
        self.name = name
        self.start_memory = 0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_memory = torch.cuda.memory_allocated(self.device)
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_memory = torch.cuda.memory_allocated(self.device)
            delta = (end_memory - self.start_memory) / 1024**3
            prefix = f"{self.name}: " if self.name else ""
            logger.info(f"{prefix}Memory delta: {delta:+.3f}GB")


def estimate_model_memory(model: nn.Module, dtype: torch.dtype = torch.float32) -> float:
    """Estimate model memory usage in GB.

    Args:
        model: Model to estimate.
        dtype: Data type for calculation.

    Returns:
        Estimated memory in GB.
    """
    param_count = sum(p.numel() for p in model.parameters())
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
    }.get(dtype, 4)

    # Parameters + gradients + optimizer states (assuming AdamW)
    # AdamW: 2 state tensors per param (momentum, variance)
    training_multiplier = 1 + 1 + 2  # params + grads + optimizer

    total_bytes = param_count * bytes_per_param * training_multiplier
    return total_bytes / 1024**3
