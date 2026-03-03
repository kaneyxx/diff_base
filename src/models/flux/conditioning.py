"""Unified conditioning interface for all FLUX variants.

This module provides version-agnostic conditioning utilities. Training code
should import from here rather than from v1/ or v2/ directly.

Usage:
    from diff_base.src.models.flux.conditioning import (
        create_position_ids,
        rearrange_latent_to_sequence,
        rearrange_sequence_to_latent,
    )

    # Version-aware position IDs
    ids = create_position_ids(
        version="v1",
        batch_size=B, height=h, width=w,
        device=device, dtype=dtype,
    )
"""

from typing import Optional

import torch

# Re-export shared utilities that work across all versions
from .v2.conditioning import (
    rearrange_latent_to_sequence,
    rearrange_sequence_to_latent,
)


def create_position_ids(
    version: str,
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    time_offset: float = 0.0,
) -> torch.Tensor:
    """Create position IDs appropriate for the model version.

    Args:
        version: "v1" or "v2".
        batch_size: Batch size.
        height: Spatial height in patches.
        width: Spatial width in patches.
        device: Target device.
        dtype: Data type.
        time_offset: Stream/time index (0.0 for target, 1.0 for reference).

    Returns:
        Position IDs tensor. Shape [B, height*width, 3] for v1,
        [B, height*width, 3] for v2 (same format currently).
    """
    if version == "v1":
        from .v1.conditioning import create_position_ids as _create
    elif version == "v2":
        from .v2.conditioning import create_position_ids as _create
    else:
        raise ValueError(f"Unknown FLUX version: {version}")

    return _create(
        batch_size=batch_size,
        height=height,
        width=width,
        device=device,
        dtype=dtype,
        time_offset=time_offset,
    )


__all__ = [
    "create_position_ids",
    "rearrange_latent_to_sequence",
    "rearrange_sequence_to_latent",
]
