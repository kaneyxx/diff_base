"""Flux-specific embedding implementations.

This module provides embedding utilities for FLUX models including:
- Positional embeddings (RoPE with per-axis frequencies)
- 3D position ID to RoPE conversion for Kontext mode

RoPE implementation matches HuggingFace diffusers:
- axes_dim=(16, 56, 56) for FLUX.1 (stream, height, width)
- axes_dim=(32, 32, 32, 32) for FLUX.2 (4D position IDs)
- Uses repeat_interleave for adjacent-pair duplication
- Position IDs are 3D [stream, h, w] for FLUX.1 (not 4D)

Shared utilities (get_timestep_embedding, MLPEmbedder) live in
src/models/components/embeddings.py to avoid duplication.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Re-export shared utilities for backward compatibility
from ...components.embeddings import get_timestep_embedding, MLPEmbedder


class FluxPosEmbed(nn.Module):
    """Positional embedding for Flux models.

    Matches HuggingFace FluxPosEmbed exactly:
    - Takes position IDs tensor [seq, n_axes] or [B, seq, n_axes]
    - Computes per-axis RoPE with repeat_interleave_real=True
    - Concatenates cos/sin across axes
    """

    def __init__(
        self,
        theta: float = 10000.0,
        axes_dim: Tuple[int, ...] = (16, 56, 56),
    ):
        """Initialize FluxPosEmbed.

        Args:
            theta: Base for frequency computation.
            axes_dim: Per-axis dimension allocation (default FLUX.1: 16+56+56=128).
        """
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(
        self,
        ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE from position IDs.

        Matches HuggingFace: for each axis i, compute
        get_1d_rotary_pos_embed(axes_dim[i], ids[:, i], repeat_interleave_real=True)
        then concatenate.

        Args:
            ids: Position IDs [seq, n_axes] or [B, seq, n_axes].

        Returns:
            Tuple of (cos, sin) embeddings.
        """
        return compute_rope_from_position_ids(
            ids, sum(self.axes_dim), self.theta, axes_dim=self.axes_dim
        )


def compute_axis_freqs(
    positions: torch.Tensor,
    dim: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute rotary cos/sin for a single axis.

    Matches HuggingFace get_1d_rotary_pos_embed with repeat_interleave_real=True:
    - Computes dim//2 base frequencies
    - repeat_interleave(2) to get dim values where each freq is adjacent-duplicated

    Args:
        positions: Position values [seq] or [B, seq].
        dim: Number of output dimensions (must be even).
        theta: Base for frequency computation.

    Returns:
        Tuple of (cos, sin) each of shape matching positions + [dim].
    """
    if dim == 0:
        shape = list(positions.shape) + [0]
        z = torch.zeros(shape, device=positions.device, dtype=torch.float32)
        return z, z

    # Compute inverse frequencies for half the dimensions
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=positions.device).float() / dim)
    )

    # Compute outer product: positions x inv_freq
    if positions.dim() == 1:
        # [seq] x [dim//2] -> [seq, dim//2]
        freqs = torch.outer(positions.float(), inv_freq)
    else:
        # [B, seq] x [dim//2] -> [B, seq, dim//2]
        freqs = positions.float().unsqueeze(-1) * inv_freq

    # Adjacent-pair duplication (repeat_interleave) matching HuggingFace
    # Each frequency is duplicated for the adjacent real/imaginary pair
    cos_freqs = freqs.cos().repeat_interleave(2, dim=-1).float()
    sin_freqs = freqs.sin().repeat_interleave(2, dim=-1).float()

    return cos_freqs, sin_freqs


def compute_rope_from_position_ids(
    position_ids: torch.Tensor,
    dim: int,
    theta: float = 10000.0,
    axes_dim: Optional[Tuple[int, ...]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute rotary embeddings from 3D position IDs.

    Matches HuggingFace FluxPosEmbed: processes each axis independently
    with get_1d_rotary_pos_embed(repeat_interleave_real=True), then
    concatenates the cos/sin along the feature dimension.

    Default axes_dim for FLUX.1 is (16, 56, 56) = 128 total (head_dim).
    Position IDs are 3D: [stream_id, height, width] where stream_id=0
    for target, stream_id=1 for reference (Kontext mode).

    Args:
        position_ids: Position IDs [B, seq, 3] or [seq, 3] with [stream, h, w].
            Also accepts 4D [t, h, w, l] for backward compatibility (l is ignored).
        dim: Total embedding dimension per head (e.g. 128 for FLUX.1).
        theta: Base for frequency computation.
        axes_dim: Per-axis dimension allocation. Defaults to (16, 56, 56) for FLUX.1.

    Returns:
        Tuple of (cos, sin) embeddings, each [B, seq, dim] or [seq, dim].
    """
    if axes_dim is None:
        axes_dim = (16, 56, 56)

    n_axes = min(position_ids.shape[-1], len(axes_dim))

    cos_parts = []
    sin_parts = []
    for i in range(n_axes):
        cos_i, sin_i = compute_axis_freqs(position_ids[..., i], axes_dim[i], theta)
        cos_parts.append(cos_i)
        sin_parts.append(sin_i)

    freqs_cos = torch.cat(cos_parts, dim=-1)
    freqs_sin = torch.cat(sin_parts, dim=-1)

    return freqs_cos, freqs_sin



# Re-export create_position_ids from conditioning for backward compatibility
from ..v2.conditioning import create_position_ids as create_image_position_ids
