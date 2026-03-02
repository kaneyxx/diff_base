"""Flux-specific embedding implementations.

This module provides embedding utilities for FLUX models including:
- Timestep embeddings (sinusoidal)
- Positional embeddings (RoPE with per-axis frequencies)
- Combined timestep/guidance embeddings
- 3D position ID to RoPE conversion for Kontext mode

RoPE implementation matches HuggingFace diffusers:
- axes_dim=(16, 56, 56) for FLUX.1 (stream, height, width)
- Uses repeat_interleave for adjacent-pair duplication
- Position IDs are 3D [stream, h, w] (not 4D)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def get_timestep_embedding(
    timestep: torch.Tensor,
    dim: int = 256,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0,
    max_period: int = 10000,
) -> torch.Tensor:
    """Get sinusoidal timestep embedding.

    Matches the HuggingFace diffusers implementation used by FLUX models.
    FLUX uses flip_sin_to_cos=True and downscale_freq_shift=0.

    Args:
        timestep: Timestep values [B].
        dim: Embedding dimension.
        flip_sin_to_cos: If True, output [cos, sin] order (FLUX default).
        downscale_freq_shift: Shift for frequency denominator (FLUX uses 0).
        max_period: Maximum period for sinusoidal embeddings.

    Returns:
        Embeddings [B, dim].
    """
    half_dim = dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timestep.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timestep[:, None].float() * emb[None, :]

    if flip_sin_to_cos:
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
    else:
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))

    return emb


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


class CombinedTimestepGuidanceEmbeddings(nn.Module):
    """Combined timestep and guidance embeddings for Flux.

    Projects timestep and guidance values to hidden dimension.
    """

    def __init__(self, hidden_size: int, pooled_projection_dim: int = 768):
        """Initialize embeddings.

        Args:
            hidden_size: Output hidden dimension.
            pooled_projection_dim: Pooled text embedding dimension.
        """
        super().__init__()

        self.time_embedder = MLPEmbedder(256, hidden_size)
        self.guidance_embedder = MLPEmbedder(256, hidden_size)
        self.pooled_text_embedder = nn.Linear(pooled_projection_dim, hidden_size)

    def forward(
        self,
        timestep: torch.Tensor,
        guidance: Optional[torch.Tensor],
        pooled_projection: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            timestep: Timestep values [B].
            guidance: Optional guidance scale [B].
            pooled_projection: Pooled text embeddings [B, pooled_dim].

        Returns:
            Combined embeddings [B, hidden_size].
        """
        # Get timestep embedding
        t_emb = get_timestep_embedding(timestep)
        t_emb = self.time_embedder(t_emb)

        # Add guidance embedding if provided
        if guidance is not None:
            g_emb = get_timestep_embedding(guidance)
            t_emb = t_emb + self.guidance_embedder(g_emb)

        # Add pooled text embedding
        t_emb = t_emb + self.pooled_text_embedder(pooled_projection)

        return t_emb


class MLPEmbedder(nn.Module):
    """MLP embedder for scalar inputs."""

    def __init__(self, in_dim: int, hidden_dim: int):
        """Initialize MLP embedder.

        Args:
            in_dim: Input dimension.
            hidden_dim: Output hidden dimension.
        """
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.in_layer(x)
        x = self.silu(x)
        x = self.out_layer(x)
        return x


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


def create_image_position_ids(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    time_offset: float = 0.0,
) -> torch.Tensor:
    """Create 3D position IDs for image patches.

    This is a convenience function for creating position IDs directly
    in the embedding module (useful for text-to-image without Kontext).

    Args:
        batch_size: Batch size.
        height: Spatial height in patches.
        width: Spatial width in patches.
        device: Target device.
        dtype: Data type.
        time_offset: Stream index value (0.0 for target, 1.0 for reference).

    Returns:
        Position IDs tensor of shape [B, height*width, 3].
    """
    # Match HuggingFace: create [height, width, 3] then reshape
    ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
    ids[..., 0] = time_offset  # stream index
    ids[..., 1] = ids[..., 1] + torch.arange(height, device=device, dtype=dtype)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(width, device=device, dtype=dtype)[None, :]

    # Reshape to [height*width, 3]
    ids = ids.reshape(height * width, 3)

    # Expand to batch: [B, height*width, 3]
    return ids.unsqueeze(0).expand(batch_size, -1, -1)
