"""Flux-specific embedding implementations.

This module provides embedding utilities for FLUX models including:
- Timestep embeddings (sinusoidal)
- Positional embeddings (2D RoPE)
- Combined timestep/guidance embeddings
- 4D position ID to RoPE conversion for Kontext mode
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def get_timestep_embedding(
    timestep: torch.Tensor,
    dim: int = 256,
    max_period: int = 10000,
) -> torch.Tensor:
    """Get sinusoidal timestep embedding.

    Args:
        timestep: Timestep values [B].
        dim: Embedding dimension.
        max_period: Maximum period for sinusoidal embeddings.

    Returns:
        Embeddings [B, dim].
    """
    half_dim = dim // 2
    emb = math.log(max_period) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timestep.device) * -emb)
    emb = timestep[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))

    return emb


class FluxPosEmbed(nn.Module):
    """Positional embedding for Flux models.

    Supports both 1D and 2D positional embeddings with RoPE-style encoding.
    """

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        axes_dims: Optional[Tuple[int, ...]] = None,
    ):
        """Initialize FluxPosEmbed.

        Args:
            dim: Embedding dimension per head.
            theta: Base for frequency computation.
            axes_dims: Optional tuple of dimensions for each axis.
        """
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dims = axes_dims or (dim // 2, dim // 2)

    def _compute_2d_rope(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 2D rotary position embeddings.

        Args:
            height: Spatial height.
            width: Spatial width.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Tuple of (cos, sin) embeddings.
        """
        # Height frequencies
        h_dim = self.axes_dims[0]
        h_inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, h_dim, 2, device=device).float() / h_dim)
        )
        h_seq = torch.arange(height, device=device).float()
        h_freqs = torch.outer(h_seq, h_inv_freq)

        # Width frequencies
        w_dim = self.axes_dims[1] if len(self.axes_dims) > 1 else self.axes_dims[0]
        w_inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, w_dim, 2, device=device).float() / w_dim)
        )
        w_seq = torch.arange(width, device=device).float()
        w_freqs = torch.outer(w_seq, w_inv_freq)

        # Create 2D grid
        h_freqs = h_freqs.unsqueeze(1).expand(-1, width, -1)  # [H, W, h_dim/2]
        w_freqs = w_freqs.unsqueeze(0).expand(height, -1, -1)  # [H, W, w_dim/2]

        # Combine and flatten
        freqs = torch.cat([
            h_freqs.reshape(-1, h_dim // 2),
            w_freqs.reshape(-1, w_dim // 2),
        ], dim=-1)

        # Double for sin/cos pairs
        freqs = torch.cat([freqs, freqs], dim=-1)

        return freqs.cos().to(dtype), freqs.sin().to(dtype)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get positional embeddings.

        Args:
            seq_len: Sequence length (used if height/width not specified).
            device: Target device.
            dtype: Target dtype.
            height: Optional spatial height for 2D embeddings.
            width: Optional spatial width for 2D embeddings.

        Returns:
            Tuple of (cos, sin) embeddings.
        """
        if height is not None and width is not None:
            return self._compute_2d_rope(height, width, device, dtype)

        # 1D rope
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        )
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, inv_freq)
        freqs = torch.cat([freqs, freqs], dim=-1)

        return freqs.cos().to(dtype), freqs.sin().to(dtype)


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
) -> torch.Tensor:
    """Compute rotary frequencies for a single axis.

    Args:
        positions: Position values [B, seq].
        dim: Number of frequency dimensions to compute.
        theta: Base for frequency computation.

    Returns:
        Frequencies tensor [B, seq, dim].
    """
    if dim == 0:
        return torch.zeros(
            positions.shape[0], positions.shape[1], 0,
            device=positions.device, dtype=positions.dtype
        )

    # Compute inverse frequencies for this axis
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=positions.device).float() / dim)
    )

    # Compute frequencies: [B, seq] x [dim/2] -> [B, seq, dim/2]
    freqs = positions.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)

    # Double for sin/cos pairs: [B, seq, dim]
    freqs = torch.cat([freqs, freqs], dim=-1)

    return freqs


def compute_rope_from_position_ids(
    position_ids: torch.Tensor,
    dim: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute rotary embeddings from explicit 4D position IDs.

    Allocates embedding dimensions across t, h, w, l axes for proper
    spatial-temporal encoding of target vs reference images.

    The dimension allocation is:
    - dim_t: dim // 8 (temporal, smaller since binary for most cases)
    - dim_h: dim // 4 (height, important for spatial structure)
    - dim_w: dim // 4 (width, important for spatial structure)
    - dim_l: remaining (sequence index)

    Args:
        position_ids: 4D position IDs [B, seq, 4] with [t, h, w, l].
        dim: Total embedding dimension (must be even).
        theta: Base for frequency computation.

    Returns:
        Tuple of (cos, sin) embeddings, each [B, seq, dim].
    """
    B, seq_len, _ = position_ids.shape

    # Allocate dimensions across axes
    # t gets smallest allocation (binary 0/1 in most cases)
    # h and w get equal allocation for spatial structure
    # l gets remainder
    dim_t = dim // 8
    dim_h = dim // 4
    dim_w = dim // 4
    dim_l = dim - dim_t - dim_h - dim_w

    # Ensure even dimensions for sin/cos pairs
    dim_t = (dim_t // 2) * 2
    dim_h = (dim_h // 2) * 2
    dim_w = (dim_w // 2) * 2
    dim_l = dim - dim_t - dim_h - dim_w
    dim_l = (dim_l // 2) * 2

    # If dimensions don't add up exactly, adjust dim_l
    total = dim_t + dim_h + dim_w + dim_l
    if total < dim:
        dim_l += (dim - total)

    # Compute frequencies for each axis
    freqs_t = compute_axis_freqs(position_ids[..., 0], dim_t, theta)
    freqs_h = compute_axis_freqs(position_ids[..., 1], dim_h, theta)
    freqs_w = compute_axis_freqs(position_ids[..., 2], dim_w, theta)
    freqs_l = compute_axis_freqs(position_ids[..., 3], dim_l, theta)

    # Concatenate all frequencies: [B, seq, dim]
    freqs = torch.cat([freqs_t, freqs_h, freqs_w, freqs_l], dim=-1)

    return freqs.cos(), freqs.sin()


def create_image_position_ids(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    time_offset: float = 0.0,
) -> torch.Tensor:
    """Create 4D position IDs for image patches.

    This is a convenience function for creating position IDs directly
    in the embedding module (useful for text-to-image without Kontext).

    Args:
        batch_size: Batch size.
        height: Spatial height in patches.
        width: Spatial width in patches.
        device: Target device.
        dtype: Data type.
        time_offset: Time dimension value (0.0 for target).

    Returns:
        Position IDs tensor of shape [B, height*width, 4].
    """
    num_patches = height * width

    # Create grid indices
    y_indices = torch.arange(height, device=device)
    x_indices = torch.arange(width, device=device)
    y_grid, x_grid = torch.meshgrid(y_indices, x_indices, indexing="ij")

    # Create position components
    t = torch.full((num_patches,), time_offset, device=device, dtype=dtype)
    h = y_grid.reshape(-1).to(dtype)
    w = x_grid.reshape(-1).to(dtype)
    l = torch.zeros(num_patches, device=device, dtype=dtype)

    # Stack: [num_patches, 4]
    ids = torch.stack([t, h, w, l], dim=-1)

    # Expand to batch: [B, num_patches, 4]
    return ids.unsqueeze(0).expand(batch_size, -1, -1)
