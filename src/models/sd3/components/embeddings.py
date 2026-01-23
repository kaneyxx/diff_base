"""SD3-specific embedding implementations.

SD3 uses:
- PatchEmbed: Converts latent images to patch sequences
- CombinedTimestepTextProjEmbeddings: Combines timestep and pooled text embeddings
- Positional embeddings for spatial locations
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10000,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 1.0,
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: Timestep values [B].
        embedding_dim: Dimension of the embeddings.
        max_period: Controls the minimum frequency.
        flip_sin_to_cos: If True, use cos before sin.
        downscale_freq_shift: Shift for frequency downscaling.

    Returns:
        Timestep embeddings [B, embedding_dim].
    """
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]

    if flip_sin_to_cos:
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
    else:
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if embedding_dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))

    return emb


class PatchEmbed(nn.Module):
    """Patch embedding layer for converting latent images to patch sequences.

    Similar to ViT but for latent space inputs. Converts [B, C, H, W] to [B, N, D]
    where N = (H/patch_size) * (W/patch_size) and D = hidden_size.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        hidden_size: int = 1536,
        bias: bool = True,
    ):
        """Initialize PatchEmbed.

        Args:
            patch_size: Size of each patch.
            in_channels: Number of input channels (VAE latent channels).
            hidden_size: Output hidden dimension.
            bias: Whether to use bias in projection.
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        self.proj = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Convert latent to patch sequence.

        Args:
            latent: Latent tensor [B, C, H, W].

        Returns:
            Patch sequence [B, num_patches, hidden_size].
        """
        # Project patches: [B, C, H, W] -> [B, hidden_size, H/patch, W/patch]
        x = self.proj(latent)
        # Flatten to sequence: [B, hidden_size, H/patch, W/patch] -> [B, N, hidden_size]
        x = x.flatten(2).transpose(1, 2)
        return x


class TimestepEmbedding(nn.Module):
    """Timestep embedding with MLP projection.

    Projects sinusoidal embeddings to hidden dimension.
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_size: int = 1536,
        act_fn: str = "silu",
    ):
        """Initialize TimestepEmbedding.

        Args:
            in_channels: Input channels (sinusoidal embedding dim).
            hidden_size: Output hidden dimension.
            act_fn: Activation function.
        """
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, hidden_size, bias=True)

        if act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "gelu":
            self.act = nn.GELU(approximate="tanh")
        else:
            self.act = nn.SiLU()

        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """Project timestep embedding.

        Args:
            sample: Sinusoidal embedding [B, in_channels].

        Returns:
            Projected embedding [B, hidden_size].
        """
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class CombinedTimestepTextProjEmbeddings(nn.Module):
    """Combined timestep and pooled text projection embeddings.

    SD3 combines:
    1. Timestep embedding (sinusoidal -> MLP)
    2. Pooled text embeddings from CLIP-G

    The pooled text comes from OpenCLIP bigG (1280 dim) concatenated with
    CLIP-L pooled output, but SD3 typically uses just CLIP-G pooled (2048 total
    from CLIP-L 768 + CLIP-G 1280).
    """

    def __init__(
        self,
        embedding_dim: int = 1536,
        pooled_projection_dim: int = 2048,
        timestep_in_channels: int = 256,
    ):
        """Initialize CombinedTimestepTextProjEmbeddings.

        Args:
            embedding_dim: Output embedding dimension.
            pooled_projection_dim: Dimension of pooled text projections.
            timestep_in_channels: Dimension of sinusoidal timestep embeddings.
        """
        super().__init__()
        self.time_proj = TimestepEmbedding(
            in_channels=timestep_in_channels,
            hidden_size=embedding_dim,
        )
        self.text_proj = nn.Linear(pooled_projection_dim, embedding_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        pooled_projection: torch.Tensor,
    ) -> torch.Tensor:
        """Combine timestep and text embeddings.

        Args:
            timestep: Timestep embedding [B, timestep_in_channels].
            pooled_projection: Pooled text embedding [B, pooled_projection_dim].

        Returns:
            Combined embedding [B, embedding_dim].
        """
        time_embed = self.time_proj(timestep)
        text_embed = self.text_proj(pooled_projection)
        return time_embed + text_embed


class PositionalEmbedding2D(nn.Module):
    """2D positional embeddings using sinusoidal encoding.

    Creates learnable or fixed positional embeddings for 2D spatial locations.
    SD3 uses fixed sin/cos positional embeddings up to a maximum size.
    """

    def __init__(
        self,
        hidden_size: int,
        max_size: int = 192,
        theta: float = 10000.0,
    ):
        """Initialize PositionalEmbedding2D.

        Args:
            hidden_size: Hidden dimension.
            max_size: Maximum spatial size (patches, not pixels).
            theta: Base for frequency computation.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_size = max_size
        self.theta = theta

        # Precompute fixed positional embeddings
        pos_embed = self._make_pos_embed(max_size, hidden_size, theta)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _make_pos_embed(
        self,
        max_size: int,
        hidden_size: int,
        theta: float,
    ) -> torch.Tensor:
        """Create 2D sinusoidal position embeddings.

        Args:
            max_size: Maximum size in each dimension.
            hidden_size: Hidden dimension.
            theta: Base for frequency computation.

        Returns:
            Position embeddings [max_size*max_size, hidden_size].
        """
        grid_h = torch.arange(max_size, dtype=torch.float32)
        grid_w = torch.arange(max_size, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(grid_h, grid_w, indexing="ij"), dim=-1)
        grid = grid.reshape(-1, 2)  # [max_size*max_size, 2]

        # Compute frequencies
        half_dim = hidden_size // 4  # Split across H, W and sin, cos
        emb_h = self._get_1d_sincos_embed(grid[:, 0], half_dim, theta)
        emb_w = self._get_1d_sincos_embed(grid[:, 1], half_dim, theta)

        # Concatenate H and W embeddings
        emb = torch.cat([emb_h, emb_w], dim=-1)  # [max_size*max_size, hidden_size]

        return emb

    def _get_1d_sincos_embed(
        self,
        pos: torch.Tensor,
        dim: int,
        theta: float,
    ) -> torch.Tensor:
        """Get 1D sinusoidal embeddings.

        Args:
            pos: Position indices [N].
            dim: Embedding dimension per sin/cos.
            theta: Base for frequency computation.

        Returns:
            Embeddings [N, dim*2].
        """
        freqs = 1.0 / (theta ** (torch.arange(0, dim, dtype=torch.float32) / dim))
        angles = pos[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb

    def forward(
        self,
        num_patches_h: int,
        num_patches_w: int,
    ) -> torch.Tensor:
        """Get positional embeddings for given spatial size.

        Args:
            num_patches_h: Number of patches in height.
            num_patches_w: Number of patches in width.

        Returns:
            Position embeddings [1, num_patches_h * num_patches_w, hidden_size].
        """
        # Extract relevant portion from precomputed embeddings
        # Assuming row-major ordering
        indices = []
        for h in range(num_patches_h):
            for w in range(num_patches_w):
                indices.append(h * self.max_size + w)

        indices = torch.tensor(indices, dtype=torch.long, device=self.pos_embed.device)
        pos_embed = self.pos_embed[indices]

        return pos_embed.unsqueeze(0)

    def forward_from_seq_len(self, seq_len: int) -> torch.Tensor:
        """Get positional embeddings assuming square spatial layout.

        Args:
            seq_len: Total sequence length (num_patches_h * num_patches_w).

        Returns:
            Position embeddings [1, seq_len, hidden_size].
        """
        # Assume square layout
        size = int(math.sqrt(seq_len))
        if size * size != seq_len:
            # Non-square, just use first seq_len embeddings
            return self.pos_embed[:seq_len].unsqueeze(0)
        return self.forward(size, size)
