"""Embedding layers for diffusion models."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class Timesteps(nn.Module):
    """Sinusoidal timestep embeddings."""

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0.0,
        scale: float = 1.0,
        max_period: int = 10000,
    ):
        """Initialize timestep embeddings.

        Args:
            num_channels: Output embedding dimension.
            flip_sin_to_cos: Whether to flip sin/cos order.
            downscale_freq_shift: Frequency shift for downscaling.
            scale: Scale factor for embeddings.
            max_period: Maximum period for sinusoidal embeddings.
        """
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Create sinusoidal embeddings.

        Args:
            timesteps: Timestep values [B].

        Returns:
            Embeddings [B, num_channels].
        """
        half_dim = self.num_channels // 2
        exponent = -math.log(self.max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - self.downscale_freq_shift)

        emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
        emb = self.scale * emb

        if self.flip_sin_to_cos:
            emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        else:
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.num_channels % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))

        return emb


class TimestepEmbedding(nn.Module):
    """MLP timestep embedding projection."""

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: Optional[int] = None,
        post_act_fn: Optional[str] = None,
    ):
        """Initialize timestep embedding projection.

        Args:
            in_channels: Input dimension from sinusoidal embedding.
            time_embed_dim: Output embedding dimension.
            act_fn: Activation function.
            out_dim: Optional different output dimension.
            post_act_fn: Optional activation after output.
        """
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)

        if act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "mish":
            self.act = nn.Mish()
        elif act_fn == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {act_fn}")

        out_dim = out_dim or time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, out_dim)

        self.post_act = None
        if post_act_fn == "silu":
            self.post_act = nn.SiLU()

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            sample: Input embeddings [B, in_channels].

        Returns:
            Projected embeddings [B, out_dim].
        """
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)

        return sample


class TextProjection(nn.Module):
    """Project pooled text embeddings."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        act_fn: str = "silu",
    ):
        """Initialize text projection.

        Args:
            in_features: Input dimension.
            hidden_size: Output dimension.
            act_fn: Activation function.
        """
        super().__init__()
        self.linear_1 = nn.Linear(in_features, hidden_size)

        if act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class PositionalEmbedding(nn.Module):
    """Learnable positional embeddings."""

    def __init__(
        self,
        num_positions: int,
        embedding_dim: int,
    ):
        """Initialize positional embeddings.

        Args:
            num_positions: Maximum sequence length.
            embedding_dim: Embedding dimension.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_positions, embedding_dim)

    def forward(
        self,
        position_ids: Optional[torch.Tensor] = None,
        seq_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Get positional embeddings.

        Args:
            position_ids: Explicit position IDs [B, seq_len].
            seq_length: Sequence length (if position_ids not provided).

        Returns:
            Positional embeddings.
        """
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=self.embedding.weight.device)
        return self.embedding(position_ids)


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE)."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        theta: float = 10000.0,
    ):
        """Initialize rotary embeddings.

        Args:
            dim: Embedding dimension (per head).
            max_seq_len: Maximum sequence length.
            theta: Base for frequency computation.
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequencies
        freqs = self._compute_freqs(max_seq_len)
        self.register_buffer("freqs_cos", freqs.cos(), persistent=False)
        self.register_buffer("freqs_sin", freqs.sin(), persistent=False)

    def _compute_freqs(self, seq_len: int) -> torch.Tensor:
        """Compute rotary frequencies.

        Args:
            seq_len: Sequence length.

        Returns:
            Frequency tensor [seq_len, dim].
        """
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        t = torch.arange(seq_len).float()
        freqs = torch.outer(t, inv_freq)
        return torch.cat([freqs, freqs], dim=-1)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get rotary embedding components.

        Args:
            seq_len: Sequence length.
            device: Target device.

        Returns:
            Tuple of (cos, sin) components.
        """
        if seq_len > self.max_seq_len:
            # Recompute for longer sequences
            freqs = self._compute_freqs(seq_len).to(device)
            return freqs.cos(), freqs.sin()

        return (
            self.freqs_cos[:seq_len].to(device),
            self.freqs_sin[:seq_len].to(device),
        )


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to input tensor.

    Args:
        x: Input tensor [B, heads, seq_len, dim].
        cos: Cosine component [seq_len, dim].
        sin: Sine component [seq_len, dim].

    Returns:
        Tensor with rotary embeddings applied.
    """
    # Rotate pairs of dimensions
    x_rot = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1)
    x_rot = x_rot.reshape(x.shape)

    # Add position dimension if needed
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)

    return x * cos + x_rot * sin


class CombinedTimestepTextProjEmbeddings(nn.Module):
    """Combined timestep and text embeddings for SDXL."""

    def __init__(
        self,
        timestep_channels: int,
        text_embed_dim: int,
        time_embed_dim: int,
    ):
        """Initialize combined embeddings.

        Args:
            timestep_channels: Timestep embedding input channels.
            text_embed_dim: Pooled text embedding dimension.
            time_embed_dim: Output embedding dimension.
        """
        super().__init__()

        self.time_proj = Timesteps(timestep_channels)
        self.time_embed = TimestepEmbedding(timestep_channels, time_embed_dim)
        self.text_embed = TextProjection(text_embed_dim, time_embed_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        pooled_text_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            timestep: Timestep values [B].
            pooled_text_embed: Pooled text embeddings [B, text_dim].

        Returns:
            Combined embeddings [B, time_embed_dim].
        """
        time_embed = self.time_proj(timestep)
        time_embed = self.time_embed(time_embed)
        text_embed = self.text_embed(pooled_text_embed)

        return time_embed + text_embed


class PatchEmbed(nn.Module):
    """2D image patch embedding (for transformer-based models)."""

    def __init__(
        self,
        height: int = 224,
        width: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        flatten: bool = True,
        bias: bool = True,
    ):
        """Initialize patch embedding.

        Args:
            height: Input image height.
            width: Input image width.
            patch_size: Size of each patch.
            in_channels: Number of input channels.
            embed_dim: Output embedding dimension.
            flatten: Whether to flatten spatial dimensions.
            bias: Whether to use bias in projection.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.flatten = flatten

        self.num_patches = (height // patch_size) * (width // patch_size)

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Patch embeddings [B, num_patches, embed_dim] if flatten=True,
            otherwise [B, embed_dim, H', W'].
        """
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x


class MLPEmbedder(nn.Module):
    """MLP embedder for scalar inputs (used in Flux)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
    ):
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
        emb = torch.nn.functional.pad(emb, (0, 1))

    return emb
