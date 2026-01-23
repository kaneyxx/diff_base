"""Shared layer implementations for Flux models."""

from typing import Tuple

import torch
import torch.nn as nn


class AdaLayerNormZero(nn.Module):
    """Adaptive layer norm with zero initialization for joint blocks.

    Used in Flux joint transformer blocks. Outputs 6 modulation tensors:
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    """

    def __init__(self, hidden_size: int):
        """Initialize AdaLayerNormZero.

        Args:
            hidden_size: Hidden dimension.
        """
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, 6 * hidden_size)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [B, seq_len, hidden_size].
            emb: Conditioning embedding [B, hidden_size].

        Returns:
            Tuple of (normalized_x, gate_msa, shift_mlp, scale_mlp, gate_mlp).
        """
        emb = self.silu(emb)
        emb = self.linear(emb)[:, None, :]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    """Single adaptive layer norm for single stream blocks.

    Used in Flux single transformer blocks. Outputs 3 modulation tensors:
    shift, scale, gate
    """

    def __init__(self, hidden_size: int):
        """Initialize AdaLayerNormZeroSingle.

        Args:
            hidden_size: Hidden dimension.
        """
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, 3 * hidden_size)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [B, seq_len, hidden_size].
            emb: Conditioning embedding [B, hidden_size].

        Returns:
            Tuple of (normalized_x, gate).
        """
        emb = self.silu(emb)
        emb = self.linear(emb)[:, None, :]
        shift, scale, gate = emb.chunk(3, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x, gate


class AdaLayerNormContinuous(nn.Module):
    """Continuous adaptive layer norm used in FLUX.2.

    Unlike discrete timestep-based normalization, this takes continuous
    embeddings and produces shift/scale modulations.
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ):
        """Initialize AdaLayerNormContinuous.

        Args:
            embedding_dim: Input/output dimension.
            conditioning_embedding_dim: Conditioning embedding dimension.
            elementwise_affine: Whether to use learnable affine params.
            eps: Epsilon for layer norm.
            bias: Whether to use bias.
        """
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(
            conditioning_embedding_dim, embedding_dim * 2, bias=bias
        )
        self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self,
        x: torch.Tensor,
        conditioning_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, seq_len, embedding_dim].
            conditioning_embedding: Conditioning [B, conditioning_embedding_dim].

        Returns:
            Normalized and modulated tensor.
        """
        emb = self.linear(self.silu(conditioning_embedding))
        scale, shift = emb.unsqueeze(1).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class QKNorm(nn.Module):
    """Query-Key normalization layer.

    Applies RMSNorm to queries and keys before attention computation.
    Used in FLUX.2 for training stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize QKNorm.

        Args:
            dim: Dimension to normalize.
            eps: Epsilon for numerical stability.
        """
        super().__init__()
        self.query_norm = RMSNorm(dim, eps=eps)
        self.key_norm = RMSNorm(dim, eps=eps)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize queries and keys.

        Args:
            q: Query tensor.
            k: Key tensor.

        Returns:
            Tuple of (normalized_q, normalized_k).
        """
        return self.query_norm(q), self.key_norm(k)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize RMSNorm.

        Args:
            dim: Dimension to normalize.
            eps: Epsilon for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
