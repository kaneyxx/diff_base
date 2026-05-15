"""Shared layer implementations for Flux models."""


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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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


class Flux2Modulation(nn.Module):
    """Shared modulation for FLUX.2 transformer blocks.

    FLUX.2 uses shared modulation across all blocks (unlike FLUX.1 which has
    per-block AdaLayerNormZero). Three instances are used:
    - double_stream_modulation_img (mod_param_sets=2, for attn+FF)
    - double_stream_modulation_txt (mod_param_sets=2, for attn+FF)
    - single_stream_modulation (mod_param_sets=1, parallel attn+FF)

    HuggingFace state dict key: {name}.linear.weight
    """

    def __init__(self, dim: int, mod_param_sets: int = 2, bias: bool = False):
        """Initialize Flux2Modulation.

        Args:
            dim: Hidden dimension.
            mod_param_sets: Number of modulation sets (2 for double, 1 for single).
            bias: Whether to use bias in linear projection.
        """
        super().__init__()
        self.mod_param_sets = mod_param_sets
        self.linear = nn.Linear(dim, dim * 3 * self.mod_param_sets, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, temb: torch.Tensor) -> tuple:
        """Compute modulation parameters.

        Args:
            temb: Conditioning embedding [B, dim].

        Returns:
            Tuple of tuples, each containing (shift, scale, gate) for one set.
            For mod_param_sets=2: ((shift1, scale1, gate1), (shift2, scale2, gate2))
            For mod_param_sets=1: ((shift, scale, gate),)
        """
        mod = self.act_fn(temb)
        mod = self.linear(mod)
        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
        mod_params = torch.chunk(mod, 3 * self.mod_param_sets, dim=-1)
        return tuple(
            mod_params[3 * i : 3 * (i + 1)] for i in range(self.mod_param_sets)
        )
