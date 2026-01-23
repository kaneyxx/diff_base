"""SD3-specific layer implementations.

SD3 uses RMSNorm for QK normalization, AdaLayerNorm for conditioning,
and SwiGLU for the feedforward network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Used for query-key normalization in SD3 attention for numerical stability.
    Unlike LayerNorm, RMSNorm only normalizes by RMS without centering.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        """Initialize RMSNorm.

        Args:
            dim: Feature dimension.
            eps: Epsilon for numerical stability.
            elementwise_affine: Whether to include learnable scale.
        """
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor [..., dim].

        Returns:
            Normalized tensor with same shape.
        """
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms

        if self.weight is not None:
            x = x * self.weight

        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply shift and scale modulation.

    Args:
        x: Input tensor [B, seq_len, dim].
        shift: Shift values [B, 1, dim] or [B, dim].
        scale: Scale values [B, 1, dim] or [B, dim].

    Returns:
        Modulated tensor.
    """
    if shift.dim() == 2:
        shift = shift.unsqueeze(1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


class AdaLayerNormContinuous(nn.Module):
    """Adaptive Layer Normalization with continuous conditioning.

    Used for the final output projection in SD3, where the normalization
    parameters are predicted from the conditioning signal.
    """

    def __init__(
        self,
        hidden_size: int,
        conditioning_size: int | None = None,
        eps: float = 1e-6,
    ):
        """Initialize AdaLayerNormContinuous.

        Args:
            hidden_size: Hidden dimension.
            conditioning_size: Conditioning dimension (defaults to hidden_size).
            eps: Epsilon for layer norm.
        """
        super().__init__()
        conditioning_size = conditioning_size or hidden_size

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)
        self.linear = nn.Linear(conditioning_size, hidden_size * 2, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """Apply adaptive normalization.

        Args:
            x: Input tensor [B, seq_len, hidden_size].
            conditioning: Conditioning tensor [B, conditioning_size].

        Returns:
            Normalized and modulated tensor.
        """
        # Get scale and shift from conditioning
        emb = self.linear(F.silu(conditioning))
        scale, shift = emb.chunk(2, dim=-1)

        # Apply normalization with modulation
        x = self.norm(x)
        x = modulate(x, shift, scale)

        return x


class AdaLayerNormZero(nn.Module):
    """Adaptive Layer Normalization with zero initialization for gating.

    Produces 6 outputs: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    Used in SD3 JointTransformerBlock for both image and text streams.
    """

    def __init__(self, hidden_size: int, num_embeds_ada_norm: int | None = None):
        """Initialize AdaLayerNormZero.

        Args:
            hidden_size: Hidden dimension.
            num_embeds_ada_norm: Unused, for API compatibility.
        """
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply adaptive normalization with 6-way modulation.

        Args:
            x: Input tensor [B, seq_len, hidden_size].
            emb: Conditioning embedding [B, hidden_size].

        Returns:
            Tuple of (normalized_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa, scale_msa).
            Note: The actual returns are rearranged for the block.
        """
        emb = self.silu(emb)
        emb = self.linear(emb)

        if emb.dim() == 2:
            emb = emb.unsqueeze(1)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)

        x = self.norm(x)
        x = modulate(x, shift_msa, scale_msa)

        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class SwiGLU(nn.Module):
    """SwiGLU activation function.

    SwiGLU(x) = SiLU(W1 @ x) * (W2 @ x)
    """

    def __init__(self, in_features: int, hidden_features: int, bias: bool = True):
        """Initialize SwiGLU.

        Args:
            in_features: Input dimension.
            hidden_features: Hidden dimension (actual hidden is hidden_features * 2).
            bias: Whether to use bias in linear layers.
        """
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU.

        Args:
            x: Input tensor [B, seq_len, in_features].

        Returns:
            Output tensor [B, seq_len, in_features].
        """
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation.

    Standard MLP: Linear -> GELU -> Linear
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        """Initialize FeedForward.

        Args:
            hidden_size: Hidden dimension.
            intermediate_size: Intermediate dimension (defaults to 4 * hidden_size).
            dropout: Dropout probability.
            activation: Activation function ("gelu", "gelu_tanh", "swiglu").
        """
        super().__init__()
        intermediate_size = intermediate_size or int(hidden_size * 4)

        if activation == "swiglu":
            self.net = SwiGLU(hidden_size, intermediate_size)
        else:
            if activation == "gelu_tanh":
                act_fn = nn.GELU(approximate="tanh")
            else:
                act_fn = nn.GELU()

            self.net = nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                act_fn,
                nn.Dropout(dropout),
                nn.Linear(intermediate_size, hidden_size),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network.

        Args:
            x: Input tensor [B, seq_len, hidden_size].

        Returns:
            Output tensor [B, seq_len, hidden_size].
        """
        return self.net(x)
