"""Transformer components for diffusion models."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Feed-forward network with configurable activation."""

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: float = 4.0,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        """Initialize feed-forward network.

        Args:
            dim: Input dimension.
            dim_out: Output dimension (defaults to dim).
            mult: Hidden dimension multiplier.
            dropout: Dropout probability.
            activation_fn: Activation function type.
            final_dropout: Whether to add dropout after final layer.
        """
        super().__init__()
        dim_out = dim_out or dim
        inner_dim = int(dim * mult)

        if activation_fn == "geglu":
            self.net = nn.ModuleList([
                GEGLU(dim, inner_dim),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out),
            ])
        elif activation_fn == "gelu":
            self.net = nn.ModuleList([
                nn.Linear(dim, inner_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out),
            ])
        elif activation_fn == "silu":
            self.net = nn.ModuleList([
                nn.Linear(dim, inner_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out),
            ])
        else:
            raise ValueError(f"Unknown activation: {activation_fn}")

        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class GEGLU(nn.Module):
    """Gated GELU activation."""

    def __init__(self, dim_in: int, dim_out: int):
        """Initialize GEGLU.

        Args:
            dim_in: Input dimension.
            dim_out: Output dimension (half of projection).
        """
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class AdaLayerNorm(nn.Module):
    """Adaptive layer normalization with timestep conditioning."""

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int = 1000,
        output_dim: Optional[int] = None,
    ):
        """Initialize adaptive layer norm.

        Args:
            embedding_dim: Embedding dimension.
            num_embeddings: Number of timestep embeddings.
            output_dim: Output dimension (defaults to embedding_dim).
        """
        super().__init__()
        output_dim = output_dim or embedding_dim

        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim * 2)
        self.norm = nn.LayerNorm(output_dim, elementwise_affine=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Input tensor [B, seq_len, dim].
            timestep: Timestep tensor [B].

        Returns:
            Normalized and scaled tensor.
        """
        emb = self.emb(timestep)
        emb = self.silu(emb)
        emb = self.linear(emb)
        scale, shift = emb.unsqueeze(1).chunk(2, dim=-1)

        hidden_states = self.norm(hidden_states) * (1 + scale) + shift
        return hidden_states


class AdaLayerNormZero(nn.Module):
    """Adaptive layer normalization with zero initialization (for DiT)."""

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
    ):
        """Initialize adaptive layer norm zero.

        Args:
            embedding_dim: Hidden dimension.
            num_embeddings: Number of embeddings (unused, for API compatibility).
        """
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            hidden_states: Input tensor [B, seq_len, dim].
            emb: Conditioning embedding [B, dim].

        Returns:
            Tuple of (normalized_states, gate_msa, shift_mlp, scale_mlp, gate_mlp).
        """
        emb = self.silu(emb)
        emb = self.linear(emb)[:, None, :]

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(
            6, dim=-1
        )

        hidden_states = self.norm(hidden_states) * (1 + scale_msa) + shift_msa

        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormSingle(nn.Module):
    """Single adaptive layer norm for Flux-style models."""

    def __init__(
        self,
        embedding_dim: int,
        use_additional_conditions: bool = False,
    ):
        """Initialize single adaptive layer norm.

        Args:
            embedding_dim: Hidden dimension.
            use_additional_conditions: Whether to use additional conditions.
        """
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            hidden_states: Input tensor [B, seq_len, dim].
            emb: Conditioning embedding [B, dim].

        Returns:
            Tuple of (normalized_states, gate).
        """
        emb = self.silu(emb)
        emb = self.linear(emb)[:, None, :]

        shift, scale, gate = emb.chunk(3, dim=-1)

        hidden_states = self.norm(hidden_states) * (1 + scale) + shift

        return hidden_states, gate


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize RMSNorm.

        Args:
            dim: Normalization dimension.
            eps: Epsilon for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class Transformer2DModel(nn.Module):
    """2D Transformer model for spatial features."""

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        norm_num_groups: int = 32,
        use_linear_projection: bool = False,
    ):
        """Initialize 2D Transformer.

        Args:
            num_attention_heads: Number of attention heads.
            attention_head_dim: Dimension per head.
            in_channels: Input channels (for conv projection).
            out_channels: Output channels.
            num_layers: Number of transformer blocks.
            dropout: Dropout probability.
            cross_attention_dim: Cross-attention context dimension.
            attention_bias: Whether to use bias in attention.
            activation_fn: Activation function for FFN.
            norm_num_groups: Groups for input GroupNorm.
            use_linear_projection: Use linear vs conv for input projection.
        """
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_linear_projection = use_linear_projection

        # Input projection
        self.norm = nn.GroupNorm(norm_num_groups, in_channels, eps=1e-6, affine=True)

        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1)

        # Transformer blocks
        from .attention import BasicTransformerBlock

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                dim=inner_dim,
                num_heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                context_dim=cross_attention_dim,
                activation_fn=activation_fn,
            )
            for _ in range(num_layers)
        ])

        # Output projection
        if use_linear_projection:
            self.proj_out = nn.Linear(inner_dim, self.out_channels)
        else:
            self.proj_out = nn.Conv2d(inner_dim, self.out_channels, kernel_size=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Input tensor [B, C, H, W].
            encoder_hidden_states: Cross-attention context.
            attention_mask: Attention mask.
            return_dict: Whether to return dict (unused, for compatibility).

        Returns:
            Output tensor [B, C, H, W].
        """
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        # Norm and project
        hidden_states = self.norm(hidden_states)

        if self.use_linear_projection:
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                batch, height * width, -1
            )
            hidden_states = self.proj_in(hidden_states)
        else:
            hidden_states = self.proj_in(hidden_states)
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                batch, height * width, -1
            )

        # Transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )

        # Project out
        if self.use_linear_projection:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width, -1).permute(
                0, 3, 1, 2
            )
        else:
            hidden_states = hidden_states.reshape(batch, height, width, -1).permute(
                0, 3, 1, 2
            )
            hidden_states = self.proj_out(hidden_states)

        return hidden_states + residual
