"""Attention mechanisms for diffusion models."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Multi-head self-attention layer."""

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        """Initialize self-attention.

        Args:
            query_dim: Input dimension.
            heads: Number of attention heads.
            dim_head: Dimension per head.
            dropout: Dropout probability.
            bias: Whether to use bias in projections.
        """
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Input tensor [B, seq_len, dim].
            attention_mask: Optional mask [B, seq_len] or [B, 1, seq_len, seq_len].

        Returns:
            Output tensor [B, seq_len, dim].
        """
        batch_size, seq_len, _ = hidden_states.shape

        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        # Reshape to [B, heads, seq_len, dim_head]
        q = q.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        hidden_states = torch.matmul(attn_weights, v)

        # Reshape back
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_len, -1)
        hidden_states = self.to_out(hidden_states)

        return hidden_states


class CrossAttention(nn.Module):
    """Multi-head cross-attention layer."""

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        """Initialize cross-attention.

        Args:
            query_dim: Query input dimension.
            cross_attention_dim: Key/value input dimension (defaults to query_dim).
            heads: Number of attention heads.
            dim_head: Dimension per head.
            dropout: Dropout probability.
            bias: Whether to use bias in projections.
        """
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim or query_dim
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Query tensor [B, seq_len_q, dim].
            encoder_hidden_states: Key/value tensor [B, seq_len_kv, cross_dim].
            attention_mask: Optional mask.

        Returns:
            Output tensor [B, seq_len_q, dim].
        """
        batch_size, seq_len, _ = hidden_states.shape
        kv_seq_len = encoder_hidden_states.shape[1]

        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        # Reshape to [B, heads, seq_len, dim_head]
        q = q.view(batch_size, seq_len, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(batch_size, kv_seq_len, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(batch_size, kv_seq_len, self.heads, self.dim_head).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        hidden_states = torch.matmul(attn_weights, v)

        # Reshape back
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_len, -1)
        hidden_states = self.to_out(hidden_states)

        return hidden_states


class JointAttention(nn.Module):
    """Joint attention for processing image and text together (used in Flux).

    HuggingFace-aligned naming convention:
    - Image stream: to_q, to_k, to_v, to_out (ModuleList)
    - Text stream: add_q_proj, add_k_proj, add_v_proj, to_add_out
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
    ):
        """Initialize joint attention.

        Args:
            dim: Hidden dimension.
            num_heads: Number of attention heads.
            qkv_bias: Whether to use bias in QKV projections.
            proj_bias: Whether to use bias in output projection.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Image stream - separate Q/K/V projections (HF naming)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        # Text stream - add_* naming (HF convention)
        self.add_q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.add_k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.add_v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # Output projections - ModuleList for to_out.0 naming
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=proj_bias), nn.Identity()])
        self.to_add_out = nn.Linear(dim, dim, bias=proj_bias)

    def forward(
        self,
        image_hidden_states: torch.Tensor,
        text_hidden_states: torch.Tensor,
        image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        text_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with joint attention.

        Args:
            image_hidden_states: Image tokens [B, img_seq, dim].
            text_hidden_states: Text tokens [B, txt_seq, dim].
            image_rotary_emb: Optional rotary embeddings for image.
            text_rotary_emb: Optional rotary embeddings for text.

        Returns:
            Tuple of (image_output, text_output).
        """
        batch_size = image_hidden_states.shape[0]
        img_seq_len = image_hidden_states.shape[1]
        txt_seq_len = text_hidden_states.shape[1]

        # Compute separate Q, K, V for image
        q_img = self.to_q(image_hidden_states)
        k_img = self.to_k(image_hidden_states)
        v_img = self.to_v(image_hidden_states)

        # Compute separate Q, K, V for text
        q_txt = self.add_q_proj(text_hidden_states)
        k_txt = self.add_k_proj(text_hidden_states)
        v_txt = self.add_v_proj(text_hidden_states)

        # Reshape for multi-head attention
        def reshape_for_attention(x, seq_len):
            return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q_img = reshape_for_attention(q_img, img_seq_len)
        k_img = reshape_for_attention(k_img, img_seq_len)
        v_img = reshape_for_attention(v_img, img_seq_len)

        q_txt = reshape_for_attention(q_txt, txt_seq_len)
        k_txt = reshape_for_attention(k_txt, txt_seq_len)
        v_txt = reshape_for_attention(v_txt, txt_seq_len)

        # Apply rotary embeddings if provided
        if image_rotary_emb is not None:
            from .embeddings import apply_rotary_emb
            q_img = apply_rotary_emb(q_img, image_rotary_emb)
            k_img = apply_rotary_emb(k_img, image_rotary_emb)

        if text_rotary_emb is not None:
            from .embeddings import apply_rotary_emb
            q_txt = apply_rotary_emb(q_txt, text_rotary_emb)
            k_txt = apply_rotary_emb(k_txt, text_rotary_emb)

        # Concatenate for joint attention
        q = torch.cat([q_txt, q_img], dim=2)  # [B, heads, txt+img, dim]
        k = torch.cat([k_txt, k_img], dim=2)
        v = torch.cat([v_txt, v_img], dim=2)

        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        hidden_states = torch.matmul(attn_weights, v)

        # Split back into text and image
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, txt_seq_len + img_seq_len, -1
        )
        txt_output = hidden_states[:, :txt_seq_len]
        img_output = hidden_states[:, txt_seq_len:]

        # Project outputs using ModuleList for image (to_out.0)
        img_output = self.to_out[0](img_output)
        img_output = self.to_out[1](img_output)  # Identity

        # Project text output
        txt_output = self.to_add_out(txt_output)

        return img_output, txt_output


class AttentionBlock(nn.Module):
    """Attention block with residual connection and normalization."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        use_self_attention: bool = True,
        use_cross_attention: bool = True,
    ):
        """Initialize attention block.

        Args:
            dim: Input/output dimension.
            num_heads: Number of attention heads.
            dim_head: Dimension per head.
            dropout: Dropout probability.
            context_dim: Cross-attention context dimension.
            use_self_attention: Whether to include self-attention.
            use_cross_attention: Whether to include cross-attention.
        """
        super().__init__()
        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention

        if use_self_attention:
            self.norm1 = nn.LayerNorm(dim)
            self.self_attn = SelfAttention(
                query_dim=dim,
                heads=num_heads,
                dim_head=dim_head,
                dropout=dropout,
            )

        if use_cross_attention:
            self.norm2 = nn.LayerNorm(dim)
            self.cross_attn = CrossAttention(
                query_dim=dim,
                cross_attention_dim=context_dim,
                heads=num_heads,
                dim_head=dim_head,
                dropout=dropout,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Input tensor [B, seq_len, dim].
            encoder_hidden_states: Optional context for cross-attention.
            attention_mask: Optional attention mask.

        Returns:
            Output tensor [B, seq_len, dim].
        """
        if self.use_self_attention:
            norm_hidden = self.norm1(hidden_states)
            hidden_states = hidden_states + self.self_attn(norm_hidden, attention_mask)

        if self.use_cross_attention and encoder_hidden_states is not None:
            norm_hidden = self.norm2(hidden_states)
            hidden_states = hidden_states + self.cross_attn(
                norm_hidden, encoder_hidden_states
            )

        return hidden_states


class BasicTransformerBlock(nn.Module):
    """Transformer block with self-attention, cross-attention, and FFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        ff_inner_mult: float = 4.0,
    ):
        """Initialize transformer block.

        Args:
            dim: Input/output dimension.
            num_heads: Number of attention heads.
            dim_head: Dimension per head.
            dropout: Dropout probability.
            context_dim: Cross-attention context dimension.
            activation_fn: Activation function for FFN.
            ff_inner_mult: FFN inner dimension multiplier.
        """
        super().__init__()

        # Self attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = SelfAttention(
            query_dim=dim,
            heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        # Cross attention
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=context_dim,
            heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        # Feed forward
        self.norm3 = nn.LayerNorm(dim)
        inner_dim = int(dim * ff_inner_mult)
        self.ff = FeedForward(dim, inner_dim, activation_fn=activation_fn, dropout=dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Input tensor [B, seq_len, dim].
            encoder_hidden_states: Context for cross-attention.
            attention_mask: Optional attention mask.

        Returns:
            Output tensor [B, seq_len, dim].
        """
        # Self attention
        norm_hidden = self.norm1(hidden_states)
        hidden_states = hidden_states + self.attn1(norm_hidden, attention_mask)

        # Cross attention
        norm_hidden = self.norm2(hidden_states)
        hidden_states = hidden_states + self.attn2(
            norm_hidden,
            encoder_hidden_states if encoder_hidden_states is not None else norm_hidden,
        )

        # Feed forward
        norm_hidden = self.norm3(hidden_states)
        hidden_states = hidden_states + self.ff(norm_hidden)

        return hidden_states


class FeedForward(nn.Module):
    """Feed-forward network with GELU/GEGLU activation."""

    def __init__(
        self,
        dim: int,
        inner_dim: int,
        activation_fn: str = "geglu",
        dropout: float = 0.0,
    ):
        """Initialize feed-forward network.

        Args:
            dim: Input/output dimension.
            inner_dim: Hidden dimension.
            activation_fn: Activation function ("gelu", "geglu").
            dropout: Dropout probability.
        """
        super().__init__()

        if activation_fn == "geglu":
            self.net = nn.Sequential(
                GEGLU(dim, inner_dim),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, inner_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim),
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(hidden_states)


class GEGLU(nn.Module):
    """GEGLU activation function."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
