"""SD3-specific attention implementations.

SD3 uses MM-DiT (Multimodal Diffusion Transformer) architecture with:
- QKNorm: RMSNorm applied to queries and keys for stability
- JointTransformerBlock: Processes image and text jointly
- Self-attention with modulation from conditioning
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, AdaLayerNormZero, FeedForward, modulate


class QKNorm(nn.Module):
    """Query-Key normalization using RMSNorm.

    SD3 applies RMSNorm to queries and keys separately before computing
    attention scores. This helps with numerical stability during training.
    """

    def __init__(self, head_dim: int, eps: float = 1e-6):
        """Initialize QKNorm.

        Args:
            head_dim: Dimension per attention head.
            eps: Epsilon for RMSNorm.
        """
        super().__init__()
        self.query_norm = RMSNorm(head_dim, eps=eps)
        self.key_norm = RMSNorm(head_dim, eps=eps)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply QK normalization.

        Args:
            query: Query tensor [B, num_heads, seq_len, head_dim].
            key: Key tensor [B, num_heads, seq_len, head_dim].

        Returns:
            Tuple of (normalized_query, normalized_key).
        """
        return self.query_norm(query), self.key_norm(key)


class Attention(nn.Module):
    """Multi-head attention with optional QK normalization.

    SD3-style attention that supports:
    - QK normalization for numerical stability
    - Optional separate context (for cross-attention)
    - Head dimension configuration
    """

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        qk_norm: bool = True,
        context_dim: Optional[int] = None,
        context_pre_only: bool = False,
    ):
        """Initialize Attention.

        Args:
            query_dim: Query input dimension.
            heads: Number of attention heads.
            dim_head: Dimension per head.
            dropout: Dropout probability.
            bias: Whether to use bias in projections.
            qk_norm: Whether to apply QK normalization.
            context_dim: Context dimension for cross-attention.
            context_pre_only: If True, only project context once (optimization).
        """
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5

        context_dim = context_dim or query_dim

        # Query projection
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)

        # Key/Value projections (from context for cross-attn)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=bias)

        # QK normalization
        self.qk_norm = QKNorm(dim_head) if qk_norm else None

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim, bias=bias),
            nn.Dropout(dropout),
        )

        self.context_pre_only = context_pre_only

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply attention.

        Args:
            hidden_states: Query input [B, seq_len, query_dim].
            context: Key/value input for cross-attention [B, ctx_len, context_dim].
            attention_mask: Optional attention mask.

        Returns:
            Attention output [B, seq_len, query_dim].
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Self-attention or cross-attention
        if context is None:
            context = hidden_states

        # Project to Q, K, V
        query = self.to_q(hidden_states)
        key = self.to_k(context)
        value = self.to_v(context)

        # Reshape to [B, heads, seq_len, dim_head]
        query = query.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2)

        # Apply QK normalization
        if self.qk_norm is not None:
            query, key = self.qk_norm(query, key)

        # Scaled dot-product attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            scale=self.scale,
        )

        # Reshape back: [B, heads, seq_len, dim_head] -> [B, seq_len, inner_dim]
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_len, self.inner_dim)

        # Output projection
        hidden_states = self.to_out(hidden_states)

        return hidden_states


class JointTransformerBlock(nn.Module):
    """Joint transformer block for SD3 MM-DiT.

    Processes image and text tokens jointly with:
    1. AdaLN modulation for both streams
    2. Concatenated self-attention (image + text attend to each other)
    3. Separate MLPs for image and text streams
    4. Gating for residual connections

    This is the core building block of the MM-DiT architecture.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        context_dim: int = 4096,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
    ):
        """Initialize JointTransformerBlock.

        Args:
            hidden_size: Hidden dimension for image stream.
            num_heads: Number of attention heads.
            context_dim: Text context dimension (T5 hidden size).
            mlp_ratio: MLP expansion ratio.
            qk_norm: Whether to apply QK normalization.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.context_dim = context_dim

        # Image stream normalization with AdaLN
        self.norm1_img = AdaLayerNormZero(hidden_size)
        self.norm2_img = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Text stream normalization with AdaLN
        # Note: We project conditioning from hidden_size to context_dim
        self.txt_cond_proj = nn.Linear(hidden_size, context_dim)
        self.norm1_txt = AdaLayerNormZero(context_dim)
        self.norm2_txt = nn.LayerNorm(context_dim, elementwise_affine=False, eps=1e-6)

        # Joint attention - project both streams to same dimension for attention
        head_dim = hidden_size // num_heads
        self.heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        # Image projections
        self.to_q_img = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_k_img = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_v_img = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_out_img = nn.Linear(inner_dim, hidden_size, bias=False)

        # Text projections (project context_dim to same inner_dim)
        self.to_q_txt = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_k_txt = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v_txt = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out_txt = nn.Linear(inner_dim, context_dim, bias=False)

        # QK normalization
        self.qk_norm = QKNorm(head_dim) if qk_norm else None

        # MLPs for each stream
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp_img = FeedForward(hidden_size, mlp_hidden, activation="gelu_tanh")

        mlp_hidden_txt = int(context_dim * mlp_ratio)
        self.mlp_txt = FeedForward(context_dim, mlp_hidden_txt, activation="gelu_tanh")

        self._gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of joint transformer block.

        Args:
            hidden_states: Image tokens [B, img_seq, hidden_size].
            encoder_hidden_states: Text tokens [B, txt_seq, context_dim].
            temb: Conditioning embedding [B, hidden_size].

        Returns:
            Tuple of (image_output, text_output).
        """
        # Save residuals
        img_residual = hidden_states
        txt_residual = encoder_hidden_states

        # Project conditioning for text stream (hidden_size -> context_dim)
        txt_temb = self.txt_cond_proj(temb)

        # AdaLN modulation
        img_hidden, img_gate_msa, img_shift_mlp, img_scale_mlp, img_gate_mlp = \
            self.norm1_img(hidden_states, temb)
        txt_hidden, txt_gate_msa, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = \
            self.norm1_txt(encoder_hidden_states, txt_temb)

        # Joint self-attention
        img_attn, txt_attn = self._joint_attention(img_hidden, txt_hidden)

        # Apply attention with gating
        hidden_states = img_residual + img_gate_msa * img_attn
        encoder_hidden_states = txt_residual + txt_gate_msa * txt_attn

        # MLP for image stream
        img_norm = self.norm2_img(hidden_states)
        img_norm = modulate(img_norm, img_shift_mlp, img_scale_mlp)
        img_mlp = self.mlp_img(img_norm)
        hidden_states = hidden_states + img_gate_mlp * img_mlp

        # MLP for text stream
        txt_norm = self.norm2_txt(encoder_hidden_states)
        txt_norm = modulate(txt_norm, txt_shift_mlp, txt_scale_mlp)
        txt_mlp = self.mlp_txt(txt_norm)
        encoder_hidden_states = encoder_hidden_states + txt_gate_mlp * txt_mlp

        return hidden_states, encoder_hidden_states

    def _joint_attention(
        self,
        img_hidden: torch.Tensor,
        txt_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute joint attention between image and text.

        Args:
            img_hidden: Normalized image tokens [B, img_seq, hidden_size].
            txt_hidden: Normalized text tokens [B, txt_seq, context_dim].

        Returns:
            Tuple of (img_attn_output, txt_attn_output).
        """
        batch_size = img_hidden.shape[0]
        img_seq_len = img_hidden.shape[1]
        txt_seq_len = txt_hidden.shape[1]

        # Project to Q, K, V for both streams
        q_img = self.to_q_img(img_hidden)
        k_img = self.to_k_img(img_hidden)
        v_img = self.to_v_img(img_hidden)

        q_txt = self.to_q_txt(txt_hidden)
        k_txt = self.to_k_txt(txt_hidden)
        v_txt = self.to_v_txt(txt_hidden)

        # Reshape to [B, heads, seq_len, head_dim]
        q_img = q_img.view(batch_size, img_seq_len, self.heads, self.head_dim).transpose(1, 2)
        k_img = k_img.view(batch_size, img_seq_len, self.heads, self.head_dim).transpose(1, 2)
        v_img = v_img.view(batch_size, img_seq_len, self.heads, self.head_dim).transpose(1, 2)

        q_txt = q_txt.view(batch_size, txt_seq_len, self.heads, self.head_dim).transpose(1, 2)
        k_txt = k_txt.view(batch_size, txt_seq_len, self.heads, self.head_dim).transpose(1, 2)
        v_txt = v_txt.view(batch_size, txt_seq_len, self.heads, self.head_dim).transpose(1, 2)

        # Apply QK normalization
        if self.qk_norm is not None:
            q_img, k_img = self.qk_norm(q_img, k_img)
            q_txt, k_txt = self.qk_norm(q_txt, k_txt)

        # Concatenate for joint attention: [B, heads, img_seq + txt_seq, head_dim]
        q = torch.cat([q_img, q_txt], dim=2)
        k = torch.cat([k_img, k_txt], dim=2)
        v = torch.cat([v_img, v_txt], dim=2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=0.0,
            scale=scale,
        )

        # Split back to image and text
        attn_output = attn_output.transpose(1, 2)  # [B, seq, heads, head_dim]
        attn_img = attn_output[:, :img_seq_len].reshape(batch_size, img_seq_len, -1)
        attn_txt = attn_output[:, img_seq_len:].reshape(batch_size, txt_seq_len, -1)

        # Output projections
        attn_img = self.to_out_img(attn_img)
        attn_txt = self.to_out_txt(attn_txt)

        return attn_img, attn_txt

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for this block."""
        self._gradient_checkpointing = True
