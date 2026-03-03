"""FLUX.2-specific transformer block implementations.

These blocks match HuggingFace's Flux2TransformerBlock and
Flux2SingleTransformerBlock for correct weight loading.

Key differences from FLUX.1 blocks:
- Shared modulation (Flux2Modulation) instead of per-block AdaLayerNormZero
- SwiGLU feed-forward (Flux2FeedForward) instead of GELU
- All bias=False
- Single blocks use fused to_qkv_mlp_proj projection
- Joint blocks have add_q_proj (FLUX.1 also has it in HF)
- norm2/norm2_context present (unlike FLUX.1 HF which skips them)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.layers import RMSNorm
from ...components.embeddings import apply_rotary_emb


class Flux2SwiGLU(nn.Module):
    """SwiGLU activation for FLUX.2 feed-forward networks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


class Flux2FeedForward(nn.Module):
    """FLUX.2 feed-forward with SwiGLU activation.

    State dict keys: ff.linear_in.weight, ff.linear_out.weight (no biases).
    """

    def __init__(self, dim: int, mult: float = 3.0, bias: bool = False):
        """Initialize Flux2FeedForward.

        Args:
            dim: Input/output dimension.
            mult: Hidden dimension multiplier.
            bias: Whether to use bias.
        """
        super().__init__()
        inner_dim = int(dim * mult)
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=bias)  # *2 for SwiGLU gate
        self.act_fn = Flux2SwiGLU()
        self.linear_out = nn.Linear(inner_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.act_fn(x)
        x = self.linear_out(x)
        return x


class Flux2Attention(nn.Module):
    """FLUX.2 joint attention for double-stream blocks.

    Matches HuggingFace Flux2Attention with added_kv_proj_dim.

    State dict keys:
    - attn.to_q.weight, attn.to_k.weight, attn.to_v.weight
    - attn.norm_q.weight, attn.norm_k.weight
    - attn.to_out.0.weight (ModuleList)
    - attn.add_q_proj.weight, attn.add_k_proj.weight, attn.add_v_proj.weight
    - attn.norm_added_q.weight, attn.norm_added_k.weight
    - attn.to_add_out.weight
    """

    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        """Initialize Flux2Attention.

        Args:
            dim: Hidden dimension.
            num_heads: Number of attention heads.
            bias: Whether to use bias in projections.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.inner_dim = dim
        self.scale = self.head_dim ** -0.5

        # Image stream projections
        self.to_q = nn.Linear(dim, dim, bias=bias)
        self.to_k = nn.Linear(dim, dim, bias=bias)
        self.to_v = nn.Linear(dim, dim, bias=bias)
        self.norm_q = RMSNorm(self.head_dim)
        self.norm_k = RMSNorm(self.head_dim)
        # ModuleList for to_out.0 key naming
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=bias), nn.Identity()])

        # Text stream projections (added_kv_proj)
        self.add_q_proj = nn.Linear(dim, dim, bias=bias)
        self.add_k_proj = nn.Linear(dim, dim, bias=bias)
        self.add_v_proj = nn.Linear(dim, dim, bias=bias)
        self.norm_added_q = RMSNorm(self.head_dim)
        self.norm_added_k = RMSNorm(self.head_dim)
        self.to_add_out = nn.Linear(dim, dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        text_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with joint attention.

        Args:
            hidden_states: Image tokens [B, img_seq, dim].
            encoder_hidden_states: Text tokens [B, txt_seq, dim].
            image_rotary_emb: RoPE for image tokens.
            text_rotary_emb: RoPE for text tokens.

        Returns:
            Tuple of (image_output, text_output).
        """
        batch_size = hidden_states.shape[0]
        img_seq_len = hidden_states.shape[1]
        txt_seq_len = encoder_hidden_states.shape[1]

        # Image Q/K/V
        q_img = self.to_q(hidden_states)
        k_img = self.to_k(hidden_states)
        v_img = self.to_v(hidden_states)

        # Text Q/K/V
        q_txt = self.add_q_proj(encoder_hidden_states)
        k_txt = self.add_k_proj(encoder_hidden_states)
        v_txt = self.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head
        def reshape(x, seq_len):
            return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q_img = reshape(q_img, img_seq_len)
        k_img = reshape(k_img, img_seq_len)
        v_img = reshape(v_img, img_seq_len)
        q_txt = reshape(q_txt, txt_seq_len)
        k_txt = reshape(k_txt, txt_seq_len)
        v_txt = reshape(v_txt, txt_seq_len)

        # QK normalization
        q_img = self.norm_q(q_img)
        k_img = self.norm_k(k_img)
        q_txt = self.norm_added_q(q_txt)
        k_txt = self.norm_added_k(k_txt)

        # Apply RoPE
        if image_rotary_emb is not None:
            q_img = apply_rotary_emb(q_img, image_rotary_emb)
            k_img = apply_rotary_emb(k_img, image_rotary_emb)
        if text_rotary_emb is not None:
            q_txt = apply_rotary_emb(q_txt, text_rotary_emb)
            k_txt = apply_rotary_emb(k_txt, text_rotary_emb)

        # Joint attention
        q = torch.cat([q_txt, q_img], dim=2)
        k = torch.cat([k_txt, k_img], dim=2)
        v = torch.cat([v_txt, v_img], dim=2)

        attn_out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        # Split and reshape
        attn_out = attn_out.transpose(1, 2).reshape(
            batch_size, txt_seq_len + img_seq_len, -1
        )
        txt_output = attn_out[:, :txt_seq_len]
        img_output = attn_out[:, txt_seq_len:]

        # Project outputs
        img_output = self.to_out[0](img_output)
        img_output = self.to_out[1](img_output)  # Identity
        txt_output = self.to_add_out(txt_output)

        return img_output, txt_output


class Flux2ParallelSelfAttention(nn.Module):
    """FLUX.2 parallel self-attention for single-stream blocks.

    Uses a fused to_qkv_mlp_proj projection that combines Q/K/V and MLP
    gate projections into a single matmul for efficiency.

    State dict keys:
    - attn.to_qkv_mlp_proj.weight
    - attn.norm_q.weight, attn.norm_k.weight
    - attn.to_out.weight (plain Linear, not ModuleList)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 3.0,
        bias: bool = False,
    ):
        """Initialize Flux2ParallelSelfAttention.

        Args:
            dim: Hidden dimension.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden dimension multiplier.
            bias: Whether to use bias.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.inner_dim = dim
        self.scale = self.head_dim ** -0.5

        # MLP dimensions
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_mult_factor = 2  # For SwiGLU gate

        # Fused QKV + MLP projection
        fused_out_dim = 3 * dim + self.mlp_hidden_dim * self.mlp_mult_factor
        self.to_qkv_mlp_proj = nn.Linear(dim, fused_out_dim, bias=bias)

        # QK normalization
        self.norm_q = RMSNorm(self.head_dim)
        self.norm_k = RMSNorm(self.head_dim)

        # SwiGLU activation for MLP path
        self.act_fn = Flux2SwiGLU()

        # Output projection: takes concatenated [attn_out, mlp_out]
        self.to_out = nn.Linear(dim + self.mlp_hidden_dim, dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Input tensor [B, seq_len, dim].
            rotary_emb: Optional rotary embeddings (cos, sin).

        Returns:
            Output tensor [B, seq_len, dim].
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Fused projection
        projected = self.to_qkv_mlp_proj(hidden_states)

        # Split into QKV and MLP parts
        qkv, mlp_hidden = torch.split(
            projected,
            [3 * self.inner_dim, self.mlp_hidden_dim * self.mlp_mult_factor],
            dim=-1,
        )
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # QK normalization
        q = self.norm_q(q)
        k = self.norm_k(k)

        # Apply RoPE
        if rotary_emb is not None:
            q = apply_rotary_emb(q, rotary_emb)
            k = apply_rotary_emb(k, rotary_emb)

        # Attention
        attn_out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, -1)

        # MLP path (SwiGLU)
        mlp_out = self.act_fn(mlp_hidden)

        # Combine and project
        combined = torch.cat([attn_out, mlp_out], dim=-1)
        output = self.to_out(combined)

        return output


class Flux2TransformerBlock(nn.Module):
    """FLUX.2 double-stream transformer block.

    Matches HuggingFace Flux2TransformerBlock structure.
    Modulation is computed externally (shared across blocks) and passed in.

    State dict keys:
    - attn.* (Flux2Attention)
    - ff.* (Flux2FeedForward)
    - ff_context.* (Flux2FeedForward)
    - norm1, norm1_context, norm2, norm2_context have no parameters
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 3.0):
        """Initialize Flux2TransformerBlock.

        Args:
            hidden_size: Hidden dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Feed-forward hidden dimension multiplier.
        """
        super().__init__()
        self.hidden_size = hidden_size

        # Layer norms (no learnable parameters)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm1_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Joint attention
        self.attn = Flux2Attention(hidden_size, num_heads, bias=False)

        # Feed-forward (SwiGLU)
        self.ff = Flux2FeedForward(hidden_size, mult=mlp_ratio, bias=False)
        self.ff_context = Flux2FeedForward(hidden_size, mult=mlp_ratio, bias=False)

    def forward(
        self,
        img_hidden_states: torch.Tensor,
        txt_hidden_states: torch.Tensor,
        img_mod: Tuple,
        txt_mod: Tuple,
        img_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        txt_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            img_hidden_states: Image tokens [B, img_seq, hidden_size].
            txt_hidden_states: Text tokens [B, txt_seq, hidden_size].
            img_mod: Image modulation tuple of 2 sets:
                ((shift_attn, scale_attn, gate_attn), (shift_ff, scale_ff, gate_ff))
            txt_mod: Text modulation tuple (same structure).
            img_rotary_emb: RoPE for image tokens.
            txt_rotary_emb: RoPE for text tokens.

        Returns:
            Tuple of (image_output, text_output).
        """
        # Unpack modulation parameters
        (img_shift_attn, img_scale_attn, img_gate_attn) = img_mod[0]
        (img_shift_ff, img_scale_ff, img_gate_ff) = img_mod[1]
        (txt_shift_attn, txt_scale_attn, txt_gate_attn) = txt_mod[0]
        (txt_shift_ff, txt_scale_ff, txt_gate_ff) = txt_mod[1]

        # === Attention ===
        img_residual = img_hidden_states
        txt_residual = txt_hidden_states

        # Norm + modulate (pre-attention)
        img_normed = self.norm1(img_hidden_states) * (1 + img_scale_attn) + img_shift_attn
        txt_normed = self.norm1_context(txt_hidden_states) * (1 + txt_scale_attn) + txt_shift_attn

        # Joint attention
        img_attn, txt_attn = self.attn(
            img_normed, txt_normed,
            image_rotary_emb=img_rotary_emb,
            text_rotary_emb=txt_rotary_emb,
        )

        # Residual + gate
        img_hidden_states = img_residual + img_gate_attn * img_attn
        txt_hidden_states = txt_residual + txt_gate_attn * txt_attn

        # === Feed-forward ===
        img_residual = img_hidden_states
        txt_residual = txt_hidden_states

        # Norm + modulate (pre-FF)
        img_normed = self.norm2(img_hidden_states) * (1 + img_scale_ff) + img_shift_ff
        txt_normed = self.norm2_context(txt_hidden_states) * (1 + txt_scale_ff) + txt_shift_ff

        # Feed-forward
        img_ff = self.ff(img_normed)
        txt_ff = self.ff_context(txt_normed)

        # Residual + gate
        img_hidden_states = img_residual + img_gate_ff * img_ff
        txt_hidden_states = txt_residual + txt_gate_ff * txt_ff

        return img_hidden_states, txt_hidden_states


class Flux2SingleTransformerBlock(nn.Module):
    """FLUX.2 single-stream transformer block.

    Uses parallel attention+MLP via fused projection.
    Modulation is computed externally and passed in.

    State dict keys:
    - attn.* (Flux2ParallelSelfAttention)
    - norm has no parameters (elementwise_affine=False)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 3.0,
    ):
        """Initialize Flux2SingleTransformerBlock.

        Args:
            hidden_size: Hidden dimension.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden dimension multiplier.
        """
        super().__init__()
        self.hidden_size = hidden_size

        # Layer norm (no learnable parameters)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Parallel self-attention + MLP
        self.attn = Flux2ParallelSelfAttention(
            hidden_size, num_heads, mlp_ratio=mlp_ratio, bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        mod: Tuple,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Input tensor [B, seq_len, hidden_size].
            mod: Modulation tuple: (shift, scale, gate).
            rotary_emb: Optional rotary embeddings.

        Returns:
            Output tensor [B, seq_len, hidden_size].
        """
        shift, scale, gate = mod

        residual = hidden_states

        # Norm + modulate
        hidden_states = self.norm(hidden_states) * (1 + scale) + shift

        # Parallel attention + MLP
        output = self.attn(hidden_states, rotary_emb=rotary_emb)

        # Residual + gate
        hidden_states = residual + gate * output

        return hidden_states
