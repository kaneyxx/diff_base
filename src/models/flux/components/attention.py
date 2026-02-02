"""Flux-specific attention block implementations."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import AdaLayerNormZero, AdaLayerNormZeroSingle, RMSNorm
from ...components.attention import JointAttention
from ...components.embeddings import apply_rotary_emb


class FluxFeedForwardGELU(nn.Module):
    """GELU wrapper with internal projection for ff.net.0.proj naming."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x), approximate="tanh")


class FluxFeedForward(nn.Module):
    """HF-aligned FeedForward producing ff.net.0.proj, ff.net.2 keys.

    Structure:
    - net.0: FluxFeedForwardGELU (with .proj inside)
    - net.1: Dropout (Identity in inference)
    - net.2: Linear output projection
    """

    def __init__(self, dim: int, mult: float = 4.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.ModuleList([
            FluxFeedForwardGELU(dim, inner_dim),  # net.0 with .proj
            nn.Dropout(0.0),                       # net.1
            nn.Linear(inner_dim, dim),             # net.2
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net[0](x)
        x = self.net[1](x)
        x = self.net[2](x)
        return x


class FluxJointTransformerBlock(nn.Module):
    """Joint transformer block for processing image and text together.

    Used in both FLUX.1 and FLUX.2. Processes image and text tokens
    jointly through attention, then separately through MLPs.

    HuggingFace-aligned naming convention:
    - norm1 (image), norm1_context (text) - AdaLayerNormZero
    - attn.norm_q, attn.norm_k, attn.norm_added_q, attn.norm_added_k - QK norm
    - ff (image MLP), ff_context (text MLP)

    Note: No norm2/norm2_context - modulation is applied directly after attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = False,
    ):
        """Initialize joint transformer block.

        Args:
            hidden_size: Hidden dimension.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden dimension multiplier.
            qk_norm: Whether to use QK normalization (FLUX.2).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_norm = qk_norm

        # Layer norms - HF naming: norm1 (image), norm1_context (text)
        self.norm1 = AdaLayerNormZero(hidden_size)
        self.norm1_context = AdaLayerNormZero(hidden_size)

        # Joint attention
        self.attn = JointAttention(
            dim=hidden_size,
            num_heads=num_heads,
        )

        # Optional QK normalization for FLUX.2
        if qk_norm:
            from .layers import QKNorm
            self.qk_norm_layer = QKNorm(hidden_size // num_heads)
        else:
            self.qk_norm_layer = None

        # MLPs - HF naming: ff (image), ff_context (text)
        # Note: No norm2/norm2_context - HuggingFace FLUX applies modulation directly
        self.ff = FluxFeedForward(hidden_size, mlp_ratio)
        self.ff_context = FluxFeedForward(hidden_size, mlp_ratio)

    def forward(
        self,
        img_hidden_states: torch.Tensor,
        txt_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        img_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        txt_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            img_hidden_states: Image tokens [B, img_seq, hidden_size].
            txt_hidden_states: Text tokens [B, txt_seq, hidden_size].
            temb: Time embedding [B, hidden_size].
            img_rotary_emb: Rotary embeddings for image.
            txt_rotary_emb: Rotary embeddings for text.

        Returns:
            Tuple of (image_output, text_output).
        """
        img_residual = img_hidden_states
        txt_residual = txt_hidden_states

        # Norm + modulation (using HF-aligned names)
        img_hidden, img_gate_msa, img_shift_mlp, img_scale_mlp, img_gate_mlp = \
            self.norm1(img_hidden_states, temb)
        txt_hidden, txt_gate_msa, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = \
            self.norm1_context(txt_hidden_states, temb)

        # Joint attention
        img_attn, txt_attn = self.attn(
            img_hidden,
            txt_hidden,
            img_rotary_emb,
            txt_rotary_emb,
        )

        # Apply attention with gating
        img_hidden_states = img_residual + img_gate_msa * img_attn
        txt_hidden_states = txt_residual + txt_gate_msa * txt_attn

        # MLP for image - apply modulation directly (no norm2 in HuggingFace)
        img_hidden = img_hidden_states * (1 + img_scale_mlp) + img_shift_mlp
        img_mlp = self.ff(img_hidden)
        img_hidden_states = img_hidden_states + img_gate_mlp * img_mlp

        # MLP for text - apply modulation directly (no norm2_context in HuggingFace)
        txt_hidden = txt_hidden_states * (1 + txt_scale_mlp) + txt_shift_mlp
        txt_mlp = self.ff_context(txt_hidden)
        txt_hidden_states = txt_hidden_states + txt_gate_mlp * txt_mlp

        return img_hidden_states, txt_hidden_states


class FluxSingleAttention(nn.Module):
    """Single attention with separate Q/K/V for HF compatibility.

    HuggingFace-aligned naming:
    - attn.to_q, attn.to_k, attn.to_v
    - attn.norm_q, attn.norm_k (RMSNorm for QK normalization)

    Note: FLUX.1 single blocks do NOT have a to_out projection.
    The output is directly the reshaped attention output.
    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Separate Q/K/V projections (HF naming)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        # QK normalization (FLUX.1 single blocks always have this)
        self.norm_q = RMSNorm(self.head_dim)
        self.norm_k = RMSNorm(self.head_dim)

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

        # Compute separate Q, K, V
        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply QK normalization (always applied in FLUX.1 single blocks)
        q = self.norm_q(q)
        k = self.norm_k(k)

        # Apply rotary embeddings
        if rotary_emb is not None:
            cos, sin = rotary_emb
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        # Reshape output (no to_out projection in FLUX.1 single blocks)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, -1)

        return attn_out


class FluxSingleTransformerBlock(nn.Module):
    """Single stream transformer block for processing concatenated tokens.

    Used in both FLUX.1 and FLUX.2. Processes already-concatenated
    image and text tokens through self-attention.

    HuggingFace-aligned naming convention:
    - norm.linear (AdaLayerNormZeroSingle)
    - attn.to_q, attn.to_k, attn.to_v, attn.norm_q, attn.norm_k
    - proj_mlp, proj_out
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        """Initialize single transformer block.

        Args:
            hidden_size: Hidden dimension.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden dimension multiplier.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Layer norm
        self.norm = AdaLayerNormZeroSingle(hidden_size)

        # Attention with separate Q/K/V and built-in QK norm (HF naming)
        self.attn = FluxSingleAttention(hidden_size, num_heads)

        # MLP - HF naming: proj_mlp + activation, then proj_out combines attn + mlp
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.proj_mlp = nn.Linear(hidden_size, mlp_hidden)
        self.act_mlp = nn.GELU(approximate="tanh")
        # proj_out takes concatenated [attn_out, mlp_out] as input
        self.proj_out = nn.Linear(hidden_size + mlp_hidden, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Input tensor [B, seq_len, hidden_size].
            temb: Time embedding [B, hidden_size].
            rotary_emb: Optional rotary embeddings.

        Returns:
            Output tensor.
        """
        residual = hidden_states

        # Norm + modulation
        hidden_states, gate = self.norm(hidden_states, temb)

        # Self attention using FluxSingleAttention (QK norm is built-in)
        attn_out = self.attn(
            hidden_states,
            rotary_emb=rotary_emb,
        )

        # MLP (using HF-aligned names)
        mlp_out = self.proj_mlp(hidden_states)
        mlp_out = self.act_mlp(mlp_out)

        # Concatenate attention + MLP, then project (HF approach)
        combined = torch.cat([attn_out, mlp_out], dim=-1)
        proj_out = self.proj_out(combined)

        # Combine with gating
        hidden_states = residual + gate * proj_out

        return hidden_states
