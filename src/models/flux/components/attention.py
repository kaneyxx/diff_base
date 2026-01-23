"""Flux-specific attention block implementations."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import AdaLayerNormZero, AdaLayerNormZeroSingle
from ...components.attention import JointAttention
from ...components.embeddings import apply_rotary_emb


class FluxJointTransformerBlock(nn.Module):
    """Joint transformer block for processing image and text together.

    Used in both FLUX.1 and FLUX.2. Processes image and text tokens
    jointly through attention, then separately through MLPs.
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

        # Layer norms for image and text
        self.norm1_img = AdaLayerNormZero(hidden_size)
        self.norm1_txt = AdaLayerNormZero(hidden_size)

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

        # MLPs
        mlp_hidden = int(hidden_size * mlp_ratio)

        self.norm2_img = nn.LayerNorm(hidden_size)
        self.mlp_img = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )

        self.norm2_txt = nn.LayerNorm(hidden_size)
        self.mlp_txt = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )

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

        # Norm + modulation
        img_hidden, img_gate_msa, img_shift_mlp, img_scale_mlp, img_gate_mlp = \
            self.norm1_img(img_hidden_states, temb)
        txt_hidden, txt_gate_msa, txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = \
            self.norm1_txt(txt_hidden_states, temb)

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

        # MLP for image
        img_hidden = self.norm2_img(img_hidden_states)
        img_hidden = img_hidden * (1 + img_scale_mlp) + img_shift_mlp
        img_mlp = self.mlp_img(img_hidden)
        img_hidden_states = img_hidden_states + img_gate_mlp * img_mlp

        # MLP for text
        txt_hidden = self.norm2_txt(txt_hidden_states)
        txt_hidden = txt_hidden * (1 + txt_scale_mlp) + txt_shift_mlp
        txt_mlp = self.mlp_txt(txt_hidden)
        txt_hidden_states = txt_hidden_states + txt_gate_mlp * txt_mlp

        return img_hidden_states, txt_hidden_states


class FluxSingleTransformerBlock(nn.Module):
    """Single stream transformer block for processing concatenated tokens.

    Used in both FLUX.1 and FLUX.2. Processes already-concatenated
    image and text tokens through self-attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_norm: bool = False,
    ):
        """Initialize single transformer block.

        Args:
            hidden_size: Hidden dimension.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden dimension multiplier.
            qk_norm: Whether to use QK normalization (FLUX.2).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qk_norm = qk_norm

        # Layer norm
        self.norm = AdaLayerNormZeroSingle(hidden_size)

        # Self attention projections
        self.to_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.to_out = nn.Linear(hidden_size, hidden_size)

        # Optional QK normalization
        if qk_norm:
            from .layers import QKNorm
            self.qk_norm_layer = QKNorm(self.head_dim)
        else:
            self.qk_norm_layer = None

        # MLP
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )

        self.scale = self.head_dim ** -0.5

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
        batch_size, seq_len, _ = hidden_states.shape

        # Norm + modulation
        hidden_states, gate = self.norm(hidden_states, temb)

        # Self attention
        qkv = self.to_qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply QK norm if enabled
        if self.qk_norm_layer is not None:
            q, k = self.qk_norm_layer(q, k)

        # Apply rotary embeddings
        if rotary_emb is not None:
            cos, sin = rotary_emb
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_out = self.to_out(attn_out)

        # MLP
        mlp_out = self.mlp(hidden_states)

        # Combine with gating
        hidden_states = residual + gate * (attn_out + mlp_out)

        return hidden_states
