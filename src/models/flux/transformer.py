"""Flux Transformer (DiT) architecture."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..components.attention import JointAttention
from ..components.embeddings import RotaryEmbedding, MLPEmbedder
from ..components.transformer import RMSNorm


class FluxSingleTransformerBlock(nn.Module):
    """Single stream transformer block (processes image only)."""

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
        head_dim = hidden_size // num_heads

        # Layer norms
        self.norm = AdaLayerNormZeroSingle(hidden_size)

        # Self attention
        self.attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True,
        )

        # MLP
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )

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

        # Self attention
        attn_out, _ = self.attn(hidden_states, hidden_states, hidden_states)

        # MLP
        mlp_out = self.mlp(hidden_states)

        # Combine with gating
        hidden_states = residual + gate * (attn_out + mlp_out)

        return hidden_states


class FluxJointTransformerBlock(nn.Module):
    """Joint transformer block (processes image and text together)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        """Initialize joint transformer block.

        Args:
            hidden_size: Hidden dimension.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden dimension multiplier.
        """
        super().__init__()
        self.hidden_size = hidden_size

        # Layer norms for image
        self.norm1_img = AdaLayerNormZero(hidden_size)

        # Layer norms for text
        self.norm1_txt = AdaLayerNormZero(hidden_size)

        # Joint attention
        self.attn = JointAttention(
            dim=hidden_size,
            num_heads=num_heads,
        )

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


class AdaLayerNormZero(nn.Module):
    """Adaptive layer norm with zero initialization."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, 6 * hidden_size)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.silu(emb)
        emb = self.linear(emb)[:, None, :]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=-1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Module):
    """Single adaptive layer norm for single stream blocks."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, 3 * hidden_size)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.silu(emb)
        emb = self.linear(emb)[:, None, :]
        shift, scale, gate = emb.chunk(3, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x, gate


class FluxTransformer(nn.Module):
    """Flux DiT (Diffusion Transformer) architecture."""

    def __init__(self, config: DictConfig):
        """Initialize Flux transformer.

        Args:
            config: Transformer configuration.
        """
        super().__init__()
        self.config = config

        hidden_size = config.get("hidden_size", 3072)
        num_heads = config.get("num_attention_heads", 24)
        num_layers = config.get("num_layers", 19)
        num_single_layers = config.get("num_single_layers", 38)
        in_channels = config.get("in_channels", 64)
        pooled_projection_dim = config.get("pooled_projection_dim", 768)

        # Input projection
        self.x_embedder = nn.Linear(in_channels, hidden_size)

        # Time/guidance embedding
        self.time_embed = MLPEmbedder(256, hidden_size)

        if config.get("guidance_embeds", True):
            self.guidance_embed = MLPEmbedder(256, hidden_size)
        else:
            self.guidance_embed = None

        # Pooled text projection
        self.pooled_text_embed = nn.Linear(pooled_projection_dim, hidden_size)

        # Positional embedding (RoPE)
        head_dim = hidden_size // num_heads
        self.rope = RotaryEmbedding(head_dim)

        # Joint attention blocks
        self.joint_blocks = nn.ModuleList([
            FluxJointTransformerBlock(hidden_size, num_heads)
            for _ in range(num_layers)
        ])

        # Single stream blocks
        self.single_blocks = nn.ModuleList([
            FluxSingleTransformerBlock(hidden_size, num_heads)
            for _ in range(num_single_layers)
        ])

        # Output projection
        self.norm_out = nn.LayerNorm(hidden_size)
        self.proj_out = nn.Linear(hidden_size, in_channels)

        self._gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Latent tensor [B, seq_len, in_channels].
            timestep: Timestep values [B].
            encoder_hidden_states: Text embeddings [B, txt_seq, hidden_size].
            pooled_projections: Pooled text embeddings [B, pooled_dim].
            guidance: Optional guidance scale [B].

        Returns:
            Predicted output [B, seq_len, in_channels].
        """
        # Embed inputs
        hidden_states = self.x_embedder(hidden_states)

        # Time embedding
        temb = self._get_timestep_embedding(timestep)
        temb = self.time_embed(temb)

        if guidance is not None and self.guidance_embed is not None:
            guidance_emb = self._get_timestep_embedding(guidance)
            temb = temb + self.guidance_embed(guidance_emb)

        temb = temb + self.pooled_text_embed(pooled_projections)

        # Get rotary embeddings
        img_seq_len = hidden_states.shape[1]
        txt_seq_len = encoder_hidden_states.shape[1]

        img_rotary_emb = self.rope(img_seq_len, hidden_states.device)
        txt_rotary_emb = self.rope(txt_seq_len, hidden_states.device)

        # Joint attention blocks
        txt_hidden = encoder_hidden_states
        for block in self.joint_blocks:
            hidden_states, txt_hidden = block(
                hidden_states,
                txt_hidden,
                temb,
                img_rotary_emb,
                txt_rotary_emb,
            )

        # Concatenate for single stream
        hidden_states = torch.cat([txt_hidden, hidden_states], dim=1)

        # Single stream blocks
        for block in self.single_blocks:
            hidden_states = block(hidden_states, temb)

        # Extract image tokens
        hidden_states = hidden_states[:, txt_seq_len:]

        # Project output
        hidden_states = self.norm_out(hidden_states)
        output = self.proj_out(hidden_states)

        return output

    def _get_timestep_embedding(
        self,
        timestep: torch.Tensor,
        dim: int = 256,
    ) -> torch.Tensor:
        """Get sinusoidal timestep embedding."""
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timestep.device) * -emb)
        emb = timestep[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing."""
        self._gradient_checkpointing = True
