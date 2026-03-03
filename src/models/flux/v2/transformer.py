"""FLUX.2 Transformer (DiT) architecture — HuggingFace aligned.

Matches HuggingFace diffusers Flux2Transformer2DModel for correct weight loading.

Key differences from FLUX.1:
- Shared modulation (Flux2Modulation) instead of per-block AdaLayerNormZero
- SwiGLU feed-forward instead of GELU
- All bias=False
- 4D RoPE axes_dim=(32,32,32,32), rope_theta=2000
- Single blocks use fused to_qkv_mlp_proj projection
- norm2/norm2_context present in double blocks
- Flux2TimestepGuidanceEmbeddings for time/guidance embedding

Variant configurations:
- dev: 8 joint + 48 single blocks, 32 latent channels, Mistral-3 text encoder
- klein-4b: 5 joint + 20 single blocks, 128 latent channels, Qwen3-4B
- klein-9b: 6 joint + 24 single blocks, 128 latent channels, Qwen3-8B

Image Editing Support:
- Kontext Mode: Reference image editing via sequence-wise concatenation
- Fill Mode: Inpainting via channel-wise concatenation

HuggingFace state dict key naming:
- transformer_blocks.{i}.* (double-stream)
- single_transformer_blocks.{i}.* (single-stream)
- time_guidance_embed.* (timestep + guidance embedding)
- double_stream_modulation_img.*, double_stream_modulation_txt.*
- single_stream_modulation.*
- pos_embed (no trainable params)
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..base_transformer import FluxTransformerBase
from ..components.embeddings import (
    FluxPosEmbed,
    compute_rope_from_position_ids,
)
from ..components.layers import AdaLayerNormContinuous, Flux2Modulation
from ...components.embeddings import TimestepEmbedding, Timesteps
from .blocks import Flux2TransformerBlock, Flux2SingleTransformerBlock
from .conditioning import create_position_ids


# FLUX.2 RoPE defaults
FLUX2_AXES_DIM = (32, 32, 32, 32)
FLUX2_ROPE_THETA = 2000


class Flux2TimestepGuidanceEmbeddings(nn.Module):
    """Combined timestep and guidance embeddings for FLUX.2.

    Matches HuggingFace structure with state dict keys:
    - time_guidance_embed.time_proj.* (no trainable params)
    - time_guidance_embed.timestep_embedder.linear_1/linear_2
    - time_guidance_embed.guidance_embedder.linear_1/linear_2

    Note: Unlike FLUX.1, there is no pooled text embedding here.
    The pooled text projection is handled separately by context_embedder.
    """

    def __init__(self, in_channels: int = 256, embedding_dim: int = 3072, bias: bool = False):
        """Initialize Flux2TimestepGuidanceEmbeddings.

        Args:
            in_channels: Sinusoidal embedding dimension.
            embedding_dim: Output dimension (model hidden_size).
            bias: Whether to use bias in projections.
        """
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=in_channels,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=embedding_dim,
            sample_proj_bias=bias,
        )
        self.guidance_embedder = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=embedding_dim,
            sample_proj_bias=bias,
        )

    def forward(
        self,
        timestep: torch.Tensor,
        guidance: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            timestep: Timestep values [B] (already scaled by 1000).
            guidance: Guidance scale [B] (already scaled by 1000).

        Returns:
            Combined time+guidance embedding [B, embedding_dim].
        """
        t_emb = self.time_proj(timestep)
        t_emb = self.timestep_embedder(t_emb)

        g_emb = self.time_proj(guidance)
        g_emb = self.guidance_embedder(g_emb)

        return t_emb + g_emb


class Flux2Transformer(FluxTransformerBase):
    """FLUX.2 DiT (Diffusion Transformer) — HuggingFace aligned.

    Supports dev, klein-4b, and klein-9b variants with different
    block counts and dimensions.
    """

    # Variant-specific configurations
    VARIANT_CONFIGS = {
        "dev": {
            "num_layers": 8,
            "num_single_layers": 48,
            "hidden_size": 3072,
            "num_attention_heads": 24,
            "attention_head_dim": 128,
            "in_channels": 128,  # 32 latent channels * 4
            "pooled_projection_dim": 4096,
            "guidance_embeds": True,
            "mlp_ratio": 3.0,
        },
        "klein-4b": {
            "num_layers": 5,
            "num_single_layers": 20,
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "attention_head_dim": 128,
            "in_channels": 512,  # 128 latent channels * 4
            "pooled_projection_dim": 4096,
            "guidance_embeds": True,
            "mlp_ratio": 3.0,
        },
        "klein-9b": {
            "num_layers": 6,
            "num_single_layers": 24,
            "hidden_size": 2560,
            "num_attention_heads": 20,
            "attention_head_dim": 128,
            "in_channels": 512,  # 128 latent channels * 4
            "pooled_projection_dim": 4096,
            "guidance_embeds": True,
            "mlp_ratio": 3.0,
        },
    }

    def __init__(self, config: DictConfig, variant: str = "dev"):
        """Initialize FLUX.2 transformer.

        Args:
            config: Transformer configuration.
            variant: Model variant ("dev", "klein-4b", or "klein-9b").
        """
        super().__init__()
        self.config = config
        self.variant = variant

        # Get variant defaults, allow config overrides
        variant_cfg = self.VARIANT_CONFIGS.get(variant, self.VARIANT_CONFIGS["dev"])

        hidden_size = config.get("hidden_size", variant_cfg["hidden_size"])
        num_heads = config.get("num_attention_heads", variant_cfg["num_attention_heads"])
        num_layers = config.get("num_layers", variant_cfg["num_layers"])
        num_single_layers = config.get("num_single_layers", variant_cfg["num_single_layers"])
        in_channels = config.get("in_channels", variant_cfg["in_channels"])
        pooled_projection_dim = config.get("pooled_projection_dim", variant_cfg["pooled_projection_dim"])
        guidance_embeds = config.get("guidance_embeds", variant_cfg["guidance_embeds"])
        mlp_ratio = config.get("mlp_ratio", variant_cfg["mlp_ratio"])

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.guidance_embeds = guidance_embeds

        # RoPE configuration — FLUX.2 uses 4D axes, theta=2000
        self.axes_dim = tuple(config.get("axes_dims_rope", FLUX2_AXES_DIM))
        self.rope_theta = config.get("rope_theta", FLUX2_ROPE_THETA)

        # Input projection (bias=False for FLUX.2)
        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=False)

        # Fill mode embedder
        self.fill_extra_channels = config.get("fill_extra_channels", 0)
        if self.fill_extra_channels > 0:
            self.x_embedder_fill = nn.Linear(
                in_channels + self.fill_extra_channels, hidden_size, bias=False,
            )
        else:
            self.x_embedder_fill = None

        # Context embedder for text hidden states (bias=False)
        text_embed_dim = config.get("joint_attention_dim", 4096)
        self.context_embedder = nn.Linear(text_embed_dim, hidden_size, bias=False)

        # Pooled text projection (separate from context_embedder)
        self.pooled_text_embed = nn.Linear(pooled_projection_dim, hidden_size, bias=False)

        # Time/guidance embedding (HF: time_guidance_embed)
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=256,
            embedding_dim=hidden_size,
            bias=False,
        )

        # Positional embedding (RoPE) — no trainable params
        self.pos_embed = FluxPosEmbed(
            theta=self.rope_theta,
            axes_dim=self.axes_dim,
        )

        # Shared modulation (HF naming)
        self.double_stream_modulation_img = Flux2Modulation(
            dim=hidden_size, mod_param_sets=2, bias=False,
        )
        self.double_stream_modulation_txt = Flux2Modulation(
            dim=hidden_size, mod_param_sets=2, bias=False,
        )
        self.single_stream_modulation = Flux2Modulation(
            dim=hidden_size, mod_param_sets=1, bias=False,
        )

        # Joint attention blocks (HF: transformer_blocks)
        self.transformer_blocks = nn.ModuleList([
            Flux2TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_layers)
        ])

        # Single stream blocks (HF: single_transformer_blocks)
        self.single_transformer_blocks = nn.ModuleList([
            Flux2SingleTransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(num_single_layers)
        ])

        # Output normalization and projection
        self.norm_out = AdaLayerNormContinuous(
            hidden_size, hidden_size,
            elementwise_affine=False, eps=1e-6, bias=True,
        )
        self.proj_out = nn.Linear(hidden_size, in_channels, bias=False)

        self._gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        img_ids: Optional[torch.Tensor] = None,
        txt_ids: Optional[torch.Tensor] = None,
        img_cond_seq: Optional[torch.Tensor] = None,
        img_cond_seq_ids: Optional[torch.Tensor] = None,
        img_cond: Optional[torch.Tensor] = None,
        return_hidden_states_at: Optional[List[int]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[int, torch.Tensor]]]:
        """Forward pass matching HuggingFace Flux2Transformer2DModel.

        Timestep scaling:
        - Caller passes timestep in [0, 1] range
        - Model internally scales by 1000 (matching HuggingFace)

        Args:
            hidden_states: Latent tensor [B, seq_len, in_channels].
            timestep: Timestep values [B] in [0, 1] range.
            encoder_hidden_states: Text embeddings [B, txt_seq, text_dim].
            pooled_projections: Pooled text embeddings [B, pooled_dim].
            guidance: Optional guidance scale [B] in raw scale.
            img_ids: Optional pre-computed image position IDs [B, img_seq, 3].
            txt_ids: Optional pre-computed text position IDs [B, txt_seq, 3].
            img_cond_seq: Kontext conditioning sequence [B, ref_seq, in_channels].
            img_cond_seq_ids: Position IDs for Kontext conditioning [B, ref_seq, 3].
            img_cond: Fill conditioning [B, seq_len, in_channels + fill_extra].
            return_hidden_states_at: Optional list of joint block indices to capture.

        Returns:
            Predicted output [B, img_seq, in_channels].
            If return_hidden_states_at is set, returns (output, captured_states).
        """
        B = hidden_states.shape[0]
        device = hidden_states.device
        dtype = hidden_states.dtype

        # === Timestep scaling (HuggingFace does *1000 inside model) ===
        timestep = timestep.to(dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(dtype) * 1000
        else:
            guidance = torch.ones(B, device=device, dtype=dtype) * 1000

        # === Handle Fill mode (channel-wise concatenation) ===
        if img_cond is not None:
            hidden_states = torch.cat([hidden_states, img_cond], dim=-1)
            if self.x_embedder_fill is not None:
                hidden_states = self.x_embedder_fill(hidden_states)
            else:
                hidden_states = self.x_embedder(hidden_states)
        else:
            hidden_states = self.x_embedder(hidden_states)

        # === Compute image position IDs if not provided ===
        base_seq_len = hidden_states.shape[1]
        if img_ids is None:
            h = w = int(math.sqrt(base_seq_len))
            if h * w != base_seq_len:
                h, w = 1, base_seq_len
            img_ids = create_position_ids(
                batch_size=B, height=h, width=w,
                device=device, dtype=dtype, time_offset=0.0,
            )

        # === Handle Kontext mode (sequence-wise concatenation) ===
        if img_cond_seq is not None:
            ref_embedded = self.x_embedder(img_cond_seq)
            hidden_states = torch.cat([hidden_states, ref_embedded], dim=1)

            if img_cond_seq_ids is not None:
                img_ids = torch.cat([img_ids, img_cond_seq_ids], dim=1)
            else:
                ref_seq_len = img_cond_seq.shape[1]
                ref_h = ref_w = int(math.sqrt(ref_seq_len))
                if ref_h * ref_w != ref_seq_len:
                    ref_h, ref_w = 1, ref_seq_len
                ref_ids = create_position_ids(
                    batch_size=B, height=ref_h, width=ref_w,
                    device=device, dtype=dtype, time_offset=1.0,
                )
                img_ids = torch.cat([img_ids, ref_ids], dim=1)

        # === Text position IDs ===
        txt_seq_len = encoder_hidden_states.shape[1]
        if txt_ids is None:
            # FLUX.2 uses 3D position IDs (same as FLUX.1 format)
            txt_ids = torch.zeros(B, txt_seq_len, 3, device=device, dtype=dtype)

        # === Compute RoPE ===
        # For FLUX.2 with 4D axes, we pad the 3D IDs to 4D by adding a zero column
        n_axes = len(self.axes_dim)
        if img_ids.shape[-1] < n_axes:
            pad_width = n_axes - img_ids.shape[-1]
            img_ids = torch.cat([
                img_ids,
                torch.zeros(*img_ids.shape[:-1], pad_width, device=device, dtype=dtype),
            ], dim=-1)
        if txt_ids.shape[-1] < n_axes:
            pad_width = n_axes - txt_ids.shape[-1]
            txt_ids = torch.cat([
                txt_ids,
                torch.zeros(*txt_ids.shape[:-1], pad_width, device=device, dtype=dtype),
            ], dim=-1)

        # Compute separate text and image RoPE
        img_rotary_emb = compute_rope_from_position_ids(
            img_ids, sum(self.axes_dim), self.rope_theta, axes_dim=self.axes_dim,
        )
        txt_rotary_emb = compute_rope_from_position_ids(
            txt_ids, sum(self.axes_dim), self.rope_theta, axes_dim=self.axes_dim,
        )

        # === Time/guidance embedding ===
        temb = self.time_guidance_embed(timestep, guidance)

        # Add pooled text projection
        temb = temb + self.pooled_text_embed(pooled_projections)

        # === Project text sequence embeddings ===
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # === Compute shared modulation (once for all blocks) ===
        img_double_mod = self.double_stream_modulation_img(temb)
        txt_double_mod = self.double_stream_modulation_txt(temb)
        single_mod = self.single_stream_modulation(temb)

        # === Joint attention blocks ===
        txt_hidden = encoder_hidden_states
        captured_hidden_states = {} if return_hidden_states_at is not None else None

        for block_idx, block in enumerate(self.transformer_blocks):
            hidden_states, txt_hidden = block(
                hidden_states, txt_hidden,
                img_mod=img_double_mod,
                txt_mod=txt_double_mod,
                img_rotary_emb=img_rotary_emb,
                txt_rotary_emb=txt_rotary_emb,
            )
            if captured_hidden_states is not None and block_idx in return_hidden_states_at:
                captured_hidden_states[block_idx] = hidden_states

        # === Concatenate for single stream ===
        hidden_states = torch.cat([txt_hidden, hidden_states], dim=1)

        # Combined RoPE for single stream
        combined_rotary_emb = (
            torch.cat([txt_rotary_emb[0], img_rotary_emb[0]], dim=-2),
            torch.cat([txt_rotary_emb[1], img_rotary_emb[1]], dim=-2),
        )

        # === Single stream blocks ===
        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states,
                mod=single_mod[0],  # single_stream_modulation returns ((shift, scale, gate),)
                rotary_emb=combined_rotary_emb,
            )

        # === Extract image tokens (remove text prefix) ===
        hidden_states = hidden_states[:, txt_seq_len:]

        # === Project output ===
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if captured_hidden_states is not None:
            return output, captured_hidden_states
        return output

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing."""
        self._gradient_checkpointing = True
