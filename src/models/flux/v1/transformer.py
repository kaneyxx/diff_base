"""FLUX.1 Transformer (DiT) architecture.

FLUX.1 variants (dev, schnell) use:
- 19 joint transformer blocks
- 38 single transformer blocks
- 3072 hidden size, 24 attention heads
- T5-XXL + CLIP-L text encoders
- 16 latent channels

Image Editing Support:
- Kontext Mode: Reference image editing via sequence-wise concatenation
  - img_cond_seq is concatenated along sequence dimension (dim=1)
  - 4D Position IDs [t, h, w, l] distinguish target (t=0) from reference (t=1)
- Fill Mode: NOT supported in FLUX.1

HuggingFace Alignment:
This transformer uses naming conventions compatible with HuggingFace's
FluxTransformer2DModel for direct weight loading from pretrained checkpoints.

Key naming mappings:
- transformer_blocks (not joint_blocks)
- single_transformer_blocks (not single_blocks)
- time_text_embed.timestep_embedder/guidance_embedder/text_embedder
- Blocks use to_q/to_k/to_v (not combined to_qkv)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..components.attention import FluxJointTransformerBlock, FluxSingleTransformerBlock
from ..components.embeddings import (
    FluxPosEmbed,
    get_timestep_embedding,
    compute_rope_from_position_ids,
)
from ..components.layers import AdaLayerNormContinuous
from ...components.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    RotaryEmbedding,
)
from .conditioning import create_position_ids


class Flux1Transformer(nn.Module):
    """FLUX.1 DiT (Diffusion Transformer) architecture.

    Configuration for FLUX.1 variants:
    - dev: 19 joint + 38 single blocks, guidance enabled
    - schnell: 19 joint + 38 single blocks, guidance disabled

    Uses HuggingFace-compatible naming for direct weight loading.
    """

    # Default configurations for FLUX.1 variants
    VARIANT_CONFIGS = {
        "dev": {
            "num_layers": 19,
            "num_single_layers": 38,
            "hidden_size": 3072,
            "num_attention_heads": 24,
            "in_channels": 64,
            "guidance_embeds": True,
        },
        "schnell": {
            "num_layers": 19,
            "num_single_layers": 38,
            "hidden_size": 3072,
            "num_attention_heads": 24,
            "in_channels": 64,
            "guidance_embeds": False,
        },
    }

    def __init__(self, config: DictConfig, variant: str = "dev"):
        """Initialize FLUX.1 transformer.

        Args:
            config: Transformer configuration.
            variant: Model variant ("dev" or "schnell").
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
        pooled_projection_dim = config.get("pooled_projection_dim", 768)
        guidance_embeds = config.get("guidance_embeds", variant_cfg["guidance_embeds"])

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.guidance_embeds = guidance_embeds

        # Input projection for latents
        self.x_embedder = nn.Linear(in_channels, hidden_size)

        # Context embedder for text (T5 4096 -> hidden_size 3072)
        joint_attention_dim = config.get("joint_attention_dim", 4096)
        self.context_embedder = nn.Linear(joint_attention_dim, hidden_size)

        # Combined time/guidance/text embedding (HF naming: time_text_embed)
        self.time_text_embed = CombinedTimestepGuidanceTextProjEmbeddings(
            embedding_dim=hidden_size,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
        )

        # Positional embedding (RoPE)
        head_dim = hidden_size // num_heads
        self.rope = RotaryEmbedding(head_dim)

        # Joint attention blocks (HF naming: transformer_blocks)
        self.transformer_blocks = nn.ModuleList([
            FluxJointTransformerBlock(hidden_size, num_heads)
            for _ in range(num_layers)
        ])

        # Single stream blocks (HF naming: single_transformer_blocks)
        self.single_transformer_blocks = nn.ModuleList([
            FluxSingleTransformerBlock(hidden_size, num_heads)
            for _ in range(num_single_layers)
        ])

        # Output projection (HF-aligned: AdaLayerNormContinuous produces norm_out.linear.*)
        self.norm_out = AdaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=hidden_size,
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
        )
        self.proj_out = nn.Linear(hidden_size, in_channels)

        self._gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        img_cond_seq: Optional[torch.Tensor] = None,
        img_cond_seq_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Latent tensor [B, seq_len, in_channels].
            timestep: Timestep values [B].
            encoder_hidden_states: Text embeddings [B, txt_seq, joint_attention_dim].
                Will be projected from joint_attention_dim (4096) to hidden_size (3072).
            pooled_projections: Pooled text embeddings [B, pooled_dim].
            guidance: Optional guidance scale [B].
            img_cond_seq: Kontext conditioning sequence [B, ref_seq, in_channels].
                Concatenated along sequence dimension.
            img_cond_seq_ids: Position IDs for Kontext conditioning [B, ref_seq, 4].
                4D format [t, h, w, l] used for positional encoding.

        Returns:
            Predicted output [B, seq_len, in_channels].
            Note: If Kontext mode, output includes both base and reference tokens.
            The caller (pipeline) is responsible for slicing to get only base tokens.
        """
        B = hidden_states.shape[0]
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Embed inputs
        hidden_states = self.x_embedder(hidden_states)

        # Compute base image position IDs (time_offset=0 for target image)
        base_seq_len = hidden_states.shape[1]
        h = w = int(math.sqrt(base_seq_len))
        if h * w != base_seq_len:
            h, w = 1, base_seq_len

        base_position_ids = create_position_ids(
            batch_size=B,
            height=h,
            width=w,
            device=device,
            dtype=dtype,
            time_offset=0.0,  # Target image uses time_offset=0
        )

        # Handle Kontext mode (sequence-wise concatenation)
        if img_cond_seq is not None:
            # Embed reference image sequence
            ref_embedded = self.x_embedder(img_cond_seq)
            # Concatenate along sequence dimension
            hidden_states = torch.cat([hidden_states, ref_embedded], dim=1)

            # Concatenate position IDs if provided
            if img_cond_seq_ids is not None:
                combined_position_ids = torch.cat([base_position_ids, img_cond_seq_ids], dim=1)
            else:
                # Create default position IDs for reference with time_offset=1.0
                ref_seq_len = img_cond_seq.shape[1]
                ref_h = ref_w = int(math.sqrt(ref_seq_len))
                if ref_h * ref_w != ref_seq_len:
                    ref_h, ref_w = 1, ref_seq_len
                ref_position_ids = create_position_ids(
                    batch_size=B,
                    height=ref_h,
                    width=ref_w,
                    device=device,
                    dtype=dtype,
                    time_offset=1.0,  # Reference image uses time_offset=1.0
                )
                combined_position_ids = torch.cat([base_position_ids, ref_position_ids], dim=1)
        else:
            combined_position_ids = base_position_ids

        # Combined time/guidance/text embedding (using HF-aligned time_text_embed)
        temb = self.time_text_embed(
            timestep=timestep,
            pooled_projection=pooled_projections,
            guidance=guidance,
        )

        # Compute rotary embeddings from 4D position IDs
        img_seq_len = hidden_states.shape[1]  # Includes Kontext if present
        txt_seq_len = encoder_hidden_states.shape[1]

        # Get head dimension for RoPE
        head_dim = self.hidden_size // self.num_heads

        # Compute image RoPE from position IDs (uses 4D [t, h, w, l] format)
        img_rotary_emb = compute_rope_from_position_ids(
            combined_position_ids, head_dim, self.rope.theta
        )

        # Text uses standard 1D RoPE
        txt_rotary_emb = self.rope(txt_seq_len, device)

        # Project text embeddings from T5 dim (4096) to hidden dim (3072)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # Joint attention blocks (using HF-aligned transformer_blocks)
        txt_hidden = encoder_hidden_states
        for block in self.transformer_blocks:
            hidden_states, txt_hidden = block(
                hidden_states,
                txt_hidden,
                temb,
                img_rotary_emb,
                txt_rotary_emb,
            )

        # Concatenate for single stream
        hidden_states = torch.cat([txt_hidden, hidden_states], dim=1)

        # Create combined position IDs for single stream (text + image)
        txt_position_ids = torch.zeros(B, txt_seq_len, 4, device=device, dtype=dtype)
        txt_position_ids[..., 3] = torch.arange(txt_seq_len, device=device).float()

        combined_stream_position_ids = torch.cat([txt_position_ids, combined_position_ids], dim=1)
        combined_rotary_emb = compute_rope_from_position_ids(
            combined_stream_position_ids, head_dim, self.rope.theta
        )

        # Single stream blocks (using HF-aligned single_transformer_blocks)
        for block in self.single_transformer_blocks:
            hidden_states = block(hidden_states, temb, combined_rotary_emb)

        # Extract image tokens (all image tokens including Kontext if present)
        hidden_states = hidden_states[:, txt_seq_len:]

        # Project output (HF-aligned: norm_out takes temb as conditioning)
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        return output

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing."""
        self._gradient_checkpointing = True
