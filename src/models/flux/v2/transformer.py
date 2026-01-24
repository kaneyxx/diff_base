"""FLUX.2 Transformer (DiT) architecture.

FLUX.2 variants differ from FLUX.1 in:
- Different block counts (fewer joint, more single)
- Different text encoders (Mistral/Qwen instead of T5+CLIP)
- Different latent channel counts
- QK normalization for stability

Variant configurations:
- dev: 8 joint + 48 single blocks, 32 latent channels, Mistral-3 text encoder
- klein-4b: 5 joint + 20 single blocks, 128 latent channels, Qwen3-4B
- klein-9b: 6 joint + 24 single blocks, 128 latent channels, Qwen3-8B

Image Editing Support:
- Kontext Mode: Reference image editing via sequence-wise concatenation
  - img_cond_seq is concatenated along sequence dimension (dim=1)
  - Position IDs distinguish base image (id=0) from reference (id=1)
- Fill Mode: Inpainting via channel-wise concatenation
  - img_cond (latent + mask) is concatenated along channel dimension (dim=-1)
  - Uses x_embedder_fill for expanded input channels
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..components.attention import FluxJointTransformerBlock, FluxSingleTransformerBlock
from ..components.embeddings import FluxPosEmbed, get_timestep_embedding
from ..components.layers import AdaLayerNormContinuous
from ...components.embeddings import MLPEmbedder, RotaryEmbedding
from .conditioning import get_fill_extra_channels


class Flux2Transformer(nn.Module):
    """FLUX.2 DiT (Diffusion Transformer) architecture.

    Supports dev, klein-4b, and klein-9b variants with different
    block counts and dimensions.
    """

    # Variant-specific configurations
    VARIANT_CONFIGS = {
        "dev": {
            "num_layers": 8,  # Joint/double blocks
            "num_single_layers": 48,  # Single blocks
            "hidden_size": 3072,
            "num_attention_heads": 24,
            "attention_head_dim": 128,
            "in_channels": 128,  # 32 latent channels * 4
            "pooled_projection_dim": 4096,  # Mistral hidden size
            "guidance_embeds": True,
            "qk_norm": True,
        },
        "klein-4b": {
            "num_layers": 5,
            "num_single_layers": 20,
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "attention_head_dim": 128,
            "in_channels": 512,  # 128 latent channels * 4
            "pooled_projection_dim": 4096,  # Qwen hidden size
            "guidance_embeds": True,
            "qk_norm": True,
        },
        "klein-9b": {
            "num_layers": 6,
            "num_single_layers": 24,
            "hidden_size": 2560,
            "num_attention_heads": 20,
            "attention_head_dim": 128,
            "in_channels": 512,  # 128 latent channels * 4
            "pooled_projection_dim": 4096,  # Qwen hidden size
            "guidance_embeds": True,
            "qk_norm": True,
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
        qk_norm = config.get("qk_norm", variant_cfg["qk_norm"])

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.guidance_embeds = guidance_embeds
        self.qk_norm = qk_norm

        # Input projection
        self.x_embedder = nn.Linear(in_channels, hidden_size)

        # Fill mode embedder (handles extra mask channels)
        # This is used when channel-wise concatenation is applied
        self.fill_extra_channels = config.get("fill_extra_channels", 0)
        if self.fill_extra_channels > 0:
            self.x_embedder_fill = nn.Linear(
                in_channels + self.fill_extra_channels,
                hidden_size,
            )
        else:
            self.x_embedder_fill = None

        # Time/guidance embedding
        self.time_embed = MLPEmbedder(256, hidden_size)

        if guidance_embeds:
            self.guidance_embed = MLPEmbedder(256, hidden_size)
        else:
            self.guidance_embed = None

        # Pooled text projection (for Mistral/Qwen embeddings)
        self.pooled_text_embed = nn.Linear(pooled_projection_dim, hidden_size)

        # Context projection (for text encoder hidden states)
        text_embed_dim = config.get("joint_attention_dim", 4096)
        self.context_embedder = nn.Linear(text_embed_dim, hidden_size)

        # Positional embedding (RoPE)
        head_dim = config.get("attention_head_dim", hidden_size // num_heads)
        self.rope = RotaryEmbedding(head_dim)

        # Joint attention blocks (with QK norm for FLUX.2)
        self.joint_blocks = nn.ModuleList([
            FluxJointTransformerBlock(hidden_size, num_heads, qk_norm=qk_norm)
            for _ in range(num_layers)
        ])

        # Single stream blocks (with QK norm for FLUX.2)
        self.single_blocks = nn.ModuleList([
            FluxSingleTransformerBlock(hidden_size, num_heads, qk_norm=qk_norm)
            for _ in range(num_single_layers)
        ])

        # Output normalization and projection
        self.norm_out = AdaLayerNormContinuous(
            hidden_size,
            hidden_size,
            elementwise_affine=False,
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
        img_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Latent tensor [B, seq_len, in_channels].
            timestep: Timestep values [B].
            encoder_hidden_states: Text embeddings [B, txt_seq, text_dim].
            pooled_projections: Pooled text embeddings [B, pooled_dim].
            guidance: Optional guidance scale [B].
            img_cond_seq: Kontext conditioning sequence [B, ref_seq, in_channels].
                Concatenated along sequence dimension.
            img_cond_seq_ids: Position IDs for Kontext conditioning [B, ref_seq, 3].
                Used for positional encoding.
            img_cond: Fill conditioning [B, seq_len, in_channels + fill_extra].
                Concatenated along channel dimension.

        Returns:
            Predicted output [B, seq_len, in_channels].
            Note: Output has same seq_len as input hidden_states, not including
            any concatenated Kontext conditioning. Slicing is handled by caller.
        """
        # Track original sequence length for potential slicing
        original_seq_len = hidden_states.shape[1]

        # Handle Fill mode (channel-wise concatenation)
        if img_cond is not None:
            # Concatenate conditioning along channel dimension
            hidden_states = torch.cat([hidden_states, img_cond], dim=-1)
            # Use fill embedder if available, otherwise standard
            if self.x_embedder_fill is not None:
                hidden_states = self.x_embedder_fill(hidden_states)
            else:
                # Fall back: just use regular embedder (may fail if dims mismatch)
                hidden_states = self.x_embedder(hidden_states)
        else:
            # Standard embedding
            hidden_states = self.x_embedder(hidden_states)

        # Handle Kontext mode (sequence-wise concatenation)
        if img_cond_seq is not None:
            # Embed reference image sequence
            ref_embedded = self.x_embedder(img_cond_seq)
            # Concatenate along sequence dimension
            hidden_states = torch.cat([hidden_states, ref_embedded], dim=1)

        # Project text encoder hidden states to hidden_size
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # Time embedding
        temb = get_timestep_embedding(timestep)
        temb = self.time_embed(temb)

        if guidance is not None and self.guidance_embed is not None:
            guidance_emb = get_timestep_embedding(guidance)
            temb = temb + self.guidance_embed(guidance_emb)

        temb = temb + self.pooled_text_embed(pooled_projections)

        # Get rotary embeddings
        img_seq_len = hidden_states.shape[1]  # Includes Kontext if present
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

        # Combined rotary for single stream
        combined_seq_len = txt_seq_len + img_seq_len
        combined_rotary_emb = self.rope(combined_seq_len, hidden_states.device)

        # Single stream blocks
        for block in self.single_blocks:
            hidden_states = block(hidden_states, temb, combined_rotary_emb)

        # Extract image tokens (all image tokens including Kontext if present)
        hidden_states = hidden_states[:, txt_seq_len:]

        # Project output with adaptive normalization
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        # Note: If Kontext mode, output includes both base and reference tokens.
        # The caller (pipeline) is responsible for slicing to get only base tokens:
        #   output = output[:, :original_seq_len]
        # This is intentionally NOT done here to give caller flexibility.

        return output

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing."""
        self._gradient_checkpointing = True
