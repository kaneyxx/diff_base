"""FLUX.1 Transformer (DiT) architecture.

FLUX.1 variants (dev, schnell) use:
- 19 joint transformer blocks
- 38 single transformer blocks
- 3072 hidden size, 24 attention heads
- T5-XXL + CLIP-L text encoders
- 16 latent channels
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..components.attention import FluxJointTransformerBlock, FluxSingleTransformerBlock
from ..components.embeddings import FluxPosEmbed, get_timestep_embedding
from ...components.embeddings import MLPEmbedder, RotaryEmbedding


class Flux1Transformer(nn.Module):
    """FLUX.1 DiT (Diffusion Transformer) architecture.

    Configuration for FLUX.1 variants:
    - dev: 19 joint + 38 single blocks, guidance enabled
    - schnell: 19 joint + 38 single blocks, guidance disabled
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

        # Input projection
        self.x_embedder = nn.Linear(in_channels, hidden_size)

        # Time/guidance embedding
        self.time_embed = MLPEmbedder(256, hidden_size)

        if guidance_embeds:
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
        temb = get_timestep_embedding(timestep)
        temb = self.time_embed(temb)

        if guidance is not None and self.guidance_embed is not None:
            guidance_emb = get_timestep_embedding(guidance)
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

        # Combined rotary for single stream
        combined_seq_len = txt_seq_len + img_seq_len
        combined_rotary_emb = self.rope(combined_seq_len, hidden_states.device)

        # Single stream blocks
        for block in self.single_blocks:
            hidden_states = block(hidden_states, temb, combined_rotary_emb)

        # Extract image tokens
        hidden_states = hidden_states[:, txt_seq_len:]

        # Project output
        hidden_states = self.norm_out(hidden_states)
        output = self.proj_out(hidden_states)

        return output

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing."""
        self._gradient_checkpointing = True
