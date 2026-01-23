"""SD3 MM-DiT (Multimodal Diffusion Transformer) implementation.

SD3.5 uses MM-DiT architecture which is fundamentally different from:
- SDXL: UNet-based architecture
- FLUX: DiT with joint/single blocks

Key SD3.5 MM-DiT characteristics:
- hidden_size = 64 * depth
- num_heads = depth
- JointTransformerBlock processes image and text jointly
- QK normalization for numerical stability
- Positional embeddings up to 192x192 patches
"""

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .components.embeddings import (
    PatchEmbed,
    CombinedTimestepTextProjEmbeddings,
    PositionalEmbedding2D,
    get_timestep_embedding,
)
from .components.attention import JointTransformerBlock
from .components.layers import AdaLayerNormContinuous


# Variant configurations
SD3_VARIANT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "large": {
        "depth": 38,
        "hidden_size": 2432,  # 64 * 38
        "num_heads": 38,
        "patch_size": 2,
        "pos_embed_max_size": 192,
        "context_dim": 4096,  # T5-XXL hidden size
        "pooled_projection_dim": 2048,  # CLIP-L (768) + CLIP-G (1280)
    },
    "large-turbo": {
        "depth": 38,
        "hidden_size": 2432,
        "num_heads": 38,
        "patch_size": 2,
        "pos_embed_max_size": 192,
        "context_dim": 4096,
        "pooled_projection_dim": 2048,
    },
    "medium": {
        "depth": 24,
        "hidden_size": 1536,  # 64 * 24
        "num_heads": 24,
        "patch_size": 2,
        "pos_embed_max_size": 192,
        "context_dim": 4096,
        "pooled_projection_dim": 2048,
    },
}


class SD3Transformer(nn.Module):
    """SD3 MM-DiT (Multimodal Diffusion Transformer).

    The core transformer architecture for Stable Diffusion 3.5.

    Architecture overview:
    1. Patch embedding converts latent [B, 16, H, W] to [B, N, hidden_size]
    2. Positional embeddings added to patch sequence
    3. Timestep + pooled text embeddings create conditioning
    4. Joint transformer blocks process image + text together
    5. Final layer norm + linear projection outputs noise prediction
    """

    def __init__(
        self,
        config: DictConfig,
        variant: str = "large",
    ):
        """Initialize SD3 Transformer.

        Args:
            config: Transformer configuration.
            variant: Model variant ("large", "large-turbo", "medium").
        """
        super().__init__()
        self.config = config
        self.variant = variant

        # Get variant-specific defaults
        variant_cfg = SD3_VARIANT_CONFIGS.get(variant, SD3_VARIANT_CONFIGS["large"])

        # Core dimensions (can be overridden by config)
        self.depth = config.get("depth", variant_cfg["depth"])
        self.hidden_size = config.get("hidden_size", variant_cfg["hidden_size"])
        self.num_heads = config.get("num_heads", variant_cfg["num_heads"])
        self.patch_size = config.get("patch_size", variant_cfg["patch_size"])
        self.in_channels = config.get("in_channels", 16)  # VAE latent channels
        self.context_dim = config.get("context_dim", variant_cfg["context_dim"])
        self.pooled_projection_dim = config.get(
            "pooled_projection_dim",
            variant_cfg["pooled_projection_dim"]
        )
        pos_embed_max_size = config.get(
            "pos_embed_max_size",
            variant_cfg["pos_embed_max_size"]
        )
        self.qk_norm = config.get("qk_norm", True)

        # Calculate head dimension
        self.head_dim = self.hidden_size // self.num_heads

        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            hidden_size=self.hidden_size,
        )

        # Positional embedding
        self.pos_embed = PositionalEmbedding2D(
            hidden_size=self.hidden_size,
            max_size=pos_embed_max_size,
        )

        # Timestep + pooled text conditioning
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.hidden_size,
            pooled_projection_dim=self.pooled_projection_dim,
        )

        # Context embedder (project T5 embeddings to hidden_size for conditioning)
        self.context_embedder = nn.Linear(self.context_dim, self.hidden_size)

        # Joint transformer blocks
        self.transformer_blocks = nn.ModuleList([
            JointTransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                context_dim=self.context_dim,
                qk_norm=self.qk_norm,
            )
            for _ in range(self.depth)
        ])

        # Output layers
        self.norm_out = AdaLayerNormContinuous(
            hidden_size=self.hidden_size,
            conditioning_size=self.hidden_size,
        )
        self.proj_out = nn.Linear(
            self.hidden_size,
            self.patch_size * self.patch_size * self.in_channels,
        )

        self._gradient_checkpointing = False

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        # Initialize linear layers with small std
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        # Zero-init output projection for residual connections
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor:
        """Forward pass of SD3 transformer.

        Args:
            hidden_states: Latent tensor [B, C, H, W] where C=16 (VAE latent channels).
            timestep: Timestep values [B] (can be float for flow matching).
            encoder_hidden_states: T5 text embeddings [B, seq_len, 4096].
            pooled_projections: Pooled CLIP embeddings [B, 2048].
            return_dict: Unused, for API compatibility.

        Returns:
            Predicted noise/velocity [B, C, H, W].
        """
        batch_size = hidden_states.shape[0]
        height, width = hidden_states.shape[2], hidden_states.shape[3]

        # Calculate patch dimensions
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size

        # 1. Patch embedding: [B, C, H, W] -> [B, N, hidden_size]
        hidden_states = self.patch_embed(hidden_states)

        # 2. Add positional embedding
        pos_embed = self.pos_embed(num_patches_h, num_patches_w)
        hidden_states = hidden_states + pos_embed.to(hidden_states.dtype)

        # 3. Compute conditioning from timestep + pooled text
        timestep_emb = get_timestep_embedding(timestep, 256)
        temb = self.time_text_embed(timestep_emb, pooled_projections)

        # 4. Process through joint transformer blocks
        for block in self.transformer_blocks:
            if self._gradient_checkpointing and self.training:
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    use_reentrant=False,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                )

        # 5. Final layer norm with conditioning
        hidden_states = self.norm_out(hidden_states, temb)

        # 6. Project to output dimension
        hidden_states = self.proj_out(hidden_states)

        # 7. Unpatchify: [B, N, patch_size*patch_size*C] -> [B, C, H, W]
        hidden_states = self._unpatchify(
            hidden_states,
            num_patches_h,
            num_patches_w,
        )

        return hidden_states

    def _unpatchify(
        self,
        hidden_states: torch.Tensor,
        num_patches_h: int,
        num_patches_w: int,
    ) -> torch.Tensor:
        """Convert patch sequence back to image tensor.

        Args:
            hidden_states: Patch sequence [B, N, patch_size*patch_size*C].
            num_patches_h: Number of patches in height.
            num_patches_w: Number of patches in width.

        Returns:
            Image tensor [B, C, H, W].
        """
        batch_size = hidden_states.shape[0]
        p = self.patch_size
        c = self.in_channels

        # Reshape: [B, N, p*p*C] -> [B, h, w, p, p, C]
        hidden_states = hidden_states.reshape(
            batch_size, num_patches_h, num_patches_w, p, p, c
        )

        # Permute and reshape: [B, h, w, p, p, C] -> [B, C, H, W]
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        hidden_states = hidden_states.reshape(
            batch_size, c, num_patches_h * p, num_patches_w * p
        )

        return hidden_states

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        self._gradient_checkpointing = True
        for block in self.transformer_blocks:
            if hasattr(block, "enable_gradient_checkpointing"):
                block.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False

    @property
    def device(self) -> torch.device:
        """Get device of model parameters."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype of model parameters."""
        return next(self.parameters()).dtype

    def get_param_count(self) -> dict:
        """Get parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "total_millions": total / 1e6,
        }


def create_sd3_transformer(
    config: DictConfig,
    variant: str = "large",
) -> SD3Transformer:
    """Factory function to create SD3 transformer.

    Args:
        config: Transformer configuration.
        variant: Model variant ("large", "large-turbo", "medium").

    Returns:
        SD3Transformer instance.
    """
    return SD3Transformer(config, variant=variant)
