"""Base transformer interface for all FLUX variants.

Training code should program against FluxTransformerBase, not version-specific
classes like Flux1Transformer or Flux2Transformer. This enables architecture
swaps (FLUX.1 -> FLUX.2 -> future) without changing training logic.

Usage:
    from diff_base.src.models.flux import create_flux_transformer

    transformer = create_flux_transformer(version="v1", config=config, variant="dev")
    output = transformer(
        hidden_states=noisy_latent,
        timestep=timesteps,
        encoder_hidden_states=text_embeds,
        pooled_projections=pooled_embeds,
    )
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class FluxTransformerBase(nn.Module, ABC):
    """Abstract base class for all FLUX transformers (v1, v2, future).

    Defines the unified forward signature that training code uses.
    Version-specific details (position ID format, RoPE config, block types)
    are handled internally by each implementation.

    Properties:
        guidance_embeds: Whether the model supports guidance conditioning.
        in_channels: Number of input channels (64 for v1, 128 for v2-dev).
        hidden_size: Hidden dimension of the transformer.
    """

    guidance_embeds: bool
    in_channels: int
    hidden_size: int

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor | None = None,
        img_ids: torch.Tensor | None = None,
        txt_ids: torch.Tensor | None = None,
        img_cond_seq: torch.Tensor | None = None,
        img_cond_seq_ids: torch.Tensor | None = None,
        return_hidden_states_at: list[int] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Unified forward signature for all FLUX variants.

        Timestep convention:
        - Caller passes timestep in [0, 1] range
        - Model internally scales as needed (e.g., *1000 for HuggingFace compat)

        Position ID convention:
        - img_ids/txt_ids are optional; model creates defaults if not provided
        - Format varies by version (3D for v1, 4D for v2) but is abstracted away

        Args:
            hidden_states: Patchified latent sequence [B, seq, in_channels].
            timestep: Timestep values [B] in [0, 1] range.
            encoder_hidden_states: Text encoder hidden states [B, txt_seq, text_dim].
            pooled_projections: Pooled text embeddings [B, pooled_dim].
            guidance: Optional guidance scale [B] (ignored if model has no guidance).
            img_ids: Optional pre-computed image position IDs.
            txt_ids: Optional pre-computed text position IDs.
            img_cond_seq: Kontext conditioning sequence [B, ref_seq, in_channels].
            img_cond_seq_ids: Position IDs for Kontext reference.
            return_hidden_states_at: Optional list of joint block indices to capture
                intermediate hidden states (for REPA training).

        Returns:
            Predicted output [B, img_seq, in_channels].
            If return_hidden_states_at is set, returns
            (output, {block_idx: hidden_states}) tuple.
        """
        ...

    @abstractmethod
    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to reduce memory usage."""
        ...
