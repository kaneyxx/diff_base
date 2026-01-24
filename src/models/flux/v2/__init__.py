"""FLUX.2 model implementations (dev, klein-4B, klein-9B).

Image Editing Support:
- conditioning module provides utilities for Kontext and Fill modes
- Kontext: Reference image editing via sequence-wise concatenation
- Fill: Inpainting via channel-wise concatenation with masks
"""

from .model import Flux2Model
from .transformer import Flux2Transformer
from .vae import Flux2VAE
from .text_encoder import Flux2TextEncoders
from .conditioning import (
    rearrange_latent_to_sequence,
    rearrange_sequence_to_latent,
    create_position_ids,
    prepare_kontext_conditioning,
    prepare_fill_conditioning,
    get_fill_extra_channels,
)

__all__ = [
    # Model components
    "Flux2Model",
    "Flux2Transformer",
    "Flux2VAE",
    "Flux2TextEncoders",
    # Conditioning utilities
    "rearrange_latent_to_sequence",
    "rearrange_sequence_to_latent",
    "create_position_ids",
    "prepare_kontext_conditioning",
    "prepare_fill_conditioning",
    "get_fill_extra_channels",
]
