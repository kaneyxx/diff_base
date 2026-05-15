"""FLUX.2 model implementations (dev, klein-4B, klein-9B).

Image Editing Support:
- conditioning module provides utilities for Kontext and Fill modes
- Kontext: Reference image editing via sequence-wise concatenation
- Fill: Inpainting via channel-wise concatenation with masks
"""

from .bfl_export import convert_internal_to_bfl, to_bfl_checkpoint
from .blocks import (
    Flux2Attention,
    Flux2FeedForward,
    Flux2ParallelSelfAttention,
    Flux2SingleTransformerBlock,
    Flux2TransformerBlock,
)
from .conditioning import (
    create_position_ids,
    get_fill_extra_channels,
    prepare_fill_conditioning,
    prepare_kontext_conditioning,
    rearrange_latent_to_sequence,
    rearrange_sequence_to_latent,
)
from .model import Flux2Model
from .text_encoder import Flux2TextEncoders
from .transformer import Flux2Transformer
from .vae import Flux2VAE
from .weight_mapping import (
    DOUBLE_BLOCK_SUFFIX_MAP,
    SINGLE_BLOCK_SUFFIX_MAP,
    STATIC_MAP,
    Flux2FormatEnum,
    StateDict,
    convert_state_dict,
    detect_format,
    load_flux2_checkpoint,
)

__all__ = [
    # Model components
    "Flux2Model",
    "Flux2Transformer",
    "Flux2VAE",
    "Flux2TextEncoders",
    # Block components
    "Flux2TransformerBlock",
    "Flux2SingleTransformerBlock",
    "Flux2Attention",
    "Flux2ParallelSelfAttention",
    "Flux2FeedForward",
    # Conditioning utilities
    "rearrange_latent_to_sequence",
    "rearrange_sequence_to_latent",
    "create_position_ids",
    "prepare_kontext_conditioning",
    "prepare_fill_conditioning",
    "get_fill_extra_channels",
    # Weight mapping
    "Flux2FormatEnum",
    "StateDict",
    "STATIC_MAP",
    "DOUBLE_BLOCK_SUFFIX_MAP",
    "SINGLE_BLOCK_SUFFIX_MAP",
    "detect_format",
    "convert_state_dict",
    "load_flux2_checkpoint",
    # BFL export
    "convert_internal_to_bfl",
    "to_bfl_checkpoint",
]
