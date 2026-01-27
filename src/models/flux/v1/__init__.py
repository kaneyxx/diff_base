"""FLUX.1 model implementations (dev, schnell).

Image Editing Support:
- Kontext Mode: Reference image editing via sequence-wise concatenation
- Fill Mode: NOT supported (use FLUX.2 for Fill mode)
"""

from .model import Flux1Model
from .transformer import Flux1Transformer
from .vae import Flux1VAE
from .text_encoder import Flux1TextEncoders
from .conditioning import (
    rearrange_latent_to_sequence,
    rearrange_sequence_to_latent,
    create_position_ids,
    prepare_kontext_conditioning,
)

__all__ = [
    # Model components
    "Flux1Model",
    "Flux1Transformer",
    "Flux1VAE",
    "Flux1TextEncoders",
    # Conditioning utilities
    "rearrange_latent_to_sequence",
    "rearrange_sequence_to_latent",
    "create_position_ids",
    "prepare_kontext_conditioning",
]
