"""FLUX.1 image conditioning utilities.

Provides Kontext mode for FLUX.1 (reference image editing).

Note: FLUX.1 does NOT support Fill mode (channel-wise conditioning).
Only Kontext mode (sequence-wise concatenation) is available.

Position ID Format (4D):
- t: Temporal/time offset (0.0 for target image, 1.0+ for reference images)
- h: Height coordinate in patches
- w: Width coordinate in patches
- l: Sequence/layer index within patch position
"""

from typing import Tuple

import torch
import torch.nn as nn

# Re-export utilities from v2 conditioning (same format works for v1)
from ..v2.conditioning import (
    rearrange_latent_to_sequence,
    rearrange_sequence_to_latent,
    create_position_ids,
)


def prepare_kontext_conditioning(
    reference_images: torch.Tensor,
    vae: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    patch_size: int = 2,
    time_offset: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare Kontext conditioning for FLUX.1.

    Similar to FLUX.2 but uses 16 latent channels (vs 32 for FLUX.2 dev).

    Encodes reference images through VAE and converts to sequence format
    with appropriate 4D position IDs [t, h, w, l].

    In FLUX.1 Kontext:
    1. Reference images are encoded via VAE (16 latent channels)
    2. Latents are patchified to sequence format
    3. Position IDs have time_offset=1.0 (vs 0.0 for target)
    4. In denoise loop, ref sequence is concatenated with base sequence

    Args:
        reference_images: Reference images [B, 3, H, W] in range [-1, 1].
        vae: FLUX.1 VAE model for encoding.
        device: Target device.
        dtype: Data type.
        patch_size: Patch size for sequence conversion.
        time_offset: Time offset for reference images (default 1.0).

    Returns:
        Tuple of:
        - img_cond_seq: Encoded reference sequence [B, ref_seq, patch_dim]
        - img_cond_seq_ids: Position IDs [B, ref_seq, 4]
    """
    batch_size = reference_images.shape[0]

    # Move to device/dtype
    reference_images = reference_images.to(device=device, dtype=dtype)

    # Encode through VAE
    with torch.no_grad():
        if hasattr(vae, "encode_to_latent"):
            latent = vae.encode_to_latent(reference_images)
        else:
            # Standard VAE encode
            posterior = vae.encode(reference_images)
            if hasattr(posterior, "sample"):
                latent = posterior.sample()
            else:
                latent = posterior
            # Apply scaling if available (FLUX.1 VAE uses shift and scale)
            if hasattr(vae, "scaling_factor"):
                if hasattr(vae, "shift_factor"):
                    latent = (latent - vae.shift_factor) * vae.scaling_factor
                else:
                    latent = latent * vae.scaling_factor

    # Get spatial dimensions after encoding
    _, C, H, W = latent.shape
    h, w = H // patch_size, W // patch_size

    # Convert to sequence format
    img_cond_seq = rearrange_latent_to_sequence(latent, patch_size=patch_size)

    # Create 4D position IDs with time_offset for reference images
    img_cond_seq_ids = create_position_ids(
        batch_size=batch_size,
        height=h,
        width=w,
        device=device,
        dtype=dtype,
        time_offset=time_offset,  # Reference images use time_offset=1.0
    )

    return img_cond_seq, img_cond_seq_ids


__all__ = [
    "rearrange_latent_to_sequence",
    "rearrange_sequence_to_latent",
    "create_position_ids",
    "prepare_kontext_conditioning",
]
