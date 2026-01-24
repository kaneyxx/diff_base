"""FLUX.2 image conditioning utilities.

This module provides utilities for image editing with FLUX.2:

1. Kontext Mode: Reference image editing via sequence-wise concatenation
   - Encode reference images via VAE
   - Rearrange to sequence format
   - Create position IDs with ref_image_id=1
   - Concatenate along sequence dimension in denoise loop

2. Fill Mode: Inpainting via channel-wise concatenation
   - Apply mask to reference image
   - Encode and add mask as extra channels
   - Concatenate along channel dimension in denoise loop
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


def rearrange_latent_to_sequence(
    latent: torch.Tensor,
    patch_size: int = 2,
) -> torch.Tensor:
    """Rearrange latent tensor from spatial to sequence format.

    Converts [B, C, H, W] to [B, seq, C*ph*pw] where:
    - seq = (H // patch_size) * (W // patch_size)
    - patch features = C * patch_size * patch_size

    This is the standard patchification used by FLUX transformers.

    Args:
        latent: Latent tensor of shape [B, C, H, W].
        patch_size: Size of patches (default 2 for FLUX).

    Returns:
        Sequence tensor of shape [B, num_patches, patch_dim].

    Example:
        >>> latent = torch.randn(2, 32, 64, 64)
        >>> seq = rearrange_latent_to_sequence(latent, patch_size=2)
        >>> # seq.shape = (2, 1024, 128)  # 32*32 patches, 32*2*2 features
    """
    B, C, H, W = latent.shape
    ph, pw = patch_size, patch_size

    assert H % ph == 0, f"Height {H} must be divisible by patch_size {ph}"
    assert W % pw == 0, f"Width {W} must be divisible by patch_size {pw}"

    h, w = H // ph, W // pw

    # Rearrange: [B, C, H, W] -> [B, h, w, C, ph, pw] -> [B, h*w, C*ph*pw]
    latent = latent.view(B, C, h, ph, w, pw)
    latent = latent.permute(0, 2, 4, 1, 3, 5)  # [B, h, w, C, ph, pw]
    latent = latent.reshape(B, h * w, C * ph * pw)

    return latent


def rearrange_sequence_to_latent(
    sequence: torch.Tensor,
    height: int,
    width: int,
    channels: int,
    patch_size: int = 2,
) -> torch.Tensor:
    """Rearrange sequence tensor back to spatial latent format.

    Converts [B, seq, C*ph*pw] back to [B, C, H, W].
    This is the inverse of rearrange_latent_to_sequence.

    Args:
        sequence: Sequence tensor of shape [B, num_patches, patch_dim].
        height: Target spatial height in patches (H // patch_size).
        width: Target spatial width in patches (W // patch_size).
        channels: Number of latent channels.
        patch_size: Size of patches (default 2 for FLUX).

    Returns:
        Latent tensor of shape [B, C, H*patch_size, W*patch_size].
    """
    B = sequence.shape[0]
    ph, pw = patch_size, patch_size

    # Reshape: [B, h*w, C*ph*pw] -> [B, h, w, C, ph, pw] -> [B, C, H, W]
    sequence = sequence.view(B, height, width, channels, ph, pw)
    sequence = sequence.permute(0, 3, 1, 4, 2, 5)  # [B, C, h, ph, w, pw]
    sequence = sequence.reshape(B, channels, height * ph, width * pw)

    return sequence


def create_position_ids(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    ref_image_id: int = 0,
) -> torch.Tensor:
    """Create position IDs for image patches.

    Position IDs have shape [B, seq, 3] where:
    - ids[..., 0] = image index (0 for base, 1+ for references)
    - ids[..., 1] = row index (y position)
    - ids[..., 2] = column index (x position)

    In FLUX Kontext, reference images use ref_image_id=1 to distinguish
    them from the generated image (ref_image_id=0).

    Args:
        batch_size: Batch size.
        height: Spatial height in patches.
        width: Spatial width in patches.
        device: Target device.
        dtype: Data type.
        ref_image_id: Image index (0 for base, 1 for reference).

    Returns:
        Position IDs tensor of shape [B, height*width, 3].
    """
    num_patches = height * width

    # Create grid indices
    y_indices = torch.arange(height, device=device)
    x_indices = torch.arange(width, device=device)
    y_grid, x_grid = torch.meshgrid(y_indices, x_indices, indexing="ij")

    # Flatten to sequence
    y_flat = y_grid.reshape(-1)  # [num_patches]
    x_flat = x_grid.reshape(-1)  # [num_patches]

    # Create image index tensor (all same for a single image)
    img_idx = torch.full((num_patches,), ref_image_id, device=device)

    # Stack: [num_patches, 3]
    ids = torch.stack([img_idx, y_flat, x_flat], dim=-1)

    # Expand to batch: [B, num_patches, 3]
    ids = ids.unsqueeze(0).expand(batch_size, -1, -1)

    return ids.to(dtype)


def prepare_kontext_conditioning(
    reference_images: torch.Tensor,
    vae: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    patch_size: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare Kontext conditioning from reference images.

    Encodes reference images through VAE and converts to sequence format
    with appropriate position IDs.

    In FLUX Kontext:
    1. Reference images are encoded via VAE
    2. Latents are patchified to sequence format
    3. Position IDs have ref_image_id=1 (vs 0 for base)
    4. In denoise loop, ref sequence is concatenated with base sequence

    Args:
        reference_images: Reference images [B, 3, H, W] in range [-1, 1].
        vae: VAE model for encoding.
        device: Target device.
        dtype: Data type.
        patch_size: Patch size for sequence conversion.

    Returns:
        Tuple of:
        - img_cond_seq: Encoded reference sequence [B, ref_seq, patch_dim]
        - img_cond_seq_ids: Position IDs [B, ref_seq, 3]
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
            # Apply scaling if available
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

    # Create position IDs with ref_image_id=1
    img_cond_seq_ids = create_position_ids(
        batch_size=batch_size,
        height=h,
        width=w,
        device=device,
        dtype=dtype,
        ref_image_id=1,  # Reference images use id=1
    )

    return img_cond_seq, img_cond_seq_ids


def prepare_fill_conditioning(
    reference_image: torch.Tensor,
    mask: torch.Tensor,
    vae: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    patch_size: int = 2,
) -> torch.Tensor:
    """Prepare Fill/Inpainting conditioning.

    In FLUX Fill (inpainting):
    1. Reference image is masked: masked_image = image * (1 - mask)
    2. Masked image is encoded via VAE
    3. Mask is downsampled and converted to sequence format
    4. Latent sequence and mask sequence are concatenated along channel dim
    5. In denoise loop, this is concatenated with noisy latent along channel dim

    Args:
        reference_image: Reference image [B, 3, H, W] in range [-1, 1].
        mask: Binary mask [B, 1, H, W] where 1 = inpaint region.
        vae: VAE model for encoding.
        device: Target device.
        dtype: Data type.
        patch_size: Patch size for sequence conversion.

    Returns:
        img_cond: Conditioning tensor [B, seq, latent_dim + mask_dim]
            where mask_dim = patch_size * patch_size
    """
    batch_size = reference_image.shape[0]

    # Move to device/dtype
    reference_image = reference_image.to(device=device, dtype=dtype)
    mask = mask.to(device=device, dtype=dtype)

    # Apply mask: keep unmasked regions, zero out masked regions
    # Mask=1 means inpaint, so we keep where mask=0
    masked_image = reference_image * (1 - mask)

    # Encode masked image through VAE
    with torch.no_grad():
        if hasattr(vae, "encode_to_latent"):
            latent = vae.encode_to_latent(masked_image)
        else:
            posterior = vae.encode(masked_image)
            if hasattr(posterior, "sample"):
                latent = posterior.sample()
            else:
                latent = posterior
            if hasattr(vae, "scaling_factor"):
                if hasattr(vae, "shift_factor"):
                    latent = (latent - vae.shift_factor) * vae.scaling_factor
                else:
                    latent = latent * vae.scaling_factor

    # Get spatial dimensions
    _, C, H, W = latent.shape

    # Downsample mask to latent resolution
    # VAE typically downsamples by 8, so mask should match
    vae_scale = reference_image.shape[2] // H
    mask_downsampled = torch.nn.functional.interpolate(
        mask,
        size=(H, W),
        mode="nearest",
    )

    # Convert latent to sequence
    latent_seq = rearrange_latent_to_sequence(latent, patch_size=patch_size)

    # Convert mask to sequence (patchify the mask)
    # Mask has 1 channel, so patch_dim = 1 * patch_size * patch_size
    mask_seq = rearrange_latent_to_sequence(mask_downsampled, patch_size=patch_size)

    # Concatenate latent and mask along channel dimension
    # latent_seq: [B, seq, C*ph*pw]
    # mask_seq: [B, seq, 1*ph*pw]
    img_cond = torch.cat([latent_seq, mask_seq], dim=-1)

    return img_cond


def get_fill_extra_channels(patch_size: int = 2) -> int:
    """Get number of extra channels for Fill mode conditioning.

    The mask is patchified, so the extra channels = patch_size * patch_size.

    Args:
        patch_size: Patch size used in patchification.

    Returns:
        Number of extra channels for Fill mode.
    """
    return patch_size * patch_size
