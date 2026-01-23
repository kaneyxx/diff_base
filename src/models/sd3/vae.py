"""SD3 VAE implementation.

SD3 uses a VAE with 16 latent channels (same as FLUX), but differs from SDXL:
- 16 latent channels (vs 4 in SDXL)
- Different scaling factors
- No post_quant_conv in some implementations

The VAE architecture itself reuses the Encoder/Decoder from SDXL.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..sdxl.vae import Encoder, Decoder, DiagonalGaussianDistribution


class SD3VAE(nn.Module):
    """SD3 VAE (AutoencoderKL) with 16 latent channels.

    Key differences from SDXL VAE:
    - 16 latent channels instead of 4
    - Different scaling_factor and shift_factor
    - Optional use of post_quant_conv
    """

    def __init__(self, config: DictConfig):
        """Initialize SD3 VAE.

        Args:
            config: VAE configuration.
        """
        super().__init__()
        self.config = config

        in_channels = config.get("in_channels", 3)
        out_channels = config.get("out_channels", 3)
        latent_channels = config.get("latent_channels", 16)  # SD3 uses 16 channels

        block_out_channels = config.get("block_out_channels", (128, 256, 512, 512))
        if isinstance(block_out_channels, list):
            block_out_channels = tuple(block_out_channels)

        layers_per_block = config.get("layers_per_block", 2)
        norm_num_groups = config.get("norm_num_groups", 32)

        # SD3-specific scaling factors
        self.scaling_factor = config.get("scaling_factor", 1.5305)
        self.shift_factor = config.get("shift_factor", 0.0609)

        # Whether to use quant_conv layers
        self.use_quant_conv = config.get("use_quant_conv", True)

        # Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            double_z=True,  # For DiagonalGaussianDistribution
        )

        # Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
        )

        # Quantization convolutions (optional for SD3)
        if self.use_quant_conv:
            self.quant_conv = nn.Conv2d(
                2 * latent_channels, 2 * latent_channels, kernel_size=1
            )
            self.post_quant_conv = nn.Conv2d(
                latent_channels, latent_channels, kernel_size=1
            )
        else:
            self.quant_conv = None
            self.post_quant_conv = None

        self.latent_channels = latent_channels

    def encode(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
    ) -> DiagonalGaussianDistribution:
        """Encode image to latent distribution.

        Args:
            x: Input image [B, 3, H, W] in range [-1, 1].
            return_dict: Unused, for API compatibility.

        Returns:
            Diagonal Gaussian distribution over latents.
        """
        h = self.encoder(x)

        if self.quant_conv is not None:
            moments = self.quant_conv(h)
        else:
            moments = h

        return DiagonalGaussianDistribution(moments)

    def decode(
        self,
        z: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor:
        """Decode latent to image.

        Args:
            z: Latent tensor [B, latent_channels, h, w].
            return_dict: Unused, for API compatibility.

        Returns:
            Decoded image [B, 3, H, W] in range [-1, 1].
        """
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)

        dec = self.decoder(z)
        return dec

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = True,
    ) -> torch.Tensor:
        """Forward pass (encode then decode).

        Args:
            sample: Input image [B, 3, H, W].
            sample_posterior: Whether to sample from posterior vs use mode.

        Returns:
            Reconstructed image [B, 3, H, W].
        """
        posterior = self.encode(sample)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        # Apply SD3 scaling (encode direction)
        z = (z - self.shift_factor) * self.scaling_factor

        # Decode (with inverse scaling)
        dec = self.decode(z / self.scaling_factor + self.shift_factor)

        return dec

    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image and return scaled latent.

        This is the typical interface for training/inference.

        Args:
            x: Input image [B, 3, H, W] in range [-1, 1].

        Returns:
            Scaled latent tensor [B, latent_channels, h, w].
        """
        posterior = self.encode(x)
        z = posterior.sample()

        # Apply SD3 scaling
        z = (z - self.shift_factor) * self.scaling_factor

        return z

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from scaled latent to image.

        Args:
            z: Scaled latent tensor [B, latent_channels, h, w].

        Returns:
            Decoded image [B, 3, H, W] in range [-1, 1].
        """
        # Inverse SD3 scaling
        z = z / self.scaling_factor + self.shift_factor

        return self.decode(z)

    def get_latent_shape(
        self,
        batch_size: int,
        height: int,
        width: int,
    ) -> tuple:
        """Get the latent shape for a given image size.

        Args:
            batch_size: Batch size.
            height: Image height in pixels.
            width: Image width in pixels.

        Returns:
            Tuple of (batch_size, latent_channels, latent_h, latent_w).
        """
        # VAE downsamples by 8x
        return (
            batch_size,
            self.latent_channels,
            height // 8,
            width // 8,
        )
