"""Flux VAE implementation."""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..sdxl.vae import Encoder, Decoder, DiagonalGaussianDistribution


class FluxVAE(nn.Module):
    """Flux VAE with different scaling factors than SDXL."""

    def __init__(self, config: DictConfig):
        """Initialize Flux VAE.

        Args:
            config: VAE configuration.
        """
        super().__init__()
        self.config = config

        in_channels = config.get("in_channels", 3)
        out_channels = config.get("out_channels", 3)
        latent_channels = config.get("latent_channels", 16)  # Flux uses 16 channels

        block_out_channels = config.get("block_out_channels", (128, 256, 512, 512))
        if isinstance(block_out_channels, list):
            block_out_channels = tuple(block_out_channels)

        layers_per_block = config.get("layers_per_block", 2)
        norm_num_groups = config.get("norm_num_groups", 32)

        # Flux-specific scaling
        self.scaling_factor = config.get("scaling_factor", 0.3611)
        self.shift_factor = config.get("shift_factor", 0.1159)

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)

    def encode(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
    ) -> DiagonalGaussianDistribution:
        """Encode image to latent distribution.

        Args:
            x: Input image [B, 3, H, W] in [-1, 1].
            return_dict: Unused, for API compatibility.

        Returns:
            Diagonal Gaussian distribution.
        """
        h = self.encoder(x)
        moments = self.quant_conv(h)
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
            Decoded image [B, 3, H, W] in [-1, 1].
        """
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
            sample: Input image.
            sample_posterior: Whether to sample from posterior.

        Returns:
            Reconstructed image.
        """
        posterior = self.encode(sample)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        # Apply Flux scaling
        z = (z - self.shift_factor) * self.scaling_factor
        dec = self.decode(z / self.scaling_factor + self.shift_factor)
        return dec

    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image and scale latent.

        Args:
            x: Input image.

        Returns:
            Scaled latent.
        """
        posterior = self.encode(x)
        z = posterior.sample()
        z = (z - self.shift_factor) * self.scaling_factor
        return z

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Unscale and decode latent.

        Args:
            z: Scaled latent.

        Returns:
            Decoded image.
        """
        z = z / self.scaling_factor + self.shift_factor
        return self.decode(z)
