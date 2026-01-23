"""FLUX.2 VAE implementation.

FLUX.2 variants use different latent channel counts:
- dev: 32 latent channels
- klein-4b/9b: 128 latent channels

The higher channel counts in klein variants allow for more
compact spatial representations while maintaining quality.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ...sdxl.vae import Encoder, Decoder, DiagonalGaussianDistribution


class Flux2VAE(nn.Module):
    """FLUX.2 VAE with variant-specific latent channels.

    Supports 32 channels for dev, 128 channels for klein variants.
    """

    # Variant-specific configurations
    VARIANT_CONFIGS = {
        "dev": {
            "latent_channels": 32,
            "scaling_factor": 0.3611,
            "shift_factor": 0.1159,
        },
        "klein-4b": {
            "latent_channels": 128,
            "scaling_factor": 0.3611,
            "shift_factor": 0.1159,
        },
        "klein-9b": {
            "latent_channels": 128,
            "scaling_factor": 0.3611,
            "shift_factor": 0.1159,
        },
    }

    def __init__(self, config: DictConfig, variant: str = "dev"):
        """Initialize FLUX.2 VAE.

        Args:
            config: VAE configuration.
            variant: Model variant ("dev", "klein-4b", or "klein-9b").
        """
        super().__init__()
        self.config = config
        self.variant = variant

        # Get variant defaults
        variant_cfg = self.VARIANT_CONFIGS.get(variant, self.VARIANT_CONFIGS["dev"])

        in_channels = config.get("in_channels", 3)
        out_channels = config.get("out_channels", 3)
        latent_channels = config.get("latent_channels", variant_cfg["latent_channels"])

        # FLUX.2 may use different encoder/decoder configurations
        # for different latent channel counts
        if latent_channels <= 32:
            block_out_channels = config.get("block_out_channels", (128, 256, 512, 512))
        else:
            # Larger latent space may need different encoder structure
            block_out_channels = config.get("block_out_channels", (128, 256, 512, 512, 512))

        if isinstance(block_out_channels, list):
            block_out_channels = tuple(block_out_channels)

        layers_per_block = config.get("layers_per_block", 2)
        norm_num_groups = config.get("norm_num_groups", 32)

        # Scaling factors
        self.scaling_factor = config.get("scaling_factor", variant_cfg["scaling_factor"])
        self.shift_factor = config.get("shift_factor", variant_cfg["shift_factor"])
        self.latent_channels = latent_channels

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

        # Apply scaling
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
