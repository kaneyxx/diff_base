"""SDXL VAE (AutoencoderKL) implementation."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..components.resnet import ResnetBlock2D, Downsample2D, Upsample2D
from ..components.attention import SelfAttention


class Encoder(nn.Module):
    """VAE Encoder."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 4,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        double_z: bool = True,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()

        output_channel = block_out_channels[0]
        for i, out_ch in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = out_ch
            is_final_block = i == len(block_out_channels) - 1

            down_block = nn.Module()
            down_block.resnets = nn.ModuleList()

            for _ in range(layers_per_block):
                down_block.resnets.append(
                    ResnetBlock2D(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        groups=norm_num_groups,
                    )
                )
                input_channel = output_channel

            down_block.downsamplers = None
            if not is_final_block:
                down_block.downsamplers = nn.ModuleList([
                    Downsample2D(output_channel, use_conv=True, out_channels=output_channel)
                ])

            self.down_blocks.append(down_block)

        # Mid block
        self.mid_block = nn.Module()
        self.mid_block.resnets = nn.ModuleList([
            ResnetBlock2D(
                in_channels=block_out_channels[-1],
                out_channels=block_out_channels[-1],
                groups=norm_num_groups,
            ),
            ResnetBlock2D(
                in_channels=block_out_channels[-1],
                out_channels=block_out_channels[-1],
                groups=norm_num_groups,
            ),
        ])
        self.mid_block.attentions = nn.ModuleList([
            AttentionBlock(
                block_out_channels[-1],
                num_head_channels=None,
                norm_num_groups=norm_num_groups,
            )
        ])

        # Output
        self.conv_norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[-1], eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, kernel_size=3, padding=1)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.conv_in(sample)

        # Down
        for down_block in self.down_blocks:
            for resnet in down_block.resnets:
                sample = resnet(sample)
            if down_block.downsamplers is not None:
                for downsampler in down_block.downsamplers:
                    sample = downsampler(sample)

        # Mid
        sample = self.mid_block.resnets[0](sample)
        sample = self.mid_block.attentions[0](sample)
        sample = self.mid_block.resnets[1](sample)

        # Output
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class Decoder(nn.Module):
    """VAE Decoder."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
    ):
        super().__init__()

        reversed_block_out_channels = list(reversed(block_out_channels))

        self.conv_in = nn.Conv2d(in_channels, reversed_block_out_channels[0], kernel_size=3, padding=1)

        # Mid block
        self.mid_block = nn.Module()
        self.mid_block.resnets = nn.ModuleList([
            ResnetBlock2D(
                in_channels=reversed_block_out_channels[0],
                out_channels=reversed_block_out_channels[0],
                groups=norm_num_groups,
            ),
            ResnetBlock2D(
                in_channels=reversed_block_out_channels[0],
                out_channels=reversed_block_out_channels[0],
                groups=norm_num_groups,
            ),
        ])
        self.mid_block.attentions = nn.ModuleList([
            AttentionBlock(
                reversed_block_out_channels[0],
                num_head_channels=None,
                norm_num_groups=norm_num_groups,
            )
        ])

        # Up blocks
        self.up_blocks = nn.ModuleList()

        output_channel = reversed_block_out_channels[0]
        for i, out_ch in enumerate(reversed_block_out_channels):
            input_channel = output_channel
            output_channel = out_ch
            is_final_block = i == len(reversed_block_out_channels) - 1

            up_block = nn.Module()
            up_block.resnets = nn.ModuleList()

            for _ in range(layers_per_block + 1):
                up_block.resnets.append(
                    ResnetBlock2D(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        groups=norm_num_groups,
                    )
                )
                input_channel = output_channel

            up_block.upsamplers = None
            if not is_final_block:
                up_block.upsamplers = nn.ModuleList([
                    Upsample2D(output_channel, use_conv=True, out_channels=output_channel)
                ])

            self.up_blocks.append(up_block)

        # Output
        self.conv_norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[0], eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.conv_in(sample)

        # Mid
        sample = self.mid_block.resnets[0](sample)
        sample = self.mid_block.attentions[0](sample)
        sample = self.mid_block.resnets[1](sample)

        # Up
        for up_block in self.up_blocks:
            for resnet in up_block.resnets:
                sample = resnet(sample)
            if up_block.upsamplers is not None:
                for upsampler in up_block.upsamplers:
                    sample = upsampler(sample)

        # Output
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class AttentionBlock(nn.Module):
    """Attention block for VAE."""

    def __init__(
        self,
        channels: int,
        num_head_channels: Optional[int] = None,
        norm_num_groups: int = 32,
        rescale_output_factor: float = 1.0,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = channels // (num_head_channels or channels)
        self.rescale_output_factor = rescale_output_factor

        self.group_norm = nn.GroupNorm(norm_num_groups, channels, eps=1e-6, affine=True)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.proj_attn = nn.Linear(channels, channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, channel, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        # Multi-head attention
        head_dim = channel // self.num_heads
        query = query.view(batch, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch, -1, self.num_heads, head_dim).transpose(1, 2)

        scale = head_dim ** -0.5
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)

        hidden_states = torch.matmul(attn_weights, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch, -1, channel)

        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch, channel, height, width)

        return (hidden_states + residual) / self.rescale_output_factor


class DiagonalGaussianDistribution:
    """Gaussian distribution with diagonal covariance."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self) -> torch.Tensor:
        if self.deterministic:
            return self.mean
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self, other=None) -> torch.Tensor:
        if other is None:
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3],
            )
        return 0.5 * torch.sum(
            torch.pow(self.mean - other.mean, 2) / other.var
            + self.var / other.var
            - 1.0
            - self.logvar
            + other.logvar,
            dim=[1, 2, 3],
        )

    def mode(self) -> torch.Tensor:
        return self.mean


class SDXLVAE(nn.Module):
    """SDXL VAE (AutoencoderKL).

    Encodes images to latent space and decodes back.
    """

    def __init__(self, config: DictConfig):
        """Initialize SDXL VAE.

        Args:
            config: VAE configuration.
        """
        super().__init__()
        self.config = config

        in_channels = config.get("in_channels", 3)
        out_channels = config.get("out_channels", 3)
        latent_channels = config.get("latent_channels", 4)

        block_out_channels = config.get("block_out_channels", (128, 256, 512, 512))
        if isinstance(block_out_channels, list):
            block_out_channels = tuple(block_out_channels)

        layers_per_block = config.get("layers_per_block", 2)
        norm_num_groups = config.get("norm_num_groups", 32)

        self.scaling_factor = config.get("scaling_factor", 0.13025)

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
            z: Latent tensor [B, 4, h, w].
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
        z = z * self.scaling_factor
        dec = self.decode(z / self.scaling_factor)
        return dec
