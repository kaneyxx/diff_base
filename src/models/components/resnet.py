"""ResNet blocks and sampling layers for UNet architectures."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetBlock2D(nn.Module):
    """2D ResNet block with optional time embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: Optional[int] = None,
        groups: int = 32,
        groups_out: Optional[int] = None,
        eps: float = 1e-6,
        dropout: float = 0.0,
        use_shortcut: Optional[bool] = None,
        output_scale_factor: float = 1.0,
        up: bool = False,
        down: bool = False,
    ):
        """Initialize ResNet block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (defaults to in_channels).
            temb_channels: Time embedding channels (None to disable).
            groups: Number of groups for GroupNorm.
            groups_out: Groups for output norm (defaults to groups).
            eps: Epsilon for GroupNorm.
            dropout: Dropout probability.
            use_shortcut: Force skip connection even if channels match.
            output_scale_factor: Scale factor for output.
            up: Whether to upsample.
            down: Whether to downsample.
        """
        super().__init__()

        out_channels = out_channels or in_channels
        groups_out = groups_out or groups
        self.output_scale_factor = output_scale_factor

        # Determine if we need a skip connection
        if use_shortcut is None:
            use_shortcut = in_channels != out_channels
        self.use_shortcut = use_shortcut

        # First convolution block
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time embedding projection
        self.time_emb_proj = None
        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)

        # Second convolution block
        self.norm2 = nn.GroupNorm(groups_out, out_channels, eps=eps)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Shortcut connection
        if self.use_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )
        else:
            self.conv_shortcut = None

        # Upsampling/downsampling
        self.upsample = None
        self.downsample = None
        if up:
            self.upsample = Upsample2D(in_channels, use_conv=False)
        if down:
            self.downsample = Downsample2D(in_channels, use_conv=False)

        self.nonlinearity = nn.SiLU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Input tensor [B, C, H, W].
            temb: Optional time embedding [B, temb_channels].

        Returns:
            Output tensor [B, out_channels, H, W].
        """
        input_tensor = hidden_states

        # Upsampling/downsampling
        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states)
            input_tensor = self.upsample(input_tensor)
        elif self.downsample is not None:
            hidden_states = self.downsample(hidden_states)
            input_tensor = self.downsample(input_tensor)

        # First conv block
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        # Add time embedding
        if temb is not None and self.time_emb_proj is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)
            hidden_states = hidden_states + temb[:, :, None, None]

        # Second conv block
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        # Skip connection
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output = (input_tensor + hidden_states) / self.output_scale_factor

        return output


class Downsample2D(nn.Module):
    """2D downsampling layer."""

    def __init__(
        self,
        channels: int,
        use_conv: bool = True,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
    ):
        """Initialize downsample layer.

        Args:
            channels: Number of input channels.
            use_conv: Whether to use convolution (vs average pooling).
            out_channels: Number of output channels.
            padding: Padding for convolution.
            name: Name prefix for the conv layer.
        """
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding

        if use_conv:
            self.conv = nn.Conv2d(
                self.channels,
                self.out_channels,
                kernel_size=3,
                stride=2,
                padding=padding,
            )
        else:
            self.conv = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Input tensor [B, C, H, W].

        Returns:
            Downsampled tensor [B, C, H/2, W/2].
        """
        if self.use_conv:
            hidden_states = self.conv(hidden_states)
        else:
            hidden_states = F.avg_pool2d(hidden_states, kernel_size=2, stride=2)
        return hidden_states


class Upsample2D(nn.Module):
    """2D upsampling layer."""

    def __init__(
        self,
        channels: int,
        use_conv: bool = True,
        out_channels: Optional[int] = None,
        name: str = "conv",
        interpolate: bool = True,
    ):
        """Initialize upsample layer.

        Args:
            channels: Number of input channels.
            use_conv: Whether to use convolution after upsampling.
            out_channels: Number of output channels.
            name: Name prefix for the conv layer.
            interpolate: Whether to use interpolation (vs transpose conv).
        """
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.interpolate = interpolate

        if use_conv:
            self.conv = nn.Conv2d(
                self.channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
            )
        else:
            self.conv = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_size: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Input tensor [B, C, H, W].
            output_size: Optional target output size (H, W).

        Returns:
            Upsampled tensor [B, C, H*2, W*2].
        """
        if self.interpolate:
            if output_size is None:
                hidden_states = F.interpolate(
                    hidden_states, scale_factor=2.0, mode="nearest"
                )
            else:
                hidden_states = F.interpolate(
                    hidden_states, size=output_size, mode="nearest"
                )

        if self.conv is not None:
            hidden_states = self.conv(hidden_states)

        return hidden_states


class DownEncoderBlock2D(nn.Module):
    """Encoder block with ResNet blocks and downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: Optional[int] = None,
        num_layers: int = 1,
        resnet_groups: int = 32,
        dropout: float = 0.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
    ):
        """Initialize encoder block.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            temb_channels: Time embedding channels.
            num_layers: Number of ResNet blocks.
            resnet_groups: Groups for GroupNorm.
            dropout: Dropout probability.
            add_downsample: Whether to add downsampling.
            downsample_padding: Padding for downsample conv.
        """
        super().__init__()

        self.resnets = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    dropout=dropout,
                )
            )

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([
                Downsample2D(
                    out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                )
            ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Args:
            hidden_states: Input tensor.
            temb: Time embedding.

        Returns:
            Tuple of (output, list of intermediate outputs for skip connections).
        """
        output_states = []

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states.append(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class UpDecoderBlock2D(nn.Module):
    """Decoder block with ResNet blocks and upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channels: int,
        temb_channels: Optional[int] = None,
        num_layers: int = 1,
        resnet_groups: int = 32,
        dropout: float = 0.0,
        add_upsample: bool = True,
    ):
        """Initialize decoder block.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            prev_output_channels: Channels from skip connection.
            temb_channels: Time embedding channels.
            num_layers: Number of ResNet blocks.
            resnet_groups: Groups for GroupNorm.
            dropout: Dropout probability.
            add_upsample: Whether to add upsampling.
        """
        super().__init__()

        self.resnets = nn.ModuleList()

        for i in range(num_layers):
            # First block receives skip connection
            skip_channels = prev_output_channels if i == 0 else 0
            in_ch = in_channels + skip_channels if i == 0 else out_channels

            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    dropout=dropout,
                )
            )

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([
                Upsample2D(out_channels, use_conv=True, out_channels=out_channels)
            ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        skip_states: Optional[list[torch.Tensor]] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: Input tensor.
            skip_states: Skip connection tensors from encoder.
            temb: Time embedding.

        Returns:
            Output tensor.
        """
        for i, resnet in enumerate(self.resnets):
            # Add skip connection on first block
            if skip_states is not None and i == 0:
                skip = skip_states.pop() if skip_states else None
                if skip is not None:
                    hidden_states = torch.cat([hidden_states, skip], dim=1)

            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states
