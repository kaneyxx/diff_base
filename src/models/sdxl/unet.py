"""SDXL UNet2DConditionModel implementation."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..components.embeddings import (
    Timesteps,
    TimestepEmbedding,
)
from ..components.resnet import (
    ResnetBlock2D,
    Downsample2D,
    Upsample2D,
)
from ..components.transformer import Transformer2DModel


class CrossAttnDownBlock2D(nn.Module):
    """Downsampling block with cross attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 2,
        resnet_groups: int = 32,
        transformer_layers: int = 1,
        num_attention_heads: int = 8,
        cross_attention_dim: int = 2048,
        add_downsample: bool = True,
    ):
        super().__init__()

        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_ch,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                )
            )
            self.attentions.append(
                Transformer2DModel(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers,
                    cross_attention_dim=cross_attention_dim,
                    use_linear_projection=True,
                )
            )

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([
                Downsample2D(out_channels, use_conv=True, out_channels=out_channels)
            ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        output_states = []

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states)
            output_states.append(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class DownBlock2D(nn.Module):
    """Basic downsampling block without attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int = 2,
        resnet_groups: int = 32,
        add_downsample: bool = True,
    ):
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
                )
            )

        self.downsamplers = None
        if add_downsample:
            self.downsamplers = nn.ModuleList([
                Downsample2D(out_channels, use_conv=True, out_channels=out_channels)
            ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        output_states = []

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states.append(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states


class CrossAttnUpBlock2D(nn.Module):
    """Upsampling block with cross attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        num_layers: int = 2,
        resnet_groups: int = 32,
        transformer_layers: int = 1,
        num_attention_heads: int = 8,
        cross_attention_dim: int = 2048,
        add_upsample: bool = True,
    ):
        super().__init__()

        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList()

        for i in range(num_layers):
            skip_channels = prev_output_channel if i == 0 else out_channels
            in_ch = in_channels if i == 0 else out_channels

            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_ch + skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                )
            )
            self.attentions.append(
                Transformer2DModel(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=out_channels // num_attention_heads,
                    in_channels=out_channels,
                    num_layers=transformer_layers,
                    cross_attention_dim=cross_attention_dim,
                    use_linear_projection=True,
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
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class UpBlock2D(nn.Module):
    """Basic upsampling block without attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        num_layers: int = 2,
        resnet_groups: int = 32,
        add_upsample: bool = True,
    ):
        super().__init__()

        self.resnets = nn.ModuleList()

        for i in range(num_layers):
            skip_channels = prev_output_channel if i == 0 else out_channels
            in_ch = in_channels if i == 0 else out_channels

            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_ch + skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
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
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for resnet in self.resnets:
            res_hidden = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = torch.cat([hidden_states, res_hidden], dim=1)
            hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class UNetMidBlock2DCrossAttn(nn.Module):
    """Middle block with cross attention."""

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        resnet_groups: int = 32,
        transformer_layers: int = 1,
        num_attention_heads: int = 8,
        cross_attention_dim: int = 2048,
    ):
        super().__init__()

        self.resnets = nn.ModuleList([
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                groups=resnet_groups,
            ),
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                groups=resnet_groups,
            ),
        ])

        self.attentions = nn.ModuleList([
            Transformer2DModel(
                num_attention_heads=num_attention_heads,
                attention_head_dim=in_channels // num_attention_heads,
                in_channels=in_channels,
                num_layers=transformer_layers,
                cross_attention_dim=cross_attention_dim,
                use_linear_projection=True,
            )
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.attentions[0](hidden_states, encoder_hidden_states)
        hidden_states = self.resnets[1](hidden_states, temb)
        return hidden_states


class SDXLUNet(nn.Module):
    """SDXL UNet2DConditionModel.

    Locally defined UNet architecture matching SDXL specifications.
    """

    def __init__(self, config: DictConfig):
        """Initialize SDXL UNet.

        Args:
            config: UNet configuration.
        """
        super().__init__()
        self.config = config

        # Default SDXL configuration
        in_channels = config.get("in_channels", 4)
        out_channels = config.get("out_channels", 4)
        model_channels = config.get("model_channels", 320)
        attention_head_dim = config.get("attention_head_dim", 64)
        cross_attention_dim = config.get("cross_attention_dim", 2048)

        # Channel multipliers for each level
        block_out_channels = (320, 640, 1280)
        layers_per_block = 2
        transformer_layers_per_block = [1, 2, 10]

        time_embed_dim = model_channels * 4

        # Time embedding
        self.time_proj = Timesteps(model_channels, flip_sin_to_cos=True)
        self.time_embedding = TimestepEmbedding(model_channels, time_embed_dim)

        # Additional SDXL embeddings (pooled text + time ids)
        addition_time_embed_dim = 256
        self.add_time_proj = Timesteps(addition_time_embed_dim)
        self.add_embedding = TimestepEmbedding(
            in_channels=2816,  # pooled_embed (1280) + 6 * addition_time_embed_dim
            time_embed_dim=time_embed_dim,
        )

        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        # Down blocks
        self.down_blocks = nn.ModuleList()

        output_channel = model_channels
        for i, mult in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = mult
            is_final_block = i == len(block_out_channels) - 1

            num_heads = output_channel // attention_head_dim

            if i > 0:  # First block has no attention
                self.down_blocks.append(
                    CrossAttnDownBlock2D(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        temb_channels=time_embed_dim,
                        num_layers=layers_per_block,
                        transformer_layers=transformer_layers_per_block[i],
                        num_attention_heads=num_heads,
                        cross_attention_dim=cross_attention_dim,
                        add_downsample=not is_final_block,
                    )
                )
            else:
                self.down_blocks.append(
                    DownBlock2D(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        temb_channels=time_embed_dim,
                        num_layers=layers_per_block,
                        add_downsample=not is_final_block,
                    )
                )

        # Mid block
        mid_channels = block_out_channels[-1]
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=mid_channels,
            temb_channels=time_embed_dim,
            transformer_layers=transformer_layers_per_block[-1],
            num_attention_heads=mid_channels // attention_head_dim,
            cross_attention_dim=cross_attention_dim,
        )

        # Up blocks
        self.up_blocks = nn.ModuleList()

        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_transformer_layers = list(reversed(transformer_layers_per_block))

        output_channel = reversed_block_out_channels[0]
        for i, mult in enumerate(reversed_block_out_channels):
            prev_output_channel = output_channel
            output_channel = mult
            input_channel = reversed_block_out_channels[
                min(i + 1, len(reversed_block_out_channels) - 1)
            ]

            is_final_block = i == len(reversed_block_out_channels) - 1
            num_heads = output_channel // attention_head_dim

            if i < len(reversed_block_out_channels) - 1:
                self.up_blocks.append(
                    CrossAttnUpBlock2D(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        prev_output_channel=prev_output_channel,
                        temb_channels=time_embed_dim,
                        num_layers=layers_per_block + 1,
                        transformer_layers=reversed_transformer_layers[i],
                        num_attention_heads=num_heads,
                        cross_attention_dim=cross_attention_dim,
                        add_upsample=not is_final_block,
                    )
                )
            else:
                self.up_blocks.append(
                    UpBlock2D(
                        in_channels=input_channel,
                        out_channels=output_channel,
                        prev_output_channel=prev_output_channel,
                        temb_channels=time_embed_dim,
                        num_layers=layers_per_block + 1,
                        add_upsample=not is_final_block,
                    )
                )

        # Output
        self.conv_norm_out = nn.GroupNorm(32, model_channels, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)

        self._gradient_checkpointing = False

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Optional[dict] = None,
        down_block_additional_residuals: Optional[list[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            sample: Noisy latent [B, 4, H, W].
            timestep: Timestep values [B].
            encoder_hidden_states: Text embeddings [B, seq_len, 2048].
            added_cond_kwargs: Additional conditioning (pooled text, time ids).
            down_block_additional_residuals: ControlNet residuals.
            mid_block_additional_residual: ControlNet mid block residual.

        Returns:
            Predicted noise [B, 4, H, W].
        """
        # Time embedding
        t_emb = self.time_proj(timestep.to(sample.dtype))
        t_emb = self.time_embedding(t_emb)

        # SDXL additional embeddings
        if added_cond_kwargs is not None:
            text_embeds = added_cond_kwargs.get("text_embeds")
            time_ids = added_cond_kwargs.get("time_ids")

            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape(time_ids.shape[0], -1)

            add_embeds = torch.cat([text_embeds, time_embeds], dim=-1)
            add_embeds = self.add_embedding(add_embeds)

            t_emb = t_emb + add_embeds

        # Input conv
        sample = self.conv_in(sample)

        # Down blocks
        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            sample, res_samples = down_block(
                sample, t_emb, encoder_hidden_states
            )
            down_block_res_samples += res_samples

        # Add ControlNet residuals if provided
        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()
            for i, (down_res, ctrl_res) in enumerate(
                zip(down_block_res_samples, down_block_additional_residuals)
            ):
                new_down_block_res_samples += (down_res + ctrl_res,)
            down_block_res_samples = new_down_block_res_samples

        # Mid block
        sample = self.mid_block(sample, t_emb, encoder_hidden_states)

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # Up blocks
        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]

            sample = up_block(sample, res_samples, t_emb, encoder_hidden_states)

        # Output
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing."""
        self._gradient_checkpointing = True

        for block in self.down_blocks:
            if hasattr(block, "gradient_checkpointing"):
                block.gradient_checkpointing = True

        if hasattr(self.mid_block, "gradient_checkpointing"):
            self.mid_block.gradient_checkpointing = True

        for block in self.up_blocks:
            if hasattr(block, "gradient_checkpointing"):
                block.gradient_checkpointing = True
