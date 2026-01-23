"""Tests for model architectures."""

import pytest
import torch
from pathlib import Path
from omegaconf import OmegaConf

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.components.attention import SelfAttention, CrossAttention, JointAttention
from src.models.components.embeddings import TimestepEmbedding, Timesteps, get_timestep_embedding
from src.models.components.resnet import ResnetBlock2D, Downsample2D, Upsample2D
from src.models.components.transformer import FeedForward
from src.models.components.attention import BasicTransformerBlock


class TestAttention:
    """Tests for attention mechanisms."""

    def test_self_attention_forward(self):
        """Test self-attention forward pass."""
        batch_size = 2
        seq_len = 16
        dim = 64
        num_heads = 4

        attn = SelfAttention(dim=dim, num_heads=num_heads)
        x = torch.randn(batch_size, seq_len, dim)

        output = attn(x)

        assert output.shape == (batch_size, seq_len, dim)

    def test_cross_attention_forward(self):
        """Test cross-attention forward pass."""
        batch_size = 2
        seq_len = 16
        context_len = 32
        dim = 64
        context_dim = 128
        num_heads = 4

        attn = CrossAttention(
            dim=dim,
            context_dim=context_dim,
            num_heads=num_heads,
        )
        x = torch.randn(batch_size, seq_len, dim)
        context = torch.randn(batch_size, context_len, context_dim)

        output = attn(x, context)

        assert output.shape == (batch_size, seq_len, dim)

    def test_joint_attention_forward(self):
        """Test joint attention forward pass."""
        batch_size = 2
        seq_len = 16
        context_len = 32
        dim = 64
        num_heads = 4

        attn = JointAttention(dim=dim, num_heads=num_heads)
        x = torch.randn(batch_size, seq_len, dim)
        context = torch.randn(batch_size, context_len, dim)

        out_x, out_ctx = attn(x, context)

        assert out_x.shape == (batch_size, seq_len, dim)
        assert out_ctx.shape == (batch_size, context_len, dim)


class TestEmbeddings:
    """Tests for embedding layers."""

    def test_timestep_embedding(self):
        """Test sinusoidal timestep embedding."""
        batch_size = 4
        dim = 128

        timesteps = torch.randint(0, 1000, (batch_size,))
        embedding = get_timestep_embedding(timesteps, dim)

        assert embedding.shape == (batch_size, dim)

    def test_timesteps_module(self):
        """Test Timesteps module."""
        num_channels = 128
        timesteps_module = Timesteps(num_channels)

        batch_size = 4
        timesteps = torch.randint(0, 1000, (batch_size,))
        embedding = timesteps_module(timesteps)

        assert embedding.shape == (batch_size, num_channels)

    def test_timestep_embedding_mlp(self):
        """Test TimestepEmbedding MLP."""
        in_channels = 128
        time_embed_dim = 512

        mlp = TimestepEmbedding(in_channels, time_embed_dim)
        x = torch.randn(4, in_channels)

        output = mlp(x)

        assert output.shape == (4, time_embed_dim)


class TestResnet:
    """Tests for ResNet blocks."""

    def test_resnet_block_forward(self):
        """Test ResNet block forward pass."""
        batch_size = 2
        in_channels = 64
        out_channels = 128
        height, width = 32, 32
        temb_channels = 256

        block = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
        )

        x = torch.randn(batch_size, in_channels, height, width)
        temb = torch.randn(batch_size, temb_channels)

        output = block(x, temb)

        assert output.shape == (batch_size, out_channels, height, width)

    def test_resnet_block_same_channels(self):
        """Test ResNet block with same input/output channels."""
        batch_size = 2
        channels = 64
        height, width = 32, 32

        block = ResnetBlock2D(
            in_channels=channels,
            out_channels=channels,
            temb_channels=None,
        )

        x = torch.randn(batch_size, channels, height, width)

        output = block(x, None)

        assert output.shape == (batch_size, channels, height, width)

    def test_downsample(self):
        """Test downsampling layer."""
        batch_size = 2
        channels = 64
        height, width = 32, 32

        downsample = Downsample2D(channels, use_conv=True)
        x = torch.randn(batch_size, channels, height, width)

        output = downsample(x)

        assert output.shape == (batch_size, channels, height // 2, width // 2)

    def test_upsample(self):
        """Test upsampling layer."""
        batch_size = 2
        channels = 64
        height, width = 16, 16

        upsample = Upsample2D(channels, use_conv=True)
        x = torch.randn(batch_size, channels, height, width)

        output = upsample(x)

        assert output.shape == (batch_size, channels, height * 2, width * 2)


class TestTransformer:
    """Tests for transformer components."""

    def test_feedforward(self):
        """Test FeedForward network."""
        dim = 256
        mult = 4

        ff = FeedForward(dim=dim, mult=mult)
        x = torch.randn(2, 16, dim)

        output = ff(x)

        assert output.shape == (2, 16, dim)

    def test_basic_transformer_block(self):
        """Test BasicTransformerBlock."""
        dim = 256
        num_heads = 4
        context_dim = 512

        block = BasicTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            context_dim=context_dim,
        )

        x = torch.randn(2, 16, dim)
        context = torch.randn(2, 32, context_dim)

        output = block(x, context=context)

        assert output.shape == (2, 16, dim)


class TestModelFactory:
    """Tests for model factory function."""

    def test_model_registry_has_sdxl(self):
        """Test that SDXL is in model registry."""
        from src.models import MODEL_REGISTRY

        assert "sdxl" in MODEL_REGISTRY

    def test_model_registry_has_flux(self):
        """Test that Flux is in model registry."""
        from src.models import MODEL_REGISTRY

        assert "flux" in MODEL_REGISTRY


class TestSDXLComponents:
    """Tests for SDXL-specific components."""

    def test_sdxl_unet_blocks_import(self):
        """Test that SDXL UNet components can be imported."""
        from src.models.sdxl.unet import (
            CrossAttnDownBlock2D,
            CrossAttnUpBlock2D,
            UNetMidBlock2DCrossAttn,
        )

        # Just verify imports work
        assert CrossAttnDownBlock2D is not None
        assert CrossAttnUpBlock2D is not None
        assert UNetMidBlock2DCrossAttn is not None

    def test_sdxl_vae_import(self):
        """Test that SDXL VAE can be imported."""
        from src.models.sdxl.vae import SDXLVAE, Encoder, Decoder

        assert SDXLVAE is not None
        assert Encoder is not None
        assert Decoder is not None


class TestFluxComponents:
    """Tests for Flux-specific components."""

    def test_flux_transformer_import(self):
        """Test that Flux transformer can be imported."""
        from src.models.flux.transformer import (
            FluxTransformer,
            FluxJointTransformerBlock,
            FluxSingleTransformerBlock,
        )

        assert FluxTransformer is not None
        assert FluxJointTransformerBlock is not None
        assert FluxSingleTransformerBlock is not None

    def test_flux_vae_import(self):
        """Test that Flux VAE can be imported."""
        from src.models.flux.vae import FluxVAE

        assert FluxVAE is not None


class TestSD3Components:
    """Tests for SD3-specific components."""

    def test_sd3_components_import(self):
        """Test that SD3 components can be imported."""
        from src.models.sd3.components import (
            RMSNorm,
            AdaLayerNormContinuous,
            AdaLayerNormZero,
            FeedForward,
            PatchEmbed,
            CombinedTimestepTextProjEmbeddings,
            QKNorm,
            JointTransformerBlock,
        )

        assert RMSNorm is not None
        assert AdaLayerNormContinuous is not None
        assert AdaLayerNormZero is not None
        assert FeedForward is not None
        assert PatchEmbed is not None
        assert CombinedTimestepTextProjEmbeddings is not None
        assert QKNorm is not None
        assert JointTransformerBlock is not None

    def test_sd3_transformer_import(self):
        """Test that SD3 transformer can be imported."""
        from src.models.sd3.mmdit import SD3Transformer, SD3_VARIANT_CONFIGS

        assert SD3Transformer is not None
        assert "large" in SD3_VARIANT_CONFIGS
        assert "large-turbo" in SD3_VARIANT_CONFIGS
        assert "medium" in SD3_VARIANT_CONFIGS

    def test_sd3_vae_import(self):
        """Test that SD3 VAE can be imported."""
        from src.models.sd3.vae import SD3VAE

        assert SD3VAE is not None

    def test_sd3_text_encoder_import(self):
        """Test that SD3 text encoder can be imported."""
        from src.models.sd3.text_encoder import SD3TextEncoders

        assert SD3TextEncoders is not None

    def test_sd3_model_import(self):
        """Test that SD3 model can be imported."""
        from src.models.sd3 import SD3Model, create_sd3_model, get_available_variants

        assert SD3Model is not None
        assert create_sd3_model is not None
        variants = get_available_variants()
        assert "large" in variants
        assert "medium" in variants

    def test_sd3_in_model_registry(self):
        """Test that SD3 is available via main model factory."""
        from src.models import get_available_sd3_variants

        variants = get_available_sd3_variants()
        assert "large" in variants
        assert "large-turbo" in variants
        assert "medium" in variants

    def test_sd3_rms_norm(self):
        """Test RMSNorm forward pass."""
        from src.models.sd3.components import RMSNorm

        batch_size = 2
        seq_len = 16
        dim = 64

        norm = RMSNorm(dim)
        x = torch.randn(batch_size, seq_len, dim)

        output = norm(x)

        assert output.shape == (batch_size, seq_len, dim)

    def test_sd3_qk_norm(self):
        """Test QKNorm forward pass."""
        from src.models.sd3.components import QKNorm

        batch_size = 2
        num_heads = 4
        seq_len = 16
        head_dim = 64

        qk_norm = QKNorm(head_dim)
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)

        q_out, k_out = qk_norm(q, k)

        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_sd3_patch_embed(self):
        """Test PatchEmbed forward pass."""
        from src.models.sd3.components import PatchEmbed

        batch_size = 2
        in_channels = 16
        hidden_size = 256
        patch_size = 2
        height, width = 32, 32

        patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
        )
        x = torch.randn(batch_size, in_channels, height, width)

        output = patch_embed(x)

        expected_num_patches = (height // patch_size) * (width // patch_size)
        assert output.shape == (batch_size, expected_num_patches, hidden_size)

    def test_sd3_joint_transformer_block(self):
        """Test JointTransformerBlock forward pass."""
        from src.models.sd3.components import JointTransformerBlock

        batch_size = 2
        img_seq_len = 64
        txt_seq_len = 32
        hidden_size = 256
        context_dim = 512
        num_heads = 4

        block = JointTransformerBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            context_dim=context_dim,
        )

        img = torch.randn(batch_size, img_seq_len, hidden_size)
        txt = torch.randn(batch_size, txt_seq_len, context_dim)
        temb = torch.randn(batch_size, hidden_size)

        img_out, txt_out = block(img, txt, temb)

        assert img_out.shape == (batch_size, img_seq_len, hidden_size)
        assert txt_out.shape == (batch_size, txt_seq_len, context_dim)


class TestSD3Model:
    """Tests for SD3 model creation and forward pass."""

    def test_sd3_medium_creation(self):
        """Test SD3.5-Medium model creation."""
        from src.models import create_model

        config = OmegaConf.create({
            'model': {
                'type': 'sd3',
                'variant': 'medium',
                'transformer': {
                    'depth': 24,
                    'hidden_size': 1536,
                    'num_heads': 24,
                    'patch_size': 2,
                    'in_channels': 16,
                    'context_dim': 4096,
                    'pooled_projection_dim': 2048,
                },
                'vae': {
                    'latent_channels': 16,
                },
                'text_encoder': {},
            }
        })

        model = create_model(config)

        assert model.variant == "medium"
        assert model.transformer.depth == 24
        assert model.transformer.hidden_size == 1536

    def test_sd3_medium_forward(self):
        """Test SD3.5-Medium forward pass."""
        from src.models import create_model

        config = OmegaConf.create({
            'model': {
                'type': 'sd3',
                'variant': 'medium',
                'transformer': {
                    'depth': 24,
                    'hidden_size': 1536,
                    'num_heads': 24,
                    'patch_size': 2,
                    'in_channels': 16,
                    'context_dim': 4096,
                    'pooled_projection_dim': 2048,
                },
                'vae': {
                    'latent_channels': 16,
                },
                'text_encoder': {},
            }
        })

        model = create_model(config)

        # Small input for testing
        batch_size = 1
        latents = torch.randn(batch_size, 16, 8, 8)
        timesteps = torch.tensor([500.0])
        encoder_hidden_states = torch.randn(batch_size, 64, 4096)
        pooled_projections = torch.randn(batch_size, 2048)

        with torch.no_grad():
            output = model(
                latents=latents,
                timesteps=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
            )

        assert output.shape == latents.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
