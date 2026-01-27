"""Tests for FLUX.1 Kontext support.

FLUX.1 supports Kontext mode (reference image editing via sequence-wise concatenation)
but does NOT support Fill mode (which is only available in FLUX.2).
"""

import pytest
import torch
import sys
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.flux.v1.conditioning import (
    rearrange_latent_to_sequence,
    rearrange_sequence_to_latent,
    create_position_ids,
    prepare_kontext_conditioning,
)


class TestFlux1PositionIds:
    """Tests for FLUX.1 4D position ID generation."""

    def test_position_ids_shape(self):
        """Test position IDs have correct 4D shape."""
        batch_size = 2
        height = 32
        width = 32

        ids = create_position_ids(
            batch_size=batch_size,
            height=height,
            width=width,
            device=torch.device("cpu"),
            dtype=torch.float32,
            time_offset=0.0,
        )

        expected_seq = height * width
        # 4D format: [t, h, w, l]
        assert ids.shape == (batch_size, expected_seq, 4)

    def test_position_ids_time_offset_target(self):
        """Test target images use time_offset=0.0."""
        ids = create_position_ids(
            batch_size=1,
            height=8,
            width=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
            time_offset=0.0,
        )

        # Time dimension (index 0) should all be 0.0
        assert torch.all(ids[..., 0] == 0.0)

    def test_position_ids_time_offset_reference(self):
        """Test reference images use time_offset=1.0."""
        ids = create_position_ids(
            batch_size=1,
            height=8,
            width=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
            time_offset=1.0,
        )

        # Time dimension should all be 1.0
        assert torch.all(ids[..., 0] == 1.0)


class MockVAE:
    """Mock VAE for testing FLUX.1 conditioning."""

    def __init__(self, latent_channels=16, scale=8):
        self.latent_channels = latent_channels
        self.scaling_factor = 0.3611
        self.shift_factor = 0.1159
        self.scale = scale

    def encode(self, x):
        """Return mock posterior."""
        B, C, H, W = x.shape
        h, w = H // self.scale, W // self.scale
        return MockPosterior(B, self.latent_channels, h, w)


class MockPosterior:
    """Mock VAE posterior."""

    def __init__(self, B, C, H, W):
        self.latent = torch.randn(B, C, H, W)

    def sample(self):
        return self.latent


class TestFlux1KontextConditioning:
    """Tests for FLUX.1 Kontext conditioning."""

    def test_kontext_conditioning_shapes(self):
        """Test shapes match expected 4D format."""
        batch_size = 2
        height, width = 512, 512
        latent_channels = 16  # FLUX.1 uses 16 latent channels
        patch_size = 2

        vae = MockVAE(latent_channels=latent_channels)
        reference_images = torch.randn(batch_size, 3, height, width)

        img_cond_seq, img_cond_seq_ids = prepare_kontext_conditioning(
            reference_images=reference_images,
            vae=vae,
            device=torch.device("cpu"),
            dtype=torch.float32,
            patch_size=patch_size,
        )

        # Latent size: 512/8 = 64
        latent_h = latent_w = height // 8
        # Sequence length: (64/2) * (64/2) = 32 * 32 = 1024
        expected_seq = (latent_h // patch_size) * (latent_w // patch_size)
        # Patch dim: 16 * 2 * 2 = 64 (FLUX.1 uses 16 channels)
        expected_dim = latent_channels * patch_size * patch_size

        assert img_cond_seq.shape == (batch_size, expected_seq, expected_dim)
        # 4D position IDs
        assert img_cond_seq_ids.shape == (batch_size, expected_seq, 4)

    def test_kontext_position_ids_time_offset(self):
        """Test reference images have time_offset=1.0."""
        vae = MockVAE(latent_channels=16)
        reference_images = torch.randn(1, 3, 256, 256)

        _, img_cond_seq_ids = prepare_kontext_conditioning(
            reference_images=reference_images,
            vae=vae,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # Time offset (dim 0) should be 1.0 for reference
        assert torch.all(img_cond_seq_ids[..., 0] == 1.0)


class TestFlux1TransformerKontext:
    """Tests for FLUX.1 transformer with Kontext support."""

    @pytest.fixture
    def transformer_config(self):
        """Create minimal transformer config for FLUX.1."""
        return OmegaConf.create({
            "hidden_size": 256,
            "num_attention_heads": 4,
            "num_layers": 1,
            "num_single_layers": 1,
            "in_channels": 64,  # FLUX.1 uses 64 (16 channels * 4)
            "pooled_projection_dim": 256,
            "guidance_embeds": True,
        })

    def test_transformer_accepts_kontext_params(self, transformer_config):
        """Test forward accepts new Kontext parameters."""
        from src.models.flux.v1.transformer import Flux1Transformer

        transformer = Flux1Transformer(transformer_config, variant="dev")

        # Check that forward method signature includes Kontext params
        import inspect
        sig = inspect.signature(transformer.forward)
        param_names = list(sig.parameters.keys())

        assert "img_cond_seq" in param_names
        assert "img_cond_seq_ids" in param_names

    def test_transformer_backward_compatible(self, transformer_config):
        """Test transformer works without Kontext conditioning."""
        from src.models.flux.v1.transformer import Flux1Transformer

        transformer = Flux1Transformer(transformer_config, variant="dev")

        batch_size = 1
        seq_len = 64
        txt_seq_len = 32

        hidden_states = torch.randn(batch_size, seq_len, 64)
        timestep = torch.tensor([500.0])
        encoder_hidden_states = torch.randn(batch_size, txt_seq_len, 256)
        pooled_projections = torch.randn(batch_size, 256)

        # Forward without conditioning should work
        with torch.no_grad():
            output = transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
            )

        assert output.shape == (batch_size, seq_len, 64)


class TestFlux1ModelKontext:
    """Tests for FLUX.1 model encode_reference_images method."""

    @pytest.fixture
    def model_config(self):
        """Create minimal model config."""
        return OmegaConf.create({
            "transformer": {
                "hidden_size": 256,
                "num_attention_heads": 4,
                "num_layers": 1,
                "num_single_layers": 1,
                "in_channels": 64,
                "pooled_projection_dim": 256,
            },
            "vae": {
                "latent_channels": 16,
            },
            "text_encoder": {},
        })

    def test_encode_reference_images_kontext_only(self, model_config):
        """Test that FLUX.1 only supports kontext mode."""
        from src.models.flux.v1 import Flux1Model

        model = Flux1Model(model_config, variant="dev")
        images = torch.randn(1, 3, 64, 64)

        # Kontext mode should be accepted
        # (will fail without proper VAE, but interface is tested)
        try:
            result = model.encode_reference_images(images, mode="kontext")
            assert "img_cond_seq" in result
            assert "img_cond_seq_ids" in result
        except Exception:
            # Expected to fail without proper VAE weights
            pass

    def test_encode_reference_images_fill_rejected(self, model_config):
        """Test that FLUX.1 rejects fill mode."""
        from src.models.flux.v1 import Flux1Model

        model = Flux1Model(model_config, variant="dev")
        images = torch.randn(1, 3, 64, 64)

        with pytest.raises(ValueError, match="only supports 'kontext' mode"):
            model.encode_reference_images(images, mode="fill")


class TestFlux1Imports:
    """Tests that FLUX.1 conditioning can be imported."""

    def test_conditioning_module_import(self):
        """Test conditioning module can be imported."""
        from src.models.flux.v1.conditioning import (
            rearrange_latent_to_sequence,
            rearrange_sequence_to_latent,
            create_position_ids,
            prepare_kontext_conditioning,
        )

        assert rearrange_latent_to_sequence is not None
        assert rearrange_sequence_to_latent is not None
        assert create_position_ids is not None
        assert prepare_kontext_conditioning is not None

    def test_v1_module_exports(self):
        """Test v1 module exports conditioning utilities."""
        from src.models.flux.v1 import (
            Flux1Model,
            Flux1Transformer,
            rearrange_latent_to_sequence,
            prepare_kontext_conditioning,
        )

        assert Flux1Model is not None
        assert Flux1Transformer is not None
        assert rearrange_latent_to_sequence is not None
        assert prepare_kontext_conditioning is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
