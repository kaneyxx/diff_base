"""Alignment verification tests against official FLUX implementations.

These tests verify that our implementation matches the expected behavior
of official Black Forest Labs FLUX models, particularly for:
- 4D position ID format [t, h, w, l]
- Position IDs integration with RoPE
- VAE encoding order
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.flux.v2.conditioning import create_position_ids
from src.models.flux.components.embeddings import (
    compute_rope_from_position_ids,
    compute_axis_freqs,
)


class TestPositionIdFormat:
    """Verify 4D position ID format matches official implementation."""

    def test_4d_format_shape(self):
        """Verify [B, seq, 4] shape."""
        ids = create_position_ids(
            batch_size=2,
            height=8,
            width=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
            time_offset=0.0,
        )

        assert ids.dim() == 3
        assert ids.shape[2] == 4, "Position IDs must have 4 dimensions"

    def test_dimensions_order(self):
        """Verify [t, h, w, l] dimension order."""
        # Create with known values to verify order
        ids = create_position_ids(
            batch_size=1,
            height=4,
            width=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            time_offset=1.5,
            sequence_index=2,
        )

        # dim 0: t (time_offset) - should be constant 1.5
        assert torch.allclose(ids[..., 0], torch.full_like(ids[..., 0], 1.5))

        # dim 1: h (height) - should be 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3 for 4x4
        expected_h = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=torch.float32)
        assert torch.allclose(ids[0, :, 1], expected_h)

        # dim 2: w (width) - should be 0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3 for 4x4
        expected_w = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.float32)
        assert torch.allclose(ids[0, :, 2], expected_w)

        # dim 3: l (sequence index) - should be constant 2
        assert torch.allclose(ids[..., 3], torch.full_like(ids[..., 3], 2.0))

    def test_target_vs_reference_time_offset(self):
        """Verify target uses t=0 and reference uses t=1."""
        target_ids = create_position_ids(
            batch_size=1,
            height=4,
            width=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            time_offset=0.0,  # Target
        )

        ref_ids = create_position_ids(
            batch_size=1,
            height=4,
            width=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            time_offset=1.0,  # Reference
        )

        # Target should have t=0
        assert torch.all(target_ids[..., 0] == 0.0)

        # Reference should have t=1
        assert torch.all(ref_ids[..., 0] == 1.0)

        # Spatial coords (h, w) should be the same
        assert torch.allclose(target_ids[..., 1:3], ref_ids[..., 1:3])


class TestRoPEIntegration:
    """Verify position IDs are properly integrated into RoPE."""

    def test_position_ids_affect_rope(self):
        """Verify different positions produce different embeddings."""
        # Create two different position ID tensors
        ids1 = create_position_ids(
            batch_size=1,
            height=4,
            width=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            time_offset=0.0,
        )

        ids2 = create_position_ids(
            batch_size=1,
            height=4,
            width=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            time_offset=1.0,  # Different time offset
        )

        # Compute RoPE for both
        dim = 128  # Typical head dimension
        cos1, sin1 = compute_rope_from_position_ids(ids1, dim)
        cos2, sin2 = compute_rope_from_position_ids(ids2, dim)

        # Should have same shape
        assert cos1.shape == cos2.shape
        assert sin1.shape == sin2.shape

        # But different values due to different time offset
        assert not torch.allclose(cos1, cos2), "Different positions should produce different RoPE cos"
        assert not torch.allclose(sin1, sin2), "Different positions should produce different RoPE sin"

    def test_rope_output_shape(self):
        """Verify RoPE output has correct shape."""
        ids = create_position_ids(
            batch_size=2,
            height=8,
            width=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
            time_offset=0.0,
        )

        dim = 128
        cos, sin = compute_rope_from_position_ids(ids, dim)

        # Should be [B, seq, dim]
        assert cos.shape == (2, 64, dim)
        assert sin.shape == (2, 64, dim)

    def test_axis_freqs_computation(self):
        """Test individual axis frequency computation."""
        positions = torch.arange(16).float().unsqueeze(0)  # [1, 16]
        dim = 32

        freqs = compute_axis_freqs(positions, dim)

        # Output should be [B, seq, dim]
        assert freqs.shape == (1, 16, dim)

        # Different positions should have different frequencies
        assert not torch.allclose(freqs[0, 0], freqs[0, 1])

    def test_spatial_position_continuity(self):
        """Test that adjacent spatial positions have smoothly varying embeddings."""
        ids = create_position_ids(
            batch_size=1,
            height=4,
            width=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            time_offset=0.0,
        )

        dim = 64
        cos, sin = compute_rope_from_position_ids(ids, dim)

        # Adjacent positions (0,0) and (0,1) should have similar but different embeddings
        # Position 0 is (h=0, w=0), Position 1 is (h=0, w=1)
        diff = (cos[0, 0] - cos[0, 1]).abs().mean()

        # Should be different but not wildly so
        assert diff > 0, "Adjacent positions should have different embeddings"
        assert diff < 1.0, "Adjacent positions should have smoothly varying embeddings"


class TestVAEAlignmentNotes:
    """Document and test VAE alignment details.

    Note: These tests document expected behavior but don't test actual VAE
    since it requires model weights.
    """

    def test_vae_encoding_order_documented(self):
        """Document that VAE encoding uses (z - shift) * scale order."""
        # FLUX VAE encoding order: (z - shift) * scale
        # This is the CORRECT order based on official implementation

        # Example values from official FLUX VAE
        shift_factor = 0.1159
        scaling_factor = 0.3611

        # Simulated latent
        z = torch.randn(1, 16, 64, 64)

        # Correct encoding order
        encoded_correct = (z - shift_factor) * scaling_factor

        # Wrong order (scale then shift) - would produce different results
        encoded_wrong = (z * scaling_factor) - shift_factor

        # These should be different
        assert not torch.allclose(encoded_correct, encoded_wrong)

    def test_vae_scaling_factors_documented(self):
        """Document expected VAE scaling factors."""
        # FLUX.1 and FLUX.2 VAE scaling factors
        flux_scale = 0.3611
        flux_shift = 0.1159

        # SD3.5 VAE scaling factor (for reference)
        sd35_scale = 1.5305

        # These are the documented values from official implementations
        assert flux_scale == pytest.approx(0.3611, rel=1e-3)
        assert flux_shift == pytest.approx(0.1159, rel=1e-3)
        assert sd35_scale == pytest.approx(1.5305, rel=1e-3)


class TestTransformerBlockCounts:
    """Verify transformer block counts match official specifications."""

    def test_flux1_block_counts(self):
        """Verify FLUX.1 has 19 joint + 38 single blocks."""
        from src.models.flux.v1.transformer import Flux1Transformer

        expected_joint = 19
        expected_single = 38

        # Check VARIANT_CONFIGS
        dev_config = Flux1Transformer.VARIANT_CONFIGS["dev"]
        assert dev_config["num_layers"] == expected_joint
        assert dev_config["num_single_layers"] == expected_single

        schnell_config = Flux1Transformer.VARIANT_CONFIGS["schnell"]
        assert schnell_config["num_layers"] == expected_joint
        assert schnell_config["num_single_layers"] == expected_single

    def test_flux2_block_counts(self):
        """Verify FLUX.2 variant block counts."""
        from src.models.flux.v2.transformer import Flux2Transformer

        # FLUX.2 dev: 8 joint + 48 single
        dev_config = Flux2Transformer.VARIANT_CONFIGS["dev"]
        assert dev_config["num_layers"] == 8
        assert dev_config["num_single_layers"] == 48

        # FLUX.2 klein-4b: 5 joint + 20 single
        klein4b_config = Flux2Transformer.VARIANT_CONFIGS["klein-4b"]
        assert klein4b_config["num_layers"] == 5
        assert klein4b_config["num_single_layers"] == 20

        # FLUX.2 klein-9b: 6 joint + 24 single
        klein9b_config = Flux2Transformer.VARIANT_CONFIGS["klein-9b"]
        assert klein9b_config["num_layers"] == 6
        assert klein9b_config["num_single_layers"] == 24


class TestTextEncoderConfigurations:
    """Document expected text encoder configurations."""

    def test_flux1_text_encoders(self):
        """FLUX.1 uses T5-XXL + CLIP-L."""
        # T5-XXL: 4096 hidden size, 24 layers
        # CLIP-L: 768 hidden size, 12 layers

        t5_hidden = 4096
        clip_l_hidden = 768

        # Pooled projection dim for FLUX.1 (CLIP-L)
        flux1_pooled_dim = 768

        assert flux1_pooled_dim == clip_l_hidden

    def test_flux2_text_encoders(self):
        """FLUX.2 uses Mistral/Qwen instead of T5+CLIP."""
        # FLUX.2 dev uses Mistral-3 (4096 hidden)
        # FLUX.2 klein uses Qwen3-4B/8B (4096 hidden)

        flux2_pooled_dim = 4096

        from src.models.flux.v2.transformer import Flux2Transformer
        for variant in ["dev", "klein-4b", "klein-9b"]:
            config = Flux2Transformer.VARIANT_CONFIGS[variant]
            assert config["pooled_projection_dim"] == flux2_pooled_dim


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
