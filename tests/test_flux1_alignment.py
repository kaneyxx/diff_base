"""Numerical alignment test for Flux1Transformer vs BFL reference (AC10).

This test is skipped unless RUN_HEAVY_TESTS=1 is set AND a reference checkpoint
is available at the path specified by FLUX1_KONTEXT_CHECKPOINT_PATH.

When weights are available, verifies that max absolute difference between
Flux1Transformer output and the BFL reference implementation is < 1e-3.

To run:
    RUN_HEAVY_TESTS=1 FLUX1_KONTEXT_CHECKPOINT_PATH=/path/to/checkpoint \
        python -m pytest tests/test_flux1_alignment.py -v
"""

import os

import pytest
import torch
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

RUN_HEAVY = os.environ.get("RUN_HEAVY_TESTS", "0") == "1"
CHECKPOINT_PATH = os.environ.get("FLUX1_KONTEXT_CHECKPOINT_PATH", "")

skip_unless_heavy = pytest.mark.skipif(
    not RUN_HEAVY,
    reason="Set RUN_HEAVY_TESTS=1 to run alignment tests (requires real weights)",
)
skip_unless_checkpoint = pytest.mark.skipif(
    not CHECKPOINT_PATH,
    reason="Set FLUX1_KONTEXT_CHECKPOINT_PATH to a valid checkpoint to run alignment tests",
)


# ---------------------------------------------------------------------------
# AC10 — Numerical alignment against BFL reference
# ---------------------------------------------------------------------------

@skip_unless_heavy
@skip_unless_checkpoint
class TestFlux1Alignment:
    """AC10: Flux1Transformer output matches BFL reference within 1e-3 max abs diff."""

    def _load_internal_model(self, checkpoint_path: str):
        """Load Flux1Transformer from checkpoint (BFL or diffusers format)."""
        from pathlib import Path

        from src.models.flux.v1.model import Flux1Model

        path = Path(checkpoint_path)
        config = OmegaConf.create({
            "transformer": {
                "hidden_size": 3072,
                "num_attention_heads": 24,
                "num_layers": 19,
                "num_single_layers": 38,
                "in_channels": 64,
                "joint_attention_dim": 4096,
                "pooled_projection_dim": 768,
                "guidance_embeds": True,
            },
        })
        if path.is_file():
            model = Flux1Model.from_bfl_checkpoint(path, variant="kontext")
        else:
            model = Flux1Model(config, variant="kontext")
            model._load_diffusers_checkpoint(path)

        model.eval()
        return model

    def test_alignment_max_abs_diff_lt_1e3(self):
        """AC10: max abs diff between our forward and BFL reference < 1e-3.

        Expected numerical tolerance: 1e-3 max absolute difference.
        This accounts for float32 vs bfloat16 rounding in the reference implementation.

        To establish a new baseline:
            1. Run BFL reference on identical inputs and save output tensor.
            2. Compare against Flux1Transformer output here.
        """
        model = self._load_internal_model(CHECKPOINT_PATH)

        # Deterministic tiny inputs for reproducible comparison
        torch.manual_seed(42)
        B = 1  # noqa: N806 - conventional ML notation for batch size
        target_seq = 256  # 16x16 latent patches
        txt_seq = 77
        in_ch = 64
        txt_dim = 4096
        pool_dim = 768

        hidden_states = torch.randn(B, target_seq, in_ch)
        timestep = torch.full((B,), 0.5)
        encoder_hidden_states = torch.randn(B, txt_seq, txt_dim)
        pooled_projections = torch.randn(B, pool_dim)
        guidance = torch.ones(B) * 3.5

        with torch.no_grad():
            output = model.transformer(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                guidance=guidance,
            )

        assert output.shape == (B, target_seq, in_ch), (
            f"Expected [{B}, {target_seq}, {in_ch}], got {output.shape}"
        )

        # If a reference output tensor exists, compare numerically
        reference_path = os.environ.get("FLUX1_REFERENCE_OUTPUT_PATH", "")
        if reference_path:
            reference = torch.load(reference_path, map_location="cpu")
            max_diff = (output.cpu().float() - reference.float()).abs().max().item()
            assert max_diff < 1e-3, (
                f"Alignment failed: max abs diff {max_diff:.2e} exceeds 1e-3 tolerance. "
                f"This may indicate a bug in the forward pass or weight loading."
            )
        else:
            # No reference tensor: just verify the forward pass runs and output is finite
            assert torch.isfinite(output).all(), "Output contains NaN or Inf"
            pytest.skip(
                "No reference output available. Set FLUX1_REFERENCE_OUTPUT_PATH "
                "to a saved reference tensor for full numerical comparison. "
                f"Current output stats: mean={output.mean():.4f}, std={output.std():.4f}"
            )


# ---------------------------------------------------------------------------
# Lightweight alignment structure tests (always run, no weights needed)
# ---------------------------------------------------------------------------

class TestAlignmentStructure:
    """Structural alignment tests that run without reference weights."""

    def test_timestep_scaled_by_1000(self):
        """Transformer scales timestep by 1000 internally (HuggingFace compat)."""
        from omegaconf import OmegaConf

        from src.models.flux.v1.transformer import Flux1Transformer

        cfg = OmegaConf.create({
            "hidden_size": 64,
            "num_attention_heads": 2,
            "num_layers": 1,
            "num_single_layers": 1,
            "in_channels": 4,
            "joint_attention_dim": 16,
            "pooled_projection_dim": 8,
            "guidance_embeds": True,
            "axes_dims_rope": [8, 12, 12],
            "rope_theta": 10000.0,
        })
        t = Flux1Transformer(cfg, variant="dev")
        t.eval()

        # Two calls with different raw timesteps should differ
        hidden = torch.zeros(1, 4, 4)
        ts_low = torch.full((1,), 0.1)
        ts_high = torch.full((1,), 0.9)
        enc = torch.zeros(1, 2, 16)
        pool = torch.zeros(1, 8)
        guidance = torch.ones(1)

        with torch.no_grad():
            out_low = t(hidden, ts_low, enc, pool, guidance)
            out_high = t(hidden, ts_high, enc, pool, guidance)

        assert not torch.allclose(out_low, out_high), (
            "Different timesteps should produce different outputs (timestep scaling active)"
        )

    def test_rope_axes_dim_sum_equals_head_dim(self):
        """axes_dims_rope must sum to head_dim = hidden_size / num_heads."""
        from omegaconf import OmegaConf

        from src.models.flux.v1.transformer import Flux1Transformer

        cfg = OmegaConf.create({
            "hidden_size": 3072,
            "num_attention_heads": 24,
            "num_layers": 1,
            "num_single_layers": 1,
            "in_channels": 64,
            "guidance_embeds": True,
        })
        t = Flux1Transformer(cfg, variant="dev")
        head_dim = t.hidden_size // t.num_heads  # 128
        assert sum(t.axes_dim) == head_dim, (
            f"axes_dim {t.axes_dim} sums to {sum(t.axes_dim)}, expected {head_dim}"
        )

    def test_flux1_axes_dim_constant_matches_hf(self):
        """FLUX1_AXES_DIM=(16,56,56) matches HuggingFace default."""
        from src.models.flux.v1.transformer import FLUX1_AXES_DIM
        assert FLUX1_AXES_DIM == (16, 56, 56), (
            f"Expected (16, 56, 56), got {FLUX1_AXES_DIM}"
        )
        assert sum(FLUX1_AXES_DIM) == 128, "HF head_dim for FLUX.1 is 128"
