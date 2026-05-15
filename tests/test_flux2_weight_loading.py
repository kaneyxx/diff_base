"""Tests for FLUX.2 weight_mapping and bfl_export modules.

CPU-only, no GPU or HF downloads required. Tiny synthetic Flux2Transformer
configs are used for fast execution (< 30s each).

Covers:
- Format detection for internal state dicts
- STATIC_MAP covers every top-level (non-block) key
- DOUBLE_BLOCK_SUFFIX_MAP and SINGLE_BLOCK_SUFFIX_MAP cover all block keys
- Full internal -> BFL -> internal round-trip with forward-pass equivalence
"""

import pytest
import torch
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Shared tiny-model fixture
# ---------------------------------------------------------------------------

# Tiny overrides: 1 joint block, 1 single block, hidden_size=64, 2 heads.
# in_channels=128 matches FLUX.2 dev (32 latent * 4 patch).
# axes_dims_rope must sum to head_dim = hidden_size // num_heads = 32.
# Use (8, 8, 8, 8) so the 4D RoPE sums to 32.
_TINY_CFG = OmegaConf.create({
    "hidden_size": 64,
    "num_attention_heads": 2,
    "num_layers": 1,
    "num_single_layers": 1,
    "in_channels": 128,
    "guidance_embeds": True,
    "axes_dims_rope": [8, 8, 8, 8],
})


def _make_tiny_transformer():
    """Return a tiny Flux2Transformer on CPU with random weights."""
    from src.models.flux.v2.transformer import Flux2Transformer

    return Flux2Transformer(_TINY_CFG, variant="dev")


# ---------------------------------------------------------------------------
# Test 1: detect_format identifies internal state dicts
# ---------------------------------------------------------------------------


class TestDetectFormat:
    def test_detect_format_internal(self):
        """detect_format on a real Flux2Transformer state_dict returns 'internal'."""
        from src.models.flux.v2.weight_mapping import detect_format

        model = _make_tiny_transformer()
        sd = model.state_dict()
        fmt = detect_format(sd)
        assert fmt == "internal", f"Expected 'internal', got {fmt!r}"

    def test_detect_format_bfl_keys(self):
        """detect_format on a BFL-style dict returns 'bfl'."""
        from src.models.flux.v2.weight_mapping import detect_format

        bfl_sd = {
            "img_in.weight": torch.zeros(1),
            "double_blocks.0.img_attn.to_q.weight": torch.zeros(1),
        }
        assert detect_format(bfl_sd) == "bfl"

    def test_detect_format_unknown_raises(self):
        """detect_format on an unrecognised dict raises ValueError."""
        from src.models.flux.v2.weight_mapping import detect_format

        with pytest.raises(ValueError, match="Cannot detect"):
            detect_format({"unknown.key.weight": torch.zeros(1)})


# ---------------------------------------------------------------------------
# Test 2: STATIC_MAP covers all top-level (non-block) keys
# ---------------------------------------------------------------------------


class TestStaticMapCoverage:
    def test_static_map_covers_all_top_level_keys(self):
        """Every non-block key in state_dict() maps via STATIC_MAP (inverse direction)."""
        from src.models.flux.v2.weight_mapping import STATIC_MAP

        model = _make_tiny_transformer()
        sd = model.state_dict()

        # Build inverse: internal_key -> bfl_key
        inv = {v: k for k, v in STATIC_MAP.items()}

        top_level_keys = [
            k for k in sd.keys()
            if not k.startswith(("transformer_blocks.", "single_transformer_blocks."))
        ]
        missing = [k for k in top_level_keys if k not in inv]
        assert not missing, (
            f"Top-level keys not covered by STATIC_MAP: {missing}"
        )


# ---------------------------------------------------------------------------
# Test 3: Block maps cover all block keys
# ---------------------------------------------------------------------------


class TestBlockMapCoverage:
    def test_block_map_covers_all_double_block_keys(self):
        """Every transformer_blocks.0.* suffix has an entry in DOUBLE_BLOCK_SUFFIX_MAP."""
        from src.models.flux.v2.weight_mapping import DOUBLE_BLOCK_SUFFIX_MAP

        model = _make_tiny_transformer()
        sd = model.state_dict()

        inv_double = {v: k for k, v in DOUBLE_BLOCK_SUFFIX_MAP.items()}
        double_keys = [k for k in sd.keys() if k.startswith("transformer_blocks.0.")]
        prefix_len = len("transformer_blocks.0.")
        missing = [k[prefix_len:] for k in double_keys if k[prefix_len:] not in inv_double]
        assert not missing, (
            f"Double-block suffixes not covered by DOUBLE_BLOCK_SUFFIX_MAP: {missing}"
        )

    def test_block_map_covers_all_single_block_keys(self):
        """Every single_transformer_blocks.0.* suffix has an entry in SINGLE_BLOCK_SUFFIX_MAP."""
        from src.models.flux.v2.weight_mapping import SINGLE_BLOCK_SUFFIX_MAP

        model = _make_tiny_transformer()
        sd = model.state_dict()

        inv_single = {v: k for k, v in SINGLE_BLOCK_SUFFIX_MAP.items()}
        single_keys = [k for k in sd.keys() if k.startswith("single_transformer_blocks.0.")]
        prefix_len = len("single_transformer_blocks.0.")
        missing = [k[prefix_len:] for k in single_keys if k[prefix_len:] not in inv_single]
        assert not missing, (
            f"Single-block suffixes not covered by SINGLE_BLOCK_SUFFIX_MAP: {missing}"
        )


# ---------------------------------------------------------------------------
# Test 4: internal -> BFL -> internal round-trip with forward equivalence
# ---------------------------------------------------------------------------


class TestInternalBflInternalRoundtrip:
    def test_roundtrip_state_dict_keys(self, tmp_path):
        """convert_internal_to_bfl -> load_flux2_checkpoint recovers identical key set."""
        from src.models.flux.v2.bfl_export import to_bfl_checkpoint
        from src.models.flux.v2.weight_mapping import load_flux2_checkpoint

        model = _make_tiny_transformer()
        ckpt_path = tmp_path / "flux2_test.safetensors"
        to_bfl_checkpoint(model, ckpt_path)

        reloaded = load_flux2_checkpoint(str(ckpt_path), target="internal")
        original = dict(model.state_dict().items())

        assert set(reloaded.keys()) == set(original.keys()), (
            f"Key set mismatch after round-trip.\n"
            f"Only in original: {set(original) - set(reloaded)}\n"
            f"Only in reloaded: {set(reloaded) - set(original)}"
        )

    def test_roundtrip_forward_equivalence(self, tmp_path):
        """After round-trip, both models produce identical fp32 outputs (max diff < 1e-5)."""
        from src.models.flux.v2.bfl_export import to_bfl_checkpoint
        from src.models.flux.v2.transformer import Flux2Transformer
        from src.models.flux.v2.weight_mapping import load_flux2_checkpoint

        model_orig = _make_tiny_transformer()
        model_orig.eval()

        # Save to BFL and reload
        ckpt_path = tmp_path / "flux2_roundtrip.safetensors"
        to_bfl_checkpoint(model_orig, ckpt_path)

        reloaded_sd = load_flux2_checkpoint(str(ckpt_path), target="internal")

        model_reloaded = Flux2Transformer(_TINY_CFG, variant="dev")
        model_reloaded.load_state_dict(reloaded_sd, strict=True)
        model_reloaded.eval()

        # Build tiny synthetic inputs
        B = 1
        # FLUX.2 dev: 32 latent channels, patch_size=2 → in_channels=128
        # Use 4x4 latent grid → seq_len = 4 (2x2 patches)
        patch_h, patch_w = 2, 2
        seq_len = patch_h * patch_w  # 4 tokens
        in_channels = 128
        text_dim = 4096  # joint_attention_dim default

        # pooled_projection_dim defaults to 4096 for FLUX.2 dev
        pooled_dim = 4096

        hidden_states = torch.randn(B, seq_len, in_channels, dtype=torch.float32)
        encoder_hidden_states = torch.randn(B, 8, text_dim, dtype=torch.float32)
        pooled_projections = torch.randn(B, pooled_dim, dtype=torch.float32)
        timestep = torch.ones(B, dtype=torch.float32) * 0.5
        guidance = torch.ones(B, dtype=torch.float32) * 3.5

        # Build position IDs [B, seq, 3] for a 2x2 grid
        img_ids = torch.zeros(B, seq_len, 3, dtype=torch.float32)
        for i in range(patch_h):
            for j in range(patch_w):
                idx = i * patch_w + j
                img_ids[:, idx, 1] = float(i)
                img_ids[:, idx, 2] = float(j)

        txt_ids = torch.zeros(B, 8, 3, dtype=torch.float32)

        with torch.no_grad():
            out_orig = model_orig(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                guidance=guidance,
                img_ids=img_ids,
                txt_ids=txt_ids,
            )
            out_reload = model_reloaded(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                guidance=guidance,
                img_ids=img_ids,
                txt_ids=txt_ids,
            )

        # Handle both raw tensor and dataclass/dict output
        if isinstance(out_orig, torch.Tensor):
            out_orig_t = out_orig
            out_reload_t = out_reload
        else:
            # FluxTransformerOutput or similar
            out_orig_t = out_orig[0] if isinstance(out_orig, (list, tuple)) else out_orig.sample
            out_reload_t = out_reload[0] if isinstance(out_reload, (list, tuple)) else out_reload.sample

        max_diff = (out_orig_t.float() - out_reload_t.float()).abs().max().item()
        assert max_diff < 1e-5, (
            f"Round-trip forward pass mismatch: max abs diff = {max_diff:.2e} (threshold 1e-5)"
        )
