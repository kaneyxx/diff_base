"""Tests for BFL export (F.7 / AC13, AC13b).

AC13: to_bfl_checkpoint() produces a file loadable by from_bfl_checkpoint().
AC13b: Round-trip internal -> BFL -> internal must produce identical state dict
       (within float32 precision).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from src.models.flux.v1.bfl_export import convert_internal_to_bfl, to_bfl_checkpoint
from src.models.flux.v1.weight_mapping import convert_state_dict, detect_format

# ---------------------------------------------------------------------------
# Tiny synthetic BFL state dict helpers
# ---------------------------------------------------------------------------

def _make_tiny_bfl_state_dict(
    num_double: int = 2,
    num_single: int = 2,
    hidden: int = 64,
    heads: int = 2,
    mlp_ratio: int = 4,
) -> dict[str, torch.Tensor]:
    """Create a minimal synthetic BFL-format state dict for testing.

    Covers:
    - Static top-level keys (img_in, txt_in, time_in, etc.)
    - double_blocks.N: img_attn.qkv (fused), txt_attn.qkv (fused), plus suffixes
    - single_blocks.N: linear1 (fused q/k/v/mlp), linear2, modulation
    """
    sd: dict[str, torch.Tensor] = {}
    mlp_hidden = hidden * mlp_ratio

    # Static top-level
    for key in [
        "img_in.weight", "img_in.bias",
        "txt_in.weight", "txt_in.bias",
        "time_in.in_proj.weight", "time_in.in_proj.bias",
        "time_in.out_proj.weight", "time_in.out_proj.bias",
        "vector_in.in_proj.weight", "vector_in.in_proj.bias",
        "vector_in.out_proj.weight", "vector_in.out_proj.bias",
        "guidance_in.in_proj.weight", "guidance_in.in_proj.bias",
        "guidance_in.out_proj.weight", "guidance_in.out_proj.bias",
        "final_layer.linear.weight", "final_layer.linear.bias",
        "final_layer.adaLN_modulation.1.weight",
        "final_layer.adaLN_modulation.1.bias",
    ]:
        sd[key] = torch.randn(hidden, hidden) if "weight" in key else torch.randn(hidden)

    # Double blocks
    for n in range(num_double):
        p = f"double_blocks.{n}"
        # Fused qkv: [3*hidden, hidden]
        sd[f"{p}.img_attn.qkv.weight"] = torch.randn(3 * hidden, hidden)
        sd[f"{p}.img_attn.qkv.bias"] = torch.randn(3 * hidden)
        sd[f"{p}.txt_attn.qkv.weight"] = torch.randn(3 * hidden, hidden)
        sd[f"{p}.txt_attn.qkv.bias"] = torch.randn(3 * hidden)
        # Direct-renamed suffixes
        for sfx in [
            ("img_attn.proj.weight", hidden, hidden),
            ("img_attn.proj.bias", hidden,),
            ("txt_attn.proj.weight", hidden, hidden),
            ("txt_attn.proj.bias", hidden,),
            ("img_attn.norm.query_norm.scale", hidden,),
            ("img_attn.norm.key_norm.scale", hidden,),
            ("txt_attn.norm.query_norm.scale", hidden,),
            ("txt_attn.norm.key_norm.scale", hidden,),
            ("img_mlp.0.weight", mlp_hidden, hidden),
            ("img_mlp.0.bias", mlp_hidden,),
            ("img_mlp.2.weight", hidden, mlp_hidden),
            ("img_mlp.2.bias", hidden,),
            ("txt_mlp.0.weight", mlp_hidden, hidden),
            ("txt_mlp.0.bias", mlp_hidden,),
            ("txt_mlp.2.weight", hidden, mlp_hidden),
            ("txt_mlp.2.bias", hidden,),
            ("img_mod.lin.weight", hidden * 6, hidden),
            ("img_mod.lin.bias", hidden * 6,),
            ("txt_mod.lin.weight", hidden * 6, hidden),
            ("txt_mod.lin.bias", hidden * 6,),
        ]:
            name = sfx[0]
            shape = sfx[1:]
            sd[f"{p}.{name}"] = torch.randn(*shape)

    # Single blocks
    for n in range(num_single):
        p = f"single_blocks.{n}"
        # Fused linear1: [3*hidden + mlp_hidden, hidden]
        sd[f"{p}.linear1.weight"] = torch.randn(3 * hidden + mlp_hidden, hidden)
        sd[f"{p}.linear1.bias"] = torch.randn(3 * hidden + mlp_hidden)
        for sfx in [
            ("linear2.weight", hidden, hidden + mlp_hidden),
            ("linear2.bias", hidden,),
            ("modulation.lin.weight", hidden * 3, hidden),
            ("modulation.lin.bias", hidden * 3,),
            ("pre_norm.query_norm.scale", hidden,),
            ("pre_norm.key_norm.scale", hidden,),
        ]:
            name = sfx[0]
            shape = sfx[1:]
            sd[f"{p}.{name}"] = torch.randn(*shape)

    return sd


def _bfl_to_internal(bfl_sd: dict, hidden: int = 64) -> dict:
    """Convert BFL->internal using the production converter."""
    return convert_state_dict(
        bfl_sd,
        source_format="bfl",
        num_heads=2,
        hidden_size=hidden,
        num_double_blocks=2,
        num_single_blocks=2,
    )


# ---------------------------------------------------------------------------
# AC13: to_bfl_checkpoint() produces a valid .safetensors file
# ---------------------------------------------------------------------------

class TestToBflCheckpoint:
    def test_saves_safetensors_file(self):
        """to_bfl_checkpoint must write a .safetensors file."""
        bfl_sd = _make_tiny_bfl_state_dict()
        internal_sd = _bfl_to_internal(bfl_sd)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "flux1-test.safetensors"
            result = to_bfl_checkpoint(internal_sd, out, num_double_blocks=2, num_single_blocks=2)
            assert result == out
            assert out.exists(), "to_bfl_checkpoint must write a .safetensors file"

    def test_output_has_bfl_keys(self):
        """Exported file must contain BFL-style keys (double_blocks / img_in)."""
        bfl_sd = _make_tiny_bfl_state_dict()
        internal_sd = _bfl_to_internal(bfl_sd)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "flux1-test.safetensors"
            to_bfl_checkpoint(internal_sd, out, num_double_blocks=2, num_single_blocks=2)

            from safetensors.torch import load_file
            reloaded = load_file(str(out))

            assert any(k.startswith("double_blocks.") for k in reloaded), (
                "Exported file must have double_blocks.* keys"
            )
            assert any(k.startswith("single_blocks.") for k in reloaded), (
                "Exported file must have single_blocks.* keys"
            )
            assert "img_in.weight" in reloaded, (
                "Exported file must have img_in.weight"
            )

    def test_accepts_nn_module(self):
        """to_bfl_checkpoint must accept an nn.Module and use its state_dict()."""
        class _TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.x_embedder = nn.Linear(64, 64)

        model = _TinyModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "tiny.safetensors"
            # Should not raise; will produce a file with whatever keys x_embedder has
            to_bfl_checkpoint(model, out, num_double_blocks=0, num_single_blocks=0)
            assert out.exists()

    def test_creates_parent_dir(self):
        """to_bfl_checkpoint must create parent directories if they don't exist."""
        bfl_sd = _make_tiny_bfl_state_dict()
        internal_sd = _bfl_to_internal(bfl_sd)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "subdir" / "nested" / "out.safetensors"
            to_bfl_checkpoint(internal_sd, out, num_double_blocks=2, num_single_blocks=2)
            assert out.exists()


# ---------------------------------------------------------------------------
# AC13b: Round-trip BFL -> internal -> BFL -> reload == original BFL
# ---------------------------------------------------------------------------

class TestBFLRoundTrip:
    def test_round_trip_preserves_all_keys(self):
        """BFL -> internal -> BFL must produce the same set of keys."""
        bfl_sd = _make_tiny_bfl_state_dict()
        internal_sd = _bfl_to_internal(bfl_sd)
        reconstructed_bfl = convert_internal_to_bfl(
            internal_sd, num_double_blocks=2, num_single_blocks=2
        )
        assert set(reconstructed_bfl.keys()) == set(bfl_sd.keys()), (
            f"Key mismatch after round-trip.\n"
            f"Missing: {set(bfl_sd.keys()) - set(reconstructed_bfl.keys())}\n"
            f"Extra: {set(reconstructed_bfl.keys()) - set(bfl_sd.keys())}"
        )

    def test_round_trip_preserves_tensor_values(self):
        """BFL -> internal -> BFL must produce tensors equal to the originals."""
        bfl_sd = _make_tiny_bfl_state_dict()
        internal_sd = _bfl_to_internal(bfl_sd)
        reconstructed_bfl = convert_internal_to_bfl(
            internal_sd, num_double_blocks=2, num_single_blocks=2
        )
        for key, orig_tensor in bfl_sd.items():
            assert key in reconstructed_bfl, f"Key {key!r} missing after round-trip"
            rt_tensor = reconstructed_bfl[key]
            assert orig_tensor.shape == rt_tensor.shape, (
                f"Shape mismatch for {key}: {orig_tensor.shape} vs {rt_tensor.shape}"
            )
            assert torch.allclose(orig_tensor, rt_tensor, atol=1e-6), (
                f"Value mismatch for {key}: max_err="
                f"{(orig_tensor - rt_tensor).abs().max().item():.2e}"
            )

    def test_round_trip_fused_qkv_img_attn(self):
        """img_attn.qkv.weight must be reconstructed exactly after split+concat."""
        bfl_sd = _make_tiny_bfl_state_dict(num_double=1, num_single=0)
        orig_qkv = bfl_sd["double_blocks.0.img_attn.qkv.weight"]

        internal_sd = _bfl_to_internal(bfl_sd, hidden=64)
        reconstructed_bfl = convert_internal_to_bfl(internal_sd, num_double_blocks=1, num_single_blocks=0)

        rt_qkv = reconstructed_bfl["double_blocks.0.img_attn.qkv.weight"]
        assert torch.allclose(orig_qkv, rt_qkv, atol=1e-6), (
            "img_attn.qkv.weight must survive split+concat round-trip"
        )

    def test_round_trip_fused_qkv_txt_attn(self):
        """txt_attn.qkv.weight must be reconstructed exactly after split+concat."""
        bfl_sd = _make_tiny_bfl_state_dict(num_double=1, num_single=0)
        orig_qkv = bfl_sd["double_blocks.0.txt_attn.qkv.weight"]

        internal_sd = _bfl_to_internal(bfl_sd, hidden=64)
        reconstructed_bfl = convert_internal_to_bfl(internal_sd, num_double_blocks=1, num_single_blocks=0)

        rt_qkv = reconstructed_bfl["double_blocks.0.txt_attn.qkv.weight"]
        assert torch.allclose(orig_qkv, rt_qkv, atol=1e-6), (
            "txt_attn.qkv.weight must survive split+concat round-trip"
        )

    def test_round_trip_fused_linear1_single_block(self):
        """single_blocks linear1 [q,k,v,mlp] must survive round-trip."""
        bfl_sd = _make_tiny_bfl_state_dict(num_double=0, num_single=1)
        orig_linear1 = bfl_sd["single_blocks.0.linear1.weight"]

        internal_sd = _bfl_to_internal(bfl_sd, hidden=64)
        reconstructed_bfl = convert_internal_to_bfl(internal_sd, num_double_blocks=0, num_single_blocks=1)

        rt_linear1 = reconstructed_bfl["single_blocks.0.linear1.weight"]
        assert torch.allclose(orig_linear1, rt_linear1, atol=1e-6), (
            "single_blocks.linear1 must survive q/k/v/mlp split+concat round-trip"
        )

    def test_round_trip_via_file(self):
        """BFL -> internal -> save .safetensors -> reload -> compare to original BFL."""
        bfl_sd = _make_tiny_bfl_state_dict()
        internal_sd = _bfl_to_internal(bfl_sd)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "round_trip.safetensors"
            to_bfl_checkpoint(internal_sd, out, num_double_blocks=2, num_single_blocks=2)

            from safetensors.torch import load_file
            reloaded = load_file(str(out))

        for key, orig_tensor in bfl_sd.items():
            assert key in reloaded, f"Key {key!r} missing in reloaded file"
            assert torch.allclose(orig_tensor.float(), reloaded[key].float(), atol=1e-6), (
                f"Value mismatch for {key} after file round-trip"
            )

    def test_detect_format_on_exported_file(self):
        """Exported BFL file must be detected as 'bfl' format."""
        bfl_sd = _make_tiny_bfl_state_dict()
        internal_sd = _bfl_to_internal(bfl_sd)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "flux1.safetensors"
            to_bfl_checkpoint(internal_sd, out, num_double_blocks=2, num_single_blocks=2)

            from safetensors.torch import load_file
            reloaded = load_file(str(out))

        fmt = detect_format(reloaded)
        assert fmt == "bfl", f"Expected 'bfl', got {fmt!r}"


# ---------------------------------------------------------------------------
# convert_internal_to_bfl unit tests
# ---------------------------------------------------------------------------

class TestConvertInternalToBfl:
    def test_static_keys_inverted(self):
        """Static key inversion must map all top-level internal keys back to BFL."""
        from src.models.flux.v1.bfl_export import _INTERNAL_TO_BFL_STATIC
        sd = {k: torch.zeros(4) for k in _INTERNAL_TO_BFL_STATIC}
        result = convert_internal_to_bfl(sd, num_double_blocks=0, num_single_blocks=0)
        assert set(result.keys()) == set(_INTERNAL_TO_BFL_STATIC.values()), (
            "Static key inversion produced unexpected keys"
        )

    def test_no_double_or_single_blocks(self):
        """Empty double/single block input yields only static key output."""
        from src.models.flux.v1.bfl_export import _INTERNAL_TO_BFL_STATIC
        sd = {k: torch.zeros(4) for k in _INTERNAL_TO_BFL_STATIC}
        result = convert_internal_to_bfl(sd, num_double_blocks=0, num_single_blocks=0)
        assert not any(k.startswith("double_blocks") for k in result)
        assert not any(k.startswith("single_blocks") for k in result)
