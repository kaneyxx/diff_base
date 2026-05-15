"""Tests for FLUX.1 Kontext integration.

Covers:
- AC1: output shape is target-only when img_cond_seq is provided
- AC2: create_flux_transformer(version="v1", variant="kontext") works
- AC11: block_hooks parameter (Phase F, tested once hooks are wired)
- AC12: schnell guidance bug fixed
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.models.flux import create_flux_transformer
from src.models.flux.v1.transformer import Flux1Transformer

# Tiny config to avoid OOM in unit tests.
# Constraints:
#   head_dim = hidden_size / num_attention_heads = 128 / 4 = 32
#   sum(axes_dims_rope) must equal head_dim = 32
TINY_CONFIG = OmegaConf.create({
    "hidden_size": 128,
    "num_attention_heads": 4,
    "num_layers": 2,
    "num_single_layers": 2,
    "in_channels": 64,
    "joint_attention_dim": 64,
    "pooled_projection_dim": 32,
    "axes_dims_rope": [8, 12, 12],  # sum=32 == head_dim
    "rope_theta": 10000.0,
})


def _make_inputs(
    batch: int = 1,
    target_h: int = 2,
    target_w: int = 2,
    txt_seq: int = 4,
    guidance: bool = True,
) -> dict:
    """Build minimal synthetic inputs for Flux1Transformer.forward().

    Dimensions match TINY_CONFIG:
      in_channels=64, joint_attention_dim=64, pooled_projection_dim=32.
    """
    target_seq = target_h * target_w
    in_channels = 64
    txt_dim = 64
    pool_dim = 32

    hidden_states = torch.randn(batch, target_seq, in_channels)
    timestep = torch.rand(batch)
    encoder_hidden_states = torch.randn(batch, txt_seq, txt_dim)
    pooled_projections = torch.randn(batch, pool_dim)
    g = torch.ones(batch) * 3.5 if guidance else None
    return {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "encoder_hidden_states": encoder_hidden_states,
        "pooled_projections": pooled_projections,
        "guidance": g,
    }


class TestAC1OutputSlicing:
    """AC1: forward() returns target-only shape when img_cond_seq is provided."""

    def test_no_kontext_output_shape(self):
        """Without Kontext, output shape matches input hidden_states."""
        model = Flux1Transformer(TINY_CONFIG, variant="dev")
        model.eval()

        inputs = _make_inputs(batch=1, target_h=2, target_w=2)
        target_seq = inputs["hidden_states"].shape[1]

        with torch.no_grad():
            out = model(**inputs)

        in_ch = TINY_CONFIG.in_channels
        assert out.shape == (1, target_seq, in_ch), (
            f"Expected [1, {target_seq}, {in_ch}], got {out.shape}"
        )

    def test_kontext_output_shape_is_target_only(self):
        """With img_cond_seq, output must have target_seq tokens, NOT target+ref."""
        model = Flux1Transformer(TINY_CONFIG, variant="dev")
        model.eval()

        batch = 1
        target_h, target_w = 2, 2
        ref_h, ref_w = 3, 3
        target_seq = target_h * target_w
        ref_seq = ref_h * ref_w

        in_ch = TINY_CONFIG.in_channels
        inputs = _make_inputs(batch=batch, target_h=target_h, target_w=target_w)
        img_cond_seq = torch.randn(batch, ref_seq, in_ch)

        with torch.no_grad():
            out = model(**inputs, img_cond_seq=img_cond_seq)

        assert out.shape == (batch, target_seq, in_ch), (
            f"Output should be target-only [{batch}, {target_seq}, {in_ch}], "
            f"got {out.shape}. Reference tokens must not appear in output."
        )
        assert out.shape[1] != target_seq + ref_seq, (
            "Output includes reference tokens — slicing is broken."
        )

    def test_kontext_with_explicit_position_ids(self):
        """img_cond_seq with explicit img_cond_seq_ids still returns target-only shape."""
        model = Flux1Transformer(TINY_CONFIG, variant="dev")
        model.eval()

        in_ch = TINY_CONFIG.in_channels
        batch, target_seq, ref_seq = 1, 4, 9

        inputs = _make_inputs(batch=batch, target_h=2, target_w=2)
        img_cond_seq = torch.randn(batch, ref_seq, in_ch)
        img_cond_seq_ids = torch.zeros(batch, ref_seq, 3)
        img_cond_seq_ids[:, :, 0] = 1.0  # stream=1 for reference

        with torch.no_grad():
            out = model(**inputs, img_cond_seq=img_cond_seq, img_cond_seq_ids=img_cond_seq_ids)

        assert out.shape == (batch, target_seq, in_ch), (
            f"Expected [{batch}, {target_seq}, {in_ch}], got {out.shape}"
        )

    def test_captured_hidden_states_are_target_only(self):
        """return_hidden_states_at captures are target-only (not target+ref)."""
        model = Flux1Transformer(TINY_CONFIG, variant="dev")
        model.eval()

        in_ch = TINY_CONFIG.in_channels
        hidden_size = TINY_CONFIG.hidden_size
        batch, target_seq, ref_seq = 1, 4, 9
        inputs = _make_inputs(batch=batch, target_h=2, target_w=2)
        img_cond_seq = torch.randn(batch, ref_seq, in_ch)

        with torch.no_grad():
            out, captured = model(
                **inputs,
                img_cond_seq=img_cond_seq,
                return_hidden_states_at=[0],
            )

        assert 0 in captured, "Block 0 should be captured"
        cap = captured[0]
        # Captured states are in hidden_size dimension (post x_embedder projection)
        assert cap.shape == (batch, target_seq, hidden_size), (
            f"Captured state should be target-only [{batch}, {target_seq}, {hidden_size}], "
            f"got {cap.shape}"
        )


class TestAC2KontextVariant:
    """AC2: create_flux_transformer(version='v1', variant='kontext') works."""

    def test_create_flux_transformer_kontext(self):
        """Factory creates a Flux1Transformer with variant='kontext'."""
        transformer = create_flux_transformer(
            version="v1",
            config=TINY_CONFIG,
            variant="kontext",
        )
        assert isinstance(transformer, Flux1Transformer)
        assert transformer.variant == "kontext"
        assert transformer.guidance_embeds is True

    def test_kontext_variant_accepts_guidance(self):
        """Kontext variant (like dev) accepts guidance input."""
        transformer = create_flux_transformer(
            version="v1",
            config=TINY_CONFIG,
            variant="kontext",
        )
        transformer.eval()

        inputs = _make_inputs(batch=1, guidance=True)
        with torch.no_grad():
            out = transformer(**inputs)

        assert out.shape[0] == 1

    def test_unknown_variant_raises(self):
        """Passing an unknown variant raises ValueError."""
        with pytest.raises(ValueError, match="Unknown FLUX.1 variant"):
            Flux1Transformer(TINY_CONFIG, variant="unknown-variant")


class TestAC12SchnellGuidance:
    """AC12: schnell variant has guidance_embeds=False and rejects guidance arg."""

    def test_schnell_guidance_embeds_false(self):
        """Flux1Transformer(variant='schnell') has guidance_embeds=False."""
        # schnell config must set guidance_embeds: false
        schnell_config = OmegaConf.merge(TINY_CONFIG, OmegaConf.create({
            "guidance_embeds": False,
        }))
        model = Flux1Transformer(schnell_config, variant="schnell")
        assert model.guidance_embeds is False

    def test_schnell_forward_without_guidance(self):
        """Schnell forward pass works fine without guidance."""
        schnell_config = OmegaConf.merge(TINY_CONFIG, OmegaConf.create({
            "guidance_embeds": False,
        }))
        model = Flux1Transformer(schnell_config, variant="schnell")
        model.eval()

        inputs = _make_inputs(batch=1, guidance=False)
        with torch.no_grad():
            out = model(**inputs)

        assert out.shape[0] == 1

    def test_schnell_rejects_guidance_arg(self):
        """Schnell raises ValueError when guidance is passed."""
        schnell_config = OmegaConf.merge(TINY_CONFIG, OmegaConf.create({
            "guidance_embeds": False,
        }))
        model = Flux1Transformer(schnell_config, variant="schnell")
        model.eval()

        inputs = _make_inputs(batch=1, guidance=True)
        with pytest.raises(ValueError, match="guidance_embeds=False"):
            with torch.no_grad():
                model(**inputs)

    def test_dev_still_accepts_guidance(self):
        """Dev variant still works correctly with guidance (backward compat)."""
        model = Flux1Transformer(TINY_CONFIG, variant="dev")
        model.eval()

        inputs = _make_inputs(batch=1, guidance=True)
        with torch.no_grad():
            out = model(**inputs)

        assert out.shape[0] == 1


# ---------------------------------------------------------------------------
# AC11 — block_hooks extensibility (Phase F)
# ---------------------------------------------------------------------------

class TestBlockHookInjection:
    """AC11: block_hooks parameter allows downstream conditioning injection."""

    def test_block_hook_noop_runs(self):
        """No-op hook (returns zeros) does not crash forward."""
        model = Flux1Transformer(TINY_CONFIG, variant="dev")
        model.eval()

        def noop_hook(block_idx, hidden_states, txt_hidden, temb):
            return torch.zeros_like(hidden_states)

        inputs = _make_inputs(batch=1, guidance=True)
        with torch.no_grad():
            out = model(**inputs, block_hooks={"joint": [noop_hook], "single": [noop_hook]})

        target_seq = inputs["hidden_states"].shape[1]
        assert out.shape == (1, target_seq, TINY_CONFIG.in_channels)

    def test_block_hook_modifies_output(self):
        """Hook returning non-zero delta changes forward output."""
        model = Flux1Transformer(TINY_CONFIG, variant="dev")
        model.eval()

        inputs = _make_inputs(batch=1, guidance=True)

        with torch.no_grad():
            out_base = model(**inputs)

        BIG = 1e4  # noqa: N806

        def big_delta_hook(block_idx, hidden_states, txt_hidden, temb):
            if block_idx == 0:
                return torch.full_like(hidden_states, BIG)
            return torch.zeros_like(hidden_states)

        with torch.no_grad():
            out_hooked = model(**inputs, block_hooks={"joint": [big_delta_hook]})

        assert not torch.allclose(out_base, out_hooked), (
            "Hook with large delta should change the output"
        )

    def test_block_hook_bad_return_raises(self):
        """Hook returning non-Tensor raises TypeError."""
        model = Flux1Transformer(TINY_CONFIG, variant="dev")
        model.eval()

        def bad_hook(block_idx, hidden_states, txt_hidden, temb):
            return 42  # wrong type

        inputs = _make_inputs(batch=1, guidance=True)
        with pytest.raises(TypeError, match="torch.Tensor delta"):
            model(**inputs, block_hooks={"joint": [bad_hook]})

    def test_no_hooks_identical_to_baseline(self):
        """Backward compat: block_hooks=None gives identical output to omitting arg."""
        model = Flux1Transformer(TINY_CONFIG, variant="dev")
        model.eval()

        inputs = _make_inputs(batch=1, guidance=True)

        with torch.no_grad():
            out_none = model(**inputs, block_hooks=None)
            out_no_arg = model(**inputs)

        assert torch.allclose(out_none, out_no_arg), (
            "block_hooks=None must produce identical output to omitting the arg"
        )

    def test_single_block_hook_receives_none_txt(self):
        """Single-block hooks receive txt_hidden=None (single stream is img-only)."""
        model = Flux1Transformer(TINY_CONFIG, variant="dev")
        model.eval()

        received_txt = []

        def capture_hook(block_idx, hidden_states, txt_hidden, temb):
            received_txt.append(txt_hidden)
            return torch.zeros_like(hidden_states)

        inputs = _make_inputs(batch=1, guidance=True)
        model(**inputs, block_hooks={"single": [capture_hook]})

        assert all(v is None for v in received_txt), (
            "Single-block hooks should receive txt_hidden=None"
        )

    def test_joint_hook_receives_txt_hidden_tensor(self):
        """Joint-block hooks receive txt_hidden as a Tensor."""
        model = Flux1Transformer(TINY_CONFIG, variant="dev")
        model.eval()

        received_txt = []

        def capture_hook(block_idx, hidden_states, txt_hidden, temb):
            received_txt.append(txt_hidden)
            return torch.zeros_like(hidden_states)

        inputs = _make_inputs(batch=1, guidance=True)
        model(**inputs, block_hooks={"joint": [capture_hook]})

        assert all(isinstance(v, torch.Tensor) for v in received_txt), (
            "Joint-block hooks should receive txt_hidden as a Tensor"
        )


# ---------------------------------------------------------------------------
# register_conditioning_module (Phase F)
# ---------------------------------------------------------------------------

class TestRegisterConditioningModule:
    """register_conditioning_module stores nn.Modules on the transformer."""

    def test_register_stores_module(self):
        """register_conditioning_module stores module in conditioning_modules."""
        model = Flux1Transformer(TINY_CONFIG, variant="dev")
        adapter = nn.Linear(64, 64)
        model.register_conditioning_module("my_adapter", adapter)

        assert "my_adapter" in model.conditioning_modules
        assert model.conditioning_modules["my_adapter"] is adapter

    def test_conditioning_module_in_state_dict(self):
        """Registered module parameters appear in state_dict."""
        model = Flux1Transformer(TINY_CONFIG, variant="dev")
        adapter = nn.Linear(64, 32, bias=False)
        model.register_conditioning_module("ctrl", adapter)

        sd = model.state_dict()
        assert any(k.startswith("conditioning_modules.ctrl.") for k in sd), (
            "Registered module params must appear in state_dict"
        )

    def test_register_bad_type_raises(self):
        """register_conditioning_module raises TypeError for non-Module."""
        model = Flux1Transformer(TINY_CONFIG, variant="dev")
        with pytest.raises(TypeError):
            model.register_conditioning_module("bad", object())

    def test_register_duplicate_name_raises(self):
        """Registering same name twice raises ValueError."""
        model = Flux1Transformer(TINY_CONFIG, variant="dev")
        model.register_conditioning_module("dup", nn.Linear(4, 4))
        with pytest.raises(ValueError, match="already registered"):
            model.register_conditioning_module("dup", nn.Linear(4, 4))

    def test_conditioning_module_moves_with_model(self):
        """Registered modules move to the same device as the transformer."""
        model = Flux1Transformer(TINY_CONFIG, variant="dev")
        adapter = nn.Linear(64, 64)
        model.register_conditioning_module("adapter", adapter)

        model.cpu()
        for param in model.conditioning_modules["adapter"].parameters():
            assert param.device == torch.device("cpu")
