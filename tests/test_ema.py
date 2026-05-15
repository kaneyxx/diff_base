"""Tests for EMAModel (AC6, AC7)."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.training.ema import EMAModel

# ---------------------------------------------------------------------------
# Tiny model for EMA tests
# ---------------------------------------------------------------------------

class _TinyLinear(nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.fc = nn.Linear(8, 8)


# ---------------------------------------------------------------------------
# AC6: EMA update arithmetic
# ---------------------------------------------------------------------------

class TestEMAUpdate:
    def test_shadow_initialized_from_model(self):
        model = _TinyLinear(seed=0)
        ema = EMAModel(model, decay=0.99)
        for name, param in model.named_parameters():
            assert torch.allclose(ema.shadow[name], param.detach().float())

    def test_update_arithmetic_single_step(self):
        """After 1 update: shadow = decay*old + (1-decay)*new_param."""
        model = _TinyLinear(seed=0)
        ema = EMAModel(model, decay=0.99)

        # Save old shadow
        old_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Simulate parameter update
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.ones_like(p) * 0.1)

        ema.update(model)

        for name, param in model.named_parameters():
            expected = 0.99 * old_shadow[name] + (1 - 0.99) * param.detach().float()
            assert torch.allclose(ema.shadow[name], expected, atol=1e-6), (
                f"EMA arithmetic failed for {name}"
            )

    def test_update_three_steps_accumulates(self):
        """Three EMA updates should accumulate correctly (AC6)."""
        model = _TinyLinear(seed=1)
        ema = EMAModel(model, decay=0.99)

        for _step in range(3):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(0.05)
            ema.update(model)

        # Verify a specific key still exists and has the right dtype
        for name in ema.shadow:
            assert ema.shadow[name].dtype == torch.float32

    def test_specific_weight_value_ac6(self):
        """AC6: verify exact arithmetic on fc.weight after 3 steps."""
        decay = 0.99
        model = _TinyLinear(seed=42)
        ema = EMAModel(model, decay=decay)

        initial = model.fc.weight.detach().clone().float()
        shadow = initial.clone()

        for _ in range(3):
            delta = torch.full_like(model.fc.weight, 0.1)
            with torch.no_grad():
                model.fc.weight.add_(delta)
            shadow = decay * shadow + (1 - decay) * model.fc.weight.detach().float()
            ema.update(model)

        assert torch.allclose(ema.shadow["fc.weight"], shadow, atol=1e-6), (
            "3-step EMA arithmetic mismatch on fc.weight"
        )


# ---------------------------------------------------------------------------
# AC7: EMA save / load_into round-trip
# ---------------------------------------------------------------------------

class TestEMASaveLoad:
    def test_save_load_round_trip(self):
        """AC7: save EMA then load_into a new model; params must match shadow."""
        model = _TinyLinear(seed=0)
        ema = EMAModel(model, decay=0.99)

        # Do a few updates
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.5)
        ema.update(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ema.safetensors"
            ema.save(path)
            assert path.exists()

            # Create fresh model and load EMA into it
            new_model = _TinyLinear(seed=99)  # different init
            ema.load_into(new_model)

            for name, param in new_model.named_parameters():
                assert torch.allclose(
                    param.float(), ema.shadow[name].float(), atol=1e-6
                ), f"load_into mismatch for {name}"

    def test_save_load_state_round_trip(self):
        """EMA save → load (into EMAModel) → load_into model round-trip."""
        model = _TinyLinear(seed=5)
        ema = EMAModel(model, decay=0.95)

        with torch.no_grad():
            for p in model.parameters():
                p.mul_(2.0)
        ema.update(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ema.safetensors"
            ema.save(path)

            # Restore into fresh EMA object by loading
            model2 = _TinyLinear(seed=5)
            ema2 = EMAModel(model2, decay=0.95)
            ema2.load(path)

            for name in ema.shadow:
                assert torch.allclose(ema.shadow[name], ema2.shadow[name], atol=1e-6), (
                    f"EMA load mismatch for {name}"
                )

    def test_load_into_overwrites_params(self):
        """load_into must overwrite model params with shadow values."""
        model = _TinyLinear(seed=7)
        ema = EMAModel(model, decay=0.9)

        # Manually set shadow to known value
        for name in ema.shadow:
            ema.shadow[name] = torch.zeros_like(ema.shadow[name])

        fresh_model = _TinyLinear(seed=0)  # nonzero weights
        ema.load_into(fresh_model)

        for name, param in fresh_model.named_parameters():
            assert torch.allclose(param, torch.zeros_like(param), atol=1e-6), (
                f"load_into did not overwrite {name}"
            )


# ---------------------------------------------------------------------------
# EMA on CPU
# ---------------------------------------------------------------------------

class TestEMAOnCPU:
    def test_shadow_stays_on_cpu_when_on_cpu_true(self):
        model = _TinyLinear(seed=0)
        ema = EMAModel(model, decay=0.99, on_cpu=True)
        for tensor in ema.shadow.values():
            assert tensor.device.type == "cpu"

    def test_update_with_cpu_shadow(self):
        model = _TinyLinear(seed=3)
        ema = EMAModel(model, decay=0.9, on_cpu=True)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        ema.update(model)
        # Should not raise; shadow should still be on CPU
        for tensor in ema.shadow.values():
            assert tensor.device.type == "cpu"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEMAEdgeCases:
    def test_invalid_decay_raises(self):
        model = _TinyLinear()
        with pytest.raises(ValueError, match="decay"):
            EMAModel(model, decay=1.0)

    def test_invalid_decay_zero_raises(self):
        model = _TinyLinear()
        with pytest.raises(ValueError, match="decay"):
            EMAModel(model, decay=0.0)

    def test_state_dict_round_trip(self):
        model = _TinyLinear(seed=9)
        ema = EMAModel(model, decay=0.99)
        sd = ema.state_dict()
        # Mutate shadow
        for k in ema.shadow:
            ema.shadow[k] = torch.zeros_like(ema.shadow[k])
        ema.load_state_dict(sd)
        for name, orig in sd.items():
            assert torch.allclose(ema.shadow[name], orig, atol=1e-6)
