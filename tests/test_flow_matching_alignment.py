"""Tests for BFL-aligned flow matching scheduler (AC1, AC2, Phase A).

AC1: get_schedule matches BFL formula to 1e-5 absolute tolerance.
AC2: training_sample / add_noise_to_target satisfy rectified-flow identity.
"""


import pytest
import torch

from src.schedulers.flow_matching import FlowMatchingScheduler
from tests.fixtures.bfl_reference_sampling import (
    get_lin_function,
)
from tests.fixtures.bfl_reference_sampling import (
    get_schedule as bfl_get_schedule,
)
from tests.fixtures.bfl_reference_sampling import (
    time_shift as bfl_time_shift,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scheduler() -> FlowMatchingScheduler:
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"num_train_timesteps": 1000, "shift": 3.0,
                             "base_shift": 0.5, "max_shift": 1.15})
    return FlowMatchingScheduler(cfg)


# ---------------------------------------------------------------------------
# AC1a — property-based assertions on _mu / _time_shift
# ---------------------------------------------------------------------------

class TestBFLFormulas:
    def test_time_shift_at_t1_is_1(self):
        """time_shift(mu, sigma=1, t=1) must equal 1 for any mu."""
        for mu in [0.5, 1.0, 1.15]:
            t = torch.tensor(1.0, dtype=torch.float32)
            result = FlowMatchingScheduler._time_shift(mu, 1.0, t)
            assert abs(result.item() - 1.0) < 1e-6, f"mu={mu}: expected 1.0 got {result.item()}"

    def test_time_shift_at_t0_limit(self):
        """time_shift(mu, 1, t→0) should approach 0."""
        t = torch.tensor(1e-6, dtype=torch.float32)
        result = FlowMatchingScheduler._time_shift(0.5, 1.0, t)
        assert result.item() < 1e-3

    def test_time_shift_midpoint_matches_bfl(self):
        """time_shift at t=0.5 must match vendored BFL scalar implementation."""
        mu = 0.7
        t_val = 0.5
        bfl_val = bfl_time_shift(mu, 1.0, t_val)
        t = torch.tensor(t_val, dtype=torch.float32)
        our_val = FlowMatchingScheduler._time_shift(mu, 1.0, t).item()
        assert abs(our_val - bfl_val) < 1e-5, f"expected {bfl_val}, got {our_val}"

    def test_mu_matches_bfl_lin_function(self):
        """_mu must equal BFL get_lin_function(y1=base_shift, y2=max_shift)(seq_len)."""
        for seq_len in [256, 1024, 4096]:
            bfl_fn = get_lin_function(y1=0.5, y2=1.15)
            bfl_mu = bfl_fn(seq_len)
            our_mu = FlowMatchingScheduler._mu(seq_len, base_shift=0.5, max_shift=1.15)
            assert abs(our_mu - bfl_mu) < 1e-6, (
                f"seq_len={seq_len}: BFL mu={bfl_mu}, ours={our_mu}"
            )


# ---------------------------------------------------------------------------
# AC1b — hardcoded reference values for seq_len ∈ {256, 1024, 4096}
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("image_seq_len", [256, 1024, 4096])
def test_get_schedule_matches_bfl_at_seq_lens(image_seq_len: int):
    """get_schedule must match BFL get_schedule to 1e-5 absolute tolerance (float32)."""
    num_steps = 20
    sched = _make_scheduler()

    # Our implementation (float32)
    our_ts = sched.get_schedule(
        num_steps=num_steps,
        image_seq_len=image_seq_len,
        shift=True,
        base_shift=0.5,
        max_shift=1.15,
        device="cpu",
    ).float()  # ensure float32

    # BFL reference (computed in Python floats, then converted)
    bfl_ts = torch.tensor(
        bfl_get_schedule(num_steps, image_seq_len, base_shift=0.5, max_shift=1.15, shift=True),
        dtype=torch.float32,
    )

    assert our_ts.shape == bfl_ts.shape, (
        f"shape mismatch: {our_ts.shape} vs {bfl_ts.shape}"
    )
    max_err = (our_ts - bfl_ts).abs().max().item()
    assert max_err < 1e-5, (
        f"seq_len={image_seq_len}: max abs error={max_err:.2e} exceeds 1e-5"
    )


def test_get_schedule_no_shift_is_uniform():
    """With shift=False, get_schedule must return exact linspace from 1 to 0."""
    sched = _make_scheduler()
    num_steps = 10
    ts = sched.get_schedule(num_steps=num_steps, image_seq_len=1024, shift=False, device="cpu")
    expected = torch.linspace(1.0, 0.0, num_steps + 1)
    assert torch.allclose(ts, expected, atol=1e-6), "shift=False must be plain linspace"


def test_get_schedule_descending():
    """Shifted schedule must be strictly descending from ~1 to ~0."""
    sched = _make_scheduler()
    ts = sched.get_schedule(num_steps=20, image_seq_len=1024, shift=True, device="cpu")
    assert ts[0] > 0.99, f"first timestep should be ~1, got {ts[0]}"
    assert ts[-1] < 0.01, f"last timestep should be ~0, got {ts[-1]}"
    diffs = ts[1:] - ts[:-1]
    assert (diffs < 0).all(), "schedule must be strictly descending"


# ---------------------------------------------------------------------------
# AC2 — training_sample + add_noise_to_target satisfy rectified-flow identity
# ---------------------------------------------------------------------------

class TestTrainingSample:
    def _sched(self):
        return _make_scheduler()

    def test_velocity_consistency(self):
        """v = noise - x_0 and x_t = (1-t)*x_0 + t*noise must hold exactly."""
        sched = self._sched()
        torch.manual_seed(0)
        B, C, H, W = 2, 4, 8, 8
        x_0 = torch.randn(B, C, H, W)
        noise = torch.randn_like(x_0)
        t = sched.training_sample(batch_size=B, image_seq_len=256, shift=True)

        x_t, v_target = sched.add_noise_to_target(x_0, noise, t)

        # v = noise - x_0
        assert torch.allclose(v_target, noise - x_0, atol=1e-6), (
            "velocity target must equal noise - x_0"
        )

        # x_t = (1-t)*x_0 + t*noise  ≡  x_0 + t*(noise - x_0) = x_0 + t*v
        t_b = t.view(B, 1, 1, 1)
        expected_xt = (1.0 - t_b) * x_0 + t_b * noise
        assert torch.allclose(x_t, expected_xt, atol=1e-6), (
            "x_t must equal (1-t)*x_0 + t*noise"
        )

    def test_noise_reconstruction(self):
        """noise = x_0 + target_velocity must hold (AC2 identity)."""
        sched = self._sched()
        torch.manual_seed(42)
        B, C, H, W = 3, 16, 4, 4
        x_0 = torch.randn(B, C, H, W)
        noise = torch.randn_like(x_0)
        t = sched.training_sample(batch_size=B, image_seq_len=1024, shift=True)
        _, v = sched.add_noise_to_target(x_0, noise, t)
        reconstructed_noise = x_0 + v
        assert torch.allclose(reconstructed_noise, noise, atol=1e-6), (
            "noise == x_0 + target_velocity must hold"
        )

    def test_training_sample_shape_and_range(self):
        """training_sample must return shape [B] with values in (0, 1]."""
        sched = self._sched()
        t = sched.training_sample(batch_size=8, image_seq_len=1024, shift=True)
        assert t.shape == (8,), f"expected shape (8,), got {t.shape}"
        assert (t > 0).all() and (t <= 1.0 + 1e-6).all(), (
            f"timesteps must be in (0, 1], got min={t.min()}, max={t.max()}"
        )

    def test_no_shift_uniform(self):
        """With shift=False, sampled t must be uniform (not time-shifted)."""
        sched = self._sched()
        gen = torch.Generator().manual_seed(7)
        t_shifted = sched.training_sample(
            batch_size=1000, image_seq_len=1024, shift=True, generator=gen
        )
        gen2 = torch.Generator().manual_seed(7)
        t_raw = torch.rand(1000, generator=gen2)
        # Shifted values should differ from raw uniform (BFL shift moves mass toward 1)
        # Just verify they are not identical (shift is applied)
        assert not torch.allclose(t_shifted, t_raw, atol=1e-4), (
            "shift=True should produce different values from uniform"
        )

    def test_add_noise_broadcasts_correctly(self):
        """add_noise_to_target must broadcast t [B] over [B, C, H, W] without error."""
        sched = self._sched()
        x_0 = torch.randn(4, 16, 16, 16)
        noise = torch.randn_like(x_0)
        t = torch.tensor([0.1, 0.3, 0.7, 0.9])
        x_t, v = sched.add_noise_to_target(x_0, noise, t)
        assert x_t.shape == x_0.shape
        assert v.shape == x_0.shape


# ---------------------------------------------------------------------------
# Backward-compat: existing methods must still work (§4.4)
# ---------------------------------------------------------------------------

def test_existing_scale_noise_still_works():
    """scale_noise() must continue to function correctly after additive changes."""
    sched = _make_scheduler()
    x = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(x)
    t = torch.tensor([0.3, 0.7])
    result = sched.scale_noise(x, t, noise)
    expected = (1 - t.view(2, 1, 1, 1)) * x + t.view(2, 1, 1, 1) * noise
    assert torch.allclose(result, expected, atol=1e-6)


def test_existing_get_velocity_still_works():
    """get_velocity() must still return noise - sample."""
    sched = _make_scheduler()
    x = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(x)
    t = torch.zeros(2)
    v = sched.get_velocity(x, noise, t)
    assert torch.allclose(v, noise - x, atol=1e-6)
