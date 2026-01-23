"""Tests for noise schedulers."""

import pytest
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schedulers.ddpm import DDPMScheduler
from src.schedulers.euler import EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
from src.schedulers.flow_matching import FlowMatchingScheduler
from src.schedulers import create_scheduler


class TestDDPMScheduler:
    """Tests for DDPM scheduler."""

    def test_init_default(self):
        """Test default initialization."""
        scheduler = DDPMScheduler()

        assert scheduler.num_train_timesteps == 1000
        assert scheduler.beta_start == 0.00085
        assert scheduler.beta_end == 0.012

    def test_init_custom(self):
        """Test custom initialization."""
        scheduler = DDPMScheduler(
            num_train_timesteps=500,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
        )

        assert scheduler.num_train_timesteps == 500
        assert len(scheduler.betas) == 500

    def test_set_timesteps(self):
        """Test setting inference timesteps."""
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(50)

        assert len(scheduler.timesteps) == 50
        # Should be descending
        assert scheduler.timesteps[0] > scheduler.timesteps[-1]

    def test_add_noise(self):
        """Test adding noise to samples."""
        scheduler = DDPMScheduler()

        batch_size = 4
        original = torch.randn(batch_size, 4, 64, 64)
        noise = torch.randn_like(original)
        timesteps = torch.randint(0, 1000, (batch_size,))

        noisy = scheduler.add_noise(original, noise, timesteps)

        assert noisy.shape == original.shape
        # With noise added, samples should be different
        assert not torch.allclose(noisy, original)

    def test_step(self):
        """Test denoising step."""
        scheduler = DDPMScheduler()
        scheduler.set_timesteps(50)

        sample = torch.randn(2, 4, 64, 64)
        noise_pred = torch.randn_like(sample)
        t = scheduler.timesteps[25]

        output, pred_x0 = scheduler.step(noise_pred, t, sample)

        assert output.shape == sample.shape
        assert pred_x0.shape == sample.shape

    def test_get_velocity(self):
        """Test velocity computation for v-prediction."""
        scheduler = DDPMScheduler()

        sample = torch.randn(2, 4, 64, 64)
        noise = torch.randn_like(sample)
        timesteps = torch.tensor([100, 500])

        velocity = scheduler.get_velocity(sample, noise, timesteps)

        assert velocity.shape == sample.shape

    def test_get_snr(self):
        """Test signal-to-noise ratio computation."""
        scheduler = DDPMScheduler()

        timesteps = torch.tensor([0, 250, 500, 750, 999])
        snr = scheduler.get_snr(timesteps)

        assert snr.shape == timesteps.shape
        # SNR should generally decrease with timestep
        for i in range(len(snr) - 1):
            assert snr[i] >= snr[i + 1]


class TestEulerScheduler:
    """Tests for Euler discrete scheduler."""

    def test_init(self):
        """Test initialization."""
        scheduler = EulerDiscreteScheduler(num_train_timesteps=1000)

        assert scheduler.num_train_timesteps == 1000
        assert hasattr(scheduler, "sigmas")

    def test_set_timesteps(self):
        """Test setting inference timesteps."""
        scheduler = EulerDiscreteScheduler()
        scheduler.set_timesteps(25)

        assert len(scheduler.timesteps) == 25
        assert len(scheduler.sigmas) == 26  # One extra for final step

    def test_step(self):
        """Test denoising step."""
        scheduler = EulerDiscreteScheduler()
        scheduler.set_timesteps(25)

        sample = torch.randn(2, 4, 64, 64)
        noise_pred = torch.randn_like(sample)

        output, pred_x0 = scheduler.step(
            noise_pred,
            scheduler.timesteps[0],
            sample,
            step_index=0,
        )

        assert output.shape == sample.shape

    def test_ancestral_scheduler(self):
        """Test Euler ancestral scheduler."""
        scheduler = EulerAncestralDiscreteScheduler()
        scheduler.set_timesteps(25)

        sample = torch.randn(2, 4, 64, 64)
        noise_pred = torch.randn_like(sample)

        output, pred_x0 = scheduler.step(
            noise_pred,
            scheduler.timesteps[0],
            sample,
            step_index=0,
        )

        assert output.shape == sample.shape


class TestFlowMatchingScheduler:
    """Tests for Flow Matching scheduler."""

    def test_init(self):
        """Test initialization."""
        scheduler = FlowMatchingScheduler(num_train_timesteps=1000)

        assert scheduler.num_train_timesteps == 1000
        assert hasattr(scheduler, "shift")

    def test_set_timesteps(self):
        """Test setting inference timesteps."""
        scheduler = FlowMatchingScheduler()
        scheduler.set_timesteps(50)

        assert len(scheduler.timesteps) == 50

    def test_add_noise(self):
        """Test flow matching noise addition (interpolation)."""
        scheduler = FlowMatchingScheduler()

        sample = torch.randn(2, 4, 64, 64)
        noise = torch.randn_like(sample)
        timesteps = torch.tensor([100, 500])

        noisy = scheduler.add_noise(sample, noise, timesteps)

        assert noisy.shape == sample.shape

    def test_step(self):
        """Test flow matching denoising step."""
        scheduler = FlowMatchingScheduler()
        scheduler.set_timesteps(28)  # Flux typically uses 28 steps

        sample = torch.randn(2, 4, 64, 64)
        velocity_pred = torch.randn_like(sample)

        output, pred_x0 = scheduler.step(
            velocity_pred,
            scheduler.timesteps[0],
            sample,
            step_index=0,
        )

        assert output.shape == sample.shape

    def test_get_velocity(self):
        """Test velocity target computation."""
        scheduler = FlowMatchingScheduler()

        sample = torch.randn(2, 4, 64, 64)
        noise = torch.randn_like(sample)
        timesteps = torch.tensor([100, 500])

        velocity = scheduler.get_velocity(sample, noise, timesteps)

        assert velocity.shape == sample.shape


class TestSchedulerFactory:
    """Tests for scheduler factory function."""

    def test_create_ddpm(self):
        """Test creating DDPM scheduler."""
        from omegaconf import OmegaConf

        config = OmegaConf.create({
            "type": "ddpm",
            "num_train_timesteps": 1000,
            "beta_schedule": "scaled_linear",
        })

        scheduler = create_scheduler(config)

        assert isinstance(scheduler, DDPMScheduler)

    def test_create_euler(self):
        """Test creating Euler scheduler."""
        from omegaconf import OmegaConf

        config = OmegaConf.create({
            "type": "euler",
            "num_train_timesteps": 1000,
        })

        scheduler = create_scheduler(config)

        assert isinstance(scheduler, EulerDiscreteScheduler)

    def test_create_flow_matching(self):
        """Test creating Flow Matching scheduler."""
        from omegaconf import OmegaConf

        config = OmegaConf.create({
            "type": "flow_matching",
            "num_train_timesteps": 1000,
        })

        scheduler = create_scheduler(config)

        assert isinstance(scheduler, FlowMatchingScheduler)

    def test_invalid_scheduler_type(self):
        """Test error for invalid scheduler type."""
        from omegaconf import OmegaConf

        config = OmegaConf.create({
            "type": "invalid_scheduler",
        })

        with pytest.raises(ValueError):
            create_scheduler(config)


class TestSchedulerConsistency:
    """Tests for scheduler consistency."""

    def test_ddpm_deterministic_noise(self):
        """Test that DDPM noise addition is deterministic."""
        scheduler = DDPMScheduler()

        sample = torch.randn(2, 4, 64, 64)
        noise = torch.randn_like(sample)
        timesteps = torch.tensor([500, 500])

        noisy1 = scheduler.add_noise(sample, noise, timesteps)
        noisy2 = scheduler.add_noise(sample, noise, timesteps)

        assert torch.allclose(noisy1, noisy2)

    def test_euler_init_noise_sigma(self):
        """Test Euler scheduler init_noise_sigma property."""
        scheduler = EulerDiscreteScheduler()
        scheduler.set_timesteps(50)

        assert hasattr(scheduler, "init_noise_sigma")
        assert scheduler.init_noise_sigma > 0

    def test_flow_matching_init_noise_sigma(self):
        """Test Flow Matching scheduler init_noise_sigma."""
        scheduler = FlowMatchingScheduler()
        scheduler.set_timesteps(50)

        assert hasattr(scheduler, "init_noise_sigma")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
