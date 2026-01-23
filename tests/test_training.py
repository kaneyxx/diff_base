"""Tests for training infrastructure."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import OmegaConf

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.methods.lora import (
    LoRALayer,
    LoRALinear,
    inject_lora_layers,
    get_lora_parameters,
)


class TestLoRALayer:
    """Tests for LoRA layer implementation."""

    def test_lora_layer_init(self):
        """Test LoRA layer initialization."""
        in_features = 256
        out_features = 512
        rank = 8

        lora = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=8.0,
        )

        assert lora.lora_down.weight.shape == (rank, in_features)
        assert lora.lora_up.weight.shape == (out_features, rank)

        # lora_up should be initialized to zeros
        assert torch.all(lora.lora_up.weight == 0)

    def test_lora_layer_forward(self):
        """Test LoRA layer forward pass."""
        batch_size = 4
        in_features = 256
        out_features = 512
        rank = 8

        lora = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=rank,
            alpha=8.0,
        )

        x = torch.randn(batch_size, in_features)
        output = lora(x)

        assert output.shape == (batch_size, out_features)
        # Initially should be near zero due to zero initialization of lora_up
        assert output.abs().max() < 1e-6

    def test_lora_linear_forward(self):
        """Test LoRA-wrapped linear layer."""
        batch_size = 4
        in_features = 256
        out_features = 512

        base_layer = nn.Linear(in_features, out_features)
        lora_linear = LoRALinear(
            base_layer=base_layer,
            rank=8,
            alpha=8.0,
            dropout=0.0,
        )

        x = torch.randn(batch_size, in_features)

        # Forward through LoRA linear
        output = lora_linear(x)

        # Initially should match base layer (since LoRA delta is zero)
        expected = base_layer(x)
        assert torch.allclose(output, expected, atol=1e-6)


class TestLoRAInjection:
    """Tests for LoRA injection into models."""

    def test_inject_lora_simple_model(self):
        """Test injecting LoRA into a simple model."""
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(64, 64)
                self.to_k = nn.Linear(64, 64)
                self.to_v = nn.Linear(64, 64)
                self.other = nn.Linear(64, 64)

            def forward(self, x):
                return self.to_q(x) + self.to_k(x) + self.to_v(x)

        model = SimpleModel()

        # Count original params
        original_param_count = sum(p.numel() for p in model.parameters())

        # Inject LoRA
        inject_lora_layers(
            model,
            rank=4,
            alpha=4.0,
            dropout=0.0,
            target_modules=["to_q", "to_k", "to_v"],
        )

        # Check that target layers are wrapped
        assert isinstance(model.to_q, LoRALinear)
        assert isinstance(model.to_k, LoRALinear)
        assert isinstance(model.to_v, LoRALinear)
        assert isinstance(model.other, nn.Linear)  # Not wrapped

        # Check param count increased
        new_param_count = sum(p.numel() for p in model.parameters())
        assert new_param_count > original_param_count

    def test_get_lora_parameters(self):
        """Test getting only LoRA parameters."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = nn.Linear(64, 64)
                self.other = nn.Linear(64, 64)

        model = SimpleModel()
        inject_lora_layers(
            model,
            rank=4,
            alpha=4.0,
            dropout=0.0,
            target_modules=["to_q"],
        )

        lora_params = get_lora_parameters(model)

        # Should only get LoRA params
        assert len(lora_params) > 0
        for param in lora_params:
            assert param.requires_grad

        # Verify we didn't get base model params
        total_lora_numel = sum(p.numel() for p in lora_params)
        # LoRA adds rank * (in + out) params per layer
        expected_lora_numel = 4 * (64 + 64)  # rank * (in_features + out_features)
        assert total_lora_numel == expected_lora_numel


class TestTrainerFactory:
    """Tests for trainer factory."""

    def test_trainer_registry(self):
        """Test that all trainers are registered."""
        from src.training import TRAINER_REGISTRY

        expected_trainers = [
            "lora",
            "full_finetune",
            "dreambooth",
            "controlnet",
            "textual_inversion",
        ]

        for trainer_name in expected_trainers:
            assert trainer_name in TRAINER_REGISTRY


class TestSchedulers:
    """Tests for noise schedulers."""

    def test_ddpm_scheduler_add_noise(self):
        """Test DDPM scheduler noise addition."""
        from src.schedulers.ddpm import DDPMScheduler

        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
        )

        batch_size = 2
        latents = torch.randn(batch_size, 4, 64, 64)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (batch_size,))

        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        assert noisy_latents.shape == latents.shape

    def test_ddpm_scheduler_step(self):
        """Test DDPM scheduler denoising step."""
        from src.schedulers.ddpm import DDPMScheduler

        scheduler = DDPMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(50)

        batch_size = 2
        latents = torch.randn(batch_size, 4, 64, 64)
        noise_pred = torch.randn_like(latents)
        t = scheduler.timesteps[0]

        output, _ = scheduler.step(noise_pred, t, latents)

        assert output.shape == latents.shape

    def test_euler_scheduler(self):
        """Test Euler discrete scheduler."""
        from src.schedulers.euler import EulerDiscreteScheduler

        scheduler = EulerDiscreteScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(50)

        assert len(scheduler.timesteps) == 50

    def test_flow_matching_scheduler(self):
        """Test Flow matching scheduler for Flux."""
        from src.schedulers.flow_matching import FlowMatchingScheduler

        scheduler = FlowMatchingScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(50)

        batch_size = 2
        latents = torch.randn(batch_size, 4, 64, 64)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (batch_size,))

        noisy = scheduler.add_noise(latents, noise, timesteps)
        assert noisy.shape == latents.shape


class TestCheckpointing:
    """Tests for checkpoint save/load."""

    def test_save_and_load_checkpoint(self, tmp_path):
        """Test saving and loading a checkpoint."""
        from src.utils.checkpoint import save_checkpoint, load_checkpoint

        # Create a simple model
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        optimizer = torch.optim.Adam(model.parameters())

        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint"
        config = OmegaConf.create({"test": "value"})

        save_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            step=100,
            epoch=5,
            config=config,
        )

        # Load checkpoint
        loaded = load_checkpoint(checkpoint_path)

        assert loaded["step"] == 100
        assert loaded["epoch"] == 5
        assert "model_state_dict" in loaded

    def test_checkpoint_formats(self, tmp_path):
        """Test different checkpoint formats."""
        from src.utils.checkpoint import save_checkpoint

        model = nn.Linear(64, 64)

        # Save in safetensors format
        save_checkpoint(
            path=tmp_path / "ckpt_safetensors",
            model=model,
            format="safetensors",
        )

        # Check file was created
        assert (tmp_path / "ckpt_safetensors" / "model.safetensors").exists()


class TestLossWeighting:
    """Tests for loss weighting utilities."""

    def test_snr_weighting(self):
        """Test SNR loss weighting computation."""
        from src.schedulers.ddpm import DDPMScheduler

        scheduler = DDPMScheduler(num_train_timesteps=1000)

        timesteps = torch.tensor([100, 500, 900])
        snr = scheduler.get_snr(timesteps)

        assert snr.shape == timesteps.shape
        # SNR should decrease with timestep
        assert snr[0] > snr[1] > snr[2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
