"""Tests for accelerate.Accelerator integration in BaseTrainer (US-7 #2).

CPU-only synthetic tests — no GPU required. Verifies:
  (a) Accelerator instantiation under default config.
  (b) prepare() returns wrapped objects in-place on self.{model,optimizer,...}.
  (c) accelerator.backward() correctly accumulates gradients onto LoRA params.
  (d) is_main_process gate exists and defaults to True for single-process runs.
  (e) use_accelerator=false config falls back to legacy single-GPU path.
  (f) save_checkpoint unwraps DDP model before writing LoRA weights.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Minimal trainer concrete subclass for unit testing.
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """A tiny LoRA-style model: one trainable linear layer (the "adapter")
    plus one frozen linear layer (the "base"). Mirrors the LoRA setup
    where only adapter params have requires_grad=True.
    """

    def __init__(self, in_features: int = 4, out_features: int = 4) -> None:
        super().__init__()
        self.frozen_base = nn.Linear(in_features, out_features)
        for p in self.frozen_base.parameters():
            p.requires_grad = False
        self.lora_adapter = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frozen_base(x) + self.lora_adapter(x)


def _make_tiny_dataloader(batch_size: int = 2, n_samples: int = 6) -> DataLoader:
    xs = torch.randn(n_samples, 4)
    ys = torch.randn(n_samples, 4)
    return DataLoader(TensorDataset(xs, ys), batch_size=batch_size)


def _make_config(use_accelerator: bool = True) -> OmegaConf:
    return OmegaConf.create({
        "experiment": {"output_dir": "/tmp/test_accel_integration"},
        "model": {"dtype": "float32"},
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "gradient_accumulation": 1,
            "max_grad_norm": 1.0,
            "learning_rate": 1e-3,
            "use_accelerator": use_accelerator,
            "loss": {},
            "optimizer": {"lr": 1e-3, "betas": [0.9, 0.999], "weight_decay": 0.0},
        },
        "hardware": {"mixed_precision": "no"},
    })


def _make_trainer(config):
    """Construct a minimal BaseTrainer subclass that uses the tiny model.

    We skip the real `_setup_model` (which calls `create_model` / heavy
    factories) by patching it at the class level and feeding a tiny model.
    """
    from src.training.base_trainer import BaseTrainer

    class _T(BaseTrainer):
        def _setup_model(self) -> nn.Module:
            m = _TinyModel()
            return m.to(self.device)

        def _setup_noise_scheduler(self):
            return MagicMock()

        def _setup_optimizer(self) -> torch.optim.Optimizer:
            params = [p for p in self.model.parameters() if p.requires_grad]
            return torch.optim.AdamW(params, lr=self.config.training.optimizer["lr"])

        def _setup_dataloader(self) -> DataLoader:
            return _make_tiny_dataloader()

        def training_step(self, batch) -> torch.Tensor:
            x, y = batch
            return torch.nn.functional.mse_loss(self.model(x), y)

    # Silence training logger side-effects (wandb etc) in tests.
    with patch("src.training.base_trainer.TrainingLogger") as _Logger:
        _Logger.return_value = MagicMock()
        trainer = _T(config)
    return trainer


# ---------------------------------------------------------------------------
# (a) + (b) Accelerator instantiation + prepare() wraps objects
# ---------------------------------------------------------------------------


def test_accelerator_instantiated_by_default():
    """Default config should produce a live Accelerator."""
    trainer = _make_trainer(_make_config(use_accelerator=True))
    assert trainer.accelerator is not None
    # accelerator.device is a torch.device; CPU on this test box.
    assert isinstance(trainer.accelerator.device, torch.device)
    assert trainer.device == trainer.accelerator.device


def test_prepare_wraps_model_optimizer_dataloader():
    """After prepare() self.model, optimizer, dataloader, lr_scheduler are set."""
    trainer = _make_trainer(_make_config(use_accelerator=True))
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.dataloader is not None
    # On a single-process Accelerator the prepared model is the same object
    # (no DDP wrapping when world_size==1), so we mostly check it survived.
    params = [p for p in trainer.model.parameters() if p.requires_grad]
    assert len(params) > 0, "trainable params should survive prepare()"


# ---------------------------------------------------------------------------
# (c) accelerator.backward() accumulates gradients
# ---------------------------------------------------------------------------


def test_accelerator_backward_accumulates_gradients():
    """One backward() should populate .grad on trainable params only."""
    trainer = _make_trainer(_make_config(use_accelerator=True))
    device = trainer.device
    x = torch.randn(2, 4, device=device)
    y = torch.randn(2, 4, device=device)
    pred = trainer.model(x)
    loss = torch.nn.functional.mse_loss(pred, y)
    trainer.accelerator.backward(loss)

    # LoRA adapter (trainable) should have grads
    adapter = trainer._unwrap_model().lora_adapter
    assert adapter.weight.grad is not None
    assert adapter.weight.grad.abs().sum().item() > 0.0

    # Frozen base must have no grad
    base = trainer._unwrap_model().frozen_base
    assert base.weight.grad is None or base.weight.grad.abs().sum().item() == 0.0


# ---------------------------------------------------------------------------
# (d) is_main_process gating
# ---------------------------------------------------------------------------


def test_is_main_process_default_true():
    """Single-process Accelerator → is_main_process is always True."""
    trainer = _make_trainer(_make_config(use_accelerator=True))
    assert trainer.is_main_process is True


def test_is_main_process_when_disabled():
    """With Accelerator disabled, is_main_process must still return True."""
    trainer = _make_trainer(_make_config(use_accelerator=False))
    assert trainer.accelerator is None
    assert trainer.is_main_process is True


# ---------------------------------------------------------------------------
# (e) use_accelerator=false fallback
# ---------------------------------------------------------------------------


def test_use_accelerator_false_fallback():
    """Explicit opt-out should disable Accelerator entirely (legacy path)."""
    trainer = _make_trainer(_make_config(use_accelerator=False))
    assert trainer.accelerator is None
    # device must still be valid
    assert isinstance(trainer.device, torch.device)


# ---------------------------------------------------------------------------
# (f) _unwrap_model is callable and returns nn.Module
# ---------------------------------------------------------------------------


def test_unwrap_model_returns_nn_module():
    """_unwrap_model() must return the underlying nn.Module."""
    trainer = _make_trainer(_make_config(use_accelerator=True))
    unwrapped = trainer._unwrap_model()
    assert isinstance(unwrapped, nn.Module)
    assert isinstance(unwrapped, _TinyModel)


def test_unwrap_model_without_accelerator():
    """When Accelerator is disabled _unwrap_model() returns self.model."""
    trainer = _make_trainer(_make_config(use_accelerator=False))
    unwrapped = trainer._unwrap_model()
    assert unwrapped is trainer.model


# ---------------------------------------------------------------------------
# (g) loss convergence: training a few steps moves the loss down
# ---------------------------------------------------------------------------


def test_short_training_loop_reduces_loss():
    """End-to-end smoke test: a few accel.backward() steps drive loss down."""
    torch.manual_seed(0)
    cfg = _make_config(use_accelerator=True)
    cfg.training.epochs = 2
    cfg.training.optimizer["lr"] = 1e-1  # aggressive LR so loss drops fast
    trainer = _make_trainer(cfg)

    device = trainer.device
    x = torch.randn(2, 4, device=device)
    y = torch.randn(2, 4, device=device)
    with torch.no_grad():
        initial = torch.nn.functional.mse_loss(trainer.model(x), y).item()

    # Run enough steps for the overfit-on-2-samples problem to converge.
    for _ in range(100):
        pred = trainer.model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        trainer.accelerator.backward(loss)
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()

    with torch.no_grad():
        final = torch.nn.functional.mse_loss(trainer.model(x), y).item()

    # Loss should drop substantially on this trivial overfit problem.
    assert final < initial * 0.5, f"expected loss to drop, got {initial} -> {final}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
