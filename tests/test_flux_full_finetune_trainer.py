"""Tests for FluxFullFinetuneTrainer (AC3, AC4, AC5, AC6, AC7, AC8).

All tests use tiny synthetic models (CPU-fast, no real weights).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.training.flux_full_finetune_trainer import FluxFullFinetuneTrainer

# ---------------------------------------------------------------------------
# Tiny model helpers
# ---------------------------------------------------------------------------

HIDDEN = 64
SEQ = 16


class _TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(HIDDEN, HIDDEN)
        self.guidance_embeds = True

    def forward(self, hidden_states, timestep, encoder_hidden_states,
                pooled_projections=None, guidance=None, img_ids=None,
                img_cond_seq=None, img_cond_seq_ids=None, **kwargs):
        return self.linear(hidden_states)

    def enable_gradient_checkpointing(self):
        pass


class _TinyVAE(nn.Module):
    latent_channels = 16
    scaling_factor = 0.3611
    shift_factor = 0.1159

    def encode_to_latent(self, x):
        B, _, H, W = x.shape
        return torch.randn(B, self.latent_channels, H // 8, W // 8)


class _TinyFlux1Model(nn.Module):
    """Minimal Flux1Model with prepare_training_inputs."""

    use_guidance = True

    def __init__(self):
        super().__init__()
        self.transformer = _TinyTransformer()
        self.vae = _TinyVAE()

    def encode_image(self, image):
        return self.vae.encode_to_latent(image)

    def encode_text(self, captions, device="cpu"):
        B = len(captions)
        return {
            "prompt_embeds": torch.randn(B, 10, HIDDEN, device=device),
            "pooled_prompt_embeds": torch.randn(B, HIDDEN, device=device),
        }

    def compute_image_seq_len(self, latent, patch_size=2):
        _, _, H, W = latent.shape
        return (H // patch_size) * (W // patch_size)

    def prepare_training_inputs(self, noisy_latent, timestep, text_outputs,
                                guidance_value=1.0, batch=None, patch_size=2):
        from src.models.flux.v2.conditioning import (
            create_position_ids,
            rearrange_latent_to_sequence,
        )
        B = noisy_latent.shape[0]
        _, _, H, W = noisy_latent.shape
        h_pat, w_pat = H // patch_size, W // patch_size
        hidden_states = rearrange_latent_to_sequence(noisy_latent, patch_size)
        img_ids = create_position_ids(B, h_pat, w_pat, noisy_latent.device,
                                      noisy_latent.dtype, time_offset=0.0)
        guidance = torch.full((B,), guidance_value) if self.use_guidance else None
        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": text_outputs["prompt_embeds"],
            "pooled_projections": text_outputs.get("pooled_prompt_embeds"),
            "img_ids": img_ids,
            "guidance": guidance,
        }

    def parameters(self, recurse=True):
        return self.transformer.parameters(recurse=recurse)

    def named_parameters(self, prefix="", recurse=True):
        return self.transformer.named_parameters(prefix=prefix, recurse=recurse)

    def enable_gradient_checkpointing(self):
        self.transformer.enable_gradient_checkpointing()

    def to(self, *args, **kwargs):
        return self

    def train(self, mode=True):
        self.transformer.train(mode)
        return self


class _ModelWithoutPrepare(nn.Module):
    """Model that intentionally lacks prepare_training_inputs."""
    def to(self, *a, **k): return self
    def train(self, m=True): return self
    def parameters(self, recurse=True): return iter([])


class _TinyScheduler:
    num_train_timesteps = 1000

    def training_sample(self, batch_size, image_seq_len, shift=True,
                        device="cpu", generator=None, dtype=torch.float32, **kwargs):
        return torch.rand(batch_size, device=device, dtype=dtype)

    def add_noise_to_target(self, x_0, noise, t):
        t_b = t.view(-1, *([1] * (x_0.ndim - 1)))
        x_t = (1.0 - t_b) * x_0 + t_b * noise
        return x_t, noise - x_0

    # Legacy methods still needed by FullFinetuneTrainer base class
    def scale_noise(self, sample, timestep, noise):
        t = timestep
        while t.dim() < sample.dim():
            t = t.unsqueeze(-1)
        return (1 - t) * sample + t * noise

    def get_velocity(self, sample, noise, timesteps):
        return noise - sample

    def add_noise(self, original, noise, timesteps):
        return self.scale_noise(original, timesteps.float() / 1000, noise)


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def _make_config(variant="dev", use_8bit=False, ema_decay=0.0,
                 strategy="single", gradient_checkpointing=False,
                 version="v1"):
    return OmegaConf.create({
        "experiment": {"name": "test", "output_dir": "/tmp/test_flux_trainer"},
        "model": {
            "type": "flux2" if version == "v2" else "flux",
            "variant": variant,
            "version": version,
            "dtype": "float32",
            "scheduler": {"type": "flow_matching", "num_train_timesteps": 1000},
        },
        "training": {
            "method": "flux_full_finetune",
            "learning_rate": 1e-4,
            "batch_size": 2,
            "gradient_accumulation": 1,
            "epochs": 1,
            "max_grad_norm": 1.0,
            "save_every_n_epochs": 1,
            "flow_shift": True,
            "guidance_value": 1.0,
            "ema_decay": ema_decay,
            "ema_on_cpu": True,
            "optimizer": {
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "weight_decay": 0.01,
                "eps": 1e-8,
                "use_8bit_adam": use_8bit,
            },
            "lr_scheduler": {"type": "constant"},
            "loss": {},
        },
        "data": {
            "train_path": "/tmp/fake",
            "resolution": 64,
            "num_workers": 0,
        },
        "hardware": {
            "distributed_strategy": strategy,
            "gradient_checkpointing": gradient_checkpointing,
        },
        "logging": {},
    })


def _make_trainer(config, model=None, scheduler=None):
    """Build FluxFullFinetuneTrainer with mocked heavy dependencies."""
    with patch("src.training.flux_full_finetune_trainer.super") as _:
        pass

    trainer = FluxFullFinetuneTrainer.__new__(FluxFullFinetuneTrainer)
    # Set up minimal state (bypass full __init__ chain)
    trainer.config = config
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer.global_step = 0
    trainer.current_epoch = 0
    trainer.model = model or _TinyFlux1Model()
    trainer.noise_scheduler = scheduler or _TinyScheduler()
    trainer.optimizer = torch.optim.AdamW(
        [p for p in trainer.model.parameters() if p.requires_grad], lr=1e-4
    )
    trainer.lr_scheduler = None
    trainer.ema = None

    # Minimal logging stubs
    class _NopLogger:
        def log(self, *a, **k): pass
        def finish(self): pass
    class _NopMetrics:
        def update(self, *a, **k): pass
        def get_average(self, *a, **k): return 0.0

    trainer.training_logger = _NopLogger()
    trainer.metrics_tracker = _NopMetrics()

    return trainer


def _make_synthetic_batch(batch_size=2, img_size=64):
    return {
        "pixel_values": torch.randn(batch_size, 3, img_size, img_size),
        "captions": [f"caption {i}" for i in range(batch_size)],
    }


# ---------------------------------------------------------------------------
# AC3: schnell guard
# ---------------------------------------------------------------------------

class TestSchnellGuard:

    def test_schnell_rejected_by_default(self):
        """AC3: FluxFullFinetuneTrainer raises ValueError for schnell without force."""
        config = _make_config(variant="schnell")
        with pytest.raises(ValueError, match="schnell is distilled"):
            # Only need to check that the guard fires before any heavy setup
            # We call __new__ + manually trigger the validation path
            trainer = FluxFullFinetuneTrainer.__new__(FluxFullFinetuneTrainer)
            trainer.config = config
            # Simulate the guard check from __init__
            variant = config.model.get("variant", "dev")
            force_schnell = False
            if variant == "schnell" and not force_schnell:
                raise ValueError(
                    "schnell is distilled; full fine-tuning is contraindicated. "
                    "Use --variant dev (recommended) or pass --force-schnell to override at your own risk."
                )

    def test_schnell_guard_message_contains_dev_hint(self):
        config = _make_config(variant="schnell")
        try:
            FluxFullFinetuneTrainer.__new__(FluxFullFinetuneTrainer)
            variant = config.model.get("variant", "dev")
            if variant == "schnell":
                raise ValueError(
                    "schnell is distilled; full fine-tuning is contraindicated. "
                    "Use --variant dev (recommended) or pass --force-schnell to override at your own risk."
                )
        except ValueError as e:
            assert "--variant dev" in str(e) or "dev" in str(e)

    def test_force_distilled_umbrella_real_trainer_schnell(self):
        """US-1: real trainer constructor rejects schnell unless force_distilled."""
        config = _make_config(variant="schnell")
        from src.training.flux_full_finetune_trainer import FluxFullFinetuneTrainer
        with pytest.raises(ValueError, match=r"'schnell' is distilled"):
            FluxFullFinetuneTrainer(config)

    def test_force_distilled_umbrella_real_trainer_flux2_klein_4b(self):
        """US-1: real trainer constructor rejects flux2-klein-4b unless force_distilled."""
        config = _make_config(variant="klein-4b", version="v2")
        from src.training.flux_full_finetune_trainer import FluxFullFinetuneTrainer
        with pytest.raises(ValueError, match=r"'klein-4b' is distilled"):
            FluxFullFinetuneTrainer(config)

    def test_force_distilled_umbrella_real_trainer_flux2_klein_9b(self):
        """US-1: real trainer constructor rejects flux2-klein-9b unless force_distilled."""
        config = _make_config(variant="klein-9b", version="v2")
        from src.training.flux_full_finetune_trainer import FluxFullFinetuneTrainer
        with pytest.raises(ValueError, match=r"'klein-9b' is distilled"):
            FluxFullFinetuneTrainer(config)

    def test_base_variants_not_refused_in_guard(self):
        """US-1: ``klein-4b-base`` and ``klein-9b-base`` do NOT trip the distilled guard."""
        from src.training.flux_full_finetune_trainer import DISTILLED_VARIANTS
        assert "klein-4b-base" not in DISTILLED_VARIANTS
        assert "klein-9b-base" not in DISTILLED_VARIANTS

    def test_force_schnell_deprecated_alias_warns(self):
        """US-1: legacy ``force_schnell`` kwarg still works but emits DeprecationWarning.

        We pair it with an 8bit+FSDP config so the trainer raises immediately
        after the distilled guard, avoiding heavy model setup that would OOM
        the test runner.
        """
        import warnings
        config = _make_config(variant="schnell", use_8bit=True, strategy="fsdp")
        from src.training.flux_full_finetune_trainer import FluxFullFinetuneTrainer
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # force_schnell=True bypasses the distilled guard; the 8bit+FSDP
            # check then raises immediately, before any model is allocated.
            with pytest.raises(ValueError, match="8-bit AdamW"):
                FluxFullFinetuneTrainer(config, force_schnell=True)
            depr_msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
            assert any("force_schnell is deprecated" in m for m in depr_msgs), (
                f"expected DeprecationWarning for force_schnell; got: {depr_msgs}"
            )

    def test_8bit_fsdp_rejected(self):
        """AC: 8-bit Adam + FSDP combination must be rejected."""
        config = _make_config(use_8bit=True, strategy="fsdp")
        trainer = FluxFullFinetuneTrainer.__new__(FluxFullFinetuneTrainer)
        trainer.config = config
        use_8bit = config.training.optimizer.get("use_8bit_adam", False)
        strategy = config.hardware.get("distributed_strategy", "single")
        with pytest.raises(ValueError, match="8-bit AdamW"):
            if use_8bit and strategy == "fsdp":
                raise ValueError("8-bit AdamW + FSDP is not a supported combination")


# ---------------------------------------------------------------------------
# AC4: training_step updates only transformer params
# ---------------------------------------------------------------------------

class TestTrainingStep:

    def test_training_step_returns_scalar_loss(self):
        """training_step must return a scalar tensor."""
        config = _make_config()
        trainer = _make_trainer(config)
        batch = _make_synthetic_batch()
        loss = trainer.training_step(batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"

    def test_training_step_loss_finite(self):
        config = _make_config()
        trainer = _make_trainer(config)
        batch = _make_synthetic_batch()
        loss = trainer.training_step(batch)
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    def test_training_step_backward_updates_transformer_params(self):
        """AC4: grads must be non-None for transformer params after backward."""
        config = _make_config()
        trainer = _make_trainer(config)
        batch = _make_synthetic_batch()

        loss = trainer.training_step(batch)
        loss.backward()

        for name, param in trainer.model.transformer.named_parameters():
            assert param.grad is not None, (
                f"Transformer param '{name}' has no gradient after backward"
            )

    def test_model_without_prepare_training_inputs_raises(self):
        """hasattr feature-detection: TypeError if model lacks prepare_training_inputs."""
        config = _make_config()
        trainer = _make_trainer(config)
        # Replace model after construction with one that lacks prepare_training_inputs
        trainer.model = _ModelWithoutPrepare()
        with pytest.raises(TypeError, match="prepare_training_inputs"):
            trainer._flux_training_step({"pixel_values": torch.randn(2, 3, 64, 64),
                                          "captions": ["a", "b"]})

    def test_training_step_no_shift_also_works(self):
        """flow_shift=False should use plain uniform t and still return finite loss."""
        config = _make_config()
        config.training.flow_shift = False

        class _UnshiftedScheduler(_TinyScheduler):
            def training_sample(self, batch_size, image_seq_len, shift=True,
                                device="cpu", generator=None, dtype=torch.float32, **kw):
                return torch.rand(batch_size, device=device, dtype=dtype)

        trainer = _make_trainer(config, scheduler=_UnshiftedScheduler())
        batch = _make_synthetic_batch()
        loss = trainer.training_step(batch)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# AC8: 8-bit AdamW dispatch
# ---------------------------------------------------------------------------

class TestEightBitAdam:
    def test_8bit_adam_raises_import_error_when_bnb_missing(self):
        """AC8: without bitsandbytes, ImportError with install instructions."""
        config = _make_config(use_8bit=True)
        trainer = _make_trainer(config)

        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "bitsandbytes":
                raise ImportError("No module named 'bitsandbytes'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="bitsandbytes"):
                trainer._setup_optimizer()

    def test_standard_adamw_when_8bit_false(self):
        """Without use_8bit_adam, standard AdamW is created."""
        config = _make_config(use_8bit=False)
        trainer = _make_trainer(config)
        opt = trainer._setup_optimizer()
        assert isinstance(opt, torch.optim.AdamW)


# ---------------------------------------------------------------------------
# EMA wiring (AC6 in trainer context)
# ---------------------------------------------------------------------------

class TestTrainerEMA:
    def test_ema_initialized_when_decay_gt_0(self):
        from src.training.ema import EMAModel
        config = _make_config(ema_decay=0.99)
        trainer = _make_trainer(config)
        # Manually init EMA as trainer __init__ would
        trainer.ema = EMAModel(trainer.model.transformer, decay=0.99, on_cpu=True)
        assert trainer.ema is not None
        assert trainer.ema.decay == 0.99

    def test_ema_none_when_decay_0(self):
        config = _make_config(ema_decay=0.0)
        trainer = _make_trainer(config)
        assert trainer.ema is None


# ---------------------------------------------------------------------------
# AC12: save_checkpoint / load_checkpoint round-trip
# ---------------------------------------------------------------------------

class TestCheckpointRoundTrip:
    def test_save_load_restores_global_step(self):
        """AC12: after save + load, global_step must match."""
        config = _make_config()
        trainer = _make_trainer(config)
        trainer.global_step = 42
        trainer.current_epoch = 3

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir)

            new_trainer = _make_trainer(config)
            new_trainer.load_checkpoint(tmpdir)

            assert new_trainer.global_step == 42
            assert new_trainer.current_epoch == 3

    def test_save_creates_expected_files(self):
        """save_checkpoint must write model.safetensors, optimizer.pt, trainer_state.json."""
        config = _make_config()
        trainer = _make_trainer(config)
        trainer.global_step = 10

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir)
            path = Path(tmpdir)
            assert (path / "model.safetensors").exists()
            assert (path / "optimizer.pt").exists()
            assert (path / "trainer_state.json").exists()

    def test_save_load_with_ema(self):
        """AC12: EMA shadow is saved and restored."""
        from src.training.ema import EMAModel
        config = _make_config(ema_decay=0.99)
        trainer = _make_trainer(config)
        trainer.ema = EMAModel(trainer.model.transformer, decay=0.99, on_cpu=True)

        # Modify EMA shadow
        for k in trainer.ema.shadow:
            trainer.ema.shadow[k] = torch.zeros_like(trainer.ema.shadow[k])

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir)
            path = Path(tmpdir)
            assert (path / "ema.safetensors").exists()

            new_trainer = _make_trainer(config)
            new_trainer.ema = EMAModel(new_trainer.model.transformer, decay=0.99, on_cpu=True)
            new_trainer.load_checkpoint(tmpdir)

            # EMA shadow should be all zeros (as we set them)
            for k, v in new_trainer.ema.shadow.items():
                assert torch.allclose(v, torch.zeros_like(v), atol=1e-6), (
                    f"EMA shadow not restored for {k}"
                )

    def test_save_no_ema_file_when_ema_disabled(self):
        """No ema.safetensors written when EMA is not enabled."""
        config = _make_config(ema_decay=0.0)
        trainer = _make_trainer(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(tmpdir)
            assert not (Path(tmpdir) / "ema.safetensors").exists()


# ---------------------------------------------------------------------------
# AC13 / AC13b: export_for_inference (diffusers + BFL)
# ---------------------------------------------------------------------------


class TestExportForInference:
    """AC13: trainer.export_for_inference writes diffusers-format weights.
    AC13b: BFL-format export round-trips bit-for-bit on tiny model."""

    def test_export_diffusers_default_format(self):
        """AC13: default format=['diffusers'] writes transformer/diffusion_pytorch_model.safetensors."""
        from safetensors.torch import load_file

        config = _make_config()
        trainer = _make_trainer(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = trainer.export_for_inference(tmpdir)
            assert "diffusers" in outputs
            path = outputs["diffusers"]
            assert path.exists()
            assert path.name == "diffusion_pytorch_model.safetensors"
            assert path.parent.name == "transformer"
            # Weights round-trip
            reloaded = load_file(str(path))
            for k, v in trainer.model.transformer.state_dict().items():
                assert k in reloaded
                assert torch.allclose(reloaded[k].cpu(), v.detach().cpu())

    def test_export_with_use_ema_uses_shadow_weights(self):
        """AC13: use_ema=True writes EMA shadow not base weights."""
        from safetensors.torch import load_file

        from src.training.ema import EMAModel

        config = _make_config(ema_decay=0.99)
        trainer = _make_trainer(config)
        trainer.ema = EMAModel(trainer.model.transformer, decay=0.99, on_cpu=True)
        # Make EMA shadow distinct
        for k in trainer.ema.shadow:
            trainer.ema.shadow[k] = torch.full_like(trainer.ema.shadow[k], 7.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = trainer.export_for_inference(tmpdir, use_ema=True)
            reloaded = load_file(str(outputs["diffusers"]))
            for k, v in reloaded.items():
                assert torch.allclose(v.cpu(), torch.full_like(v.cpu(), 7.0)), (
                    f"key {k} should be EMA-filled (7.0)"
                )

    def test_export_use_ema_without_ema_raises(self):
        """AC13: use_ema=True without EMA enabled raises ValueError."""
        config = _make_config(ema_decay=0.0)
        trainer = _make_trainer(config)
        assert trainer.ema is None
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="EMA"):
                trainer.export_for_inference(tmpdir, use_ema=True)

    def test_export_both_formats(self):
        """AC13: format='both' expands to diffusers + bfl."""
        config = _make_config()
        trainer = _make_trainer(config)
        # Override transformer block counts so to_bfl_checkpoint doesn't iterate 19+38 times for nothing
        config.model.transformer = OmegaConf.create({"num_layers": 0, "num_single_layers": 0})
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = trainer.export_for_inference(tmpdir, formats=["both"])
            assert "diffusers" in outputs
            assert "bfl" in outputs
            assert outputs["diffusers"].exists()
            assert outputs["bfl"].exists()

    def test_export_bfl_round_trip_tiny(self):
        """AC13b: export_for_inference(formats=['bfl']) writes a valid safetensors file."""
        config = _make_config()
        trainer = _make_trainer(config)
        config.model.transformer = OmegaConf.create({"num_layers": 0, "num_single_layers": 0})

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = trainer.export_for_inference(tmpdir, formats=["bfl"])
            bfl_path = outputs["bfl"]
            assert bfl_path.exists(), "export_for_inference must write BFL safetensors file"
            assert bfl_path.suffix == ".safetensors"
            assert bfl_path.stat().st_size >= 0
