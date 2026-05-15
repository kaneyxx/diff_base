"""Tests for model.prepare_training_inputs() polymorphism (Phase D / Task #2).

Verifies that Flux1Model and Flux2Model implement the feature-detection adapter
used by FluxFullFinetuneTrainer, without modifying BaseDiffusionModel.
"""

import torch
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Tiny model helpers (CPU-fast, no weight loading)
# ---------------------------------------------------------------------------

def _make_flux1_model(guidance: bool = True):
    """Build a tiny Flux1Model with 1 joint + 1 single block."""
    import torch.nn as nn

    from src.models.flux.v1.model import Flux1Model
    from src.models.flux.v1.transformer import Flux1Transformer

    cfg = OmegaConf.create({
        "transformer": {
            "hidden_size": 64,
            "num_attention_heads": 2,
            "num_layers": 1,
            "num_single_layers": 1,
            "in_channels": 64,   # 16 latent * 2*2 patch
            "guidance_embeds": guidance,
        },
        "vae": {},
        "text_encoder": {},
    })
    # Use __new__ then call nn.Module.__init__ before assigning submodules
    model = Flux1Model.__new__(Flux1Model)
    nn.Module.__init__(model)
    model.variant = "dev" if guidance else "schnell"
    model.config = cfg
    model.transformer = Flux1Transformer(cfg.transformer, variant=model.variant)
    model.use_guidance = guidance
    model.vae = None  # not needed for text2img prepare_training_inputs test
    return model


def _make_flux2_model():
    """Build a tiny Flux2Model with 1 joint + 1 single block."""
    import torch.nn as nn

    from src.models.flux.v2.model import Flux2Model
    from src.models.flux.v2.transformer import Flux2Transformer

    cfg = OmegaConf.create({
        "transformer": {
            "hidden_size": 64,
            "num_attention_heads": 2,
            "num_layers": 1,
            "num_single_layers": 1,
            "in_channels": 128,   # 32 latent * 2*2 patch
            "guidance_embeds": True,
        },
        "vae": {},
        "text_encoder": {},
    })
    model = Flux2Model.__new__(Flux2Model)
    nn.Module.__init__(model)
    model.variant = "dev"
    model.config = cfg
    model.transformer = Flux2Transformer(cfg.transformer, variant="dev")
    model.use_guidance = True
    model.vae = None
    return model


def _fake_text_outputs(B: int, seq_len: int = 10, t5_dim: int = 4096,
                       clip_dim: int = 768) -> dict:
    return {
        "prompt_embeds": torch.randn(B, seq_len, t5_dim),
        "pooled_prompt_embeds": torch.randn(B, clip_dim),
    }


# ---------------------------------------------------------------------------
# Feature-detection contract
# ---------------------------------------------------------------------------

class TestFeatureDetection:
    def test_flux1_has_prepare_training_inputs(self):
        model = _make_flux1_model()
        assert hasattr(model, "prepare_training_inputs"), (
            "Flux1Model must expose prepare_training_inputs()"
        )

    def test_flux2_has_prepare_training_inputs(self):
        model = _make_flux2_model()
        assert hasattr(model, "prepare_training_inputs"), (
            "Flux2Model must expose prepare_training_inputs()"
        )

    def test_flux1_has_compute_image_seq_len(self):
        model = _make_flux1_model()
        assert hasattr(model, "compute_image_seq_len")

    def test_flux2_has_compute_image_seq_len(self):
        model = _make_flux2_model()
        assert hasattr(model, "compute_image_seq_len")

    def test_base_model_does_not_have_prepare_training_inputs(self):
        """BaseDiffusionModel must NOT declare prepare_training_inputs (§D.1)."""
        from src.models.base import BaseDiffusionModel
        assert not hasattr(BaseDiffusionModel, "prepare_training_inputs"), (
            "BaseDiffusionModel must not declare prepare_training_inputs; "
            "trainer uses hasattr feature-detection"
        )


# ---------------------------------------------------------------------------
# compute_image_seq_len
# ---------------------------------------------------------------------------

class TestComputeImageSeqLen:
    def test_flux1_seq_len_correct(self):
        model = _make_flux1_model()
        latent = torch.zeros(2, 16, 8, 8)  # H=8, W=8, patch=2 → 16 tokens
        assert model.compute_image_seq_len(latent) == 16

    def test_flux2_seq_len_correct(self):
        model = _make_flux2_model()
        latent = torch.zeros(2, 32, 16, 16)  # H=16, W=16, patch=2 → 64 tokens
        assert model.compute_image_seq_len(latent) == 64

    def test_seq_len_custom_patch_size(self):
        model = _make_flux1_model()
        latent = torch.zeros(1, 16, 16, 16)
        assert model.compute_image_seq_len(latent, patch_size=4) == 16


# ---------------------------------------------------------------------------
# prepare_training_inputs — Flux1Model text2img path
# ---------------------------------------------------------------------------

class TestFlux1PrepareTrainingInputs:
    def test_output_keys_text2img(self):
        model = _make_flux1_model(guidance=True)
        B, C, H, W = 2, 16, 8, 8
        noisy = torch.randn(B, C, H, W)
        t = torch.rand(B)
        text = _fake_text_outputs(B)

        inputs = model.prepare_training_inputs(noisy, t, text, guidance_value=3.5)

        assert "hidden_states" in inputs
        assert "timestep" in inputs
        assert "encoder_hidden_states" in inputs
        assert "pooled_projections" in inputs
        assert "img_ids" in inputs
        assert "guidance" in inputs
        # No Kontext keys without reference
        assert "img_cond_seq" not in inputs
        assert "img_cond_seq_ids" not in inputs

    def test_hidden_states_shape(self):
        model = _make_flux1_model()
        B, C, H, W = 2, 16, 8, 8
        noisy = torch.randn(B, C, H, W)
        t = torch.rand(B)
        text = _fake_text_outputs(B)
        inputs = model.prepare_training_inputs(noisy, t, text)
        # patch_size=2 → seq = (8//2)*(8//2) = 16, patch_dim = 16*4 = 64
        assert inputs["hidden_states"].shape == (B, 16, 64)

    def test_guidance_populated_for_dev(self):
        model = _make_flux1_model(guidance=True)
        noisy = torch.randn(2, 16, 8, 8)
        t = torch.rand(2)
        text = _fake_text_outputs(2)
        inputs = model.prepare_training_inputs(noisy, t, text, guidance_value=3.5)
        assert inputs["guidance"] is not None
        assert inputs["guidance"].shape == (2,)
        assert (inputs["guidance"] == 3.5).all()

    def test_no_guidance_for_schnell(self):
        model = _make_flux1_model(guidance=False)
        noisy = torch.randn(2, 16, 8, 8)
        t = torch.rand(2)
        text = _fake_text_outputs(2)
        inputs = model.prepare_training_inputs(noisy, t, text)
        assert inputs["guidance"] is None

    def test_img_ids_shape(self):
        model = _make_flux1_model()
        B, C, H, W = 2, 16, 8, 8
        noisy = torch.randn(B, C, H, W)
        t = torch.rand(B)
        text = _fake_text_outputs(B)
        inputs = model.prepare_training_inputs(noisy, t, text)
        # img_ids: [B, seq, 3]
        assert inputs["img_ids"].shape == (B, 16, 3)

    def test_timestep_passed_through(self):
        model = _make_flux1_model()
        noisy = torch.randn(2, 16, 8, 8)
        t = torch.tensor([0.3, 0.7])
        text = _fake_text_outputs(2)
        inputs = model.prepare_training_inputs(noisy, t, text)
        assert torch.allclose(inputs["timestep"], t)


# ---------------------------------------------------------------------------
# prepare_training_inputs — Flux2Model text2img path
# ---------------------------------------------------------------------------

class TestFlux2PrepareTrainingInputs:
    def test_output_keys_text2img(self):
        model = _make_flux2_model()
        B, C, H, W = 2, 32, 8, 8
        noisy = torch.randn(B, C, H, W)
        t = torch.rand(B)
        text = _fake_text_outputs(B)
        inputs = model.prepare_training_inputs(noisy, t, text)

        assert "hidden_states" in inputs
        assert "timestep" in inputs
        assert "encoder_hidden_states" in inputs
        assert "img_ids" in inputs
        assert "guidance" in inputs
        assert "img_cond_seq" not in inputs
        assert "img_cond" not in inputs

    def test_hidden_states_shape_flux2(self):
        model = _make_flux2_model()
        B, C, H, W = 2, 32, 8, 8
        noisy = torch.randn(B, C, H, W)
        t = torch.rand(B)
        text = _fake_text_outputs(B)
        inputs = model.prepare_training_inputs(noisy, t, text)
        # patch_size=2 → seq = 16, patch_dim = 32*4 = 128
        assert inputs["hidden_states"].shape == (B, 16, 128)

    def test_fill_cond_passed_through(self):
        """If batch contains 'img_cond', it should be passed into inputs."""
        model = _make_flux2_model()
        B, C, H, W = 2, 32, 8, 8
        noisy = torch.randn(B, C, H, W)
        t = torch.rand(B)
        text = _fake_text_outputs(B)
        img_cond = torch.randn(B, 16, 33)  # seq=16, fill channels
        batch = {"img_cond": img_cond}
        inputs = model.prepare_training_inputs(noisy, t, text, batch=batch)
        assert "img_cond" in inputs
        assert inputs["img_cond"].shape == img_cond.shape
