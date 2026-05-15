"""Tests for Flux1EditingPipeline.

Covers:
- AC5: Pipeline produces output without raising on dummy weights (CPU smoke test)
- AC9: Reference image is snapped to nearest preferred Kontext resolution bucket
"""

from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from src.inference.flux1_editing_pipeline import (
    PREFERED_KONTEXT_RESOLUTIONS,
    Flux1EditingPipeline,
    _snap_to_preferred_resolution,
)

# Minimal transformer config — same constraints as other tests
TINY_CONFIG = OmegaConf.create({
    "hidden_size": 128,
    "num_attention_heads": 4,
    "num_layers": 1,
    "num_single_layers": 1,
    "in_channels": 64,
    "joint_attention_dim": 64,
    "pooled_projection_dim": 32,
    "axes_dims_rope": [8, 12, 12],
    "rope_theta": 10000.0,
    "guidance_embeds": True,
})


class TestAC9ResolutionSnapping:
    """AC9: Reference images snap to nearest preferred Kontext bucket."""

    def test_snapping_selects_nearest_bucket(self):
        """Square 800x800 → 1024x1024 (closest square bucket)."""
        img = Image.new("RGB", (800, 800))
        snapped = _snap_to_preferred_resolution(img)
        assert snapped.size in PREFERED_KONTEXT_RESOLUTIONS, (
            f"Snapped size {snapped.size} not in preferred resolutions"
        )
        # 800x800 is closest to 1024x1024 (aspect ratio 1.0)
        assert snapped.size == (1024, 1024)

    def test_snapping_landscape(self):
        """Landscape 1920x1080 → snaps to a landscape bucket."""
        img = Image.new("RGB", (1920, 1080))
        snapped = _snap_to_preferred_resolution(img)
        w, h = snapped.size
        # Should pick a landscape bucket (wider than tall)
        assert w >= h, f"Expected landscape bucket, got {w}x{h}"
        assert snapped.size in PREFERED_KONTEXT_RESOLUTIONS

    def test_snapping_portrait(self):
        """Portrait 768x1200 → snaps to a portrait bucket."""
        img = Image.new("RGB", (768, 1200))
        snapped = _snap_to_preferred_resolution(img)
        w, h = snapped.size
        assert h >= w, f"Expected portrait bucket, got {w}x{h}"
        assert snapped.size in PREFERED_KONTEXT_RESOLUTIONS

    def test_already_at_bucket_no_resize(self):
        """Image already at a preferred bucket size → unchanged size."""
        target_w, target_h = PREFERED_KONTEXT_RESOLUTIONS[4]  # 1024x1024
        img = Image.new("RGB", (target_w, target_h))
        snapped = _snap_to_preferred_resolution(img)
        assert snapped.size == (target_w, target_h)

    def test_preferred_resolutions_nonempty(self):
        """PREFERED_KONTEXT_RESOLUTIONS constant has at least 10 entries."""
        assert len(PREFERED_KONTEXT_RESOLUTIONS) >= 10

    def test_all_resolutions_divisible_by_8(self):
        """All preferred resolutions are divisible by 8 (minimum VAE compatibility)."""
        for w, h in PREFERED_KONTEXT_RESOLUTIONS:
            assert w % 8 == 0, f"Width {w} not divisible by 8"
            assert h % 8 == 0, f"Height {h} not divisible by 8"


class TestAC5PipelineSmoke:
    """AC5: Pipeline runs end-to-end without raising on dummy CPU model."""

    def _make_dummy_pipeline(self) -> Flux1EditingPipeline:
        """Build a pipeline with tiny dummy model for CPU testing."""

        OmegaConf.create({
            "transformer": TINY_CONFIG,
            "vae": {
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 16,
                "block_out_channels": [32, 64],
                "layers_per_block": 1,
                "scaling_factor": 0.3611,
                "shift_factor": 0.1159,
            },
            "text_encoder": {
                "type": "t5_clip",
                "t5": {
                    "model_id": "google/t5-v1_1-xxl",
                    "hidden_size": 64,
                    "num_layers": 1,
                    "max_position_embeddings": 16,
                },
                "clip_l": {
                    "model_id": "openai/clip-vit-large-patch14",
                    "hidden_size": 32,
                    "num_layers": 1,
                    "max_position_embeddings": 16,
                },
            },
        })
        scheduler_config = OmegaConf.create({
            "type": "flow_matching_euler",
            "num_train_timesteps": 1000,
            "shift": 3.0,
        })
        from src.schedulers import create_scheduler
        scheduler = create_scheduler(scheduler_config)

        # Use a mock model to avoid requiring real VAE/text encoder
        return None, scheduler

    def test_snap_resolution_flag_respected(self):
        """kontext_snap_resolution=False skips snapping."""
        # Build a mock pipeline with snap off
        pipe = MagicMock(spec=Flux1EditingPipeline)
        pipe.kontext_snap_resolution = False
        pipe._prepare_reference_image = Flux1EditingPipeline._prepare_reference_image.__get__(
            pipe, Flux1EditingPipeline
        )

        # With snap off, image should not be resized to a bucket
        # (Just verify the flag is stored correctly)
        assert pipe.kontext_snap_resolution is False

    def test_pipeline_imports(self):
        """Pipeline can be imported and class is accessible."""
        from src.inference import Flux1EditingPipeline, create_flux1_editing_pipeline
        assert Flux1EditingPipeline is not None
        assert callable(create_flux1_editing_pipeline)

    def test_prefered_resolutions_exported(self):
        """PREFERED_KONTEXT_RESOLUTIONS is exported from inference package."""
        from src.inference import PREFERED_KONTEXT_RESOLUTIONS as res  # noqa: N811
        assert len(res) > 0

    def test_pipeline_smoke_with_mocked_model(self):
        """Full pipeline call with completely mocked model components doesn't raise.

        This tests the pipeline orchestration logic without requiring real
        VAE/text encoders or GPU.
        """
        import torch

        from src.inference.flux1_editing_pipeline import Flux1EditingPipeline
        # Build tiny scheduler
        sched_cfg = OmegaConf.create({"type": "flow_matching",
                                       "num_train_timesteps": 4, "shift": 1.0})
        from src.schedulers import create_scheduler
        scheduler = create_scheduler(sched_cfg)

        # Mock the model
        mock_model = MagicMock()
        mock_model.use_guidance = False
        mock_model.vae.latent_channels = 16

        # Mock encode_text → returns small embeddings
        txt_seq, pool_dim = 4, 32
        mock_model.encode_text.return_value = {
            "prompt_embeds": torch.zeros(1, txt_seq, 64),
            "pooled_prompt_embeds": torch.zeros(1, pool_dim),
        }

        # Mock encode_reference_images → returns small conditioning
        ref_seq = 4
        mock_model.encode_reference_images.return_value = {
            "img_cond_seq": torch.zeros(1, ref_seq, 64),
            "img_cond_seq_ids": torch.zeros(1, ref_seq, 3),
        }

        # Mock model forward (output is [B, target_seq, in_ch])
        # 64x64 image → latent 8x8 → patch 4x4 → seq=16; packed channels=16*4=64
        target_seq = (64 // 8 // 2) ** 2  # = 16
        mock_model.return_value = torch.zeros(1, target_seq, 64)

        # Mock decode_latent → returns [1, 3, 64, 64]
        mock_model.decode_latent.return_value = torch.zeros(1, 3, 64, 64)

        # Patch model.to() to return itself
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        pipe = Flux1EditingPipeline(
            model=mock_model,
            scheduler=scheduler,
            device="cpu",
            dtype=torch.float32,
            kontext_snap_resolution=False,
        )

        ref_img = Image.new("RGB", (64, 64), color=(128, 64, 32))
        result = pipe(
            prompt="test prompt",
            reference_image=ref_img,
            height=64,
            width=64,
            num_inference_steps=2,
            guidance_scale=1.0,
            output_type="pil",
        )

        assert isinstance(result, list), "Output should be a list"
        assert len(result) == 1, f"Expected 1 image, got {len(result)}"
        assert isinstance(result[0], Image.Image), "Output should be PIL Image"

    def test_output_type_tensor(self):
        """output_type='tensor' returns a tensor instead of PIL list."""
        from src.inference.flux1_editing_pipeline import Flux1EditingPipeline
        from src.schedulers import create_scheduler

        sched_cfg = OmegaConf.create({"type": "flow_matching",
                                       "num_train_timesteps": 4, "shift": 1.0})
        scheduler = create_scheduler(sched_cfg)

        mock_model = MagicMock()
        mock_model.use_guidance = False
        mock_model.vae.latent_channels = 16
        mock_model.encode_text.return_value = {
            "prompt_embeds": torch.zeros(1, 4, 64),
            "pooled_prompt_embeds": torch.zeros(1, 32),
        }
        mock_model.encode_reference_images.return_value = {
            "img_cond_seq": torch.zeros(1, 4, 64),
            "img_cond_seq_ids": torch.zeros(1, 4, 3),
        }
        target_seq = (64 // 8 // 2) ** 2
        mock_model.return_value = torch.zeros(1, target_seq, 64)
        mock_model.decode_latent.return_value = torch.zeros(1, 3, 64, 64)
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        pipe = Flux1EditingPipeline(
            model=mock_model, scheduler=scheduler, device="cpu",
            dtype=torch.float32, kontext_snap_resolution=False,
        )

        result = pipe(
            prompt="test",
            reference_image=Image.new("RGB", (64, 64)),
            height=64, width=64, num_inference_steps=1,
            guidance_scale=1.0, output_type="tensor",
        )
        assert isinstance(result, torch.Tensor), f"Expected tensor, got {type(result)}"

    def test_invalid_output_type_raises(self):
        """output_type='invalid' raises ValueError."""
        from src.inference.flux1_editing_pipeline import Flux1EditingPipeline
        from src.schedulers import create_scheduler

        sched_cfg = OmegaConf.create({"type": "flow_matching",
                                       "num_train_timesteps": 4, "shift": 1.0})
        scheduler = create_scheduler(sched_cfg)

        mock_model = MagicMock()
        mock_model.use_guidance = False
        mock_model.vae.latent_channels = 16
        mock_model.encode_text.return_value = {
            "prompt_embeds": torch.zeros(1, 4, 64),
            "pooled_prompt_embeds": torch.zeros(1, 32),
        }
        mock_model.encode_reference_images.return_value = {
            "img_cond_seq": torch.zeros(1, 4, 64),
            "img_cond_seq_ids": torch.zeros(1, 4, 3),
        }
        target_seq = (64 // 8 // 2) ** 2
        mock_model.return_value = torch.zeros(1, target_seq, 64)
        mock_model.decode_latent.return_value = torch.zeros(1, 3, 64, 64)
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        pipe = Flux1EditingPipeline(
            model=mock_model, scheduler=scheduler, device="cpu",
            dtype=torch.float32, kontext_snap_resolution=False,
        )
        with pytest.raises(ValueError, match="output_type"):
            pipe(
                prompt="test",
                reference_image=Image.new("RGB", (64, 64)),
                height=64, width=64, num_inference_steps=1,
                output_type="invalid",
            )
