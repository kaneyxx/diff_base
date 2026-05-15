"""Tests for KontextDataset (AC6, AC9).

All tests use synthetic in-memory data — no real checkpoints required.
"""

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from src.data.kontext_collate import kontext_collate_fn
from src.data.kontext_dataset import KontextDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(target_res: int = 64, reference_res: int = 64) -> OmegaConf:
    return OmegaConf.create({
        "data": {
            "flip_augment": False,
            "color_jitter": False,
        }
    })


def _save_rgb_png(path: Path, width: int = 128, height: int = 128) -> None:
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    img.save(path)


# ---------------------------------------------------------------------------
# AC6: KontextDataset loads (target, reference, caption) triplets
# ---------------------------------------------------------------------------


class TestKontextDatasetMetadata:
    """Load from metadata.json layout."""

    def test_loads_samples_from_metadata(self, tmp_path: Path) -> None:
        _save_rgb_png(tmp_path / "img_a_target.png")
        _save_rgb_png(tmp_path / "img_a_ref.png")
        metadata = [
            {
                "target": "img_a_target.png",
                "reference": "img_a_ref.png",
                "caption": "a sunny day",
            }
        ]
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))

        ds = KontextDataset(tmp_path, target_resolution=64, config=_make_config())
        assert len(ds) == 1

    def test_getitem_returns_correct_keys(self, tmp_path: Path) -> None:
        _save_rgb_png(tmp_path / "t.png")
        _save_rgb_png(tmp_path / "r.png")
        (tmp_path / "metadata.json").write_text(
            json.dumps([{"target": "t.png", "reference": "r.png", "caption": "test"}])
        )

        ds = KontextDataset(tmp_path, target_resolution=64, config=_make_config())
        item = ds[0]

        assert "target_image" in item
        assert "reference_image" in item
        assert "caption" in item
        assert "target_resolution" in item
        assert "reference_resolution" in item

    def test_getitem_tensor_shapes(self, tmp_path: Path) -> None:
        _save_rgb_png(tmp_path / "t.png")
        _save_rgb_png(tmp_path / "r.png")
        (tmp_path / "metadata.json").write_text(
            json.dumps([{"target": "t.png", "reference": "r.png", "caption": "hi"}])
        )

        ds = KontextDataset(tmp_path, target_resolution=64, config=_make_config())
        item = ds[0]

        assert item["target_image"].shape == (3, 64, 64)
        assert item["reference_image"].shape == (3, 64, 64)

    def test_getitem_tensor_dtype_is_float(self, tmp_path: Path) -> None:
        _save_rgb_png(tmp_path / "t.png")
        _save_rgb_png(tmp_path / "r.png")
        (tmp_path / "metadata.json").write_text(
            json.dumps([{"target": "t.png", "reference": "r.png", "caption": ""}])
        )

        ds = KontextDataset(tmp_path, target_resolution=64, config=_make_config())
        item = ds[0]

        assert item["target_image"].dtype == torch.float32
        assert item["reference_image"].dtype == torch.float32

    def test_getitem_normalized_to_minus1_plus1(self, tmp_path: Path) -> None:
        _save_rgb_png(tmp_path / "t.png")
        _save_rgb_png(tmp_path / "r.png")
        (tmp_path / "metadata.json").write_text(
            json.dumps([{"target": "t.png", "reference": "r.png", "caption": ""}])
        )

        ds = KontextDataset(tmp_path, target_resolution=64, config=_make_config())
        item = ds[0]

        assert item["target_image"].min() >= -1.0 - 1e-5
        assert item["target_image"].max() <= 1.0 + 1e-5
        assert item["reference_image"].min() >= -1.0 - 1e-5
        assert item["reference_image"].max() <= 1.0 + 1e-5

    def test_caption_is_string(self, tmp_path: Path) -> None:
        _save_rgb_png(tmp_path / "t.png")
        _save_rgb_png(tmp_path / "r.png")
        (tmp_path / "metadata.json").write_text(
            json.dumps([{"target": "t.png", "reference": "r.png", "caption": "hello world"}])
        )

        ds = KontextDataset(tmp_path, target_resolution=64, config=_make_config())
        assert ds[0]["caption"] == "hello world"

    def test_missing_reference_is_skipped(self, tmp_path: Path) -> None:
        _save_rgb_png(tmp_path / "t.png")
        # reference deliberately not created
        metadata = [{"target": "t.png", "reference": "missing_ref.png", "caption": "x"}]
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))

        with pytest.raises(ValueError):
            KontextDataset(tmp_path, target_resolution=64, config=_make_config())

    def test_multiple_samples(self, tmp_path: Path) -> None:
        for i in range(3):
            _save_rgb_png(tmp_path / f"t{i}.png")
            _save_rgb_png(tmp_path / f"r{i}.png")
        metadata = [
            {"target": f"t{i}.png", "reference": f"r{i}.png", "caption": f"cap{i}"}
            for i in range(3)
        ]
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))

        ds = KontextDataset(tmp_path, target_resolution=64, config=_make_config())
        assert len(ds) == 3


class TestKontextDatasetDirectory:
    """Load from *_target / *_ref directory convention."""

    def test_loads_from_directory_naming(self, tmp_path: Path) -> None:
        _save_rgb_png(tmp_path / "scene_target.png")
        _save_rgb_png(tmp_path / "scene_ref.png")
        (tmp_path / "scene_caption.txt").write_text("a forest at dusk")

        ds = KontextDataset(tmp_path, target_resolution=64, config=_make_config())
        assert len(ds) == 1
        assert ds[0]["caption"] == "a forest at dusk"

    def test_missing_ref_skips_target(self, tmp_path: Path) -> None:
        _save_rgb_png(tmp_path / "lonely_target.png")
        # no *_ref.png → should find 0 valid pairs

        with pytest.raises(ValueError):
            KontextDataset(tmp_path, target_resolution=64, config=_make_config())

    def test_empty_caption_when_no_txt(self, tmp_path: Path) -> None:
        _save_rgb_png(tmp_path / "pair_target.png")
        _save_rgb_png(tmp_path / "pair_ref.png")
        # no caption file

        ds = KontextDataset(tmp_path, target_resolution=64, config=_make_config())
        assert ds[0]["caption"] == ""


class TestKontextDatasetResolution:
    """Target and reference can have different resolutions."""

    def test_different_target_and_reference_resolutions(self, tmp_path: Path) -> None:
        _save_rgb_png(tmp_path / "t.png", width=256, height=256)
        _save_rgb_png(tmp_path / "r.png", width=256, height=256)
        (tmp_path / "metadata.json").write_text(
            json.dumps([{"target": "t.png", "reference": "r.png", "caption": ""}])
        )

        ds = KontextDataset(
            tmp_path,
            target_resolution=64,
            config=_make_config(),
            reference_resolution=32,
        )
        item = ds[0]
        assert item["target_image"].shape == (3, 64, 64)
        assert item["reference_image"].shape == (3, 32, 32)
        assert item["target_resolution"] == (64, 64)
        assert item["reference_resolution"] == (32, 32)


# ---------------------------------------------------------------------------
# Collate function tests
# ---------------------------------------------------------------------------


class TestKontextCollate:
    """Tests for kontext_collate_fn."""

    def _make_sample(self, caption: str = "cap") -> dict:
        return {
            "target_image": torch.randn(3, 64, 64),
            "reference_image": torch.randn(3, 64, 64),
            "caption": caption,
            "target_resolution": (64, 64),
            "reference_resolution": (64, 64),
        }

    def test_output_keys(self) -> None:
        batch = [self._make_sample(), self._make_sample()]
        out = kontext_collate_fn(batch)
        assert set(out.keys()) >= {
            "target_pixel", "reference_pixel", "captions",
            "target_resolution", "reference_resolution",
        }

    def test_target_pixel_shape(self) -> None:
        batch = [self._make_sample() for _ in range(4)]
        out = kontext_collate_fn(batch)
        assert out["target_pixel"].shape == (4, 3, 64, 64)

    def test_reference_pixel_shape(self) -> None:
        batch = [self._make_sample() for _ in range(4)]
        out = kontext_collate_fn(batch)
        assert out["reference_pixel"].shape == (4, 3, 64, 64)

    def test_captions_list_length(self) -> None:
        captions = ["a", "b", "c"]
        batch = [self._make_sample(caption=c) for c in captions]
        out = kontext_collate_fn(batch)
        assert out["captions"] == captions

    def test_resolutions_from_first_sample(self) -> None:
        batch = [self._make_sample(), self._make_sample()]
        out = kontext_collate_fn(batch)
        assert out["target_resolution"] == (64, 64)
        assert out["reference_resolution"] == (64, 64)
