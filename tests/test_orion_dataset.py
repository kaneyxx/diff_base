"""Tests for ``src.data.orion_dataset.OrionDataset``.

Builds a tiny synthetic ORION layout on disk so the test runs without any
external data dependency, then exercises:

- Sample discovery from a BAO-style split JSON
- Biomarker inference from filename
- H5 + PNG resolution with the BAO ``dataset/`` prefix convention
- Output shape, dtype, and ``[-1, 1]`` normalisation
- Registry-based factory dispatch via ``create_kontext_dataloader``
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from src.data import (
    KONTEXT_DATASET_REGISTRY,
    PairedKontextDataset,
    create_kontext_dataloader,
    register_kontext_dataset,
)
from src.data.orion_dataset import OrionDataset, _infer_biomarker_from_split


# ---------------------------------------------------------------------------
# Synthetic ORION fixture — minimal viable BAO layout
# ---------------------------------------------------------------------------

@pytest.fixture
def orion_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Create a tiny CRC01-only ORION-style dataset on disk.

    Returns:
        ``(base_dir, split_path)`` where ``base_dir`` is the BAO repo root
        and ``split_path`` is the CD45 training-split JSON.
    """
    crc_root = tmp_path / "dataset" / "CRC01"
    he_dir = crc_root / "HE"
    he_dir.mkdir(parents=True)

    # Two synthetic samples: one positive, one negative
    h5_path = crc_root / "tiles.h5"
    with h5py.File(h5_path, "w") as f:
        # 2 patches of 32x32 — biomarker + shared G/B channels
        f.create_dataset(
            "CD45_R",
            data=np.random.randint(0, 255, (2, 32, 32), dtype=np.uint8),
        )
        f.create_dataset(
            "shared_G",
            data=np.random.randint(0, 255, (2, 32, 32), dtype=np.uint8),
        )
        f.create_dataset(
            "shared_B",
            data=np.random.randint(0, 255, (2, 32, 32), dtype=np.uint8),
        )

    # Matching H&E PNGs
    for coord in ("0_0", "32_32"):
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(arr).save(he_dir / f"CRC01_HE_{coord}.png")

    # Split JSON with BAO's dataset/-prefixed paths
    split_dir = tmp_path / "dataset" / "training_splits"
    split_dir.mkdir(parents=True)
    split_path = split_dir / "CD45_train.json"
    split_data = {
        "biomarker": "CD45",
        "total_samples": 2,
        "samples": [
            {
                "crc_id": "CRC01",
                "h5_path": "dataset/CRC01/tiles.h5",
                "he_dir": "dataset/CRC01/HE",
                "index": 0,
                "coordinate": "0_0",
                "sample_type": "positive",
            },
            {
                "crc_id": "CRC01",
                "h5_path": "dataset/CRC01/tiles.h5",
                "he_dir": "dataset/CRC01/HE",
                "index": 1,
                "coordinate": "32_32",
                "sample_type": "negative",
            },
        ],
    }
    split_path.write_text(json.dumps(split_data))

    # base_dir = tmp_path/dataset; data_path passed to OrionDataset = tmp_path/dataset
    # so the relative paths inside the JSON resolve via the one-level-up fallback.
    return tmp_path / "dataset", split_path


# ---------------------------------------------------------------------------
# Inheritance + registry sanity
# ---------------------------------------------------------------------------

def test_orion_subclasses_paired_base():
    assert issubclass(OrionDataset, PairedKontextDataset)


def test_orion_is_registered_by_default():
    assert "orion" in KONTEXT_DATASET_REGISTRY
    assert KONTEXT_DATASET_REGISTRY["orion"] is OrionDataset


def test_register_kontext_dataset_rejects_non_subclass():
    class NotADataset:
        pass

    with pytest.raises(TypeError):
        register_kontext_dataset("bogus", NotADataset)  # type: ignore[arg-type]


def test_infer_biomarker_from_split_filename(tmp_path: Path):
    p = tmp_path / "FOXP3_train.json"
    p.write_text(json.dumps({"biomarker": "FOXP3", "samples": []}))
    assert _infer_biomarker_from_split(p) == "FOXP3"


def test_infer_biomarker_falls_back_to_json_body(tmp_path: Path):
    p = tmp_path / "unknown_split.json"
    p.write_text(json.dumps({"biomarker": "CD8a", "samples": []}))
    assert _infer_biomarker_from_split(p) == "CD8a"


def test_infer_biomarker_raises_on_unknown(tmp_path: Path):
    p = tmp_path / "nonsense.json"
    p.write_text(json.dumps({"biomarker": "NOT_A_REAL_MARKER", "samples": []}))
    with pytest.raises(ValueError, match="Could not infer ORION biomarker"):
        _infer_biomarker_from_split(p)


# ---------------------------------------------------------------------------
# Loading contract on a synthetic dataset
# ---------------------------------------------------------------------------

def test_orion_loads_pair_shape_and_range(orion_fixture):
    base_dir, split_path = orion_fixture
    cfg = OmegaConf.create({
        "data": {
            "train_path": str(base_dir),
            "resolution": 32,
            "dataset_type": "orion",
            "train_split_path": str(split_path),
            "biomarker": None,  # infer from filename
        },
        "training": {"batch_size": 1},
    })
    ds = OrionDataset(
        data_path=cfg.data.train_path,
        target_resolution=cfg.data.resolution,
        config=cfg,
    )
    assert ds.biomarker == "CD45"
    assert len(ds) == 2

    item = ds[0]
    assert set(item.keys()) >= {
        "target_image", "reference_image", "caption",
        "target_resolution", "reference_resolution",
    }
    assert item["target_image"].shape == (3, 32, 32)
    assert item["reference_image"].shape == (3, 32, 32)
    assert item["target_image"].dtype == torch.float32
    assert item["target_image"].min() >= -1.0 - 1e-5
    assert item["target_image"].max() <= 1.0 + 1e-5
    assert "CD45" in item["caption"]


def test_orion_factory_dispatches_via_create_kontext_dataloader(orion_fixture):
    base_dir, split_path = orion_fixture
    cfg = OmegaConf.create({
        "data": {
            "train_path": str(base_dir),
            "resolution": 32,
            "dataset_type": "orion",
            "train_split_path": str(split_path),
            "biomarker": "CD45",
            "num_workers": 0,
        },
        "training": {"batch_size": 1},
    })
    dl = create_kontext_dataloader(cfg)
    assert isinstance(dl.dataset, OrionDataset)
    batch = next(iter(dl))
    assert batch["target_pixel"].shape == (1, 3, 32, 32)
    assert batch["reference_pixel"].shape == (1, 3, 32, 32)
    assert batch["captions"] == ["Generate CD45 biomarker image from H&E"]


def test_orion_max_samples_cap(orion_fixture):
    base_dir, split_path = orion_fixture
    cfg = OmegaConf.create({
        "data": {
            "train_path": str(base_dir),
            "resolution": 32,
            "dataset_type": "orion",
            "train_split_path": str(split_path),
            "biomarker": "CD45",
            "max_samples": 1,
        },
        "training": {"batch_size": 1},
    })
    ds = OrionDataset(
        data_path=cfg.data.train_path,
        target_resolution=cfg.data.resolution,
        config=cfg,
    )
    assert len(ds) == 1


def test_orion_missing_split_raises(tmp_path: Path):
    cfg = OmegaConf.create({
        "data": {
            "train_path": str(tmp_path),
            "resolution": 32,
            "train_split_path": str(tmp_path / "does_not_exist.json"),
        },
        "training": {"batch_size": 1},
    })
    with pytest.raises(FileNotFoundError):
        OrionDataset(
            data_path=cfg.data.train_path,
            target_resolution=cfg.data.resolution,
            config=cfg,
        )


def test_orion_unknown_dataset_type_in_factory(orion_fixture):
    base_dir, split_path = orion_fixture
    cfg = OmegaConf.create({
        "data": {
            "train_path": str(base_dir),
            "resolution": 32,
            "dataset_type": "definitely_not_a_real_type",
            "train_split_path": str(split_path),
        },
        "training": {"batch_size": 1},
    })
    with pytest.raises(ValueError, match="Unknown Kontext dataset_type"):
        create_kontext_dataloader(cfg)
