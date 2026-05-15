"""ORION (Lin et al., *Cell* 2023) paired H&E ↔ multiplex-IF dataset adapter.

Wraps the BAO multi-patient training-split layout into the
``PairedKontextDataset`` contract so it drops in via ``kontext_collate_fn``.

Training-split JSON (BAO convention)::

    {
      "biomarker": "CD45",
      "samples": [
        {"crc_id": "CRC01",
         "h5_path": "CRC01/tiles.h5",
         "he_dir":  "CRC01/HE",
         "index":   2374,
         "coordinate": "61440_44032",
         "sample_type": "positive"},
        ...
      ]
    }

H5 layout per patient: ``{biomarker}_R`` + ``shared_G`` (Pan-CK) + ``shared_B``
(Hoechst), all indexable by sample index. H&E tiles sit alongside as
``{crc_id}_HE_{coordinate}.png``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from omegaconf import DictConfig
from PIL import Image

from .paired_kontext_base import PairedKontextDataset

logger = logging.getLogger(__name__)

ORION_BIOMARKERS = {
    "CD45", "CD31", "CD68", "CD4", "FOXP3", "CD8a",
    "CD45RO", "CD20", "PD-L1", "CD3e", "CD163", "PD-1", "Ki67",
}


def _infer_biomarker_from_split(split_path: Path) -> str:
    """Recover the biomarker name from a BAO training-split filename or JSON body.

    BAO's naming convention is ``{BIOMARKER}_train.json`` (e.g. ``CD45_train.json``).
    Falls back to reading the JSON's top-level ``biomarker`` field if the
    filename doesn't match.
    """
    stem = split_path.stem
    if stem.endswith("_train"):
        stem = stem[: -len("_train")]
    if stem in ORION_BIOMARKERS:
        return stem
    try:
        with open(split_path) as f:
            meta = json.load(f)
        bio = meta.get("biomarker")
        if bio in ORION_BIOMARKERS:
            return bio
    except (OSError, json.JSONDecodeError):
        pass
    raise ValueError(
        f"Could not infer ORION biomarker from {split_path}; "
        f"expected filename like ``CD45_train.json`` or a ``biomarker`` field."
    )


class OrionDataset(PairedKontextDataset):
    """ORION paired (H&E reference, biomarker target, caption) dataset.

    Reads `data.train_split_path` and `data.biomarker` from the training
    config (or `ORION_TRAIN_SPLIT` / `ORION_BIOMARKER` env vars as fallback).
    `data_path` is the directory containing the per-patient `CRC{NN}/`
    folders referenced by the split JSON.
    """

    def __init__(
        self,
        data_path: str | Path,
        target_resolution: int,
        config: DictConfig,
        reference_resolution: int | None = None,
        cache: Any | None = None,
    ) -> None:
        # Resolve config-driven knobs BEFORE calling super (super calls _discover_samples)
        data_cfg = config.data if hasattr(config, "data") else config
        get = data_cfg.get if hasattr(data_cfg, "get") else lambda *a, **k: None

        split_path_raw = get("train_split_path", None) or os.environ.get("ORION_TRAIN_SPLIT")
        if not split_path_raw:
            raise ValueError(
                "OrionDataset requires `data.train_split_path` in the config "
                "or the ORION_TRAIN_SPLIT environment variable to be set."
            )
        self.split_path = Path(split_path_raw)
        if not self.split_path.is_file():
            raise FileNotFoundError(f"ORION training split not found: {self.split_path}")

        self.biomarker: str = (
            get("biomarker", None)
            or os.environ.get("ORION_BIOMARKER")
            or _infer_biomarker_from_split(self.split_path)
        )
        if self.biomarker not in ORION_BIOMARKERS:
            raise ValueError(
                f"Unknown ORION biomarker '{self.biomarker}'. "
                f"Expected one of {sorted(ORION_BIOMARKERS)}."
            )

        self._max_samples = get("max_samples", None)
        # Lazy H5 cache keyed by absolute path (one handle per patient file).
        self._h5_cache: dict[str, h5py.File] = {}

        super().__init__(
            data_path=data_path,
            target_resolution=target_resolution,
            config=config,
            reference_resolution=reference_resolution,
            cache=cache,
        )

        logger.info(
            "OrionDataset initialised: biomarker=%s split=%s samples=%d "
            "target_res=%d ref_res=%d",
            self.biomarker,
            self.split_path.name,
            len(self.samples),
            self.target_resolution,
            self.reference_resolution,
        )

    # ------------------------------------------------------------------
    # PairedKontextDataset hooks
    # ------------------------------------------------------------------

    def _discover_samples(self) -> list[dict[str, Any]]:
        with open(self.split_path) as f:
            split = json.load(f)
        samples: list[dict[str, Any]] = list(split["samples"])
        if self._max_samples is not None and 0 < int(self._max_samples) < len(samples):
            samples = samples[: int(self._max_samples)]
        return samples

    def _load_pair(self, idx: int) -> tuple[Image.Image, Image.Image, str]:
        sample = self.samples[idx]
        h5_path = self._resolve_relative(sample["h5_path"])
        target_pil = self._load_biomarker_rgb(h5_path, sample["index"])
        reference_pil = Image.open(self._resolve_he_path(sample)).convert("RGB")
        caption = f"Generate {self.biomarker} biomarker image from H&E"
        return target_pil, reference_pil, caption

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_relative(self, relative_path: str) -> Path:
        """Resolve a split-JSON path against ``data_path`` with one-level fallback.

        BAO split files encode paths like ``dataset/CRC01/tiles.h5`` which are
        relative to the BAO repo root, *not* the data directory. We accept
        both layouts: try ``data_path / p`` first, then ``data_path.parent / p``.
        """
        rel = Path(relative_path)
        primary = self.data_path / rel
        if primary.exists():
            return primary
        # Strip a leading "dataset/" segment if present and try again under data_path.
        if rel.parts and rel.parts[0] in {"dataset", "data"}:
            stripped = self.data_path / Path(*rel.parts[1:])
            if stripped.exists():
                return stripped
        # Fallback: resolve against the parent of data_path (BAO repo root).
        fallback = self.data_path.parent / rel
        if fallback.exists():
            return fallback
        raise FileNotFoundError(
            f"Could not resolve ORION path '{relative_path}' under "
            f"{self.data_path} (or its parent / stripped variants)"
        )

    def _get_h5(self, h5_path: Path) -> h5py.File:
        key = str(h5_path)
        if key not in self._h5_cache:
            self._h5_cache[key] = h5py.File(h5_path, "r")
        return self._h5_cache[key]

    def _load_biomarker_rgb(self, h5_path: Path, index: int) -> Image.Image:
        f = self._get_h5(h5_path)
        r = f[f"{self.biomarker}_R"][index]
        g = f["shared_G"][index]
        b = f["shared_B"][index]
        arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
        return Image.fromarray(arr)

    def _resolve_he_path(self, sample: dict[str, Any]) -> Path:
        he_dir_rel = sample["he_dir"]
        coordinate = sample["coordinate"]
        crc_id = sample["crc_id"]
        folder = sample.get("folder", crc_id)
        # Resolve the HE directory with the same one-level fallback as h5.
        he_dir_path = Path(he_dir_rel)
        candidates_dir = [self.data_path / he_dir_path]
        if he_dir_path.parts and he_dir_path.parts[0] in {"dataset", "data"}:
            candidates_dir.append(self.data_path / Path(*he_dir_path.parts[1:]))
        candidates_dir.append(self.data_path.parent / he_dir_path)
        he_dir = next((d for d in candidates_dir if d.is_dir()), candidates_dir[0])

        # Try BAO's three filename conventions in order.
        for name in (
            f"{folder}_HE_{coordinate}.png",
            f"{crc_id}_HE_{coordinate}.png",
            f"CRC01_HE_{coordinate}.png",
        ):
            candidate = he_dir / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"H&E image not found for crc_id={crc_id} coord={coordinate} in {he_dir}"
        )

    def __del__(self) -> None:
        # Defensive: __del__ can fire on a partially-constructed instance if
        # __init__ raised before `_h5_cache` was assigned.
        cache = getattr(self, "_h5_cache", None)
        if not cache:
            return
        for f in cache.values():
            try:
                f.close()
            except Exception:
                pass
