"""Kontext paired dataset for FLUX.1 image-editing training.

Loads ``(target, reference, caption)`` triplets for Kontext mode training.
Inherits the standard contract from :class:`PairedKontextDataset` so the
``kontext_collate_fn`` consumes both this and other paired datasets
(e.g. :class:`OrionDataset`) uniformly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torchvision.transforms as T  # noqa: N812
from PIL import Image

from .paired_kontext_base import PairedKontextDataset
from .transforms import create_transforms


class KontextDataset(PairedKontextDataset):
    """Paired-image dataset using BFL Kontext naming conventions.

    Supports two layouts:

    1. ``metadata.json`` at the dataset root with
       ``[{"target": "...", "reference": "...", "caption": "..."}]`` entries.
    2. Directory pairs with ``*_target.<ext>`` / ``*_ref.<ext>`` /
       optional ``*_caption.txt`` siblings.

    The target image transform uses :func:`create_transforms` so training
    augmentation flags from ``config`` are honoured; the reference image
    uses the base-class default (resize + center-crop + normalize).
    """

    def _discover_samples(self) -> list[dict[str, Any]]:
        metadata_path = self.data_path / "metadata.json"
        if metadata_path.exists():
            return self._discover_from_metadata(metadata_path)
        return self._discover_from_directory()

    def _discover_from_metadata(self, metadata_path: Path) -> list[dict[str, Any]]:
        with open(metadata_path) as f:
            metadata = json.load(f)

        samples: list[dict[str, Any]] = []
        for item in metadata:
            target_path = self.data_path / item["target"]
            reference_path = self.data_path / item["reference"]
            if not target_path.exists() or not reference_path.exists():
                continue
            samples.append({
                "target_path": target_path,
                "reference_path": reference_path,
                "caption": item.get("caption", ""),
            })

        if not samples:
            raise ValueError(
                f"No valid (target, reference) pairs found via {metadata_path}"
            )
        return samples

    def _discover_from_directory(self) -> list[dict[str, Any]]:
        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        samples: list[dict[str, Any]] = []

        for target_path in sorted(self.data_path.rglob("*")):
            if target_path.suffix.lower() not in image_extensions:
                continue
            if not target_path.stem.endswith("_target"):
                continue

            stem_base = target_path.stem[: -len("_target")]
            reference_path: Path | None = None
            for ext in image_extensions:
                candidate = target_path.parent / f"{stem_base}_ref{ext}"
                if candidate.exists():
                    reference_path = candidate
                    break
            if reference_path is None:
                continue

            caption_path = target_path.parent / f"{stem_base}_caption.txt"
            caption = caption_path.read_text().strip() if caption_path.exists() else ""

            samples.append({
                "target_path": target_path,
                "reference_path": reference_path,
                "caption": caption,
            })

        if not samples:
            raise ValueError(
                f"No paired *_target / *_ref images found in {self.data_path}"
            )
        return samples

    # ------------------------------------------------------------------
    # PairedKontextDataset hooks
    # ------------------------------------------------------------------

    def _load_pair(self, idx: int) -> tuple[Image.Image, Image.Image, str]:
        sample = self.samples[idx]
        target_image = Image.open(sample["target_path"]).convert("RGB")
        reference_image = Image.open(sample["reference_path"]).convert("RGB")
        return target_image, reference_image, sample["caption"]

    def _make_target_transform(self) -> T.Compose:
        """Override to honour training-config augmentation flags."""
        return create_transforms(self.target_resolution, self.config)
