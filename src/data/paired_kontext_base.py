"""Abstract base for paired (target, reference, caption) Kontext datasets.

Concrete subclasses implement two hooks:

- ``_discover_samples()`` — enumerate samples on disk and return a list whose
  element type is subclass-private (used as input to ``_load_pair``).
- ``_load_pair(idx)`` — return the three PIL objects + caption for sample
  ``idx``; the base class handles transforms + tensorisation + the standard
  ``KontextDataset``-compatible return dict.

This lets `KontextDataset`, `OrionDataset`, and any future paired dataset
(`BCIDataset`, `ACROBATDataset`, …) share the contract consumed by
``kontext_collate_fn`` and ``KontextTrainerMixin``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T  # noqa: N812
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset


class PairedKontextDataset(Dataset, ABC):
    """Abstract base for paired Kontext datasets.

    Args:
        data_path: Dataset root on disk (semantics subclass-specific).
        target_resolution: Square pixel resolution for the target image.
        config: Full training config (subclasses may read extra fields).
        reference_resolution: Square resolution for the reference image.
            Defaults to ``target_resolution``.
        cache: Optional latent cache (reserved; not yet wired).
    """

    def __init__(
        self,
        data_path: str | Path,
        target_resolution: int,
        config: DictConfig,
        reference_resolution: int | None = None,
        cache: Any | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.target_resolution = target_resolution
        self.reference_resolution = reference_resolution or target_resolution
        self.config = config
        self.cache = cache

        self.samples: list[Any] = self._discover_samples()
        if not self.samples:
            raise ValueError(
                f"{type(self).__name__} discovered zero samples at {self.data_path}"
            )

        self.target_transform = self._make_target_transform()
        self.reference_transform = self._make_reference_transform()

    # ------------------------------------------------------------------
    # Abstract hooks — subclasses implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _discover_samples(self) -> list[Any]:
        """Enumerate samples on disk.

        Returns:
            List of sample handles (dicts / paths / tuples — subclass choice).
            The same handle is passed to ``_load_pair`` via ``__getitem__``.
        """

    @abstractmethod
    def _load_pair(self, idx: int) -> tuple[Image.Image, Image.Image, str]:
        """Return ``(target_pil, reference_pil, caption)`` for sample ``idx``.

        The base ``__getitem__`` applies transforms + tensorisation + packaging
        — the subclass only deals with IO.
        """

    # ------------------------------------------------------------------
    # Transform hooks — overridable for augmentation
    # ------------------------------------------------------------------

    def _make_target_transform(self) -> T.Compose:
        """Default target transform: resize + center-crop + normalise to ``[-1, 1]``."""
        return self._default_transform(self.target_resolution)

    def _make_reference_transform(self) -> T.Compose:
        """Default reference transform: resize + center-crop + normalise to ``[-1, 1]``."""
        return self._default_transform(self.reference_resolution)

    @staticmethod
    def _default_transform(resolution: int) -> T.Compose:
        return T.Compose([
            T.Resize(resolution, interpolation=T.InterpolationMode.LANCZOS),
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

    # ------------------------------------------------------------------
    # Dataset protocol — final
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        target_pil, reference_pil, caption = self._load_pair(idx)

        target_tensor: torch.Tensor = self.target_transform(target_pil)
        reference_tensor: torch.Tensor = self.reference_transform(reference_pil)

        return {
            "target_image": target_tensor,
            "reference_image": reference_tensor,
            "caption": caption,
            "target_resolution": (target_tensor.shape[1], target_tensor.shape[2]),
            "reference_resolution": (reference_tensor.shape[1], reference_tensor.shape[2]),
        }
