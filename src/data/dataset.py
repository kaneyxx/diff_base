"""Dataset classes for diffusion model training."""

import json
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from omegaconf import DictConfig

from .transforms import create_transforms, create_bucket_transforms
from .cache import LatentCache


class DiffusionDataset(Dataset):
    """Basic dataset for diffusion model training.

    Loads images and captions from a directory structure.
    Supports metadata.json or image-caption pairs.
    """

    def __init__(
        self,
        data_path: str | Path,
        resolution: int,
        config: DictConfig,
        cache: Optional[LatentCache] = None,
    ):
        """Initialize dataset.

        Args:
            data_path: Path to dataset directory.
            resolution: Target image resolution.
            config: Full training configuration.
            cache: Optional latent cache.
        """
        self.data_path = Path(data_path)
        self.resolution = resolution
        self.config = config
        self.cache = cache

        # Load samples
        self.samples = self._load_samples()

        # Setup transforms
        self.transform = create_transforms(resolution, config)

    def _load_samples(self) -> list[dict[str, Any]]:
        """Load dataset samples from metadata or directory.

        Returns:
            List of sample dictionaries with image_path and caption.
        """
        samples = []

        # Check for metadata.json
        metadata_path = self.data_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

            for item in metadata:
                image_path = self.data_path / item["image"]
                if image_path.exists():
                    samples.append({
                        "image_path": image_path,
                        "caption": item.get("caption", ""),
                    })
        else:
            # Fall back to directory structure with txt files
            image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

            for img_path in sorted(self.data_path.rglob("*")):
                if img_path.suffix.lower() in image_extensions:
                    # Look for corresponding caption file
                    txt_path = img_path.with_suffix(".txt")
                    caption = ""
                    if txt_path.exists():
                        caption = txt_path.read_text().strip()

                    samples.append({
                        "image_path": img_path,
                        "caption": caption,
                    })

        if not samples:
            raise ValueError(f"No valid samples found in {self.data_path}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with image tensor and caption.
        """
        # Check cache first
        if self.cache is not None:
            cached = self.cache.get(idx)
            if cached is not None:
                return cached

        sample = self.samples[idx]

        # Load and transform image
        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "caption": sample["caption"],
            "idx": idx,
        }


class BucketDataset(Dataset):
    """Dataset with aspect ratio bucketing for multi-resolution training.

    Groups images by aspect ratio to minimize cropping/padding.
    """

    def __init__(
        self,
        data_path: str | Path,
        base_resolution: int,
        bucket_step: int,
        config: DictConfig,
        min_dim: int = 512,
        max_dim: int = 2048,
    ):
        """Initialize bucketed dataset.

        Args:
            data_path: Path to dataset directory.
            base_resolution: Base resolution for bucket calculation.
            bucket_step: Resolution step size for buckets.
            config: Full training configuration.
            min_dim: Minimum dimension for buckets.
            max_dim: Maximum dimension for buckets.
        """
        self.data_path = Path(data_path)
        self.base_resolution = base_resolution
        self.bucket_step = bucket_step
        self.config = config

        # Load all samples
        self.samples = self._load_samples()

        # Build buckets
        from .bucket import compute_bucket_sizes, assign_to_buckets

        self.bucket_sizes = compute_bucket_sizes(
            base_resolution,
            bucket_step,
            min_dim=min_dim,
            max_dim=max_dim,
        )

        self.bucket_indices = assign_to_buckets(self.samples, self.bucket_sizes)

        # Pre-create transforms for each bucket size
        self.bucket_transforms = {
            size: create_bucket_transforms(size, config)
            for size in self.bucket_sizes
        }

        # Create flat index mapping for __getitem__
        self._flat_indices = []
        for bucket, indices in self.bucket_indices.items():
            for idx in indices:
                self._flat_indices.append((bucket, idx))

    def _load_samples(self) -> list[dict[str, Any]]:
        """Load all samples with their dimensions."""
        samples = []
        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

        # Check for metadata.json
        metadata_path = self.data_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

            for item in metadata:
                image_path = self.data_path / item["image"]
                if image_path.exists():
                    with Image.open(image_path) as img:
                        w, h = img.size
                    samples.append({
                        "image_path": image_path,
                        "caption": item.get("caption", ""),
                        "width": w,
                        "height": h,
                    })
        else:
            for img_path in sorted(self.data_path.rglob("*")):
                if img_path.suffix.lower() in image_extensions:
                    txt_path = img_path.with_suffix(".txt")
                    caption = ""
                    if txt_path.exists():
                        caption = txt_path.read_text().strip()

                    with Image.open(img_path) as img:
                        w, h = img.size

                    samples.append({
                        "image_path": img_path,
                        "caption": caption,
                        "width": w,
                        "height": h,
                    })

        return samples

    def __len__(self) -> int:
        return len(self._flat_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample with appropriate transform."""
        bucket, sample_idx = self._flat_indices[idx]
        sample = self.samples[sample_idx]

        # Load and transform with bucket-specific transform
        image = Image.open(sample["image_path"]).convert("RGB")
        transform = self.bucket_transforms[bucket]
        image = transform(image)

        return {
            "image": image,
            "caption": sample["caption"],
            "idx": sample_idx,
            "bucket": bucket,
        }

    def get_bucket_batch(
        self,
        bucket: tuple[int, int],
        indices: list[int],
    ) -> list[dict[str, Any]]:
        """Get a batch of samples from a specific bucket.

        Args:
            bucket: Target bucket size (width, height).
            indices: Sample indices within the bucket.

        Returns:
            List of sample dictionaries.
        """
        transform = self.bucket_transforms[bucket]
        batch = []

        for idx in indices:
            sample = self.samples[idx]
            image = Image.open(sample["image_path"]).convert("RGB")
            image = transform(image)

            batch.append({
                "image": image,
                "caption": sample["caption"],
                "idx": idx,
                "bucket": bucket,
            })

        return batch


class ControlNetDataset(Dataset):
    """Dataset for ControlNet training with conditioning images."""

    def __init__(
        self,
        data_path: str | Path,
        resolution: int,
        config: DictConfig,
        conditioning_type: str = "canny",
    ):
        """Initialize ControlNet dataset.

        Args:
            data_path: Path to dataset directory.
            resolution: Target resolution.
            config: Training configuration.
            conditioning_type: Type of conditioning (canny, depth, pose, etc.).
        """
        self.data_path = Path(data_path)
        self.resolution = resolution
        self.config = config
        self.conditioning_type = conditioning_type

        self.samples = self._load_samples()
        self.transform = create_transforms(resolution, config)
        self.condition_transform = self._create_condition_transform()

    def _load_samples(self) -> list[dict[str, Any]]:
        """Load samples with conditioning images."""
        samples = []

        metadata_path = self.data_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

            for item in metadata:
                image_path = self.data_path / item["image"]
                condition_path = self.data_path / item.get(
                    "conditioning", item["image"].replace("images", "conditions")
                )

                if image_path.exists() and condition_path.exists():
                    samples.append({
                        "image_path": image_path,
                        "condition_path": condition_path,
                        "caption": item.get("caption", ""),
                    })

        return samples

    def _create_condition_transform(self):
        """Create transform for conditioning images."""
        import torchvision.transforms as T

        return T.Compose([
            T.Resize(self.resolution, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(self.resolution),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        # Load target image
        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.transform(image)

        # Load conditioning image
        condition = Image.open(sample["condition_path"])
        if self.conditioning_type in ["canny", "depth"]:
            condition = condition.convert("L")
        else:
            condition = condition.convert("RGB")
        condition = self.condition_transform(condition)

        return {
            "image": image,
            "conditioning_image": condition,
            "caption": sample["caption"],
            "idx": idx,
        }
