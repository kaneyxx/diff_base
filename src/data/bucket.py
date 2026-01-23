"""Aspect ratio bucketing for multi-resolution training."""

import math
import random
from typing import Iterator

from PIL import Image


def compute_bucket_sizes(
    base_resolution: int,
    step: int,
    min_dim: int = 512,
    max_dim: int = 2048,
) -> list[tuple[int, int]]:
    """Compute valid bucket sizes maintaining approximately same pixel count.

    Args:
        base_resolution: Base resolution (e.g., 1024).
        step: Step size for dimensions.
        min_dim: Minimum dimension.
        max_dim: Maximum dimension.

    Returns:
        List of (width, height) tuples.
    """
    target_pixels = base_resolution * base_resolution
    buckets = set()

    for width in range(min_dim, max_dim + 1, step):
        # Calculate height to maintain pixel count
        height = int(target_pixels / width)
        height = (height // step) * step  # Round to nearest step

        if min_dim <= height <= max_dim:
            buckets.add((width, height))

    return sorted(list(buckets))


def assign_to_buckets(
    samples: list[dict],
    bucket_sizes: list[tuple[int, int]],
) -> dict[tuple[int, int], list[int]]:
    """Assign samples to nearest bucket by aspect ratio.

    Args:
        samples: List of sample dicts with width/height keys.
        bucket_sizes: Available bucket sizes.

    Returns:
        Dictionary mapping bucket to list of sample indices.
    """
    # Pre-compute bucket aspect ratios
    bucket_ratios = {
        size: size[0] / size[1]
        for size in bucket_sizes
    }

    bucket_indices: dict[tuple[int, int], list[int]] = {
        size: [] for size in bucket_sizes
    }

    for idx, sample in enumerate(samples):
        # Get image dimensions
        if "width" in sample and "height" in sample:
            w, h = sample["width"], sample["height"]
        else:
            # Load image to get dimensions
            with Image.open(sample["image_path"]) as img:
                w, h = img.size

        ratio = w / h

        # Find closest bucket
        best_bucket = min(
            bucket_sizes,
            key=lambda s: abs(bucket_ratios[s] - ratio)
        )

        bucket_indices[best_bucket].append(idx)

    # Remove empty buckets
    bucket_indices = {k: v for k, v in bucket_indices.items() if v}

    return bucket_indices


class BucketSampler:
    """Batch sampler that yields batches from the same bucket.

    Ensures all images in a batch have the same dimensions.
    """

    def __init__(
        self,
        bucket_indices: dict[tuple[int, int], list[int]],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        """Initialize bucket sampler.

        Args:
            bucket_indices: Mapping of bucket to sample indices.
            batch_size: Batch size.
            shuffle: Whether to shuffle samples and batch order.
            drop_last: Whether to drop incomplete batches.
        """
        self.bucket_indices = bucket_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Pre-compute batches
        self._batches = self._create_batches()

    def _create_batches(self) -> list[tuple[tuple[int, int], list[int]]]:
        """Create all batches."""
        batches = []

        for bucket, indices in self.bucket_indices.items():
            indices = indices.copy()

            if self.shuffle:
                random.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]

                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append((bucket, batch))

        return batches

    def __iter__(self) -> Iterator[tuple[tuple[int, int], list[int]]]:
        """Iterate over batches."""
        batches = self._batches.copy()

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return len(self._batches)


class AspectRatioBucketManager:
    """Manages aspect ratio buckets for training."""

    def __init__(
        self,
        base_resolution: int = 1024,
        step: int = 64,
        min_dim: int = 512,
        max_dim: int = 2048,
    ):
        """Initialize bucket manager.

        Args:
            base_resolution: Base resolution.
            step: Dimension step size.
            min_dim: Minimum dimension.
            max_dim: Maximum dimension.
        """
        self.base_resolution = base_resolution
        self.step = step
        self.min_dim = min_dim
        self.max_dim = max_dim

        self.bucket_sizes = compute_bucket_sizes(
            base_resolution, step, min_dim, max_dim
        )
        self.bucket_ratios = {
            size: size[0] / size[1]
            for size in self.bucket_sizes
        }

    def get_bucket(self, width: int, height: int) -> tuple[int, int]:
        """Get the best bucket for given dimensions.

        Args:
            width: Image width.
            height: Image height.

        Returns:
            Best matching bucket size.
        """
        ratio = width / height
        return min(
            self.bucket_sizes,
            key=lambda s: abs(self.bucket_ratios[s] - ratio)
        )

    def get_bucket_stats(
        self,
        bucket_indices: dict[tuple[int, int], list[int]]
    ) -> dict:
        """Get statistics about bucket distribution.

        Args:
            bucket_indices: Bucket to indices mapping.

        Returns:
            Dictionary with statistics.
        """
        total = sum(len(v) for v in bucket_indices.values())

        stats = {
            "total_samples": total,
            "num_buckets": len(bucket_indices),
            "bucket_counts": {
                f"{w}x{h}": len(indices)
                for (w, h), indices in bucket_indices.items()
            },
            "bucket_percentages": {
                f"{w}x{h}": len(indices) / total * 100
                for (w, h), indices in bucket_indices.items()
            },
        }

        return stats


def get_crop_coordinates(
    original_width: int,
    original_height: int,
    target_width: int,
    target_height: int,
    center_crop: bool = True,
) -> tuple[int, int, int, int]:
    """Calculate crop coordinates to fit target size.

    Args:
        original_width: Original image width.
        original_height: Original image height.
        target_width: Target width.
        target_height: Target height.
        center_crop: Whether to center the crop.

    Returns:
        Tuple of (left, top, right, bottom).
    """
    # Scale to fit target
    scale = max(target_width / original_width, target_height / original_height)
    scaled_width = int(original_width * scale)
    scaled_height = int(original_height * scale)

    # Calculate crop area
    if center_crop:
        left = (scaled_width - target_width) // 2
        top = (scaled_height - target_height) // 2
    else:
        left = random.randint(0, scaled_width - target_width)
        top = random.randint(0, scaled_height - target_height)

    return left, top, left + target_width, top + target_height
