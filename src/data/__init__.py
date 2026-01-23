"""Data pipeline for diffusion model training."""

from typing import TYPE_CHECKING

from torch.utils.data import DataLoader
from omegaconf import DictConfig

from .dataset import DiffusionDataset, BucketDataset
from .bucket import BucketSampler, compute_bucket_sizes
from .cache import LatentCache, precompute_latents
from .transforms import create_transforms, create_bucket_transforms
from .collate import create_collate_fn

if TYPE_CHECKING:
    from ..models import BaseDiffusionModel


def create_dataloader(config: DictConfig) -> DataLoader:
    """Create dataloader from configuration.

    Args:
        config: Full training configuration.

    Returns:
        Configured DataLoader.
    """
    data_config = config.data

    # Setup cache if enabled
    cache = None
    if data_config.get("cache_latents", False):
        cache = LatentCache(
            cache_dir=data_config.get("cache_dir", "./cache"),
            model_hash=str(config.model.get("pretrained_path", "unknown")),
        )

    # Create dataset based on bucketing config
    if data_config.get("bucket_resolution_steps"):
        dataset = BucketDataset(
            data_path=data_config.train_path,
            base_resolution=data_config.resolution,
            bucket_step=data_config.bucket_resolution_steps,
            config=config,
        )
        sampler = BucketSampler(
            bucket_indices=dataset.bucket_indices,
            batch_size=config.training.batch_size,
            shuffle=True,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=create_collate_fn(config),
            num_workers=data_config.get("num_workers", 4),
            pin_memory=True,
        )
    else:
        dataset = DiffusionDataset(
            data_path=data_config.train_path,
            resolution=data_config.resolution,
            config=config,
            cache=cache,
        )
        return DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            collate_fn=create_collate_fn(config),
            num_workers=data_config.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )


def create_dreambooth_dataloader(config: DictConfig) -> DataLoader:
    """Create dataloader for DreamBooth training.

    Args:
        config: Full training configuration.

    Returns:
        DreamBooth DataLoader.
    """
    from .dreambooth_dataset import DreamBoothDataset

    dataset = DreamBoothDataset(
        instance_data_dir=config.data.instance_data_dir,
        instance_prompt=config.training.dreambooth.instance_prompt,
        class_data_dir=config.data.get("class_data_dir"),
        class_prompt=config.training.dreambooth.get("class_prompt"),
        resolution=config.data.resolution,
        config=config,
    )

    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=create_collate_fn(config, dreambooth=True),
        num_workers=config.data.get("num_workers", 4),
        pin_memory=True,
    )


__all__ = [
    "DiffusionDataset",
    "BucketDataset",
    "BucketSampler",
    "LatentCache",
    "create_dataloader",
    "create_dreambooth_dataloader",
    "create_transforms",
    "create_bucket_transforms",
    "compute_bucket_sizes",
    "precompute_latents",
]
