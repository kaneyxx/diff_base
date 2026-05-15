"""Data pipeline for diffusion model training."""

from typing import TYPE_CHECKING

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .bucket import BucketSampler, compute_bucket_sizes
from .cache import LatentCache, precompute_latents
from .collate import create_collate_fn
from .dataset import BucketDataset, DiffusionDataset
from .kontext_collate import kontext_collate_fn
from .kontext_dataset import KontextDataset
from .orion_dataset import OrionDataset
from .paired_kontext_base import PairedKontextDataset
from .transforms import create_bucket_transforms, create_transforms

if TYPE_CHECKING:
    from ..models import BaseDiffusionModel

# Registry of paired-Kontext dataset classes. Keep keys lowercase.
KONTEXT_DATASET_REGISTRY: dict[str, type[PairedKontextDataset]] = {
    "kontext": KontextDataset,
    "orion": OrionDataset,
}


def register_kontext_dataset(name: str, cls: type[PairedKontextDataset]) -> None:
    """Register a new paired-Kontext dataset class under ``name``.

    Lets downstream users plug in BCI / ACROBAT / etc. without editing this
    module.
    """
    if not issubclass(cls, PairedKontextDataset):
        raise TypeError(
            f"Cannot register {cls.__name__}: must subclass PairedKontextDataset"
        )
    KONTEXT_DATASET_REGISTRY[name.lower()] = cls


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


def create_kontext_dataloader(config: DictConfig) -> DataLoader:
    """Build a paired-image dataloader for FLUX.1 Kontext-style training.

    Dispatches on ``config.data.dataset_type`` (default ``"kontext"``). The
    selected dataset class must subclass :class:`PairedKontextDataset` so
    that its samples are compatible with ``kontext_collate_fn``.

    Args:
        config: Full training configuration. Reads:
            - ``data.dataset_type`` — registry key ("kontext", "orion", ...).
            - ``data.train_path`` — dataset root passed to the dataset's
              ``data_path`` argument.
            - ``data.resolution`` — target resolution.
            - ``data.reference_resolution`` — optional reference resolution.
            - ``training.batch_size``, ``data.num_workers`` — DataLoader knobs.

    Returns:
        Configured DataLoader using ``kontext_collate_fn``.
    """
    data_config = config.data
    dataset_type = (
        data_config.get("dataset_type", "kontext") if hasattr(data_config, "get") else "kontext"
    ).lower()

    if dataset_type not in KONTEXT_DATASET_REGISTRY:
        raise ValueError(
            f"Unknown Kontext dataset_type '{dataset_type}'. "
            f"Registered: {sorted(KONTEXT_DATASET_REGISTRY)}"
        )

    dataset_cls = KONTEXT_DATASET_REGISTRY[dataset_type]
    dataset = dataset_cls(
        data_path=data_config.train_path,
        target_resolution=data_config.resolution,
        config=config,
        reference_resolution=data_config.get("reference_resolution", None),
        cache=None,
    )

    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=kontext_collate_fn,
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
    "PairedKontextDataset",
    "KontextDataset",
    "OrionDataset",
    "KONTEXT_DATASET_REGISTRY",
    "register_kontext_dataset",
    "kontext_collate_fn",
    "create_dataloader",
    "create_kontext_dataloader",
    "create_dreambooth_dataloader",
    "create_transforms",
    "create_bucket_transforms",
    "compute_bucket_sizes",
    "precompute_latents",
]
