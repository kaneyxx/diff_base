"""
CRC01 Biomarker Dataset for FLUX.2-dev LoRA Training.

This dataset loads H&E histopathology images as reference (conditioning)
and biomarker RGB images as targets for image editing training.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple, Union
import logging

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

logger = logging.getLogger(__name__)

# All available biomarkers in CRC01 dataset
BIOMARKERS = [
    "CD45", "CD31", "CD68", "CD4", "FOXP3", "CD8a",
    "CD45RO", "CD20", "PD-L1", "CD3e", "CD163", "PD-1", "Ki67"
]


class CRC01BiomarkerDataset(Dataset):
    """
    Dataset for CRC01 multiplex immunofluorescence biomarker data.

    Loads paired H&E (reference) and biomarker RGB (target) images
    for Kontext-style image editing training.

    The dataset expects:
    - biomarkers.h5: HDF5 file containing biomarker data (R channels, labels)
      and shared channels (Pan-CK as Green, Hoechst as Blue)
    - HE/: Directory containing H&E images as PNG files
      Named as: CRC01_HE_{x}_{y}.png where x_y is the coordinate

    If H&E images are not available, set he_dir=None to use biomarker
    images as both reference and target (for testing/debugging only).

    Args:
        h5_path: Path to biomarkers.h5 file
        he_dir: Directory containing H&E images (or None for debug mode)
        biomarker: Target biomarker name (e.g., "CD45")
        transform: Optional transform to apply to images
        resolution: Target resolution (default: 1024)
        filter_positive: If True, only include positive samples
        filter_negative: If True, only include negative samples
        sampled_indices_path: Path to JSON file with pre-sampled indices
        cache_images: If True, cache images in memory
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        he_dir: Optional[Union[str, Path]],
        biomarker: str,
        transform: Optional[Callable] = None,
        resolution: int = 1024,
        filter_positive: bool = False,
        filter_negative: bool = False,
        sampled_indices_path: Optional[Union[str, Path]] = None,
        cache_images: bool = False,
    ):
        self.h5_path = Path(h5_path)
        self.he_dir = Path(he_dir) if he_dir is not None else None
        self.biomarker = biomarker
        self.transform = transform
        self.resolution = resolution
        self.cache_images = cache_images
        self.debug_mode = he_dir is None

        if self.debug_mode:
            logger.warning(
                "Running in debug mode without H&E images. "
                "Using biomarker images as reference (for testing only)."
            )

        if biomarker not in BIOMARKERS:
            raise ValueError(
                f"Unknown biomarker: {biomarker}. "
                f"Available biomarkers: {BIOMARKERS}"
            )

        # Load coordinates and labels from H5 file
        self._load_metadata()

        # Apply label filtering
        if filter_positive and filter_negative:
            raise ValueError("Cannot filter for both positive and negative")

        if filter_positive:
            self._filter_by_label(1)
        elif filter_negative:
            self._filter_by_label(0)

        # Apply sampled indices if provided
        if sampled_indices_path is not None:
            self._apply_sampled_indices(sampled_indices_path)

        # Cache for images
        self._cache: Dict[int, Dict] = {}

        logger.info(
            f"CRC01BiomarkerDataset initialized: "
            f"biomarker={biomarker}, samples={len(self)}"
        )

    def _load_metadata(self) -> None:
        """Load coordinates and labels from H5 file."""
        with h5py.File(self.h5_path, 'r') as f:
            # Load coordinates
            coords_raw = f['coordinates'][:]
            self.coordinates = [
                c.decode('utf-8') if isinstance(c, bytes) else str(c)
                for c in coords_raw
            ]

            # Load labels for this biomarker
            label_key = f'{self.biomarker}_label'
            if label_key in f:
                self.labels = f[label_key][:]
            else:
                logger.warning(f"No labels found for {self.biomarker}, using all 1s")
                self.labels = np.ones(len(self.coordinates), dtype=np.int8)

        # Create index mapping
        self.indices = list(range(len(self.coordinates)))

    def _filter_by_label(self, label: int) -> None:
        """Filter samples by label value."""
        mask = self.labels == label
        self.indices = [i for i, m in enumerate(mask) if m]
        logger.info(
            f"Filtered to {len(self.indices)} samples with label={label}"
        )

    def _apply_sampled_indices(self, sampled_indices_path: Union[str, Path]) -> None:
        """Apply pre-sampled indices from JSON file.

        The sampled indices file should contain:
        {
            "seed": 42,
            "num_samples": 100,
            "indices": [1234, 5678, ...]
        }

        The indices in the JSON are the raw dataset indices (before any filtering).
        After applying sampled indices, self.indices will contain only the
        intersection of the current indices and the sampled indices.

        Args:
            sampled_indices_path: Path to JSON file with sampled indices
        """
        sampled_path = Path(sampled_indices_path)
        if not sampled_path.exists():
            raise FileNotFoundError(
                f"Sampled indices file not found: {sampled_path}\n"
                f"Run 'python scripts/sample_dataset.py' to create it."
            )

        with open(sampled_path, 'r') as f:
            sampled_data = json.load(f)

        sampled_set = set(sampled_data['indices'])
        original_count = len(self.indices)

        # Keep only indices that are in the sampled set
        self.indices = [i for i in self.indices if i in sampled_set]

        logger.info(
            f"Applied sampled indices from {sampled_path.name}: "
            f"{original_count} -> {len(self.indices)} samples "
            f"(sampled={sampled_data['num_samples']}, seed={sampled_data['seed']})"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                - reference_image: H&E image tensor [3, H, W]
                - target_image: Biomarker RGB tensor [3, H, W]
                - caption: Text prompt for generation
                - label: Binary label (0=negative, 1=positive)
                - coordinate: Patch coordinate string
        """
        if self.cache_images and idx in self._cache:
            return self._cache[idx]

        # Get actual index from filtered indices
        real_idx = self.indices[idx]
        coord = self.coordinates[real_idx]

        # Load biomarker RGB from H5 (target)
        biomarker_rgb = self._load_biomarker_rgb(real_idx)
        target_image = Image.fromarray(biomarker_rgb)

        # Load H&E reference image
        if self.debug_mode:
            # In debug mode, use biomarker image as reference too (for testing)
            he_image = target_image.copy()
        else:
            he_path = self.he_dir / f"CRC01_HE_{coord}.png"
            if not he_path.exists():
                raise FileNotFoundError(
                    f"H&E image not found: {he_path}\n"
                    f"Please ensure H&E images are in {self.he_dir}/\n"
                    f"Expected format: CRC01_HE_{{x}}_{{y}}.png"
                )
            he_image = Image.open(he_path).convert('RGB')

        # Resize if needed
        if he_image.size != (self.resolution, self.resolution):
            he_image = he_image.resize(
                (self.resolution, self.resolution),
                Image.Resampling.BILINEAR
            )

        if target_image.size != (self.resolution, self.resolution):
            target_image = target_image.resize(
                (self.resolution, self.resolution),
                Image.Resampling.BILINEAR
            )

        # Convert to tensors
        he_tensor = self._to_tensor(he_image)
        target_tensor = self._to_tensor(target_image)

        # Apply transforms if provided
        if self.transform is not None:
            he_tensor = self.transform(he_tensor)
            target_tensor = self.transform(target_tensor)

        # Create caption
        caption = f"Generate {self.biomarker} biomarker image"

        sample = {
            "reference_image": he_tensor,
            "target_image": target_tensor,
            "caption": caption,
            "label": int(self.labels[real_idx]),
            "coordinate": coord,
            "index": real_idx,
        }

        if self.cache_images:
            self._cache[idx] = sample

        return sample

    def _load_biomarker_rgb(self, index: int) -> np.ndarray:
        """Load biomarker RGB image from H5 file."""
        with h5py.File(self.h5_path, 'r') as f:
            r = f[f'{self.biomarker}_R'][index]
            g = f['shared_G'][index]
            b = f['shared_B'][index]
        return np.stack([r, g, b], axis=-1).astype(np.uint8)

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to normalized tensor [-1, 1]."""
        # Convert to numpy array
        arr = np.array(image).astype(np.float32) / 255.0

        # Normalize to [-1, 1]
        arr = arr * 2.0 - 1.0

        # Convert to tensor [C, H, W]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor

    def get_coordinate(self, idx: int) -> str:
        """Get coordinate string for a sample."""
        real_idx = self.indices[idx]
        return self.coordinates[real_idx]

    def get_label(self, idx: int) -> int:
        """Get label for a sample."""
        real_idx = self.indices[idx]
        return int(self.labels[real_idx])

    @staticmethod
    def available_biomarkers() -> List[str]:
        """Return list of available biomarkers."""
        return BIOMARKERS.copy()


def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Collate function for CRC01 dataset.

    Stacks tensors and collects other items into lists.
    """
    reference_images = torch.stack([item["reference_image"] for item in batch])
    target_images = torch.stack([item["target_image"] for item in batch])
    captions = [item["caption"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    coordinates = [item["coordinate"] for item in batch]
    indices = [item["index"] for item in batch]

    return {
        "reference_images": reference_images,
        "target_images": target_images,
        "captions": captions,
        "labels": labels,
        "coordinates": coordinates,
        "indices": indices,
    }


def create_crc01_dataloader(
    h5_path: Union[str, Path],
    he_dir: Union[str, Path],
    biomarker: str,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    resolution: int = 1024,
    filter_positive: bool = False,
    filter_negative: bool = False,
    sampled_indices_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for CRC01 biomarker dataset.

    Args:
        h5_path: Path to biomarkers.h5 file
        he_dir: Directory containing H&E images
        biomarker: Target biomarker name
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        resolution: Target image resolution
        filter_positive: Only include positive samples
        filter_negative: Only include negative samples
        sampled_indices_path: Path to JSON file with pre-sampled indices
        **kwargs: Additional arguments passed to Dataset

    Returns:
        DataLoader instance
    """
    dataset = CRC01BiomarkerDataset(
        h5_path=h5_path,
        he_dir=he_dir,
        biomarker=biomarker,
        resolution=resolution,
        filter_positive=filter_positive,
        filter_negative=filter_negative,
        sampled_indices_path=sampled_indices_path,
        **kwargs,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )

    return dataloader


# For convenience, allow direct import of dataloader creation
def from_config(config) -> DataLoader:
    """
    Create dataloader from OmegaConf config.

    Expected config structure:
        data:
            h5_path: str
            he_dir: str
            biomarker: str
            resolution: int
            filter_positive: bool
            filter_negative: bool
            sampled_indices_path: str (optional)
        training:
            batch_size: int
        hardware:
            num_workers: int
    """
    return create_crc01_dataloader(
        h5_path=config.data.h5_path,
        he_dir=config.data.he_dir,
        biomarker=config.data.biomarker,
        batch_size=config.training.get("batch_size", 1),
        shuffle=True,
        num_workers=config.hardware.get("num_workers", 4),
        resolution=config.data.get("resolution", 1024),
        filter_positive=config.data.get("filter_positive", False),
        filter_negative=config.data.get("filter_negative", False),
        sampled_indices_path=config.data.get("sampled_indices_path", None),
    )
