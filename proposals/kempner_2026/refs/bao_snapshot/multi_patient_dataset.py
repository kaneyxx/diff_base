"""
Multi-Patient CRC Biomarker Dataset for FLUX LoRA Training.

This dataset loads from training_splits JSON files which reference
multiple patient H5 files (CRC01, CRC02, ..., CRC40).

Each training split contains pre-selected balanced samples with:
- Positive samples (high biomarker signal)
- Negative samples (low biomarker signal)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Callable, Union
import logging

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

logger = logging.getLogger(__name__)

BIOMARKERS = [
    "CD45", "CD31", "CD68", "CD4", "FOXP3", "CD8a",
    "CD45RO", "CD20", "PD-L1", "CD3e", "CD163", "PD-1", "Ki67"
]


class MultiPatientBiomarkerDataset(Dataset):
    """
    Dataset for multi-patient CRC biomarker data.

    Loads from training_splits JSON files which contain references to
    multiple patient H5 files.

    Training split format:
    {
        "biomarker": "CD45",
        "total_samples": 2100,
        "samples": [
            {
                "crc_id": "CRC01",
                "h5_path": "dataset/CRC01/tiles.h5",
                "he_dir": "dataset/CRC01/HE",
                "index": 2374,
                "coordinate": "61440_44032",
                "sample_type": "positive"
            },
            ...
        ]
    }

    Args:
        training_split_path: Path to training split JSON file
        biomarker: Target biomarker name (e.g., "CD45")
        base_dir: Base directory for resolving relative paths
        transform: Optional transform to apply to images
        resolution: Target resolution (default: 1024)
        max_samples: Maximum number of samples to use (for testing)
    """

    def __init__(
        self,
        training_split_path: Union[str, Path],
        biomarker: str,
        base_dir: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        resolution: int = 1024,
        max_samples: Optional[int] = None,
    ):
        self.training_split_path = Path(training_split_path)
        self.biomarker = biomarker
        self.base_dir = Path(base_dir) if base_dir else self.training_split_path.parent.parent
        self.transform = transform
        self.resolution = resolution

        if biomarker not in BIOMARKERS:
            raise ValueError(
                f"Unknown biomarker: {biomarker}. "
                f"Available biomarkers: {BIOMARKERS}"
            )

        # Load training split
        self._load_training_split()

        # Limit samples if specified
        if max_samples is not None and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]
            logger.info(f"Limited to {max_samples} samples")

        # Cache for open H5 files (to avoid reopening)
        self._h5_cache: Dict[str, h5py.File] = {}

        logger.info(
            f"MultiPatientBiomarkerDataset initialized: "
            f"biomarker={biomarker}, samples={len(self)}"
        )

    def _load_training_split(self) -> None:
        """Load training split from JSON file."""
        if not self.training_split_path.exists():
            raise FileNotFoundError(
                f"Training split not found: {self.training_split_path}"
            )

        with open(self.training_split_path, 'r') as f:
            data = json.load(f)

        # Validate biomarker matches
        if data.get('biomarker') != self.biomarker:
            logger.warning(
                f"Biomarker mismatch: split has '{data.get('biomarker')}', "
                f"requested '{self.biomarker}'"
            )

        self.samples = data['samples']
        logger.info(f"Loaded {len(self.samples)} samples from training split")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                - reference_image: H&E image tensor [3, H, W] in [-1, 1]
                - target_image: Biomarker RGB tensor [3, H, W] in [-1, 1]
                - caption: Text prompt for generation
                - coordinate: Patch coordinate string
                - crc_id: Patient ID (e.g., "CRC01")
                - sample_type: "positive" or "negative"
        """
        sample_info = self.samples[idx]

        # Resolve paths relative to base_dir
        h5_path = self.base_dir / sample_info['h5_path']
        he_dir = self.base_dir / sample_info['he_dir']
        h5_index = sample_info['index']
        coordinate = sample_info['coordinate']
        crc_id = sample_info['crc_id']
        folder = sample_info.get('folder', crc_id)  # Use folder name for filename

        # Load biomarker RGB from H5
        biomarker_rgb = self._load_biomarker_rgb(h5_path, h5_index)
        target_image = Image.fromarray(biomarker_rgb)

        # Load H&E reference image
        # H&E files are named: {folder}_HE_{coordinate}.png
        he_filename = f"{folder}_HE_{coordinate}.png"
        he_path = he_dir / he_filename

        if not he_path.exists():
            # Try with crc_id instead of folder
            he_path = he_dir / f"{crc_id}_HE_{coordinate}.png"

        if not he_path.exists():
            # Try alternative naming convention (CRC01 style)
            he_path = he_dir / f"CRC01_HE_{coordinate}.png"

        if not he_path.exists():
            raise FileNotFoundError(
                f"H&E image not found: {he_path}\n"
                f"Coordinate: {coordinate}, CRC: {crc_id}, Folder: {folder}"
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

        # Convert to tensors normalized to [-1, 1]
        he_tensor = self._to_tensor(he_image)
        target_tensor = self._to_tensor(target_image)

        # Apply transforms if provided
        if self.transform is not None:
            he_tensor = self.transform(he_tensor)
            target_tensor = self.transform(target_tensor)

        # Create caption
        caption = f"Generate {self.biomarker} biomarker image"

        return {
            "reference_image": he_tensor,
            "target_image": target_tensor,
            "caption": caption,
            "coordinate": coordinate,
            "crc_id": crc_id,
            "sample_type": sample_info.get('sample_type', 'unknown'),
            "index": h5_index,
        }

    def _load_biomarker_rgb(self, h5_path: Path, index: int) -> np.ndarray:
        """Load biomarker RGB image from H5 file."""
        h5_key = str(h5_path)

        # Use cached file handle if available
        if h5_key not in self._h5_cache:
            self._h5_cache[h5_key] = h5py.File(h5_path, 'r')

        f = self._h5_cache[h5_key]

        r = f[f'{self.biomarker}_R'][index]
        g = f['shared_G'][index]
        b = f['shared_B'][index]

        return np.stack([r, g, b], axis=-1).astype(np.uint8)

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to normalized tensor [-1, 1]."""
        arr = np.array(image).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor

    def __del__(self):
        """Close cached H5 files."""
        for f in self._h5_cache.values():
            try:
                f.close()
            except:
                pass

    @staticmethod
    def available_biomarkers() -> List[str]:
        """Return list of available biomarkers."""
        return BIOMARKERS.copy()


def collate_fn(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """Collate function for multi-patient dataset."""
    reference_images = torch.stack([item["reference_image"] for item in batch])
    target_images = torch.stack([item["target_image"] for item in batch])
    captions = [item["caption"] for item in batch]
    coordinates = [item["coordinate"] for item in batch]
    crc_ids = [item["crc_id"] for item in batch]
    sample_types = [item["sample_type"] for item in batch]
    indices = [item["index"] for item in batch]

    return {
        "reference_images": reference_images,
        "target_images": target_images,
        "captions": captions,
        "coordinates": coordinates,
        "crc_ids": crc_ids,
        "sample_types": sample_types,
        "indices": indices,
    }


def create_multi_patient_dataloader(
    training_split_path: Union[str, Path],
    biomarker: str,
    base_dir: Optional[Union[str, Path]] = None,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    resolution: int = 1024,
    max_samples: Optional[int] = None,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for multi-patient biomarker dataset.

    Args:
        training_split_path: Path to training split JSON file
        biomarker: Target biomarker name
        base_dir: Base directory for resolving relative paths
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        resolution: Target image resolution
        max_samples: Maximum number of samples (for testing)

    Returns:
        DataLoader instance
    """
    dataset = MultiPatientBiomarkerDataset(
        training_split_path=training_split_path,
        biomarker=biomarker,
        base_dir=base_dir,
        resolution=resolution,
        max_samples=max_samples,
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
