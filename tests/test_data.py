"""Tests for data pipeline components."""

import pytest
import torch
import tempfile
import json
from pathlib import Path
from PIL import Image
import numpy as np
from omegaconf import OmegaConf

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.transforms import create_transforms, create_bucket_transforms, tensor_to_pil
from src.data.bucket import compute_bucket_sizes, AspectRatioBucketManager
from src.data.collate import default_collate_fn


class TestTransforms:
    """Tests for image transforms."""

    def test_create_transforms(self):
        """Test creating default transforms."""
        config = OmegaConf.create({"data": {}})
        transform = create_transforms(resolution=512, config=config)

        # Create a test image
        img = Image.new("RGB", (800, 600), color="red")
        tensor = transform(img)

        assert tensor.shape == (3, 512, 512)
        assert tensor.dtype == torch.float32
        # Should be normalized to [-1, 1]
        assert tensor.min() >= -1.0
        assert tensor.max() <= 1.0

    def test_create_bucket_transforms(self):
        """Test creating bucket-specific transforms."""
        config = OmegaConf.create({"data": {}})
        target_size = (768, 512)
        transform = create_bucket_transforms(target_size=target_size, config=config)

        img = Image.new("RGB", (800, 600), color="blue")
        tensor = transform(img)

        assert tensor.shape == (3, 512, 768)  # Height, Width

    def test_tensor_to_pil(self):
        """Test converting tensor back to PIL Image."""
        # Create a normalized tensor [-1, 1]
        tensor = torch.rand(3, 256, 256) * 2 - 1

        img = tensor_to_pil(tensor)

        assert isinstance(img, Image.Image)
        assert img.size == (256, 256)
        assert img.mode == "RGB"


class TestBucketing:
    """Tests for aspect ratio bucketing."""

    def test_compute_bucket_sizes(self):
        """Test computing bucket sizes."""
        base_resolution = 1024
        step = 64

        buckets = compute_bucket_sizes(
            base_resolution=base_resolution,
            step=step,
            min_dim=512,
            max_dim=2048,
        )

        assert len(buckets) > 0

        # All buckets should have dimensions as multiples of step
        for w, h in buckets:
            assert w % step == 0
            assert h % step == 0

        # Check that we have both portrait and landscape orientations
        widths = [w for w, h in buckets]
        heights = [h for w, h in buckets]
        assert min(widths) < max(heights)  # Some variety in dimensions

    def test_bucket_manager_init(self, tmp_path):
        """Test AspectRatioBucketManager initialization."""
        # Create test images
        (tmp_path / "images").mkdir()

        # Create images with different aspect ratios
        for i, size in enumerate([(512, 512), (768, 512), (512, 768)]):
            img = Image.new("RGB", size, color="white")
            img.save(tmp_path / "images" / f"img_{i}.png")
            (tmp_path / "images" / f"img_{i}.txt").write_text(f"caption {i}")

        manager = AspectRatioBucketManager(
            base_resolution=512,
            bucket_step=64,
            min_dim=256,
            max_dim=1024,
        )

        assert len(manager.bucket_sizes) > 0


class TestCollate:
    """Tests for batch collation."""

    def test_default_collate_basic(self):
        """Test basic batch collation."""
        batch = [
            {"image": torch.randn(3, 64, 64), "caption": "test 1"},
            {"image": torch.randn(3, 64, 64), "caption": "test 2"},
        ]

        collated = default_collate_fn(batch)

        assert "image" in collated
        assert collated["image"].shape == (2, 3, 64, 64)
        assert collated["caption"] == ["test 1", "test 2"]

    def test_default_collate_with_latents(self):
        """Test collation with precomputed latents."""
        batch = [
            {
                "latents": torch.randn(4, 64, 64),
                "encoder_hidden_states": torch.randn(77, 768),
                "caption": "test 1",
            },
            {
                "latents": torch.randn(4, 64, 64),
                "encoder_hidden_states": torch.randn(77, 768),
                "caption": "test 2",
            },
        ]

        collated = default_collate_fn(batch)

        assert collated["latents"].shape == (2, 4, 64, 64)
        assert collated["encoder_hidden_states"].shape == (2, 77, 768)


class TestDataset:
    """Tests for dataset classes."""

    def test_diffusion_dataset_from_metadata(self, tmp_path):
        """Test loading dataset from metadata.json."""
        from src.data.dataset import DiffusionDataset

        # Create test data
        data_dir = tmp_path / "dataset"
        data_dir.mkdir()

        # Create images
        for i in range(3):
            img = Image.new("RGB", (512, 512), color=(i * 80, i * 80, i * 80))
            img.save(data_dir / f"image_{i}.png")

        # Create metadata
        metadata = [
            {"image": f"image_{i}.png", "caption": f"Caption for image {i}"}
            for i in range(3)
        ]
        with open(data_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        config = OmegaConf.create({"data": {}})
        dataset = DiffusionDataset(
            data_path=data_dir,
            resolution=256,
            config=config,
        )

        assert len(dataset) == 3

        sample = dataset[0]
        assert "image" in sample
        assert "caption" in sample
        assert sample["image"].shape == (3, 256, 256)

    def test_diffusion_dataset_from_txt_files(self, tmp_path):
        """Test loading dataset from image/txt file pairs."""
        from src.data.dataset import DiffusionDataset

        # Create test data without metadata.json
        data_dir = tmp_path / "dataset"
        data_dir.mkdir()

        # Create images with accompanying txt files
        for i in range(2):
            img = Image.new("RGB", (512, 512), color="blue")
            img.save(data_dir / f"image_{i}.png")
            (data_dir / f"image_{i}.txt").write_text(f"Caption {i}")

        config = OmegaConf.create({"data": {}})
        dataset = DiffusionDataset(
            data_path=data_dir,
            resolution=256,
            config=config,
        )

        assert len(dataset) == 2

        sample = dataset[0]
        assert "caption" in sample


class TestCache:
    """Tests for latent caching."""

    def test_latent_cache_basic(self, tmp_path):
        """Test basic cache operations."""
        from src.data.cache import LatentCache

        cache = LatentCache(
            cache_dir=tmp_path / "cache",
            model_hash="test_model_123",
        )

        # Initially empty
        assert cache.get(0) is None
        assert not cache.has(0)

        # Store data
        data = {
            "latents": torch.randn(4, 64, 64),
            "encoder_hidden_states": torch.randn(77, 768),
        }
        cache.put(0, data)

        # Retrieve
        assert cache.has(0)
        loaded = cache.get(0)
        assert loaded is not None
        assert torch.allclose(loaded["latents"], data["latents"])


class TestDreamBoothDataset:
    """Tests for DreamBooth dataset."""

    def test_dreambooth_dataset_init(self, tmp_path):
        """Test DreamBooth dataset initialization."""
        from src.data.dreambooth_dataset import DreamBoothDataset

        # Create instance directory
        instance_dir = tmp_path / "instance"
        instance_dir.mkdir()
        for i in range(3):
            img = Image.new("RGB", (512, 512), color="red")
            img.save(instance_dir / f"img_{i}.png")

        config = OmegaConf.create({
            "data": {},
            "training": {
                "dreambooth": {
                    "instance_prompt": "a photo of sks dog",
                    "class_prompt": "a photo of dog",
                    "num_class_images": 0,  # No prior preservation for this test
                }
            }
        })

        dataset = DreamBoothDataset(
            instance_data_dir=instance_dir,
            class_data_dir=None,
            resolution=256,
            config=config,
        )

        assert len(dataset) >= 3

        sample = dataset[0]
        assert "image" in sample
        assert "caption" in sample
        assert "sks" in sample["caption"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
