"""DreamBooth dataset with prior preservation support."""

import random
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from omegaconf import DictConfig

from .transforms import create_transforms


class DreamBoothDataset(Dataset):
    """Dataset for DreamBooth fine-tuning with prior preservation.

    Combines instance images (the subject to learn) with class images
    (for prior preservation to prevent language drift).
    """

    def __init__(
        self,
        instance_data_dir: str | Path,
        instance_prompt: str,
        class_data_dir: Optional[str | Path] = None,
        class_prompt: Optional[str] = None,
        resolution: int = 1024,
        config: Optional[DictConfig] = None,
        center_crop: bool = True,
        instance_prompt_template: Optional[str] = None,
    ):
        """Initialize DreamBooth dataset.

        Args:
            instance_data_dir: Directory with instance (subject) images.
            instance_prompt: Prompt describing the instance (e.g., "a photo of sks dog").
            class_data_dir: Directory with class images for prior preservation.
            class_prompt: Prompt for class images (e.g., "a photo of dog").
            resolution: Target image resolution.
            config: Training configuration.
            center_crop: Whether to use center crop.
            instance_prompt_template: Optional template with {caption} placeholder.
        """
        self.instance_data_dir = Path(instance_data_dir)
        self.instance_prompt = instance_prompt
        self.class_data_dir = Path(class_data_dir) if class_data_dir else None
        self.class_prompt = class_prompt
        self.resolution = resolution
        self.config = config or {}
        self.center_crop = center_crop
        self.instance_prompt_template = instance_prompt_template

        # Load instance images
        self.instance_images = self._load_images(self.instance_data_dir)
        if not self.instance_images:
            raise ValueError(f"No images found in {self.instance_data_dir}")

        # Load class images
        self.class_images = []
        if self.class_data_dir and self.class_data_dir.exists():
            self.class_images = self._load_images(self.class_data_dir)

        # Setup transforms
        self.transform = create_transforms(resolution, self.config)

        # Calculate dataset length
        self._num_instance = len(self.instance_images)
        self._num_class = len(self.class_images)

    def _load_images(self, directory: Path) -> list[Path]:
        """Load image paths from directory.

        Args:
            directory: Directory to scan.

        Returns:
            List of image paths.
        """
        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        images = []

        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() in image_extensions:
                images.append(path)

        return images

    def __len__(self) -> int:
        # Return instance count; class images are sampled alongside
        return self._num_instance

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get instance sample, optionally paired with class sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with instance (and optionally class) data.
        """
        result = {}

        # Instance image
        instance_path = self.instance_images[idx % self._num_instance]
        instance_image = Image.open(instance_path).convert("RGB")
        instance_image = self.transform(instance_image)

        # Get instance prompt
        instance_prompt = self.instance_prompt
        if self.instance_prompt_template:
            # Check for caption file
            caption_path = instance_path.with_suffix(".txt")
            if caption_path.exists():
                caption = caption_path.read_text().strip()
                instance_prompt = self.instance_prompt_template.format(caption=caption)

        result["instance_image"] = instance_image
        result["instance_prompt"] = instance_prompt
        result["is_instance"] = True

        # Class image (for prior preservation)
        if self._num_class > 0:
            class_idx = random.randint(0, self._num_class - 1)
            class_path = self.class_images[class_idx]
            class_image = Image.open(class_path).convert("RGB")
            class_image = self.transform(class_image)

            result["class_image"] = class_image
            result["class_prompt"] = self.class_prompt

        return result


class DreamBoothWithPriorDataset(Dataset):
    """DreamBooth dataset that interleaves instance and class samples.

    Returns alternating instance and class samples in the batch.
    """

    def __init__(
        self,
        instance_data_dir: str | Path,
        instance_prompt: str,
        class_data_dir: str | Path,
        class_prompt: str,
        resolution: int = 1024,
        config: Optional[DictConfig] = None,
        prior_loss_weight: float = 1.0,
    ):
        """Initialize dataset.

        Args:
            instance_data_dir: Instance images directory.
            instance_prompt: Instance prompt.
            class_data_dir: Class images directory.
            class_prompt: Class prompt.
            resolution: Image resolution.
            config: Training configuration.
            prior_loss_weight: Weight for prior preservation loss.
        """
        self.instance_data_dir = Path(instance_data_dir)
        self.class_data_dir = Path(class_data_dir)
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt
        self.resolution = resolution
        self.config = config or {}
        self.prior_loss_weight = prior_loss_weight

        # Load images
        self.instance_images = self._load_images(self.instance_data_dir)
        self.class_images = self._load_images(self.class_data_dir)

        if not self.instance_images:
            raise ValueError(f"No instance images found in {self.instance_data_dir}")
        if not self.class_images:
            raise ValueError(f"No class images found in {self.class_data_dir}")

        # Transform
        self.transform = create_transforms(resolution, self.config)

    def _load_images(self, directory: Path) -> list[Path]:
        """Load image paths."""
        extensions = {".png", ".jpg", ".jpeg", ".webp"}
        return sorted([p for p in directory.rglob("*") if p.suffix.lower() in extensions])

    def __len__(self) -> int:
        # Each item returns both instance and class
        return max(len(self.instance_images), len(self.class_images))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get paired instance and class sample."""
        # Instance
        instance_idx = idx % len(self.instance_images)
        instance_path = self.instance_images[instance_idx]
        instance_image = Image.open(instance_path).convert("RGB")
        instance_image = self.transform(instance_image)

        # Class
        class_idx = idx % len(self.class_images)
        class_path = self.class_images[class_idx]
        class_image = Image.open(class_path).convert("RGB")
        class_image = self.transform(class_image)

        return {
            "image": instance_image,
            "caption": self.instance_prompt,
            "is_instance": True,
            "class_image": class_image,
            "class_caption": self.class_prompt,
            "idx": idx,
        }


def generate_class_images(
    model,
    class_prompt: str,
    output_dir: str | Path,
    num_images: int = 200,
    batch_size: int = 4,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
) -> None:
    """Generate class images for prior preservation.

    Args:
        model: Diffusion model with generation capability.
        class_prompt: Prompt for class images.
        output_dir: Directory to save generated images.
        num_images: Number of images to generate.
        batch_size: Generation batch size.
        guidance_scale: Classifier-free guidance scale.
        num_inference_steps: Number of denoising steps.
    """
    from tqdm import tqdm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_batches = (num_images + batch_size - 1) // batch_size
    generated = 0

    for batch_idx in tqdm(range(num_batches), desc="Generating class images"):
        current_batch = min(batch_size, num_images - generated)

        # Generate images
        prompts = [class_prompt] * current_batch

        # This assumes model has a generate method
        if hasattr(model, "generate"):
            images = model.generate(
                prompts,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
        else:
            # Skip generation if model doesn't support it
            break

        # Save images
        for i, image in enumerate(images):
            image_idx = generated + i
            save_path = output_dir / f"class_{image_idx:05d}.png"
            image.save(save_path)

        generated += current_batch
