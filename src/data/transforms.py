"""Image transforms for diffusion training."""

from typing import Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from omegaconf import DictConfig


def create_transforms(resolution: int, config: DictConfig) -> T.Compose:
    """Create image transforms for training.

    Args:
        resolution: Target image resolution.
        config: Training configuration.

    Returns:
        Composed transforms.
    """
    data_config = config.get("data", {})

    transforms = []

    # Optional horizontal flip augmentation
    if data_config.get("flip_augment", False):
        transforms.append(T.RandomHorizontalFlip(p=0.5))

    # Resize and crop
    transforms.extend([
        T.Resize(resolution, interpolation=T.InterpolationMode.LANCZOS),
        T.CenterCrop(resolution),
    ])

    # Optional color jitter
    if data_config.get("color_jitter", False):
        transforms.append(
            T.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
            )
        )

    # Convert to tensor and normalize to [-1, 1]
    transforms.extend([
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),  # Converts [0, 1] to [-1, 1]
    ])

    return T.Compose(transforms)


def create_bucket_transforms(
    target_size: Tuple[int, int],
    config: DictConfig,
) -> T.Compose:
    """Create transforms for a specific bucket size.

    Args:
        target_size: Target size as (width, height).
        config: Training configuration.

    Returns:
        Composed transforms.
    """
    width, height = target_size
    data_config = config.get("data", {})

    transforms = []

    # Optional horizontal flip
    if data_config.get("flip_augment", False):
        transforms.append(T.RandomHorizontalFlip(p=0.5))

    # Resize to bucket dimensions
    transforms.append(
        T.Resize(
            (height, width),
            interpolation=T.InterpolationMode.LANCZOS,
        )
    )

    # Convert to tensor and normalize
    transforms.extend([
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])

    return T.Compose(transforms)


def create_conditioning_transforms(
    resolution: int,
    conditioning_type: str = "canny",
) -> T.Compose:
    """Create transforms for conditioning images.

    Args:
        resolution: Target resolution.
        conditioning_type: Type of conditioning.

    Returns:
        Composed transforms.
    """
    transforms = [
        T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(resolution),
        T.ToTensor(),
    ]

    # Normalize based on conditioning type
    if conditioning_type in ["canny", "depth"]:
        # Single channel, keep [0, 1]
        pass
    else:
        # RGB conditioning
        transforms.append(T.Normalize([0.5], [0.5]))

    return T.Compose(transforms)


class RandomCrop:
    """Random crop with aspect ratio preservation."""

    def __init__(
        self,
        size: int | Tuple[int, int],
        scale: Tuple[float, float] = (0.8, 1.0),
        ratio: Tuple[float, float] = (0.75, 1.33),
    ):
        """Initialize random crop.

        Args:
            size: Target size.
            scale: Range of crop sizes relative to input.
            ratio: Range of aspect ratios.
        """
        self.size = size if isinstance(size, tuple) else (size, size)
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply random crop."""
        import random
        import math

        width, height = img.size
        area = width * height

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= width and h <= height:
                x = random.randint(0, width - w)
                y = random.randint(0, height - h)
                img = img.crop((x, y, x + w, y + h))
                return img.resize(self.size, Image.LANCZOS)

        # Fallback to center crop
        scale = min(width / self.size[0], height / self.size[1])
        new_w = int(self.size[0] * scale)
        new_h = int(self.size[1] * scale)
        x = (width - new_w) // 2
        y = (height - new_h) // 2
        img = img.crop((x, y, x + new_w, y + new_h))
        return img.resize(self.size, Image.LANCZOS)


class AspectRatioResize:
    """Resize maintaining aspect ratio."""

    def __init__(
        self,
        max_size: int,
        interpolation: T.InterpolationMode = T.InterpolationMode.LANCZOS,
    ):
        """Initialize aspect ratio resize.

        Args:
            max_size: Maximum dimension.
            interpolation: Interpolation mode.
        """
        self.max_size = max_size
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        """Resize image maintaining aspect ratio."""
        width, height = img.size
        scale = self.max_size / max(width, height)

        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), self.interpolation.value)

        return img


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor from [-1, 1] back to [0, 1].

    Args:
        tensor: Normalized tensor.

    Returns:
        Denormalized tensor in [0, 1].
    """
    return tensor * 0.5 + 0.5


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image.

    Args:
        tensor: Image tensor [C, H, W] in [-1, 1] or [0, 1].

    Returns:
        PIL Image.
    """
    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor[0]

    # Denormalize if needed
    if tensor.min() < 0:
        tensor = denormalize(tensor)

    # Clamp and convert
    tensor = tensor.clamp(0, 1)
    tensor = (tensor * 255).byte()

    # Convert to numpy and PIL
    if tensor.shape[0] == 1:
        # Grayscale
        return Image.fromarray(tensor[0].cpu().numpy(), mode="L")
    else:
        # RGB
        return Image.fromarray(
            tensor.permute(1, 2, 0).cpu().numpy(),
            mode="RGB"
        )


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to tensor.

    Args:
        image: PIL Image (RGB or grayscale).

    Returns:
        Image tensor [C, H, W] in range [-1, 1].
    """
    import numpy as np

    # Convert to numpy array
    if image.mode == "L":
        # Grayscale
        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
    else:
        # RGB - ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
        arr = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

    # Normalize to [-1, 1]
    tensor = tensor * 2.0 - 1.0

    return tensor
