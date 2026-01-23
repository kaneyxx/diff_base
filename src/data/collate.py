"""Batch collation functions for diffusion training."""

from typing import Any, Callable

import torch
from omegaconf import DictConfig


def create_collate_fn(
    config: DictConfig,
    dreambooth: bool = False,
) -> Callable[[list[dict]], dict[str, Any]]:
    """Create a collate function based on configuration.

    Args:
        config: Training configuration.
        dreambooth: Whether this is for DreamBooth training.

    Returns:
        Collate function.
    """
    if dreambooth:
        return dreambooth_collate_fn

    return default_collate_fn


def default_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Default collate function for diffusion training.

    Args:
        batch: List of sample dictionaries.

    Returns:
        Batched dictionary.
    """
    # Stack images
    images = torch.stack([item["image"] for item in batch])

    # Collect captions
    captions = [item["caption"] for item in batch]

    # Collect indices
    indices = [item.get("idx", i) for i, item in enumerate(batch)]

    result = {
        "images": images,
        "captions": captions,
        "indices": indices,
    }

    # Handle cached latents if present
    if "latent" in batch[0]:
        result["latents"] = torch.stack([item["latent"] for item in batch])

    if "prompt_embeds" in batch[0]:
        result["prompt_embeds"] = torch.stack([item["prompt_embeds"] for item in batch])

    if "pooled_prompt_embeds" in batch[0]:
        result["pooled_prompt_embeds"] = torch.stack(
            [item["pooled_prompt_embeds"] for item in batch]
        )

    # Handle bucket info
    if "bucket" in batch[0]:
        # All items in batch should have same bucket
        result["bucket"] = batch[0]["bucket"]

    # Handle conditioning images (ControlNet)
    if "conditioning_image" in batch[0]:
        result["conditioning_images"] = torch.stack(
            [item["conditioning_image"] for item in batch]
        )

    return result


def dreambooth_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for DreamBooth training.

    Handles instance and class images separately.

    Args:
        batch: List of sample dictionaries.

    Returns:
        Batched dictionary with instance and class data.
    """
    result = {}

    # Separate instance and class samples
    instance_batch = [item for item in batch if item.get("is_instance", True)]
    class_batch = [item for item in batch if not item.get("is_instance", True)]

    # Process instance samples
    if instance_batch:
        result["instance_images"] = torch.stack(
            [item["image"] for item in instance_batch]
        )
        result["instance_captions"] = [item["caption"] for item in instance_batch]

        if "latent" in instance_batch[0]:
            result["instance_latents"] = torch.stack(
                [item["latent"] for item in instance_batch]
            )
        if "prompt_embeds" in instance_batch[0]:
            result["instance_prompt_embeds"] = torch.stack(
                [item["prompt_embeds"] for item in instance_batch]
            )
        if "pooled_prompt_embeds" in instance_batch[0]:
            result["instance_pooled_prompt_embeds"] = torch.stack(
                [item["pooled_prompt_embeds"] for item in instance_batch]
            )

    # Process class samples (prior preservation)
    if class_batch:
        result["class_images"] = torch.stack([item["image"] for item in class_batch])
        result["class_captions"] = [item["caption"] for item in class_batch]

        if "latent" in class_batch[0]:
            result["class_latents"] = torch.stack(
                [item["latent"] for item in class_batch]
            )
        if "prompt_embeds" in class_batch[0]:
            result["class_prompt_embeds"] = torch.stack(
                [item["prompt_embeds"] for item in class_batch]
            )
        if "pooled_prompt_embeds" in class_batch[0]:
            result["class_pooled_prompt_embeds"] = torch.stack(
                [item["pooled_prompt_embeds"] for item in class_batch]
            )

    return result


def controlnet_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function for ControlNet training.

    Args:
        batch: List of sample dictionaries.

    Returns:
        Batched dictionary with images and conditioning.
    """
    result = default_collate_fn(batch)

    # Ensure conditioning images are included
    if "conditioning_image" in batch[0]:
        result["conditioning_images"] = torch.stack(
            [item["conditioning_image"] for item in batch]
        )

    return result


def bucket_collate_fn(
    bucket: tuple[int, int],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collate function for bucketed batches.

    Args:
        bucket: Target bucket size (width, height).
        samples: List of sample dictionaries.

    Returns:
        Batched dictionary with bucket info.
    """
    result = default_collate_fn(samples)
    result["bucket"] = bucket
    result["target_size"] = bucket
    return result


class DynamicBatchCollator:
    """Collator that handles varying batch compositions."""

    def __init__(
        self,
        pad_token_id: int = 0,
        max_length: int = 77,
    ):
        """Initialize dynamic batch collator.

        Args:
            pad_token_id: ID for padding tokens.
            max_length: Maximum sequence length.
        """
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate batch with dynamic handling."""
        result = {}

        # Find all keys present in batch
        all_keys = set()
        for item in batch:
            all_keys.update(item.keys())

        for key in all_keys:
            values = [item.get(key) for item in batch if key in item]

            if not values:
                continue

            if isinstance(values[0], torch.Tensor):
                # Stack tensors
                try:
                    result[key] = torch.stack(values)
                except RuntimeError:
                    # Handle size mismatch by padding
                    result[key] = self._pad_tensors(values)
            elif isinstance(values[0], str):
                result[key] = values
            elif isinstance(values[0], (int, float)):
                result[key] = torch.tensor(values)
            else:
                result[key] = values

        return result

    def _pad_tensors(
        self,
        tensors: list[torch.Tensor],
    ) -> torch.Tensor:
        """Pad tensors to same size and stack.

        Args:
            tensors: List of tensors with potentially different sizes.

        Returns:
            Stacked tensor with padding.
        """
        # Find max size for each dimension
        max_sizes = [
            max(t.shape[i] for t in tensors)
            for i in range(tensors[0].dim())
        ]

        padded = []
        for tensor in tensors:
            pad_sizes = []
            for i in range(tensor.dim() - 1, -1, -1):
                pad_sizes.extend([0, max_sizes[i] - tensor.shape[i]])

            padded_tensor = torch.nn.functional.pad(tensor, pad_sizes)
            padded.append(padded_tensor)

        return torch.stack(padded)
