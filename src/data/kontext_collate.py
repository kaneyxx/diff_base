"""Batch collation for Kontext paired (target, reference, caption) samples."""

from typing import Any

import torch


def kontext_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of Kontext samples into a batched dictionary.

    All samples in the batch must share the same target and reference
    spatial resolution (enforced by a bucketed sampler or fixed-resolution
    dataset). If resolutions differ, ``torch.stack`` will raise; callers
    are responsible for grouping by resolution.

    Args:
        batch: List of sample dicts produced by ``KontextDataset.__getitem__``.
            Each dict must contain:
            - ``target_image``: ``[3, Ht, Wt]`` float tensor in ``[-1, 1]``.
            - ``reference_image``: ``[3, Hr, Wr]`` float tensor in ``[-1, 1]``.
            - ``caption``: string.
            - ``target_resolution``: ``(Ht, Wt)`` tuple.
            - ``reference_resolution``: ``(Hr, Wr)`` tuple.

    Returns:
        Batched dict with:
        - ``target_pixel``: ``[B, 3, Ht, Wt]`` float tensor.
        - ``reference_pixel``: ``[B, 3, Hr, Wr]`` float tensor.
        - ``captions``: list of B strings.
        - ``target_resolution``: ``(Ht, Wt)`` tuple (from first sample).
        - ``reference_resolution``: ``(Hr, Wr)`` tuple (from first sample).
    """
    target_pixel = torch.stack([item["target_image"] for item in batch])
    reference_pixel = torch.stack([item["reference_image"] for item in batch])
    captions = [item["caption"] for item in batch]

    return {
        "target_pixel": target_pixel,
        "reference_pixel": reference_pixel,
        "captions": captions,
        "target_resolution": batch[0]["target_resolution"],
        "reference_resolution": batch[0]["reference_resolution"],
    }
