"""Latent and embedding caching for faster training."""

import hashlib
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import torch
from safetensors.torch import save_file, load_file
from tqdm import tqdm

if TYPE_CHECKING:
    from torch.utils.data import Dataset


class LatentCache:
    """Cache for precomputed VAE latents and text embeddings.

    Speeds up training by avoiding repeated encoding operations.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        model_hash: str,
    ):
        """Initialize latent cache.

        Args:
            cache_dir: Directory to store cached tensors.
            model_hash: Hash to identify model version.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_hash = self._compute_hash(model_hash)

    def _compute_hash(self, model_path: str) -> str:
        """Compute a short hash for the model path."""
        return hashlib.md5(model_path.encode()).hexdigest()[:8]

    def _get_cache_path(self, idx: int) -> Path:
        """Get cache file path for sample index.

        Args:
            idx: Sample index.

        Returns:
            Path to cache file.
        """
        return self.cache_dir / f"{self.model_hash}_{idx:08d}.safetensors"

    def get(self, idx: int) -> Optional[dict[str, torch.Tensor]]:
        """Load cached tensors if available.

        Args:
            idx: Sample index.

        Returns:
            Dictionary of cached tensors or None.
        """
        cache_path = self._get_cache_path(idx)
        if cache_path.exists():
            try:
                return load_file(cache_path)
            except Exception:
                return None
        return None

    def put(self, idx: int, data: dict[str, torch.Tensor]) -> None:
        """Save tensors to cache.

        Args:
            idx: Sample index.
            data: Dictionary of tensors to cache.
        """
        cache_path = self._get_cache_path(idx)
        # Ensure all tensors are on CPU and contiguous
        cpu_data = {
            k: v.cpu().contiguous() for k, v in data.items()
        }
        save_file(cpu_data, cache_path)

    def has(self, idx: int) -> bool:
        """Check if sample is cached.

        Args:
            idx: Sample index.

        Returns:
            True if cached.
        """
        return self._get_cache_path(idx).exists()

    def clear(self) -> None:
        """Clear all cached files for this model."""
        for path in self.cache_dir.glob(f"{self.model_hash}_*.safetensors"):
            path.unlink()

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats.
        """
        files = list(self.cache_dir.glob(f"{self.model_hash}_*.safetensors"))
        total_size = sum(f.stat().st_size for f in files)

        return {
            "num_cached": len(files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir),
        }


def precompute_latents(
    dataset: "Dataset",
    vae: torch.nn.Module,
    text_encoders: torch.nn.Module,
    cache: LatentCache,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
    batch_size: int = 1,
) -> None:
    """Precompute and cache all VAE latents and text embeddings.

    Args:
        dataset: Dataset to process.
        vae: VAE encoder model.
        text_encoders: Text encoder model.
        cache: Cache to store results.
        device: Computation device.
        dtype: Data type for computation.
        batch_size: Batch size for processing.
    """
    vae = vae.to(device, dtype=dtype)
    vae.eval()

    text_encoders = text_encoders.to(device, dtype=dtype)
    text_encoders.eval()

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Caching latents"):
            if cache.has(idx):
                continue

            sample = dataset[idx]

            # Encode image to latent
            image = sample["image"].unsqueeze(0).to(device, dtype=dtype)

            if hasattr(vae, "encode"):
                posterior = vae.encode(image)
                if hasattr(posterior, "sample"):
                    latent = posterior.sample()
                else:
                    latent = posterior
            else:
                latent = vae(image)

            # Get VAE scaling factor
            scaling_factor = getattr(vae, "scaling_factor", 0.13025)
            latent = latent * scaling_factor

            # Encode text
            caption = sample["caption"]
            if hasattr(text_encoders, "encode"):
                text_output = text_encoders.encode(caption, device=device)
                prompt_embeds = text_output["prompt_embeds"]
                pooled_embeds = text_output.get("pooled_prompt_embeds")
            else:
                prompt_embeds = text_encoders(caption)
                pooled_embeds = None

            # Prepare cache data
            cache_data = {
                "latent": latent.squeeze(0).cpu(),
                "prompt_embeds": prompt_embeds.squeeze(0).cpu(),
            }

            if pooled_embeds is not None:
                cache_data["pooled_prompt_embeds"] = pooled_embeds.squeeze(0).cpu()

            cache.put(idx, cache_data)


class CachedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that uses cached latents."""

    def __init__(
        self,
        base_dataset: "Dataset",
        cache: LatentCache,
        return_original: bool = False,
    ):
        """Initialize cached dataset.

        Args:
            base_dataset: Original dataset.
            cache: Latent cache.
            return_original: Whether to return original image too.
        """
        self.base_dataset = base_dataset
        self.cache = cache
        self.return_original = return_original

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get cached sample or fall back to original."""
        cached = self.cache.get(idx)

        if cached is not None:
            result = cached

            if self.return_original:
                original = self.base_dataset[idx]
                result["caption"] = original["caption"]
                result["idx"] = idx

            return result

        # Fall back to original if not cached
        return self.base_dataset[idx]


class EmbeddingCache:
    """Separate cache for text embeddings only."""

    def __init__(
        self,
        cache_dir: str | Path,
    ):
        """Initialize embedding cache.

        Args:
            cache_dir: Directory for cache files.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, dict[str, torch.Tensor]] = {}

    def _get_key(self, text: str) -> str:
        """Get cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[dict[str, torch.Tensor]]:
        """Get cached embedding for text."""
        key = self._get_key(text)

        # Check memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]

        # Check disk cache
        cache_path = self.cache_dir / f"{key}.safetensors"
        if cache_path.exists():
            data = load_file(cache_path)
            self._memory_cache[key] = data
            return data

        return None

    def put(self, text: str, data: dict[str, torch.Tensor]) -> None:
        """Cache embedding for text."""
        key = self._get_key(text)

        # Store in memory
        self._memory_cache[key] = data

        # Store on disk
        cache_path = self.cache_dir / f"{key}.safetensors"
        cpu_data = {k: v.cpu().contiguous() for k, v in data.items()}
        save_file(cpu_data, cache_path)

    def clear_memory(self) -> None:
        """Clear memory cache."""
        self._memory_cache.clear()
