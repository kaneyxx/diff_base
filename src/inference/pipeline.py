"""Unified inference pipeline for diffusion models."""

from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from omegaconf import DictConfig
from tqdm import tqdm

from ..models import create_model
from ..schedulers import create_scheduler
from ..utils.config import load_config
from ..utils.logging import get_logger
from ..data.transforms import tensor_to_pil

logger = get_logger(__name__)


class DiffusionPipeline:
    """Unified inference pipeline for SDXL and Flux models."""

    def __init__(
        self,
        model,
        scheduler,
        config: DictConfig,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize inference pipeline.

        Args:
            model: Loaded diffusion model.
            scheduler: Noise scheduler.
            config: Model configuration.
            device: Inference device.
            dtype: Data type.
        """
        self.model = model.to(device, dtype=dtype)
        self.scheduler = scheduler
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype

        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        device: str = "cuda",
        dtype: str = "float16",
    ) -> "DiffusionPipeline":
        """Load pipeline from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint or directory.
            device: Target device.
            dtype: Data type string.

        Returns:
            Loaded pipeline.
        """
        checkpoint_path = Path(checkpoint_path)

        # Load config
        config_path = checkpoint_path / "config.yaml"
        if config_path.exists():
            config = load_config(config_path)
        else:
            raise FileNotFoundError(f"Config not found at {config_path}")

        # Create model
        model = create_model(config)

        # Load weights
        model.load_pretrained(checkpoint_path)

        # Create scheduler
        scheduler = create_scheduler(config.model.scheduler)

        # Parse dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype, torch.float16)

        return cls(model, scheduler, config, device, torch_dtype)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        negative_prompt: Optional[str | list[str]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
    ) -> list[Image.Image] | torch.Tensor:
        """Generate images from text prompts.

        Args:
            prompt: Text prompt or list of prompts.
            negative_prompt: Negative prompt for CFG.
            height: Output image height.
            width: Output image width.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            num_images_per_prompt: Images per prompt.
            generator: Random number generator.
            output_type: Output format ("pil" or "tensor").

        Returns:
            List of PIL images or tensor.
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        batch_size = len(prompt) * num_images_per_prompt

        # Encode prompts
        prompt_embeds, pooled_embeds = self._encode_prompt(
            prompt,
            negative_prompt,
            num_images_per_prompt,
            guidance_scale > 1.0,
        )

        # Prepare latents
        latents = self._prepare_latents(
            batch_size,
            height,
            width,
            generator,
        )

        # Setup scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Prepare extra kwargs for SDXL
        added_cond_kwargs = self._prepare_added_conditioning(
            batch_size * (2 if guidance_scale > 1.0 else 1),
            height,
            width,
            pooled_embeds,
        )

        # Denoising loop
        for t in tqdm(timesteps, desc="Generating"):
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents

            # Predict noise
            noise_pred = self.model(
                latents=latent_model_input,
                timesteps=t.expand(latent_model_input.shape[0]),
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            )

            # Apply CFG
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # Scheduler step
            latents, _ = self.scheduler.step(noise_pred, t, latents, generator)

        # Decode latents
        images = self.model.decode_latent(latents)

        # Post-process
        if output_type == "pil":
            return [tensor_to_pil(img) for img in images]
        else:
            return images

    def _encode_prompt(
        self,
        prompt: list[str],
        negative_prompt: Optional[str | list[str]],
        num_images_per_prompt: int,
        do_cfg: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode text prompts."""
        # Encode positive prompt
        text_output = self.model.encode_text(prompt, device=self.device)
        prompt_embeds = text_output["prompt_embeds"]
        pooled_embeds = text_output.get("pooled_prompt_embeds")

        # Duplicate for num_images_per_prompt
        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        if pooled_embeds is not None:
            pooled_embeds = pooled_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        if do_cfg:
            # Encode negative prompt
            if negative_prompt is None:
                negative_prompt = [""] * len(prompt)
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * len(prompt)

            neg_text_output = self.model.encode_text(negative_prompt, device=self.device)
            neg_prompt_embeds = neg_text_output["prompt_embeds"]
            neg_pooled_embeds = neg_text_output.get("pooled_prompt_embeds")

            # Duplicate
            neg_prompt_embeds = neg_prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            if neg_pooled_embeds is not None:
                neg_pooled_embeds = neg_pooled_embeds.repeat_interleave(num_images_per_prompt, dim=0)

            # Concatenate (negative first for CFG)
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds])
            if pooled_embeds is not None:
                pooled_embeds = torch.cat([neg_pooled_embeds, pooled_embeds])

        return prompt_embeds, pooled_embeds

    def _prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Prepare initial random latents."""
        # Get latent dimensions
        latent_channels = getattr(self.model.vae, "latent_channels", 4)
        vae_scale_factor = 8  # Standard VAE downscaling

        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor

        latents = torch.randn(
            (batch_size, latent_channels, latent_height, latent_width),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        # Scale by scheduler init noise sigma
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma

        return latents

    def _prepare_added_conditioning(
        self,
        batch_size: int,
        height: int,
        width: int,
        pooled_embeds: Optional[torch.Tensor],
    ) -> Optional[dict]:
        """Prepare additional conditioning for SDXL."""
        if pooled_embeds is None:
            return None

        if not hasattr(self.model, "get_time_ids"):
            return None

        time_ids = self.model.get_time_ids(
            original_size=(height, width),
            target_size=(height, width),
            batch_size=batch_size,
            device=self.device,
            dtype=self.dtype,
        )

        return {
            "text_embeds": pooled_embeds,
            "time_ids": time_ids,
        }


def create_pipeline(
    checkpoint_path: str | Path,
    device: str = "cuda",
    dtype: str = "float16",
) -> DiffusionPipeline:
    """Create inference pipeline from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        device: Target device.
        dtype: Data type.

    Returns:
        Configured pipeline.
    """
    return DiffusionPipeline.from_pretrained(checkpoint_path, device, dtype)
