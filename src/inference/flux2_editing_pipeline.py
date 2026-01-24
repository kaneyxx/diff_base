"""FLUX.2 Image Editing Pipeline.

This pipeline supports three modes:
1. Generate: Standard text-to-image generation
2. Kontext: Reference image editing via sequence concatenation
3. Fill: Inpainting via channel concatenation with masks

Key Implementation Notes:
- Image conditioning happens at the denoise/sampling level, not in the model forward
- Conditioning tensors are concatenated to input BEFORE passing to model
- Outputs are sliced AFTER prediction to get only base image tokens
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from omegaconf import DictConfig
from tqdm import tqdm

from ..models.flux.v2 import Flux2Model
from ..models.flux.v2.conditioning import (
    rearrange_latent_to_sequence,
    rearrange_sequence_to_latent,
    prepare_kontext_conditioning,
    prepare_fill_conditioning,
)
from ..schedulers import create_scheduler
from ..utils.config import load_config
from ..utils.logging import get_logger
from ..data.transforms import tensor_to_pil, pil_to_tensor

logger = get_logger(__name__)


class Flux2EditingPipeline:
    """FLUX.2 Image Editing Pipeline.

    Supports text-to-image generation and image editing (Kontext, Fill modes).

    Example:
        >>> pipeline = Flux2EditingPipeline.from_pretrained("path/to/flux2")
        >>>
        >>> # Standard generation
        >>> images = pipeline(prompt="A cat", mode="generate")
        >>>
        >>> # Kontext (reference editing)
        >>> images = pipeline(
        ...     prompt="A cat wearing a hat",
        ...     reference_image=Image.open("cat.png"),
        ...     mode="kontext",
        ... )
        >>>
        >>> # Fill (inpainting)
        >>> images = pipeline(
        ...     prompt="A beautiful sky",
        ...     reference_image=Image.open("scene.png"),
        ...     mask=Image.open("mask.png"),
        ...     mode="fill",
        ... )
    """

    def __init__(
        self,
        model: Flux2Model,
        scheduler,
        config: DictConfig,
        device: Union[torch.device, str] = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize editing pipeline.

        Args:
            model: FLUX.2 model instance.
            scheduler: Noise scheduler.
            config: Model configuration.
            device: Target device.
            dtype: Data type.
        """
        self.model = model.to(device, dtype=dtype)
        self.scheduler = scheduler
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype

        self.model.eval()

        # Default patch size for FLUX
        self.patch_size = 2

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Union[str, Path],
        variant: str = "dev",
        device: str = "cuda",
        dtype: str = "float16",
    ) -> "Flux2EditingPipeline":
        """Load pipeline from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint.
            variant: FLUX.2 variant ("dev", "klein-4b", "klein-9b").
            device: Target device.
            dtype: Data type string.

        Returns:
            Loaded pipeline.
        """
        checkpoint_path = Path(checkpoint_path)

        # Load config if available
        config_path = checkpoint_path / "config.yaml"
        if config_path.exists():
            config = load_config(config_path)
        else:
            # Create minimal config
            config = DictConfig({
                "model": {
                    "type": "flux2",
                    "variant": variant,
                    "scheduler": {
                        "type": "flow_matching",
                        "num_train_timesteps": 1000,
                        "shift": 3.0,
                    },
                },
            })

        # Create model
        from ..models.flux.v2 import Flux2Model
        model = Flux2Model(config.model, variant=variant)

        # Load weights if available
        if checkpoint_path.is_dir():
            model.load_pretrained(checkpoint_path)

        # Create scheduler
        scheduler_config = config.model.get("scheduler", {
            "type": "flow_matching",
            "num_train_timesteps": 1000,
            "shift": 3.0,
        })
        scheduler = create_scheduler(scheduler_config)

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
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        reference_image: Optional[Union[Image.Image, torch.Tensor]] = None,
        mask: Optional[Union[Image.Image, torch.Tensor]] = None,
        mode: Literal["generate", "kontext", "fill"] = "generate",
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
    ) -> Union[List[Image.Image], torch.Tensor]:
        """Generate or edit images.

        Args:
            prompt: Text prompt or list of prompts.
            negative_prompt: Negative prompt for CFG (optional).
            reference_image: Reference image for editing modes.
            mask: Mask image for Fill mode (white = inpaint region).
            mode: Generation mode ("generate", "kontext", "fill").
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
        # Validate inputs
        if mode in ["kontext", "fill"] and reference_image is None:
            raise ValueError(f"Mode '{mode}' requires a reference_image")
        if mode == "fill" and mask is None:
            raise ValueError("Fill mode requires a mask")

        if isinstance(prompt, str):
            prompt = [prompt]

        batch_size = len(prompt) * num_images_per_prompt

        # Prepare reference image if provided
        reference_tensor = None
        mask_tensor = None
        if reference_image is not None:
            reference_tensor = self._prepare_image(reference_image, height, width)
        if mask is not None:
            mask_tensor = self._prepare_mask(mask, height, width)

        # Encode prompts
        prompt_embeds, pooled_embeds = self._encode_prompt(
            prompt,
            negative_prompt,
            num_images_per_prompt,
            do_cfg=guidance_scale > 1.0,
        )

        # Prepare initial latents
        latents = self._prepare_latents(batch_size, height, width, generator)

        # Prepare conditioning based on mode
        conditioning = self._prepare_conditioning(
            mode=mode,
            reference_tensor=reference_tensor,
            mask_tensor=mask_tensor,
            batch_size=batch_size,
        )

        # Run denoising loop
        latents = self._denoise_loop(
            latents=latents,
            prompt_embeds=prompt_embeds,
            pooled_embeds=pooled_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            conditioning=conditioning,
            generator=generator,
        )

        # Decode latents
        images = self.model.decode_latent(latents)

        # Post-process
        if output_type == "pil":
            return [tensor_to_pil(img) for img in images]
        else:
            return images

    def _prepare_image(
        self,
        image: Union[Image.Image, torch.Tensor],
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Prepare reference image for conditioning.

        Args:
            image: PIL Image or tensor.
            height: Target height.
            width: Target width.

        Returns:
            Image tensor [1, 3, H, W] in range [-1, 1].
        """
        if isinstance(image, Image.Image):
            # Resize to target size
            image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            tensor = pil_to_tensor(image)
        else:
            tensor = image

        # Ensure proper shape and range
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        # Move to device
        tensor = tensor.to(device=self.device, dtype=self.dtype)

        return tensor

    def _prepare_mask(
        self,
        mask: Union[Image.Image, torch.Tensor],
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Prepare mask for Fill mode.

        Args:
            mask: PIL Image (grayscale) or tensor.
            height: Target height.
            width: Target width.

        Returns:
            Mask tensor [1, 1, H, W] with values in [0, 1].
        """
        if isinstance(mask, Image.Image):
            # Convert to grayscale and resize
            mask = mask.convert("L").resize((width, height), Image.Resampling.NEAREST)
            tensor = torch.from_numpy(np.array(mask)).float() / 255.0
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        else:
            tensor = mask

        # Ensure proper shape
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        # Threshold to binary
        tensor = (tensor > 0.5).float()

        # Move to device
        tensor = tensor.to(device=self.device, dtype=self.dtype)

        return tensor

    def _prepare_conditioning(
        self,
        mode: str,
        reference_tensor: Optional[torch.Tensor],
        mask_tensor: Optional[torch.Tensor],
        batch_size: int,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Prepare conditioning tensors based on mode.

        Args:
            mode: Generation mode.
            reference_tensor: Reference image tensor.
            mask_tensor: Mask tensor.
            batch_size: Batch size.

        Returns:
            Dictionary of conditioning tensors.
        """
        conditioning = {
            "img_cond_seq": None,
            "img_cond_seq_ids": None,
            "img_cond": None,
        }

        if mode == "generate":
            return conditioning

        if mode == "kontext":
            # Expand reference for batch
            if reference_tensor.shape[0] == 1 and batch_size > 1:
                reference_tensor = reference_tensor.repeat(batch_size, 1, 1, 1)

            cond = self.model.encode_reference_images(
                images=reference_tensor,
                mode="kontext",
                patch_size=self.patch_size,
            )
            conditioning["img_cond_seq"] = cond["img_cond_seq"]
            conditioning["img_cond_seq_ids"] = cond["img_cond_seq_ids"]

        elif mode == "fill":
            # Expand for batch
            if reference_tensor.shape[0] == 1 and batch_size > 1:
                reference_tensor = reference_tensor.repeat(batch_size, 1, 1, 1)
            if mask_tensor.shape[0] == 1 and batch_size > 1:
                mask_tensor = mask_tensor.repeat(batch_size, 1, 1, 1)

            cond = self.model.encode_reference_images(
                images=reference_tensor,
                mode="fill",
                mask=mask_tensor,
                patch_size=self.patch_size,
            )
            conditioning["img_cond"] = cond["img_cond"]

        return conditioning

    def _encode_prompt(
        self,
        prompt: List[str],
        negative_prompt: Optional[Union[str, List[str]]],
        num_images_per_prompt: int,
        do_cfg: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode text prompts.

        Args:
            prompt: List of prompts.
            negative_prompt: Negative prompt(s).
            num_images_per_prompt: Images per prompt.
            do_cfg: Whether to do classifier-free guidance.

        Returns:
            (prompt_embeds, pooled_embeds)
        """
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

            neg_output = self.model.encode_text(negative_prompt, device=self.device)
            neg_embeds = neg_output["prompt_embeds"]
            neg_pooled = neg_output.get("pooled_prompt_embeds")

            # Duplicate
            neg_embeds = neg_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            if neg_pooled is not None:
                neg_pooled = neg_pooled.repeat_interleave(num_images_per_prompt, dim=0)

            # Concatenate (negative first for CFG)
            prompt_embeds = torch.cat([neg_embeds, prompt_embeds])
            if pooled_embeds is not None:
                pooled_embeds = torch.cat([neg_pooled, pooled_embeds])

        return prompt_embeds, pooled_embeds

    def _prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Prepare initial random latents.

        Args:
            batch_size: Batch size.
            height: Image height.
            width: Image width.
            generator: Random generator.

        Returns:
            Random latent tensor.
        """
        latent_channels = self.model.vae.latent_channels
        vae_scale_factor = 8

        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor

        latents = torch.randn(
            (batch_size, latent_channels, latent_height, latent_width),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        # Scale by scheduler init noise sigma if available
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma

        return latents

    def _denoise_loop(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: Optional[torch.Tensor],
        guidance_scale: float,
        num_inference_steps: int,
        conditioning: Dict[str, Optional[torch.Tensor]],
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Run denoising loop with conditioning.

        This is where the key conditioning logic happens:
        - Channel-wise conditioning (Fill) is concatenated to latent channels
        - Sequence-wise conditioning (Kontext) is concatenated to sequence
        - After prediction, output is sliced back to base image size

        Args:
            latents: Initial noisy latents.
            prompt_embeds: Text embeddings.
            pooled_embeds: Pooled text embeddings.
            guidance_scale: CFG scale.
            num_inference_steps: Number of steps.
            conditioning: Conditioning tensors dict.
            generator: Random generator.

        Returns:
            Denoised latents.
        """
        do_cfg = guidance_scale > 1.0

        # Setup scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Convert latents to sequence format for FLUX
        # [B, C, H, W] -> [B, seq, C*ph*pw]
        B, C, H, W = latents.shape
        h, w = H // self.patch_size, W // self.patch_size
        base_seq_len = h * w

        # Store original spatial dims for reconstruction
        original_shape = (B, C, H, W)

        for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
            # Convert current latents to sequence
            latent_seq = rearrange_latent_to_sequence(latents, patch_size=self.patch_size)

            # Prepare model input
            if do_cfg:
                latent_model_input = torch.cat([latent_seq] * 2)
            else:
                latent_model_input = latent_seq

            # Prepare conditioning for this step
            img_cond_seq = conditioning["img_cond_seq"]
            img_cond_seq_ids = conditioning["img_cond_seq_ids"]
            img_cond = conditioning["img_cond"]

            # Duplicate conditioning for CFG if needed
            if do_cfg:
                if img_cond_seq is not None:
                    img_cond_seq = torch.cat([img_cond_seq] * 2)
                    img_cond_seq_ids = torch.cat([img_cond_seq_ids] * 2)
                if img_cond is not None:
                    img_cond = torch.cat([img_cond] * 2)

            # Create timestep tensor
            timestep = t.expand(latent_model_input.shape[0])

            # Create guidance tensor for FLUX
            guidance = None
            if self.model.use_guidance:
                guidance = torch.full(
                    (latent_model_input.shape[0],),
                    guidance_scale,
                    device=self.device,
                    dtype=self.dtype,
                )

            # Forward pass through model
            noise_pred = self.model(
                latents=latent_model_input,
                timesteps=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                guidance=guidance,
                img_cond_seq=img_cond_seq,
                img_cond_seq_ids=img_cond_seq_ids,
                img_cond=img_cond,
            )

            # Slice output to base image size (remove Kontext tokens if present)
            if img_cond_seq is not None:
                noise_pred = noise_pred[:, :base_seq_len]

            # Apply CFG
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # Convert prediction back to spatial format
            noise_pred_spatial = rearrange_sequence_to_latent(
                noise_pred,
                height=h,
                width=w,
                channels=C,
                patch_size=self.patch_size,
            )

            # Scheduler step
            latents, _ = self.scheduler.step(noise_pred_spatial, t, latents, generator)

        return latents


def create_flux2_editing_pipeline(
    checkpoint_path: Union[str, Path],
    variant: str = "dev",
    device: str = "cuda",
    dtype: str = "float16",
) -> Flux2EditingPipeline:
    """Create FLUX.2 editing pipeline from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        variant: FLUX.2 variant.
        device: Target device.
        dtype: Data type.

    Returns:
        Configured pipeline.
    """
    return Flux2EditingPipeline.from_pretrained(
        checkpoint_path,
        variant=variant,
        device=device,
        dtype=dtype,
    )
