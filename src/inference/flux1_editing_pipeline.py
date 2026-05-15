"""FLUX.1 Image Editing Pipeline (Kontext mode).

Supports reference-image editing using FLUX.1-kontext checkpoints.
FLUX.1 does NOT support Fill mode (inpainting); use FLUX.2 for that.

Key implementation notes:
- Reference image is encoded once outside the denoising loop.
- Transformer output is target-only (Phase A slicing fix).
- Reference resolution is snapped to preferred Kontext buckets by default.
- Guidance embedding is only used for dev/kontext variants (not schnell).
"""

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from ..data.transforms import pil_to_tensor, tensor_to_pil
from ..models.flux.v1 import Flux1Model
from ..models.flux.v2.conditioning import (
    rearrange_latent_to_sequence,
    rearrange_sequence_to_latent,
)
from ..schedulers import create_scheduler
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Preferred Kontext reference resolutions (from BFL util.py).
# Each bucket is (width, height). Reference images are snapped to the
# closest bucket by aspect ratio when kontext_snap_resolution=True.
PREFERED_KONTEXT_RESOLUTIONS: list[tuple[int, int]] = [
    (2560, 1440),
    (1920, 1088),
    (1536, 1024),
    (1280, 1024),
    (1024, 1024),
    (1024, 1280),
    (1024, 1536),
    (1088, 1920),
    (1440, 2560),
    (1024, 768),
    (1152, 896),
    (896, 1152),
    (768, 1024),
    (1344, 768),
    (768, 1344),
    (1536, 640),
    (640, 1536),
]


def _snap_to_preferred_resolution(
    image: Image.Image,
) -> Image.Image:
    """Resize image to the nearest preferred Kontext resolution bucket.

    Selects the bucket with minimum aspect-ratio distance to preserve
    the reference image composition as closely as possible.

    Args:
        image: PIL image to resize.

    Returns:
        Resized PIL image matching the nearest preferred bucket.
    """
    w, h = image.size
    input_aspect = w / h

    best_bucket = min(
        PREFERED_KONTEXT_RESOLUTIONS,
        key=lambda b: abs(b[0] / b[1] - input_aspect),
    )
    bw, bh = best_bucket
    if (w, h) != (bw, bh):
        logger.info(f"Snapping reference from {w}x{h} to {bw}x{bh} (Kontext bucket)")
        image = image.resize((bw, bh), Image.Resampling.LANCZOS)
    return image


class Flux1EditingPipeline:
    """FLUX.1 Image Editing Pipeline using Kontext mode.

    Loads FLUX.1-kontext (or FLUX.1-dev) weights and runs inference for
    reference-image-guided editing. Reference images are concatenated
    along the sequence dimension inside the transformer.

    Example:
        >>> pipe = Flux1EditingPipeline.from_pretrained("/path/to/kontext-checkpoint")
        >>> out = pipe(
        ...     prompt="Make it look like sunset",
        ...     reference_image=Image.open("input.png"),
        ... )
        >>> out[0].save("output.png")
    """

    def __init__(
        self,
        model: Flux1Model,
        scheduler: nn.Module,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        kontext_snap_resolution: bool = True,
    ):
        """Initialize FLUX.1 editing pipeline.

        Args:
            model: Initialized Flux1Model.
            scheduler: Flow-matching scheduler.
            device: Target device.
            dtype: Computation dtype.
            kontext_snap_resolution: If True (default), resize reference images
                to the nearest preferred Kontext bucket before encoding.
        """
        self.model = model.to(device, dtype=dtype)
        self.model.eval()
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.dtype = dtype
        self.kontext_snap_resolution = kontext_snap_resolution
        self.patch_size = 2

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        variant: str = "kontext",
        device: str = "cuda",
        dtype: str = "bfloat16",
        kontext_snap_resolution: bool = True,
    ) -> "Flux1EditingPipeline":
        """Load pipeline from a pretrained checkpoint directory or single file.

        Supports both BFL native (.safetensors) and HF diffusers directory format.

        Args:
            model_path: Path to checkpoint file or diffusers directory.
            variant: FLUX.1 variant ("kontext", "dev", "schnell"). Default: "kontext".
            device: Target device string.
            dtype: Data type string ("bfloat16", "float16", "float32").
            kontext_snap_resolution: Snap reference to preferred Kontext buckets.

        Returns:
            Initialized Flux1EditingPipeline.
        """
        model_path = Path(model_path)

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        # Build default config
        config = OmegaConf.create({
            "scheduler": {
                "type": "flow_matching",
                "num_train_timesteps": 1000,
                "shift": 3.0,
            },
        })

        if model_path.is_file():
            # BFL or single-file format
            model = Flux1Model.from_bfl_checkpoint(model_path, variant=variant)
        else:
            # Diffusers directory
            config_path = model_path / "config.yaml"
            if config_path.exists():
                from ..utils.config import load_config
                config = load_config(config_path)
            model = Flux1Model(config.get("model", config), variant=variant)
            if model_path.is_dir():
                model._load_diffusers_checkpoint(model_path)

        scheduler_cfg = config.get(
            "scheduler",
            OmegaConf.create({"type": "flow_matching_euler", "num_train_timesteps": 1000, "shift": 3.0})
        )
        scheduler = create_scheduler(scheduler_cfg)

        return cls(model, scheduler, device=device, dtype=torch_dtype,
                   kontext_snap_resolution=kontext_snap_resolution)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        reference_image: Image.Image | torch.Tensor,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 2.5,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | None = None,
        output_type: str = "pil",
    ) -> list[Image.Image] | torch.Tensor:
        """Run FLUX.1 Kontext image editing.

        Args:
            prompt: Text prompt(s) describing the desired edit.
            reference_image: Reference image for editing (PIL or tensor [B,3,H,W] in [-1,1]).
            height: Output image height in pixels. Default: 1024.
            width: Output image width in pixels. Default: 1024.
            num_inference_steps: Number of denoising steps. Default: 28.
            guidance_scale: Guidance scale for dev/kontext (ignored for schnell). Default: 2.5.
            num_images_per_prompt: Number of images per prompt. Default: 1.
            generator: Optional random seed generator.
            output_type: "pil" or "tensor". Default: "pil".

        Returns:
            List of PIL images (if output_type="pil") or tensor [B,3,H,W].

        Raises:
            ValueError: If output_type is not recognized.
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt) * num_images_per_prompt

        # Prepare reference image conditioning (encoded once, outside the loop)
        ref_tensor = self._prepare_reference_image(reference_image)
        if ref_tensor.shape[0] == 1 and batch_size > 1:
            ref_tensor = ref_tensor.repeat(batch_size, 1, 1, 1)

        cond = self.model.encode_reference_images(ref_tensor, mode="kontext", patch_size=self.patch_size)
        img_cond_seq = cond["img_cond_seq"]
        img_cond_seq_ids = cond["img_cond_seq_ids"]

        # Encode text
        prompt_embeds, pooled_embeds = self._encode_prompt(prompt, num_images_per_prompt)

        # Initial latents
        latents = self._prepare_latents(batch_size, height, width, generator)

        # Denoising loop
        latents = self._denoise_loop(
            latents=latents,
            prompt_embeds=prompt_embeds,
            pooled_embeds=pooled_embeds,
            img_cond_seq=img_cond_seq,
            img_cond_seq_ids=img_cond_seq_ids,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )

        # Decode
        images = self.model.decode_latent(latents)

        if output_type == "pil":
            return [tensor_to_pil(img) for img in images]
        elif output_type == "tensor":
            return images
        else:
            raise ValueError(f"Unknown output_type: {output_type!r}. Use 'pil' or 'tensor'.")

    def _prepare_reference_image(
        self,
        image: Image.Image | torch.Tensor,
    ) -> torch.Tensor:
        """Prepare reference image tensor [1, 3, H, W] in [-1, 1].

        Args:
            image: PIL image or tensor.

        Returns:
            Float tensor on self.device.
        """
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            if self.kontext_snap_resolution:
                image = _snap_to_preferred_resolution(image)
            tensor = pil_to_tensor(image)
        else:
            tensor = image

        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return tensor.to(device=self.device, dtype=self.dtype)

    def _encode_prompt(
        self,
        prompt: list[str],
        num_images_per_prompt: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode text prompts to embeddings.

        Args:
            prompt: List of text prompts.
            num_images_per_prompt: Repetitions per prompt.

        Returns:
            (prompt_embeds [B, txt_seq, dim], pooled_embeds [B, pool_dim] or None)
        """
        text_output = self.model.encode_text(prompt, device=self.device)
        prompt_embeds = text_output["prompt_embeds"]
        pooled_embeds = text_output.get("pooled_prompt_embeds")

        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            if pooled_embeds is not None:
                pooled_embeds = pooled_embeds.repeat_interleave(num_images_per_prompt, dim=0)

        return prompt_embeds, pooled_embeds

    def _prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Prepare initial random noise latents.

        Args:
            batch_size: Number of samples.
            height: Output height in pixels.
            width: Output width in pixels.
            generator: Optional RNG for reproducibility.

        Returns:
            Latent tensor [B, C, H/8, W/8].
        """
        latent_channels = self.model.vae.latent_channels
        vae_scale_factor = 8
        latent_h = height // vae_scale_factor
        latent_w = width // vae_scale_factor

        return torch.randn(
            (batch_size, latent_channels, latent_h, latent_w),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

    def _denoise_loop(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor | None,
        img_cond_seq: torch.Tensor,
        img_cond_seq_ids: torch.Tensor,
        guidance_scale: float,
        num_inference_steps: int,
    ) -> torch.Tensor:
        """Run Euler flow-matching denoising with Kontext conditioning.

        The transformer returns target tokens only (Phase A slicing fix ensures
        reference tokens never appear in the predicted output).

        Args:
            latents: Initial noisy latents [B, C, H, W].
            prompt_embeds: T5 text embeddings [B, txt_seq, dim].
            pooled_embeds: CLIP pooled embeddings [B, pool_dim].
            img_cond_seq: Reference sequence [B, ref_seq, in_ch].
            img_cond_seq_ids: Reference position IDs [B, ref_seq, 3].
            guidance_scale: CFG guidance scale.
            num_inference_steps: Number of Euler steps.

        Returns:
            Denoised latents [B, C, H, W].
        """
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        B, C, H, W = latents.shape  # noqa: N806

        for t in tqdm(timesteps, desc="FLUX.1 Kontext denoising"):
            # Convert to sequence format [B, seq, C*patch^2]
            latent_seq = rearrange_latent_to_sequence(latents, patch_size=self.patch_size)

            timestep = t.to(self.device).expand(B)

            guidance = None
            if self.model.use_guidance:
                guidance = torch.full(
                    (B,), guidance_scale, device=self.device, dtype=self.dtype
                )

            # Transformer forward — output is target-only (no ref tokens)
            noise_pred_seq = self.model(
                latents=latent_seq,
                timesteps=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_embeds,
                guidance=guidance,
                img_cond_seq=img_cond_seq,
                img_cond_seq_ids=img_cond_seq_ids,
            )

            # Convert prediction back to spatial [B, C, H, W]
            noise_pred = rearrange_sequence_to_latent(
                noise_pred_seq,
                height=H // self.patch_size,
                width=W // self.patch_size,
                channels=C,
                patch_size=self.patch_size,
            )

            # Euler step — scheduler.step returns (prev_sample, pred_original_sample)
            step_output = self.scheduler.step(noise_pred, t, latents)
            latents = step_output[0] if isinstance(step_output, tuple) else step_output.prev_sample

        return latents


def create_flux1_editing_pipeline(
    model_path: str | Path,
    variant: str = "kontext",
    device: str = "cuda",
    dtype: str = "bfloat16",
    kontext_snap_resolution: bool = True,
) -> Flux1EditingPipeline:
    """Factory function to create a Flux1EditingPipeline.

    Convenience wrapper around Flux1EditingPipeline.from_pretrained().

    Args:
        model_path: Path to checkpoint (single .safetensors file or diffusers directory).
        variant: FLUX.1 variant ("kontext", "dev", "schnell"). Default: "kontext".
        device: Target device string.
        dtype: Data type string.
        kontext_snap_resolution: Snap reference images to preferred Kontext buckets.

    Returns:
        Initialized Flux1EditingPipeline ready for inference.

    Example:
        >>> pipe = create_flux1_editing_pipeline("/path/to/flux1-kontext-dev.safetensors")
        >>> out = pipe(prompt="make it night", reference_image=Image.open("day.png"))
        >>> out[0].save("night.png")
    """
    return Flux1EditingPipeline.from_pretrained(
        model_path=model_path,
        variant=variant,
        device=device,
        dtype=dtype,
        kontext_snap_resolution=kontext_snap_resolution,
    )
