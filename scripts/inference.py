#!/usr/bin/env python3
"""Inference script for generating images from trained models.

Supports:
- Standard text-to-image generation
- FLUX.2 Kontext: Reference image editing
- FLUX.2 Fill: Inpainting with masks

Examples:
    # Standard generation
    python scripts/inference.py --checkpoint path/to/model --prompt "A cat"

    # FLUX.2 Kontext (reference editing)
    python scripts/inference.py --checkpoint path/to/flux2 \\
        --prompt "A cat wearing a hat" \\
        --reference-image cat.png \\
        --mode kontext

    # FLUX.2 Fill (inpainting)
    python scripts/inference.py --checkpoint path/to/flux2 \\
        --prompt "A beautiful sky" \\
        --reference-image scene.png \\
        --mask mask.png \\
        --mode fill
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image

from src.inference import create_pipeline, create_flux2_editing_pipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate images using trained diffusion models"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt for classifier-free guidance",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output image path",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for inference",
    )

    # Image editing arguments
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["generate", "kontext", "fill"],
        help="Generation mode: generate (standard), kontext (reference editing), fill (inpainting)",
    )
    parser.add_argument(
        "--reference-image",
        type=str,
        default=None,
        help="Path to reference image for kontext/fill modes",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to mask image for fill mode (white = inpaint region)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="dev",
        choices=["dev", "klein-4b", "klein-9b"],
        help="FLUX.2 variant (only used for editing modes)",
    )

    return parser.parse_args()


def main():
    """Main inference entry point."""
    args = parse_args()

    # Validate editing mode arguments
    if args.mode in ["kontext", "fill"] and args.reference_image is None:
        raise ValueError(f"Mode '{args.mode}' requires --reference-image")
    if args.mode == "fill" and args.mask is None:
        raise ValueError("Fill mode requires --mask")

    logger.info(f"Loading model from {args.checkpoint}")
    logger.info(f"Mode: {args.mode}")

    # Create appropriate pipeline based on mode
    if args.mode in ["kontext", "fill"]:
        # Use FLUX.2 editing pipeline for image editing modes
        pipeline = create_flux2_editing_pipeline(
            checkpoint_path=args.checkpoint,
            variant=args.variant,
            device=args.device,
            dtype=args.dtype,
        )
        logger.info(f"Using FLUX.2 editing pipeline (variant: {args.variant})")
    else:
        # Use standard pipeline for text-to-image
        pipeline = create_pipeline(
            checkpoint_path=args.checkpoint,
            device=args.device,
            dtype=args.dtype,
        )

    # Setup generator for reproducibility
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        logger.info(f"Using seed: {args.seed}")

    # Load reference image and mask if provided
    reference_image = None
    mask = None
    if args.reference_image:
        reference_image = Image.open(args.reference_image).convert("RGB")
        logger.info(f"Reference image: {args.reference_image}")
    if args.mask:
        mask = Image.open(args.mask).convert("L")
        logger.info(f"Mask: {args.mask}")

    logger.info(f"Generating {args.num_images} image(s)...")
    logger.info(f"Prompt: {args.prompt}")
    if args.negative_prompt:
        logger.info(f"Negative: {args.negative_prompt}")

    # Generate images
    if args.mode in ["kontext", "fill"]:
        # Use editing pipeline
        images = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            reference_image=reference_image,
            mask=mask,
            mode=args.mode,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images,
            generator=generator,
        )
    else:
        # Use standard pipeline
        images = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images,
            generator=generator,
        )

    # Save images
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.num_images == 1:
        images[0].save(output_path)
        logger.info(f"Saved image to {output_path}")
    else:
        stem = output_path.stem
        suffix = output_path.suffix
        for i, img in enumerate(images):
            save_path = output_dir / f"{stem}_{i:03d}{suffix}"
            img.save(save_path)
            logger.info(f"Saved image to {save_path}")

    logger.info("Generation complete!")


if __name__ == "__main__":
    main()
