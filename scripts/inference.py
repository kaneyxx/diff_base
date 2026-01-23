#!/usr/bin/env python3
"""Inference script for generating images from trained models."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image

from src.inference import create_pipeline
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

    return parser.parse_args()


def main():
    """Main inference entry point."""
    args = parse_args()

    logger.info(f"Loading model from {args.checkpoint}")

    # Create pipeline
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

    logger.info(f"Generating {args.num_images} image(s)...")
    logger.info(f"Prompt: {args.prompt}")
    if args.negative_prompt:
        logger.info(f"Negative: {args.negative_prompt}")

    # Generate images
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
