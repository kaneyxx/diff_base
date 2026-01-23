#!/usr/bin/env python3
"""Main training script for diffusion models."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.utils.config import load_config, validate_config
from src.utils.logging import get_logger
from src.training import create_trainer


logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train diffusion models with various methods"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (for testing)",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main training entry point."""
    args = parse_args()

    # Load and validate configuration
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Override config with command line arguments
    if args.output_dir:
        config.experiment.output_dir = args.output_dir

    if args.seed is not None:
        config.experiment.seed = args.seed

    # Validate configuration
    validate_config(config)

    # Set seed
    seed = config.experiment.get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    # Log configuration
    logger.info(f"Model type: {config.model.type}")
    logger.info(f"Training method: {config.training.method}")
    logger.info(f"Output directory: {config.experiment.output_dir}")

    # Create output directory
    output_dir = Path(config.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output directory
    from src.utils.config import save_config
    save_config(config, output_dir / "config.yaml")

    # Create trainer
    logger.info("Creating trainer...")
    trainer = create_trainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    logger.info("Starting training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer._save_checkpoint()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    logger.info(f"Training complete. Outputs saved to {config.experiment.output_dir}")


if __name__ == "__main__":
    main()
