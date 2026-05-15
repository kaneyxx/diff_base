#!/usr/bin/env python3
"""Dedicated CLI for FLUX full fine-tuning.

Supports FLUX.1 dev/kontext and FLUX.2 dev/klein variants with
FLUX-specific memory optimizations and BFL-aligned flow matching.

For multi-GPU FSDP training, launch via accelerate::

    accelerate launch --config_file configs/accelerate/multi_gpu_fsdp.yaml \\
        scripts/finetune_flux.py --distributed-strategy fsdp \\
        --variant dev --pretrained-path /path/to/flux1-dev \\
        --train-data /path/to/dataset --output-dir /path/to/output
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf

from src.utils.config import load_config, save_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

EPILOG = """
Examples
--------
Single-GPU H100 (recommended for FLUX.1-dev, 80 GB):
  python scripts/finetune_flux.py \\
      --variant dev \\
      --pretrained-path /path/to/flux1-dev \\
      --train-data /path/to/dataset \\
      --output-dir ./outputs/flux1-dev-ft \\
      --resolution 1024 --batch-size 1 --gradient-accumulation 4 \\
      --lr 1e-6 --epochs 30 \\
      --use-8bit-adam --gradient-checkpointing \\
      --ema-decay 0.99 --ema-on-cpu

Multi-GPU FSDP (4 × 80 GB H100):
  accelerate launch --config_file configs/accelerate/multi_gpu_fsdp.yaml \\
      scripts/finetune_flux.py --distributed-strategy fsdp \\
      --variant dev --pretrained-path /path/to/flux1-dev \\
      --train-data /path/to/dataset --output-dir ./outputs/flux1-dev-fsdp \\
      --resolution 1024 --batch-size 2 --gradient-accumulation 2 \\
      --lr 1e-6 --epochs 30

Estimate memory before training:
  python scripts/finetune_flux.py --variant dev --pretrained-path /dummy \\
      --train-data /dummy --output-dir /tmp --estimate-memory

Print resolved config without training:
  python scripts/finetune_flux.py --config my_config.yaml \\
      --variant dev --pretrained-path /path/to/flux1-dev \\
      --train-data /path/to/dataset --output-dir ./out --print-config-only
"""

VARIANT_CHOICES = [
    "dev",
    "kontext",
    "schnell",                # distilled — refused unless --force-distilled
    "flux2-dev",
    "flux2-klein-4b",         # distilled — refused unless --force-distilled
    "flux2-klein-9b",         # distilled — refused unless --force-distilled
    "flux2-klein-4b-base",    # un-distilled, full-FT-friendly (Apache 2.0)
    "flux2-klein-9b-base",    # un-distilled, full-FT-friendly (non-commercial)
]
#: Variants whose published weights are distilled. Mirrors
#: ``src.training.flux_full_finetune_trainer.DISTILLED_VARIANTS`` for the
#: CLI's early refusal path. Keep both lists in sync.
DISTILLED_VARIANTS_CLI: frozenset[str] = frozenset({
    "schnell", "flux2-klein-4b", "flux2-klein-9b",
})


def parse_args() -> argparse.Namespace:
    """Parse FLUX fine-tuning CLI arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="FLUX full fine-tuning — production-grade trainer for FLUX.1 and FLUX.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )

    # Optional config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="FILE",
        help="OmegaConf YAML config file; CLI args take precedence over YAML values",
    )

    # Required args (only truly required when not running --estimate-memory or --print-config-only)
    required = parser.add_argument_group("required")
    required.add_argument(
        "--variant",
        type=str,
        choices=VARIANT_CHOICES,
        required=True,
        help=(
            "FLUX model variant. Distilled variants (schnell, flux2-klein-4b, "
            "flux2-klein-9b) are rejected unless --force-distilled is passed; "
            "for FLUX.2 full fine-tune use the *-base variants instead."
        ),
    )
    required.add_argument(
        "--pretrained-path",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to pretrained FLUX checkpoint (BFL safetensors or HF directory)",
    )
    required.add_argument(
        "--train-data",
        type=str,
        required=True,
        metavar="PATH",
        help="Training data: path to metadata.json or directory of paired images",
    )
    required.add_argument(
        "--output-dir",
        type=str,
        required=True,
        metavar="PATH",
        help="Directory for checkpoints, logs, and exported weights",
    )

    # Paired-dataset (Kontext) selection
    data = parser.add_argument_group("kontext dataset")
    data.add_argument(
        "--dataset-type",
        type=str,
        default="kontext",
        help=(
            "Paired-image dataset family (default: kontext). Built-in keys: "
            "'kontext' (metadata.json or *_target/_ref pairs) and 'orion' "
            "(BAO multi-patient H5 splits for FLUX.1-Kontext biomarker "
            "training). Extend via register_kontext_dataset(name, cls) — "
            "the registry is the source of truth (no argparse changes needed)."
        ),
    )
    data.add_argument(
        "--train-split",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to a dataset-specific training-split JSON. Used by "
            "--dataset-type=orion to point at e.g. BAO CD45_train.json. "
            "Falls back to ORION_TRAIN_SPLIT env var."
        ),
    )
    data.add_argument(
        "--biomarker",
        type=str,
        default=None,
        help=(
            "For --dataset-type=orion: target biomarker name "
            "(e.g. CD45). Inferred from the split filename if omitted."
        ),
    )

    # Resolution & batching
    res = parser.add_argument_group("resolution and batching")
    res.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Training image resolution (default: 1024)",
    )
    res.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-GPU batch size (default: 1)",
    )
    res.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )

    # Optimization
    opt = parser.add_argument_group("optimization")
    opt.add_argument(
        "--lr",
        type=float,
        default=1e-6,
        help="Learning rate (default: 1e-6, community default for full fine-tune)",
    )
    opt.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Maximum training steps; -1 = use --epochs (default: -1)",
    )
    opt.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)",
    )
    opt.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="LR warmup steps (default: 500)",
    )
    opt.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm (default: 1.0)",
    )

    # FLUX-specific
    flux = parser.add_argument_group("FLUX-specific")
    flux.add_argument(
        "--guidance-value",
        type=float,
        default=1.0,
        help="Fixed guidance value during training, dev variant only (default: 1.0)",
    )
    flux.add_argument(
        "--flow-shift",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply BFL-aligned time_shift to flow matching schedule (default: on)",
    )
    flux.add_argument(
        "--base-shift",
        type=float,
        default=0.5,
        help="Base shift for mu linear interpolation (default: 0.5)",
    )
    flux.add_argument(
        "--max-shift",
        type=float,
        default=1.15,
        help="Max shift for mu linear interpolation (default: 1.15)",
    )
    flux.add_argument(
        "--force-distilled",
        action="store_true",
        default=False,
        help=(
            "Bypass the distilled-variant guard. Covers FLUX.1-schnell AND "
            "FLUX.2-klein-{4b,9b} (distilled forms). Use at your own risk — "
            "full fine-tuning a distilled model loses the distillation gains "
            "without recovering score-matching capacity."
        ),
    )
    # Deprecated alias kept for backward compatibility — same dest as
    # --force-distilled so old job scripts keep working.
    flux.add_argument(
        "--force-schnell",
        dest="force_distilled",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,  # hidden; documented in CHANGELOG
    )

    # Memory & speed
    mem = parser.add_argument_group("memory and speed")
    mem.add_argument(
        "--bf16",
        dest="bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 mixed precision (default: on)",
    )
    mem.add_argument(
        "--fp32",
        dest="bf16",
        action="store_false",
        help="Disable bfloat16, use fp32",
    )
    mem.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Activation checkpointing on transformer blocks (default: on)",
    )
    mem.add_argument(
        "--use-8bit-adam",
        dest="use_8bit_adam",
        action="store_true",
        default=False,
        help="Enable bitsandbytes AdamW8bit optimizer (requires bitsandbytes)",
    )
    mem.add_argument(
        "--no-8bit-adam",
        dest="use_8bit_adam",
        action="store_false",
        help="Use 8-bit AdamW (bitsandbytes). Requires: pip install bitsandbytes>=0.43.0",
    )
    mem.add_argument(
        "--cpu-offload-optimizer",
        action="store_true",
        default=False,
        help="Offload optimizer state to CPU (slower but reduces GPU VRAM)",
    )
    mem.add_argument(
        "--xformers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable xformers memory-efficient attention (default: off; SDPA/Flash used instead)",
    )
    mem.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="torch.compile the transformer (experimental; default: off)",
    )

    # Distributed
    dist = parser.add_argument_group("distributed training")
    dist.add_argument(
        "--distributed-strategy",
        type=str,
        choices=["single", "fsdp"],
        default="single",
        help="Distributed training strategy (default: single)",
    )
    dist.add_argument(
        "--fsdp-cpu-offload",
        action="store_true",
        default=False,
        help="FSDP parameter CPU offload (reduces GPU VRAM, increases host memory)",
    )

    # EMA
    ema = parser.add_argument_group("EMA (exponential moving average)")
    ema.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="EMA decay; 0.0 disables EMA (default: 0.0; community default: 0.99)",
    )
    ema.add_argument(
        "--ema-on-cpu",
        action="store_true",
        default=False,
        help="Store EMA shadow weights on CPU to save GPU VRAM",
    )
    # Convenience aliases for test AC10 --no-ema
    ema.add_argument(
        "--no-ema",
        dest="ema_decay",
        action="store_const",
        const=0.0,
        help="Disable EMA (alias for --ema-decay 0.0)",
    )

    # Checkpointing
    ckpt = parser.add_argument_group("checkpointing")
    ckpt.add_argument(
        "--save-every-steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps (default: 1000)",
    )
    ckpt.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume training from checkpoint directory",
    )

    # Inference export
    exp = parser.add_argument_group("inference export")
    exp.add_argument(
        "--export-format",
        type=str,
        choices=["bfl", "diffusers", "both"],
        default="both",
        help="Checkpoint export format(s) at end of training (default: both)",
    )

    # Diagnostics
    diag = parser.add_argument_group("diagnostics")
    diag.add_argument(
        "--estimate-memory",
        action="store_true",
        default=False,
        help="Print memory plan table for the given configuration and exit without training",
    )
    diag.add_argument(
        "--print-config-only",
        action="store_true",
        default=False,
        help="Print the resolved config (after CLI override merge) and exit without training",
    )
    diag.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    return parser.parse_args()


def _variant_to_model_config(variant: str) -> dict:
    """Map CLI --variant to model config keys.

    Args:
        variant: CLI variant string.

    Returns:
        Dict with model.version, model.variant, and model.type keys.
    """
    mapping = {
        "dev": {"version": "v1", "variant": "dev", "type": "flux"},
        "kontext": {"version": "v1", "variant": "kontext", "type": "flux"},
        "schnell": {"version": "v1", "variant": "schnell", "type": "flux"},
        "flux2-dev": {"version": "v2", "variant": "dev", "type": "flux2"},
        "flux2-klein-4b": {"version": "v2", "variant": "klein-4b", "type": "flux2"},
        "flux2-klein-9b": {"version": "v2", "variant": "klein-9b", "type": "flux2"},
        "flux2-klein-4b-base": {"version": "v2", "variant": "klein-4b-base", "type": "flux2"},
        "flux2-klein-9b-base": {"version": "v2", "variant": "klein-9b-base", "type": "flux2"},
    }
    return mapping[variant]


def _build_config(args: argparse.Namespace) -> "OmegaConf.DictConfig":
    """Build OmegaConf DictConfig from parsed CLI args.

    Loads --config YAML first, then overrides with explicit CLI values.

    Args:
        args: Parsed argument namespace.

    Returns:
        Merged DictConfig ready for FluxFullFinetuneTrainer.
    """
    if args.config:
        base = load_config(args.config)
    else:
        base = OmegaConf.create({})

    model_keys = _variant_to_model_config(args.variant)
    mixed_precision = "bf16" if args.bf16 else "fp32"

    overrides = OmegaConf.create(
        {
            "experiment": {
                "name": f"flux-full-finetune-{args.variant}",
                "output_dir": args.output_dir,
                "seed": args.seed,
            },
            "model": {
                **model_keys,
                "pretrained_path": args.pretrained_path,
                "dtype": mixed_precision,
                "scheduler": {"type": "flow_matching"},
            },
            "training": {
                "method": "flux_full_finetune",
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "max_grad_norm": args.max_grad_norm,
                "warmup_steps": args.warmup_steps,
                "max_steps": args.max_steps,
                "scheduler": "cosine",
                "flow_shift": args.flow_shift,
                "base_shift": args.base_shift,
                "max_shift": args.max_shift,
                "guidance_value": args.guidance_value,
                "force_distilled": args.force_distilled,
                # Flat EMA keys matched to FluxFullFinetuneTrainer.__init__
                "ema_decay": args.ema_decay,
                "ema_on_cpu": args.ema_on_cpu,
                "optimizer": {
                    "type": "adamw",
                    "lr": args.lr,
                    "betas": [0.9, 0.999],
                    "weight_decay": 0.01,
                    "use_8bit_adam": args.use_8bit_adam,
                    "cpu_offload_optimizer": args.cpu_offload_optimizer,
                },
                "export": {
                    "formats": (
                        ["bfl", "diffusers"]
                        if args.export_format == "both"
                        else [args.export_format]
                    ),
                    "use_ema": args.ema_decay > 0.0,
                },
                "data": {
                    "train_path": args.train_data,
                    "resolution": args.resolution,
                    "dataset_type": args.dataset_type,
                    "train_split_path": args.train_split,
                    "biomarker": args.biomarker,
                },
                "save_every_steps": args.save_every_steps,
            },
            "hardware": {
                "mixed_precision": mixed_precision,
                "gradient_checkpointing": args.gradient_checkpointing,
                "distributed_strategy": args.distributed_strategy,
                "fsdp_cpu_offload": args.fsdp_cpu_offload,
                "xformers": args.xformers,
                "compile": args.compile,
            },
            "data": {
                "train_path": args.train_data,
                "resolution": args.resolution,
                "dataset_type": args.dataset_type,
                "train_split_path": args.train_split,
                "biomarker": args.biomarker,
            },
        }
    )

    # CLI args take precedence over YAML
    config = OmegaConf.merge(base, overrides)
    return config


def _print_memory_estimate(config: "OmegaConf.DictConfig") -> None:
    """Print memory plan table and verdict for the given config.

    Args:
        config: Resolved training config.
    """
    from src.training.flux_full_finetune_trainer import compute_memory_plan

    plan = compute_memory_plan(
        variant=config.model.variant,
        version=config.model.get("version", "v1"),
        batch_size=config.training.batch_size,
        resolution=config.data.get("resolution", 1024),
        mixed_precision=config.hardware.mixed_precision,
        use_8bit=config.training.optimizer.get("use_8bit_adam", False),
        use_ema=config.training.get("ema_decay", 0.0) > 0.0,
        ema_on_cpu=config.training.get("ema_on_cpu", False),
        distributed_strategy=config.hardware.get("distributed_strategy", "single"),
    )

    print("\nMemory Plan")
    print("=" * 60)
    for key, val in plan.items():
        if key == "verdict":
            continue
        gb = val / (1024**3)
        print(f"  {key:<35s} {gb:>7.2f} GB")
    print("-" * 60)
    print(f"  {'verdict':<35s} {plan['verdict']}")
    print("=" * 60)


def set_seed(seed: int) -> None:
    """Set random seed across torch, numpy, and random.

    Args:
        seed: Integer seed value.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    """Main entry point for FLUX full fine-tuning CLI."""
    args = parse_args()

    # Build merged config
    config = _build_config(args)

    # --print-config-only: dump resolved config and exit
    if args.print_config_only:
        print(OmegaConf.to_yaml(config))
        return

    # --estimate-memory: print plan table and exit
    if args.estimate_memory:
        _print_memory_estimate(config)
        return

    # Normal training path
    set_seed(config.experiment.seed)

    output_dir = Path(config.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    # Distilled-variant guard: fail fast before any heavy import.
    if args.variant in DISTILLED_VARIANTS_CLI and not args.force_distilled:
        logger.error(
            "'%s' is distilled; full fine-tuning is contraindicated. "
            "Pass --force-distilled to override at your own risk, or use a "
            "non-distilled variant: --variant dev (FLUX.1) or "
            "--variant flux2-klein-4b-base / flux2-klein-9b-base (FLUX.2).",
            args.variant,
        )
        sys.exit(1)

    logger.info(f"Variant:   {args.variant}")
    logger.info(f"Output:    {config.experiment.output_dir}")
    logger.info(f"Precision: {config.hardware.mixed_precision}")
    logger.info(f"Strategy:  {config.hardware.distributed_strategy}")

    from src.training.flux_full_finetune_trainer import FluxFullFinetuneTrainer

    trainer = FluxFullFinetuneTrainer(config, force_distilled=args.force_distilled)

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    logger.info("Starting training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer._save_checkpoint()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    logger.info(f"Training complete. Outputs: {config.experiment.output_dir}")


if __name__ == "__main__":
    main()
