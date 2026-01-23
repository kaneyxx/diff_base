"""Checkpoint saving and loading utilities."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file, load_file

from .logging import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    step: int = 0,
    epoch: int = 0,
    config: DictConfig | None = None,
    format: str = "safetensors",
    save_optimizer: bool = True,
) -> None:
    """Save training checkpoint.

    Args:
        path: Output directory or file path.
        model: Model to save.
        optimizer: Optimizer state to save.
        scheduler: LR scheduler state to save.
        step: Current training step.
        epoch: Current epoch.
        config: Training configuration.
        format: Save format ("safetensors", "pytorch", or "both").
        save_optimizer: Whether to save optimizer state.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Get model state dict
    if hasattr(model, "module"):
        # Handle DataParallel/DistributedDataParallel
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    # Save model weights
    if format in ("safetensors", "both"):
        save_file(model_state, path / "model.safetensors")
        logger.info(f"Saved model to {path / 'model.safetensors'}")

    if format in ("pytorch", "both"):
        torch.save(model_state, path / "model.pt")
        logger.info(f"Saved model to {path / 'model.pt'}")

    # Save training state
    training_state = {
        "step": step,
        "epoch": epoch,
    }

    if save_optimizer and optimizer is not None:
        training_state["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        training_state["scheduler"] = scheduler.state_dict()

    torch.save(training_state, path / "training_state.pt")

    # Save config
    if config is not None:
        OmegaConf.save(config, path / "config.yaml")

    logger.info(f"Checkpoint saved to {path} at step {step}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: torch.device | str = "cuda",
    strict: bool = True,
) -> dict[str, Any]:
    """Load training checkpoint.

    Args:
        path: Checkpoint directory or file path.
        model: Model to load weights into.
        optimizer: Optimizer to load state into.
        scheduler: Scheduler to load state into.
        device: Device to map tensors to.
        strict: Whether to strictly enforce state dict matching.

    Returns:
        Dictionary with checkpoint metadata (step, epoch, config).
    """
    path = Path(path)

    if path.is_file():
        checkpoint_dir = path.parent
        model_file = path
    else:
        checkpoint_dir = path
        # Try safetensors first, then pytorch
        if (path / "model.safetensors").exists():
            model_file = path / "model.safetensors"
        elif (path / "model.pt").exists():
            model_file = path / "model.pt"
        else:
            raise FileNotFoundError(f"No model file found in {path}")

    result = {}

    # Load model weights
    if model is not None:
        if model_file.suffix == ".safetensors":
            state_dict = load_file(model_file, device=str(device))
        else:
            state_dict = torch.load(model_file, map_location=device)

        # Handle DataParallel/DistributedDataParallel
        if hasattr(model, "module"):
            model.module.load_state_dict(state_dict, strict=strict)
        else:
            model.load_state_dict(state_dict, strict=strict)

        logger.info(f"Loaded model from {model_file}")

    # Load training state
    training_state_path = checkpoint_dir / "training_state.pt"
    if training_state_path.exists():
        training_state = torch.load(training_state_path, map_location=device)

        result["step"] = training_state.get("step", 0)
        result["epoch"] = training_state.get("epoch", 0)

        if optimizer is not None and "optimizer" in training_state:
            optimizer.load_state_dict(training_state["optimizer"])
            logger.info("Loaded optimizer state")

        if scheduler is not None and "scheduler" in training_state:
            scheduler.load_state_dict(training_state["scheduler"])
            logger.info("Loaded scheduler state")

    # Load config
    config_path = checkpoint_dir / "config.yaml"
    if config_path.exists():
        result["config"] = OmegaConf.load(config_path)

    return result


def save_lora_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: DictConfig | None = None,
) -> None:
    """Save only LoRA weights from a model.

    Args:
        path: Output path.
        model: Model with LoRA layers.
        config: Training configuration.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Extract LoRA weights
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            lora_state[name] = param.data

    if not lora_state:
        logger.warning("No LoRA weights found in model")
        return

    save_file(lora_state, path / "adapter_model.safetensors")
    logger.info(f"Saved {len(lora_state)} LoRA parameters to {path}")

    # Save config for compatibility with peft
    if config is not None:
        lora_config = {
            "r": config.training.lora.get("rank", 16),
            "lora_alpha": config.training.lora.get("alpha", 16),
            "target_modules": list(config.training.lora.get("target_modules", [])),
            "lora_dropout": config.training.lora.get("dropout", 0.0),
        }
        import json
        with open(path / "adapter_config.json", "w") as f:
            json.dump(lora_config, f, indent=2)


def load_lora_checkpoint(
    path: str | Path,
    model: nn.Module,
    device: torch.device | str = "cuda",
) -> None:
    """Load LoRA weights into a model.

    Args:
        path: Path to LoRA checkpoint.
        model: Model with LoRA layers to load into.
        device: Device to map tensors to.
    """
    path = Path(path)

    if path.is_dir():
        lora_path = path / "adapter_model.safetensors"
    else:
        lora_path = path

    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA weights not found at {lora_path}")

    lora_state = load_file(lora_path, device=str(device))

    # Load into model
    model_state = model.state_dict()
    for key, value in lora_state.items():
        if key in model_state:
            model_state[key] = value
        else:
            logger.warning(f"Key {key} not found in model")

    model.load_state_dict(model_state, strict=False)
    logger.info(f"Loaded LoRA weights from {lora_path}")


def convert_diffusers_to_safetensors(
    diffusers_path: str | Path,
    output_path: str | Path,
    component: str = "unet",
) -> None:
    """Convert diffusers checkpoint to single safetensors file.

    Args:
        diffusers_path: Path to diffusers model directory.
        output_path: Output safetensors file path.
        component: Component to extract (unet, vae, text_encoder).
    """
    diffusers_path = Path(diffusers_path)
    output_path = Path(output_path)

    component_path = diffusers_path / component
    if not component_path.exists():
        raise FileNotFoundError(f"Component {component} not found in {diffusers_path}")

    # Find safetensors or bin file
    safetensors_file = component_path / "diffusion_pytorch_model.safetensors"
    bin_file = component_path / "diffusion_pytorch_model.bin"

    if safetensors_file.exists():
        state_dict = load_file(safetensors_file)
    elif bin_file.exists():
        state_dict = torch.load(bin_file, map_location="cpu")
    else:
        raise FileNotFoundError(f"No weights file found in {component_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, output_path)
    logger.info(f"Converted {component} to {output_path}")
