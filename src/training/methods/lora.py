"""LoRA (Low-Rank Adaptation) implementation using peft library."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ...utils.logging import get_logger

logger = get_logger(__name__)


def inject_lora_layers(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[list[str]] = None,
    use_peft: bool = True,
) -> nn.Module:
    """Inject LoRA layers into model.

    Args:
        model: Model to inject LoRA into.
        rank: LoRA rank (r).
        alpha: LoRA alpha scaling.
        dropout: Dropout probability.
        target_modules: List of module name patterns to target.
        use_peft: Whether to use peft library (recommended).

    Returns:
        Model with LoRA layers.
    """
    if target_modules is None:
        # Default targets for diffusion models
        target_modules = [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
        ]

    if use_peft:
        return _inject_peft_lora(model, rank, alpha, dropout, target_modules)
    else:
        return _inject_manual_lora(model, rank, alpha, dropout, target_modules)


def _inject_peft_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: list[str],
) -> nn.Module:
    """Inject LoRA using peft library."""
    try:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type=None,
        )

        model = get_peft_model(model, lora_config)

        # Log LoRA info
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"LoRA injected. Trainable: {trainable:,} / {total:,} "
                   f"({100 * trainable / total:.2f}%)")

        return model

    except ImportError:
        logger.warning("peft not installed, falling back to manual LoRA")
        return _inject_manual_lora(model, rank, alpha, dropout, target_modules)


def _inject_manual_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: list[str],
) -> nn.Module:
    """Inject LoRA manually without peft."""
    import math

    class LoRALinear(nn.Module):
        """Linear layer with LoRA."""

        def __init__(
            self,
            base_layer: nn.Linear,
            r: int,
            lora_alpha: float,
            lora_dropout: float,
        ):
            super().__init__()
            self.base_layer = base_layer
            self.r = r
            self.lora_alpha = lora_alpha
            self.scaling = lora_alpha / r

            in_features = base_layer.in_features
            out_features = base_layer.out_features

            # LoRA layers
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.lora_dropout = nn.Dropout(lora_dropout)

            # Initialize
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

            # Freeze base layer
            for param in self.base_layer.parameters():
                param.requires_grad = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            base_out = self.base_layer(x)
            lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            return base_out + lora_out

    def replace_module(parent: nn.Module, name: str, module: nn.Module) -> None:
        """Replace a module in parent."""
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module)

    # Find and replace target modules
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if any(target in name for target in target_modules):
                lora_module = LoRALinear(module, rank, alpha, dropout)
                replace_module(model, name, lora_module)
                replaced += 1

    logger.info(f"Replaced {replaced} modules with LoRA")

    return model


def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Get only LoRA parameters for optimizer.

    Args:
        model: Model with LoRA layers.

    Returns:
        List of LoRA parameters.
    """
    params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            # Check if it's a LoRA parameter
            if any(lora_key in name.lower() for lora_key in ["lora_", "lora."]):
                params.append(param)
            elif param.requires_grad:
                # Include any other trainable params (shouldn't be many)
                params.append(param)

    logger.info(f"Found {len(params)} LoRA parameters")
    return params


def save_lora_weights(
    model: nn.Module,
    path: str | Path,
    config: Optional[DictConfig] = None,
) -> None:
    """Save only LoRA weights.

    Args:
        model: Model with LoRA.
        path: Output path.
        config: Training config for metadata.
    """
    from safetensors.torch import save_file
    import json

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Check if peft model
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(path)
        logger.info(f"Saved LoRA weights (peft format) to {path}")
        return

    # Manual extraction
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if "lora" in name.lower() and param.requires_grad:
            lora_state_dict[name] = param.data.cpu()

    if lora_state_dict:
        save_file(lora_state_dict, path / "adapter_model.safetensors")

        # Save config
        if config is not None:
            adapter_config = {
                "r": config.training.lora.get("rank", 16),
                "lora_alpha": config.training.lora.get("alpha", 16),
                "lora_dropout": config.training.lora.get("dropout", 0.0),
                "target_modules": list(config.training.lora.get("target_modules", [])),
                "bias": "none",
            }
            with open(path / "adapter_config.json", "w") as f:
                json.dump(adapter_config, f, indent=2)

        logger.info(f"Saved {len(lora_state_dict)} LoRA tensors to {path}")


def load_lora_weights(
    model: nn.Module,
    path: str | Path,
    device: torch.device | str = "cuda",
) -> nn.Module:
    """Load LoRA weights into model.

    Args:
        model: Model with LoRA layers.
        path: Path to saved weights.
        device: Target device.

    Returns:
        Model with loaded weights.
    """
    from safetensors.torch import load_file

    path = Path(path)

    # Check if peft model
    if hasattr(model, "load_adapter"):
        model.load_adapter(path, "default")
        logger.info(f"Loaded LoRA weights (peft format) from {path}")
        return model

    # Manual loading
    if (path / "adapter_model.safetensors").exists():
        lora_state = load_file(path / "adapter_model.safetensors", device=str(device))
    else:
        raise FileNotFoundError(f"LoRA weights not found at {path}")

    # Load into model
    model_state = model.state_dict()
    for key, value in lora_state.items():
        if key in model_state:
            model_state[key] = value
        else:
            logger.warning(f"Key not found in model: {key}")

    model.load_state_dict(model_state, strict=False)
    logger.info(f"Loaded {len(lora_state)} LoRA tensors from {path}")

    return model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into base model weights.

    Args:
        model: Model with LoRA.

    Returns:
        Model with merged weights.
    """
    if hasattr(model, "merge_and_unload"):
        # peft model
        return model.merge_and_unload()

    # Manual merge
    for name, module in model.named_modules():
        if hasattr(module, "base_layer") and hasattr(module, "lora_A"):
            # Merge LoRA into base
            weight = module.base_layer.weight.data
            lora_weight = (
                module.lora_B.weight @ module.lora_A.weight
            ) * module.scaling

            module.base_layer.weight.data = weight + lora_weight

    logger.info("Merged LoRA weights into base model")
    return model
