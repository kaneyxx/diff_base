"""Exponential Moving Average (EMA) model shadow weights."""

from pathlib import Path

import torch
import torch.nn as nn

from ..utils.logging import get_logger

logger = get_logger(__name__)


class EMAModel:
    """Maintains an exponential moving average of model parameters.

    Shadow weights are updated after each optimizer step:
    ``shadow[name] = decay * shadow[name] + (1 - decay) * param``

    Args:
        model: Source model whose trainable parameters are tracked.
        decay: EMA decay rate (community default 0.99; higher = slower update).
        on_cpu: If True, shadow weights reside on CPU to save VRAM.
        dtype: Dtype for shadow weights (default float32 for numerical stability).

    Example::

        ema = EMAModel(model.transformer, decay=0.99, on_cpu=True)
        # After each optimizer step:
        ema.update(model.transformer)
        # Save EMA checkpoint:
        ema.save("/ckpt/ema.safetensors")
        # Load EMA weights into model for inference:
        ema.load_into(model.transformer)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.99,
        on_cpu: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")

        self.decay = decay
        self.on_cpu = on_cpu
        self.dtype = dtype
        self.shadow: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                shadow = param.detach().clone().to(dtype)
                if on_cpu:
                    shadow = shadow.cpu()
                self.shadow[name] = shadow

        logger.info(
            f"EMAModel initialized: decay={decay}, on_cpu={on_cpu}, "
            f"params={len(self.shadow)}"
        )

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights from current model parameters.

        ``shadow = decay * shadow + (1 - decay) * param``

        When used with FSDP, call inside
        ``FullyShardedDataParallel.summon_full_params(model)`` context so that
        full (unsharded) parameters are visible.

        Args:
            model: Model with updated parameters (after optimizer step).
        """
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            ema_p = self.shadow[name]
            p = param.detach().to(ema_p.dtype)
            if self.on_cpu:
                p = p.cpu()
            ema_p.mul_(self.decay).add_(p, alpha=1.0 - self.decay)

    def save(self, path: str | Path) -> None:
        """Save shadow weights to a safetensors file.

        Args:
            path: Output file path (should end in ``.safetensors``).
        """
        from safetensors.torch import save_file

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.cpu() for k, v in self.shadow.items()}, str(path))
        logger.info(f"EMA shadow weights saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load shadow weights from a safetensors file into this EMAModel.

        Args:
            path: Path to a safetensors file previously written by ``save()``.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        from safetensors.torch import load_file

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"EMA checkpoint not found: {path}")

        loaded = load_file(str(path))
        for name, tensor in loaded.items():
            t = tensor.to(self.dtype)
            if self.on_cpu:
                t = t.cpu()
            self.shadow[name] = t

        logger.info(f"EMA shadow weights loaded from {path} ({len(loaded)} params)")

    def load_into(self, model: nn.Module) -> None:
        """Copy shadow weights into a model's parameters (in-place).

        Args:
            model: Target model; its trainable parameters are overwritten with
                the EMA shadow values.
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    param.copy_(
                        self.shadow[name].to(device=param.device, dtype=param.dtype)
                    )
        logger.info("EMA shadow weights loaded into model")

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a copy of the shadow weight dict (for serialization)."""
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore shadow weights from a previously saved state dict.

        Args:
            state: Dict of ``{name: tensor}`` as returned by ``state_dict()``.
        """
        for name, tensor in state.items():
            t = tensor.to(self.dtype)
            if self.on_cpu:
                t = t.cpu()
            self.shadow[name] = t
