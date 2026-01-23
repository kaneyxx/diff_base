"""Base trainer class for all training methods."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from tqdm import tqdm

from ..models import create_model
from ..data import create_dataloader
from ..schedulers import create_scheduler
from ..utils.config import config_to_dict
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..utils.logging import TrainingLogger, MetricsTracker, get_logger
from ..utils.memory import optimize_memory, get_memory_stats, clear_memory

logger = get_logger(__name__)


class BaseTrainer(ABC):
    """Abstract base trainer for all training methods.

    Provides common training infrastructure including:
    - Model creation and optimization
    - Data loading
    - Training loop with gradient accumulation
    - Checkpointing
    - Logging
    """

    def __init__(self, config: DictConfig):
        """Initialize trainer.

        Args:
            config: Full training configuration.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Parse dtype
        dtype_str = config.model.get("dtype", "float32")
        self.dtype = self._parse_dtype(dtype_str)

        # Training state
        self.global_step = 0
        self.current_epoch = 0

        # Initialize components
        logger.info("Initializing model...")
        self.model = self._setup_model()

        logger.info("Initializing scheduler...")
        self.noise_scheduler = self._setup_noise_scheduler()

        logger.info("Initializing optimizer...")
        self.optimizer = self._setup_optimizer()

        logger.info("Initializing LR scheduler...")
        self.lr_scheduler = self._setup_lr_scheduler()

        logger.info("Initializing dataloader...")
        self.dataloader = self._setup_dataloader()

        # Logging
        self.training_logger = TrainingLogger(config)
        self.metrics_tracker = MetricsTracker()

        logger.info(f"Trainer initialized on {self.device}")

    def _parse_dtype(self, dtype_str: str) -> torch.dtype:
        """Parse dtype string to torch.dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.float32)

    def _setup_model(self) -> nn.Module:
        """Create and prepare model for training.

        Returns:
            Prepared model.
        """
        model = create_model(self.config)

        # Apply memory optimizations
        model = optimize_memory(model, self.config, self.device)

        # Move to device
        model = model.to(self.device)

        return model

    def _setup_noise_scheduler(self):
        """Create noise scheduler.

        Returns:
            Noise scheduler instance.
        """
        scheduler_config = self.config.model.get("scheduler", {})
        return create_scheduler(scheduler_config)

    @abstractmethod
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer - implemented by subclasses.

        Returns:
            Configured optimizer.
        """
        pass

    def _setup_lr_scheduler(self) -> Optional[Any]:
        """Setup learning rate scheduler.

        Returns:
            LR scheduler or None.
        """
        sched_config = self.config.training.get("lr_scheduler", {})
        sched_type = sched_config.get("type", "constant")

        if sched_type == "constant":
            return torch.optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=1.0
            )
        elif sched_type == "cosine":
            total_steps = self._get_total_steps()
            warmup_steps = sched_config.get("warmup_steps", 0)
            min_lr_ratio = sched_config.get("min_lr_ratio", 0.0)

            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.config.training.learning_rate * min_lr_ratio,
            )
        elif sched_type == "linear":
            total_steps = self._get_total_steps()
            warmup_steps = sched_config.get("warmup_steps", 0)

            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=sched_config.get("end_factor", 0.0),
                total_iters=total_steps - warmup_steps,
            )
        else:
            logger.warning(f"Unknown scheduler type: {sched_type}, using constant")
            return None

    def _get_total_steps(self) -> int:
        """Calculate total training steps."""
        steps_per_epoch = len(self.dataloader) // self.config.training.get(
            "gradient_accumulation", 1
        )
        return steps_per_epoch * self.config.training.epochs

    def _setup_dataloader(self) -> DataLoader:
        """Setup data loading pipeline.

        Returns:
            Configured DataLoader.
        """
        return create_dataloader(self.config)

    @abstractmethod
    def training_step(self, batch: dict) -> torch.Tensor:
        """Single training step - returns loss.

        Args:
            batch: Batch of training data.

        Returns:
            Loss tensor.
        """
        pass

    def train(self) -> None:
        """Main training loop."""
        self.model.train()

        grad_accum = self.config.training.get("gradient_accumulation", 1)
        max_grad_norm = self.config.training.get("max_grad_norm", 1.0)
        save_every = self.config.training.get("save_every_n_epochs", 1)

        logger.info(f"Starting training for {self.config.training.epochs} epochs")
        logger.info(f"Gradient accumulation: {grad_accum}")
        logger.info(f"Effective batch size: {self.config.training.batch_size * grad_accum}")

        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0

            progress = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.training.epochs}",
            )

            for step, batch in enumerate(progress):
                # Move batch to device
                batch = self._prepare_batch(batch)

                # Forward pass with mixed precision
                with torch.autocast(
                    device_type="cuda",
                    dtype=self.dtype,
                    enabled=self.dtype != torch.float32,
                ):
                    loss = self.training_step(batch)
                    loss = loss / grad_accum

                # Backward pass
                loss.backward()

                epoch_loss += loss.item() * grad_accum

                # Optimizer step
                if (step + 1) % grad_accum == 0:
                    # Gradient clipping
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self._get_trainable_params(),
                            max_norm=max_grad_norm,
                        )

                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # Logging
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.metrics_tracker.update({
                        "loss": loss.item() * grad_accum,
                        "lr": current_lr,
                    })

                    if self.global_step % 10 == 0:
                        self.training_logger.log({
                            "loss": self.metrics_tracker.get_average("loss"),
                            "lr": current_lr,
                            "step": self.global_step,
                            "epoch": epoch,
                        }, step=self.global_step)

                progress.set_postfix(
                    loss=loss.item() * grad_accum,
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                )

            # Epoch complete
            avg_loss = epoch_loss / len(self.dataloader)
            logger.info(f"Epoch {epoch + 1} complete. Average loss: {avg_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint()

        # Final save
        self._save_checkpoint(final=True)
        self.training_logger.finish()

        logger.info("Training complete!")

    def _prepare_batch(self, batch: dict) -> dict:
        """Move batch tensors to device.

        Args:
            batch: Batch dictionary.

        Returns:
            Batch with tensors on device.
        """
        prepared = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self.device, dtype=self.dtype)
            else:
                prepared[key] = value
        return prepared

    def _get_trainable_params(self) -> list[nn.Parameter]:
        """Get trainable parameters.

        Returns:
            List of trainable parameters.
        """
        return [p for p in self.model.parameters() if p.requires_grad]

    def _save_checkpoint(self, final: bool = False) -> None:
        """Save training checkpoint.

        Args:
            final: Whether this is the final checkpoint.
        """
        output_dir = Path(self.config.experiment.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if final:
            checkpoint_dir = output_dir / "final"
        else:
            checkpoint_dir = output_dir / f"checkpoint-{self.global_step}"

        save_checkpoint(
            path=checkpoint_dir,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            step=self.global_step,
            epoch=self.current_epoch,
            config=self.config,
        )

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint.
        """
        result = load_checkpoint(
            path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            device=self.device,
        )

        self.global_step = result.get("step", 0)
        self.current_epoch = result.get("epoch", 0)

        logger.info(f"Resumed from step {self.global_step}, epoch {self.current_epoch}")

    def _add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to latents using scheduler.

        Args:
            latents: Clean latents.
            noise: Noise to add.
            timesteps: Timesteps.

        Returns:
            Noisy latents.
        """
        return self.noise_scheduler.add_noise(latents, noise, timesteps)

    def _compute_snr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute SNR-based loss weights.

        Args:
            timesteps: Timesteps.

        Returns:
            Weight tensor.
        """
        snr_gamma = self.config.training.loss.get("snr_gamma")
        if snr_gamma is None:
            return torch.ones_like(timesteps, dtype=self.dtype)

        snr = self.noise_scheduler.get_snr(timesteps)
        weights = torch.clamp(snr, max=snr_gamma) / snr
        return weights.to(self.dtype)
