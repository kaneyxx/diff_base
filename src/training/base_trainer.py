"""Base trainer class for all training methods."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import create_dataloader
from ..models import create_model
from ..schedulers import create_scheduler
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.logging import MetricsTracker, TrainingLogger, get_logger
from ..utils.memory import optimize_memory

try:  # pragma: no cover — optional import guard
    from accelerate import Accelerator, DistributedDataParallelKwargs

    _ACCELERATE_AVAILABLE = True
except ImportError:  # pragma: no cover
    Accelerator = None  # type: ignore[misc,assignment]
    DistributedDataParallelKwargs = None  # type: ignore[misc,assignment]
    _ACCELERATE_AVAILABLE = False

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

        # Parse dtype
        dtype_str = config.model.get("dtype", "float32")
        self.dtype = self._parse_dtype(dtype_str)

        # Initialise HuggingFace Accelerator (no-op wrapper for single-GPU runs).
        # When invoked via `accelerate launch --num_processes N`, each rank's
        # Accelerator picks the correct device via LOCAL_RANK. Without launch,
        # this falls back to a single-process Accelerator on the default cuda
        # device (or CPU if no GPU is available).
        self.accelerator = self._setup_accelerator()
        self.device = (
            self.accelerator.device
            if self.accelerator is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

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

        # Dataloader must come before lr_scheduler — _get_total_steps reads
        # len(self.dataloader) when computing the schedule.
        logger.info("Initializing dataloader...")
        self.dataloader = self._setup_dataloader()

        logger.info("Initializing LR scheduler...")
        self.lr_scheduler = self._setup_lr_scheduler()

        # Wrap optimizer + dataloader + lr_scheduler under Accelerator.
        # Under `accelerate launch --num_processes>1`, prepare() returns a
        # DistributedDataParallel-wrapped model and a sharded dataloader.
        # For single-process runs it is effectively a passthrough.
        #
        # Only the trainable submodule is handed to prepare(). Passing the
        # full model wrapper (FluxModel containing transformer + VAE +
        # text_encoder) breaks DDP when text_encoder.cpu_offload=true,
        # because DDP requires every parameter on the same device class.
        # VAE and text_encoder are frozen and accessed directly via
        # self.model.{vae, text_encoder} — they do not need gradient sync.
        if self.accelerator is not None:
            trainable_module = getattr(self.model, "transformer", self.model)
            prepared_module, self.optimizer, self.dataloader, self.lr_scheduler = (
                self.accelerator.prepare(
                    trainable_module,
                    self.optimizer,
                    self.dataloader,
                    self.lr_scheduler,
                )
            )
            if hasattr(self.model, "transformer"):
                self.model.transformer = prepared_module
            else:
                self.model = prepared_module

        # Logging
        self.training_logger = TrainingLogger(config)
        self.metrics_tracker = MetricsTracker()

        logger.info(f"Trainer initialized on {self.device}")

    # ------------------------------------------------------------------
    # Accelerator helpers
    # ------------------------------------------------------------------

    def _setup_accelerator(self):
        """Instantiate the HuggingFace Accelerator if available.

        Honours ``training.use_accelerator`` (default ``True``). When the
        accelerate package is not installed or the user explicitly disables
        it, returns ``None`` and the trainer falls back to the legacy
        single-GPU path.

        Mixed precision and gradient accumulation are wired from the
        existing YAML schema, so no new top-level config keys are required.
        """
        use_accelerator = self.config.training.get("use_accelerator", True)
        if not use_accelerator:
            logger.info("Accelerator disabled via config (use_accelerator=false)")
            return None
        if not _ACCELERATE_AVAILABLE:
            logger.warning(
                "accelerate not installed; falling back to single-GPU trainer. "
                "Install `accelerate` to enable multi-GPU launch."
            )
            return None

        # `hardware` section is optional in some test fixtures; tolerate absence.
        hardware_cfg = self.config.get("hardware", {}) if hasattr(
            self.config, "get"
        ) else {}
        mixed_precision = (
            hardware_cfg.get("mixed_precision", "no")
            if hasattr(hardware_cfg, "get")
            else "no"
        )
        # Accelerator expects literal strings: "no", "fp16", "bf16", "fp8".
        # Map common aliases used elsewhere in the config to those literals.
        mp_alias = {
            "bfloat16": "bf16",
            "bf16": "bf16",
            "float16": "fp16",
            "fp16": "fp16",
            "fp8": "fp8",
            "no": "no",
            "none": "no",
            "float32": "no",
            "fp32": "no",
        }
        mp = mp_alias.get(str(mixed_precision).lower(), "no")
        grad_accum = int(self.config.training.get("gradient_accumulation", 1))

        # LoRA + frozen-base training has many parameters whose grads are
        # never touched in a given step; DDP needs `find_unused_parameters`
        # set so the all-reduce skip-list is computed dynamically each step.
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            mixed_precision=mp,
            gradient_accumulation_steps=max(1, grad_accum),
            kwargs_handlers=[ddp_kwargs],
        )
        logger.info(
            "Accelerator initialised: device=%s, num_processes=%d, "
            "mixed_precision=%s, gradient_accumulation=%d",
            accelerator.device,
            accelerator.num_processes,
            mp,
            grad_accum,
        )
        return accelerator

    @property
    def is_main_process(self) -> bool:
        """True on the rank-0 process, or always when Accelerator is disabled."""
        if self.accelerator is None:
            return True
        return self.accelerator.is_main_process

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

        # Move to device. When an Accelerator is active, prepare() will move
        # the trainable parameters and wrap in DDP later. Calling .to(device)
        # here is still correct for the frozen VAE/text-encoder parameters
        # (these are not handed to prepare()).
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

    def _setup_lr_scheduler(self) -> Any | None:
        """Setup learning rate scheduler.

        Returns:
            LR scheduler or None.

        Note on warmup:
            When warmup_steps > 0 a LinearLR warmup phase is prepended via
            SequentialLR.  warmup_steps is clamped to at most total_steps-1 so
            that degenerate smoke runs (max_steps < warmup_steps) still work.
        """
        sched_config = self.config.training.get("lr_scheduler", {})
        sched_type = sched_config.get("type", "constant")
        warmup_steps = int(sched_config.get("warmup_steps", 0))

        def _with_warmup(main_scheduler, total_steps: int, warmup_steps: int):
            """Wrap main_scheduler with a linear warmup phase if requested."""
            if warmup_steps <= 0:
                return main_scheduler
            # Clamp so smoke runs (total_steps < warmup_steps) don't break.
            warmup_steps = min(warmup_steps, max(1, total_steps - 1))
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, main_scheduler],
                milestones=[warmup_steps],
            )

        if sched_type == "constant":
            return torch.optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=1.0
            )
        elif sched_type == "cosine":
            total_steps = self._get_total_steps()
            min_lr_ratio = sched_config.get("min_lr_ratio", 0.0)
            # Cosine phase covers the steps after warmup.
            cosine_steps = max(1, total_steps - min(warmup_steps, max(1, total_steps - 1)))
            main = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cosine_steps,
                eta_min=self.config.training.learning_rate * min_lr_ratio,
            )
            return _with_warmup(main, total_steps, warmup_steps)
        elif sched_type == "linear":
            total_steps = self._get_total_steps()
            effective_warmup = min(warmup_steps, max(1, total_steps - 1)) if warmup_steps > 0 else 0
            decay_steps = max(1, total_steps - effective_warmup)
            main = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=sched_config.get("end_factor", 0.0),
                total_iters=decay_steps,
            )
            return _with_warmup(main, total_steps, warmup_steps)
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

        max_steps = self.config.training.get("max_steps", -1)
        if self.is_main_process:
            logger.info(f"Starting training for {self.config.training.epochs} epochs")
            logger.info(f"Gradient accumulation: {grad_accum}")
            logger.info(f"Effective batch size: {self.config.training.batch_size * grad_accum}")
            if max_steps > 0:
                logger.info(f"max_steps={max_steps} — will stop after {max_steps} optimizer steps")

        done = False
        for epoch in range(self.config.training.epochs):
            if done:
                break
            self.current_epoch = epoch
            epoch_loss = 0.0

            progress = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.training.epochs}",
                disable=not self.is_main_process,
            )

            for step, batch in enumerate(progress):
                if max_steps > 0 and self.global_step >= max_steps:
                    done = True
                    break
                # Move batch to device (accelerate.prepare() will already have
                # produced device-aware tensors when iterating its dataloader,
                # but _prepare_batch is also called by non-accelerate paths).
                batch = self._prepare_batch(batch)

                # Forward + backward. When Accelerator is active we route
                # autocast through ``self.accelerator.autocast()`` so the
                # context honours the mixed_precision value the Accelerator
                # was constructed with. Without Accelerator we fall back to
                # the existing manual ``torch.autocast`` block.
                if self.accelerator is not None:
                    autocast_ctx = self.accelerator.autocast()
                else:
                    autocast_ctx = torch.autocast(
                        device_type="cuda",
                        dtype=self.dtype,
                        enabled=self.dtype != torch.float32,
                    )
                with autocast_ctx:
                    loss = self.training_step(batch)
                    loss = loss / grad_accum

                # Backward pass — route through Accelerator when present so
                # gradient scaling + DDP all-reduce work correctly.
                if self.accelerator is not None:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()

                epoch_loss += loss.item() * grad_accum

                # Optimizer step
                if (step + 1) % grad_accum == 0:
                    # Gradient clipping — Accelerator's helper performs the
                    # correct unscale-then-clip ordering under mixed precision.
                    if max_grad_norm > 0:
                        if self.accelerator is not None:
                            self.accelerator.clip_grad_norm_(
                                self._get_trainable_params(),
                                max_norm=max_grad_norm,
                            )
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self._get_trainable_params(),
                                max_norm=max_grad_norm,
                            )

                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # Logging (rank-0 only)
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    if self.is_main_process:
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

                if self.is_main_process:
                    progress.set_postfix(
                        loss=loss.item() * grad_accum,
                        lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    )

            # Epoch complete
            avg_loss = epoch_loss / len(self.dataloader)
            if self.is_main_process:
                logger.info(f"Epoch {epoch + 1} complete. Average loss: {avg_loss:.4f}")

            # Save checkpoint (rank-0 only). Use accelerator.wait_for_everyone()
            # to ensure all ranks finish the epoch before rank-0 writes to disk.
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
            if (epoch + 1) % save_every == 0 and self.is_main_process:
                self._save_checkpoint()

        # Final save — ensure all ranks have finished training first.
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        if self.is_main_process:
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

    def _unwrap_model(self) -> nn.Module:
        """Unwrap DDP / Accelerator wrappers to access the underlying model.

        Returns the original ``nn.Module`` regardless of whether
        ``self.model`` has been wrapped in DistributedDataParallel via
        ``accelerator.prepare()``. Safe to call on single-process runs and
        on trainers that pre-date Accelerator integration (legacy tests
        instantiate subclasses without going through ``BaseTrainer.__init__``).
        """
        accel = getattr(self, "accelerator", None)
        if accel is not None:
            return accel.unwrap_model(self.model)
        return self.model

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
            model=self._unwrap_model(),
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
