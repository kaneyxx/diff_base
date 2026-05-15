"""FLUX full fine-tune trainer with BFL-aligned flow matching, EMA, and 8-bit Adam."""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from omegaconf import DictConfig

from ..utils.logging import get_logger
from .ema import EMAModel
from .finetune_trainer import FullFinetuneTrainer
from .memory_planning import MemoryPlan
from .memory_planning import compute_memory_plan as _compute_memory_plan

logger = get_logger(__name__)

# Parameter counts for known FLUX variants (approximate)
_VARIANT_PARAMS: dict[str, int] = {
    "dev": 12_000_000_000,
    "kontext": 12_000_000_000,
    "schnell": 12_000_000_000,
    "klein-4b": 4_000_000_000,
    "klein-9b": 9_000_000_000,
}


def compute_memory_plan(
    variant: str = "dev",
    version: str = "v1",
    batch_size: int = 1,
    resolution: int = 1024,
    mixed_precision: str = "bf16",
    use_8bit: bool = False,
    use_ema: bool = False,
    ema_on_cpu: bool = True,
    distributed_strategy: str = "single",
) -> dict:
    """Estimate GPU memory requirements for FLUX full fine-tuning.

    Adapter over ``src.training.memory_planning.compute_memory_plan`` using
    the CLI-facing argument names (variant, version, mixed_precision, use_8bit).

    Args:
        variant: FLUX variant name (dev, kontext, schnell, klein-4b, klein-9b).
        version: Model version ("v1" or "v2").
        batch_size: Per-GPU batch size.
        resolution: Image resolution in pixels (square assumed).
        mixed_precision: "bf16" or "fp32".
        use_8bit: Whether 8-bit AdamW is used.
        use_ema: Whether EMA shadow weights are enabled.
        ema_on_cpu: Whether EMA shadow resides on CPU.
        distributed_strategy: "single" or "fsdp".

    Returns:
        Dict with byte-valued memory components and a "verdict" string.
        Keys: weights, gradients, optimizer, activations, ema_gpu, total, verdict.
    """
    num_params = _VARIANT_PARAMS.get(variant, 12_000_000_000)
    use_bf16 = mixed_precision == "bf16"
    gradient_checkpointing = True  # default-on in CLI

    plan: MemoryPlan = _compute_memory_plan(
        num_params=num_params,
        batch_size=batch_size,
        resolution=resolution,
        use_bf16=use_bf16,
        use_8bit_adam=use_8bit,
        gradient_checkpointing=gradient_checkpointing,
        ema_enabled=use_ema,
        ema_on_cpu=ema_on_cpu,
        distributed_strategy=distributed_strategy,
    )

    # Return dict with byte values using AC14 spec key names
    gb = 1024 ** 3
    verdict_map = {
        "fits_single_h100": "fits-80gb-single",
        "fits_4x_h100_fsdp": "fits-multi-gpu-fsdp",
        "needs_cpu_offload": "requires-cpu-offload",
        "infeasible_single_gpu": "infeasible",
    }
    return {
        "weights_bytes": int(plan.weights_gb * gb),
        "grads_bytes": int(plan.gradients_gb * gb),
        "optimizer_states_bytes": int(plan.optimizer_gb * gb),
        "activations_bytes_estimate": int(plan.activations_gb * gb),
        "ema_bytes": int(plan.ema_gpu_gb * gb),
        "total_gpu_bytes": int(plan.total_gpu_gb * gb),
        "verdict": verdict_map.get(plan.verdict, plan.verdict),
    }


#: Variants whose published weights are distilled (score-matching capacity
#: removed). Full fine-tuning of these is contraindicated; the trainer refuses
#: unless ``force_distilled=True`` is passed explicitly. The ``*-base`` FLUX.2
#: variants are the un-distilled siblings — those are NOT in this set.
DISTILLED_VARIANTS: frozenset[str] = frozenset({
    "schnell",          # FLUX.1-schnell
    "klein-4b",         # FLUX.2-klein-4B distilled
    "klein-9b",         # FLUX.2-klein-9B distilled
})


class FluxFullFinetuneTrainer(FullFinetuneTrainer):
    """Full fine-tune trainer for FLUX.1 and FLUX.2 models.

    Extends ``FullFinetuneTrainer`` with:
    - BFL-aligned flow-matching training step (shift-aware timestep sampling).
    - Distilled-variant guard at construction (full fine-tune contraindicated).
    - 8-bit AdamW support via ``bitsandbytes`` (optional dependency).
    - EMA shadow weights (optional; CPU-resident by default for VRAM savings).
    - FLUX-specific model-layer polymorphism via ``model.prepare_training_inputs()``.

    Memory composition (see §4.9 of the project plan):
    - Single-GPU efficient: bf16 + AdamW8bit + gradient checkpointing + EMA-on-CPU.
    - FSDP multi-GPU: wrap policy on transformer blocks; EMA via summon_full_params.
    - 8-bit Adam + FSDP combination is rejected at construction (unsupported).

    Supported variants:
    - FLUX.1: "dev", "kontext"  (refused: "schnell" — distilled)
    - FLUX.2: "dev", "klein-4b-base", "klein-9b-base"
              (refused: "klein-4b", "klein-9b" — distilled forms)

    Args:
        config: Full training configuration (OmegaConf DictConfig).
        force_distilled: If True, bypass the distilled-variant guard. Use at
            your own risk; full fine-tuning a distilled model erases the
            distillation gains without recovering score-matching capacity.
        force_schnell: **Deprecated** alias for ``force_distilled``. Will be
            removed in a future release.

    Raises:
        ValueError: If ``variant`` is in :data:`DISTILLED_VARIANTS` and
            ``force_distilled`` is False.
        ValueError: If variant is unsupported.
        TypeError: If the model does not implement ``prepare_training_inputs()``.
        ValueError: If 8-bit Adam + FSDP is requested simultaneously.

    Example::

        trainer = FluxFullFinetuneTrainer(config)
        trainer.train()
    """

    SUPPORTED_VARIANTS: dict[str, list[str]] = {
        "v1": ["dev", "kontext"],
        "v2": ["dev", "klein-4b-base", "klein-9b-base"],
    }

    def __init__(
        self,
        config: DictConfig,
        force_distilled: bool = False,
        force_schnell: bool | None = None,
    ) -> None:
        # Back-compat: --force-schnell maps to force_distilled with a warning.
        if force_schnell is not None:
            import warnings
            warnings.warn(
                "force_schnell is deprecated; use force_distilled (covers "
                "FLUX.1-schnell AND the FLUX.2-klein-{4b,9b} distilled forms).",
                DeprecationWarning,
                stacklevel=2,
            )
            force_distilled = force_distilled or force_schnell

        # Validate variant before any heavy setup
        variant = config.model.get("variant", "dev")
        version = config.model.get("version", "v1")

        if variant in DISTILLED_VARIANTS and not force_distilled:
            raise ValueError(
                f"'{variant}' is distilled; full fine-tuning is contraindicated. "
                "Pass --force-distilled to override at your own risk, or use a "
                "non-distilled variant: --variant dev (FLUX.1) or "
                "--variant flux2-klein-4b-base / flux2-klein-9b-base (FLUX.2)."
            )

        # Reject 8-bit Adam + FSDP combination (§4.9 matrix — unsupported)
        strategy = config.hardware.get("distributed_strategy", "single")
        use_8bit = config.training.optimizer.get("use_8bit_adam", False)
        if use_8bit and strategy == "fsdp":
            raise ValueError(
                "8-bit AdamW + FSDP is not a supported combination (§4.9 memory "
                "composition matrix). Use either:\n"
                "  • Single-GPU with 8-bit Adam (--distributed-strategy single --use-8bit-adam)\n"
                "  • FSDP with standard AdamW (--distributed-strategy fsdp)"
            )

        # EMA config
        ema_decay = config.training.get("ema_decay", 0.0)
        self._use_ema = ema_decay > 0.0
        self._ema_decay = ema_decay
        self._ema_on_cpu = config.training.get("ema_on_cpu", True)

        super().__init__(config)

        # Validate model implements prepare_training_inputs (feature-detection per D.1)
        if not hasattr(self.model, "prepare_training_inputs"):
            raise TypeError(
                f"{type(self.model).__name__} does not implement prepare_training_inputs(). "
                "FluxFullFinetuneTrainer requires a FLUX model (Flux1Model or Flux2Model)."
            )

        # Initialize EMA after model is ready
        self.ema: EMAModel | None = None
        if self._use_ema:
            transformer = getattr(self.model, "transformer", self.model)
            self.ema = EMAModel(
                transformer,
                decay=self._ema_decay,
                on_cpu=self._ema_on_cpu,
            )
            logger.info(
                f"EMA enabled: decay={self._ema_decay}, on_cpu={self._ema_on_cpu}"
            )

        logger.info(
            f"FluxFullFinetuneTrainer ready: variant={variant}, version={version}, "
            f"8bit={use_8bit}, strategy={strategy}, ema={self._use_ema}"
        )

    # ------------------------------------------------------------------
    # Model setup — freeze VAE/text encoders, optionally enable grad ckpt
    # ------------------------------------------------------------------

    def _setup_model(self) -> nn.Module:
        """Setup FLUX model: freeze VAE + text encoders, enable gradient checkpointing, FSDP."""
        model = super()._setup_model()

        if self.config.hardware.get("gradient_checkpointing", False):
            if hasattr(model, "enable_gradient_checkpointing"):
                model.enable_gradient_checkpointing()
                logger.info("Gradient checkpointing enabled on transformer")
            elif hasattr(model, "transformer") and hasattr(
                model.transformer, "enable_gradient_checkpointing"
            ):
                model.transformer.enable_gradient_checkpointing()
                logger.info("Gradient checkpointing enabled on transformer")

        strategy = self.config.hardware.get("distributed_strategy", "single")
        if strategy == "fsdp":
            from .fsdp_setup import (
                apply_fsdp_activation_checkpointing,
                setup_fsdp_model,
            )

            version = self.config.model.get("version", "v1")
            mixed_precision = self.config.hardware.get("mixed_precision", "bf16")
            mp_dtype = (
                torch.bfloat16 if mixed_precision in ("bf16", "bfloat16") else torch.float32
            )
            cpu_offload = self.config.hardware.get("cpu_offload", False)

            transformer = getattr(model, "transformer", model)
            fsdp_transformer = setup_fsdp_model(
                transformer,
                version=version,
                cpu_offload=cpu_offload,
                mixed_precision_dtype=mp_dtype,
            )
            if hasattr(model, "transformer"):
                model.transformer = fsdp_transformer
            else:
                model = fsdp_transformer

            if self.config.hardware.get("gradient_checkpointing", False):
                apply_fsdp_activation_checkpointing(fsdp_transformer, version=version)

            logger.info(f"FSDP setup complete: version={version}, cpu_offload={cpu_offload}")

        return model

    # ------------------------------------------------------------------
    # Noise scheduler — FLUX is flow matching, override DDPM default
    # ------------------------------------------------------------------

    def _setup_noise_scheduler(self):
        """Force flow_matching scheduler for FLUX (overrides BaseTrainer default of DDPM)."""
        from omegaconf import OmegaConf

        from ..schedulers import create_scheduler

        scheduler_config = self.config.model.get("scheduler", {})
        if isinstance(scheduler_config, dict) or scheduler_config is None:
            scheduler_config = OmegaConf.create(scheduler_config or {})
        if scheduler_config.get("type") in (None, "ddpm"):
            scheduler_config = OmegaConf.create(
                {**OmegaConf.to_container(scheduler_config, resolve=True), "type": "flow_matching"}
            )
        return create_scheduler(scheduler_config)

    # ------------------------------------------------------------------
    # Optimizer — optionally 8-bit AdamW via bitsandbytes
    # ------------------------------------------------------------------

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer; uses 8-bit AdamW when ``use_8bit_adam=True``."""
        opt_cfg = self.config.training.optimizer

        if opt_cfg.get("use_8bit_adam", False):
            try:
                import bitsandbytes as bnb
            except ImportError as e:
                raise ImportError(
                    "8-bit AdamW requires bitsandbytes. "
                    "Install: `pip install bitsandbytes>=0.43.0`"
                ) from e

            params = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = bnb.optim.AdamW8bit(
                params,
                lr=opt_cfg.get("lr", self.config.training.learning_rate),
                betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
                weight_decay=opt_cfg.get("weight_decay", 0.01),
            )
            logger.info(f"8-bit AdamW optimizer created ({len(params)} param groups)")
            return optimizer

        return super()._setup_optimizer()

    # ------------------------------------------------------------------
    # Training step — BFL-aligned flow matching
    # ------------------------------------------------------------------

    def training_step(self, batch: dict) -> torch.Tensor:
        """BFL-aligned flow-matching training step.

        Calls ``model.prepare_training_inputs()`` for v1/v2 dispatch.
        Subclasses (e.g., ``KontextFullFinetuneTrainer``) override via mixin.

        Args:
            batch: Training batch dict with at least ``"pixel_values"``
                (or ``"target_pixel"`` for Kontext batches) and ``"captions"``.

        Returns:
            Scalar loss tensor.
        """
        return self._flux_training_step(batch)

    def _flux_training_step(self, batch: dict) -> torch.Tensor:
        """Core BFL-aligned flow-matching step (text2img path).

        Args:
            batch: Training batch. Supports both ``pixel_values``/``captions``
                (text2img) and ``target_pixel``/``reference_pixel``/``captions``
                (Kontext — delegated to model.prepare_training_inputs via batch).

        Returns:
            Scalar MSE loss.
        """
        # Feature-detection guard (also checked in __init__, but model may be swapped)
        if not hasattr(self.model, "prepare_training_inputs"):
            raise TypeError(
                f"{type(self.model).__name__} does not implement prepare_training_inputs(). "
                "FluxFullFinetuneTrainer requires a FLUX model (Flux1Model or Flux2Model)."
            )

        # Accept batch keys from both Kontext ("target_pixel"), generic FLUX
        # ("pixel_values"), and standard DiffusionDataset collate ("images").
        for pixel_key in ("target_pixel", "pixel_values", "images"):
            if pixel_key in batch:
                break
        else:
            raise KeyError(
                f"Batch missing image tensor. Expected one of: "
                f"'target_pixel', 'pixel_values', 'images'. Got: {list(batch.keys())}"
            )
        target_pixel: torch.Tensor = batch[pixel_key]
        captions: list[str] = batch.get("captions", batch.get("caption", []))

        device = self.device
        dtype = self.dtype

        # 1. VAE-encode target image → clean latent
        with torch.no_grad():
            x_0 = self.model.encode_image(target_pixel.to(device, dtype=dtype))

        # 2. Determine image sequence length for BFL shift-mu
        seq_len = self.model.compute_image_seq_len(x_0)

        # 3. Sample timestep and build noisy latent + velocity target
        flow_shift = self.config.training.get("flow_shift", True)
        noise = torch.randn_like(x_0)
        t = self.noise_scheduler.training_sample(
            batch_size=x_0.shape[0],
            image_seq_len=seq_len,
            shift=flow_shift,
            device=device,
            dtype=dtype,
        )
        x_t, target_velocity = self.noise_scheduler.add_noise_to_target(x_0, noise, t)

        # 4. Encode text
        with torch.no_grad():
            text_outputs = self.model.encode_text(captions, device=device)

        # 5. Build transformer inputs (polymorphic: v1 vs v2, text2img vs Kontext)
        inputs = self.model.prepare_training_inputs(
            noisy_latent=x_t,
            timestep=t,
            text_outputs=text_outputs,
            guidance_value=self.config.training.get("guidance_value", 1.0),
            batch=batch,
        )

        # 6. Transformer forward
        pred = self.model.transformer(**inputs)

        # 7. Rearrange velocity target to sequence format for loss
        from ..models.flux.v2.conditioning import rearrange_latent_to_sequence
        target_seq = rearrange_latent_to_sequence(target_velocity, patch_size=2)

        loss = F.mse_loss(pred.float(), target_seq.float())
        return loss

    # ------------------------------------------------------------------
    # EMA integration — called by overridden train()
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Training loop with EMA update after each optimizer step."""
        if self.ema is None:
            # No EMA — delegate entirely to parent
            super().train()
            return

        # Replicate BaseTrainer.train() with EMA hook
        self.model.train()

        grad_accum = self.config.training.get("gradient_accumulation", 1)
        max_grad_norm = self.config.training.get("max_grad_norm", 1.0)
        save_every = self.config.training.get("save_every_n_epochs", 1)

        from tqdm import tqdm

        max_steps = self.config.training.get("max_steps", -1)
        logger.info(f"Starting training with EMA (decay={self._ema_decay})")
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
            )

            for step, batch in enumerate(progress):
                if max_steps > 0 and self.global_step >= max_steps:
                    done = True
                    break
                batch = self._prepare_batch(batch)

                # Mirror BaseTrainer.train() — route autocast/backward/clip
                # through the Accelerator when it's active.
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

                if self.accelerator is not None:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()
                epoch_loss += loss.item() * grad_accum

                if (step + 1) % grad_accum == 0:
                    if max_grad_norm > 0:
                        if self.accelerator is not None:
                            self.accelerator.clip_grad_norm_(
                                self._get_trainable_params(), max_norm=max_grad_norm
                            )
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self._get_trainable_params(), max_norm=max_grad_norm
                            )
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    # EMA update after optimizer step
                    transformer = getattr(self.model, "transformer", self.model)
                    self.ema.update(transformer)

                    self.global_step += 1

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

            avg_loss = epoch_loss / len(self.dataloader)
            logger.info(f"Epoch {epoch + 1} complete. Average loss: {avg_loss:.4f}")

            if (epoch + 1) % save_every == 0:
                self._save_checkpoint()

        self._save_checkpoint(final=True)
        self.training_logger.finish()
        logger.info("Training complete!")

    # ------------------------------------------------------------------
    # Checkpointing — EMA-aware save/load
    # ------------------------------------------------------------------

    def save_checkpoint(self, checkpoint_dir: str | Path) -> None:
        """Save full training state including EMA and trainer metadata.

        Writes:
        - ``model.safetensors`` — transformer weights
        - ``ema.safetensors`` — EMA shadow (if EMA enabled)
        - ``optimizer.pt`` — optimizer state dict
        - ``scheduler.pt`` — LR scheduler state dict (if present)
        - ``trainer_state.json`` — global_step, epoch

        Args:
            checkpoint_dir: Directory path for the checkpoint.
        """
        ckpt_path = Path(checkpoint_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)

        # Model weights
        from safetensors.torch import save_file as st_save

        transformer = getattr(self.model, "transformer", self.model)
        st_save(
            {k: v.cpu() for k, v in transformer.state_dict().items()},
            str(ckpt_path / "model.safetensors"),
        )

        # EMA shadow
        if self.ema is not None:
            self.ema.save(ckpt_path / "ema.safetensors")

        # Optimizer
        torch.save(self.optimizer.state_dict(), ckpt_path / "optimizer.pt")

        # LR scheduler
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), ckpt_path / "scheduler.pt")

        # Trainer state
        state = {"global_step": self.global_step, "epoch": self.current_epoch}
        (ckpt_path / "trainer_state.json").write_text(json.dumps(state))

        logger.info(f"Checkpoint saved to {ckpt_path} (step={self.global_step})")

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load training checkpoint and restore all state.

        Args:
            checkpoint_path: Directory path of a checkpoint saved by ``save_checkpoint()``.
        """
        ckpt_path = Path(checkpoint_path)

        # Model weights
        model_file = ckpt_path / "model.safetensors"
        if model_file.exists():
            from safetensors.torch import load_file as st_load

            transformer = getattr(self.model, "transformer", self.model)
            sd = st_load(str(model_file))
            missing, unexpected = transformer.load_state_dict(sd, strict=False)
            if missing:
                logger.warning(f"Checkpoint missing keys: {len(missing)}")
            if unexpected:
                logger.warning(f"Checkpoint unexpected keys: {len(unexpected)}")
            logger.info(f"Loaded model weights from {model_file}")

        # EMA shadow
        ema_file = ckpt_path / "ema.safetensors"
        if ema_file.exists() and self.ema is not None:
            self.ema.load(ema_file)

        # Optimizer
        opt_file = ckpt_path / "optimizer.pt"
        if opt_file.exists():
            self.optimizer.load_state_dict(
                torch.load(str(opt_file), map_location=self.device)
            )

        # LR scheduler
        sched_file = ckpt_path / "scheduler.pt"
        if sched_file.exists() and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(
                torch.load(str(sched_file), map_location="cpu")
            )

        # Trainer state
        state_file = ckpt_path / "trainer_state.json"
        if state_file.exists():
            state = json.loads(state_file.read_text())
            self.global_step = state.get("global_step", 0)
            self.current_epoch = state.get("epoch", 0)

        logger.info(
            f"Resumed from checkpoint {ckpt_path} "
            f"(step={self.global_step}, epoch={self.current_epoch})"
        )

    # Override internal _save_checkpoint to delegate to our EMA-aware version
    def _save_checkpoint(self, final: bool = False) -> None:
        output_dir = Path(self.config.experiment.output_dir)
        if final:
            ckpt_dir = output_dir / "final"
        else:
            ckpt_dir = output_dir / f"checkpoint-{self.global_step}"
        self.save_checkpoint(ckpt_dir)

    # ------------------------------------------------------------------
    # Inference export — AC13 / AC13b (diffusers + BFL native)
    # ------------------------------------------------------------------

    def export_for_inference(
        self,
        output_dir: str | Path,
        use_ema: bool = False,
        formats: list[str] | None = None,
    ) -> dict[str, Path]:
        """Export trained transformer weights for inference.

        Args:
            output_dir: Destination directory; subdirs/files written based on formats.
            use_ema: If True and EMA is enabled, use EMA shadow weights for export.
            formats: List of formats to write. Subset of {"diffusers", "bfl", "both"}.
                Default ["diffusers"]. "both" expands to ["diffusers", "bfl"].

        Returns:
            Dict mapping format name to written path:
            - "diffusers" -> output_dir/transformer/diffusion_pytorch_model.safetensors
            - "bfl" -> output_dir/flux1-finetune.safetensors
        """
        from safetensors.torch import save_file

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if formats is None:
            formats = ["diffusers"]
        if "both" in formats:
            formats = list({*formats, "diffusers", "bfl"})
            formats.remove("both")

        # Get transformer state dict (EMA shadow if requested + available)
        transformer = getattr(self.model, "transformer", self.model)
        if use_ema:
            if self.ema is None:
                raise ValueError(
                    "use_ema=True requested but EMA is not enabled "
                    "(set training.ema_decay > 0 in config)."
                )
            state_dict = {k: v.detach().cpu() for k, v in self.ema.shadow.items()}
        else:
            state_dict = {
                k: v.detach().cpu() for k, v in transformer.state_dict().items()
            }

        outputs: dict[str, Path] = {}

        if "diffusers" in formats:
            diff_dir = output_dir / "transformer"
            diff_dir.mkdir(parents=True, exist_ok=True)
            diff_path = diff_dir / "diffusion_pytorch_model.safetensors"
            save_file(state_dict, str(diff_path))
            outputs["diffusers"] = diff_path
            logger.info(f"Exported diffusers-format weights to {diff_path}")

        if "bfl" in formats:
            from ..models.flux.v1.bfl_export import to_bfl_checkpoint

            bfl_path = output_dir / "flux1-finetune.safetensors"
            num_double = self.config.model.transformer.get("num_layers", 19)
            num_single = self.config.model.transformer.get("num_single_layers", 38)
            to_bfl_checkpoint(
                state_dict,
                bfl_path,
                num_double_blocks=num_double,
                num_single_blocks=num_single,
            )
            outputs["bfl"] = bfl_path
            logger.info(f"Exported BFL-native weights to {bfl_path}")

        return outputs
