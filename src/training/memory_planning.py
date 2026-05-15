"""Memory budget estimation for FLUX full fine-tuning.

Provides ``compute_memory_plan()`` which returns a structured breakdown of
GPU memory requirements for a given model/training configuration, along with
a human-readable verdict on which hardware tier the run fits.

Numbers are derived from the §5.5 memory budget analysis in the project plan:
  FLUX.1-dev (12B params), batch_size=1, resolution=1024, seq_len=4096 tokens.

All GB values are in gibibytes (GiB, base-2) unless noted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MemoryPlan:
    """Structured memory budget for a FLUX fine-tune run.

    Args:
        weights_gb: GPU memory for model weights (bf16 or fp32).
        gradients_gb: GPU memory for gradient tensors.
        optimizer_gb: GPU memory for optimizer states (varies by type).
        activations_gb: Peak activation memory during forward/backward.
        ema_gpu_gb: GPU memory for EMA shadow weights (0 if on_cpu=True).
        total_gpu_gb: Sum of all GPU-resident components.
        verdict: One of "fits_single_h100", "fits_4x_h100_fsdp",
            "needs_cpu_offload", "infeasible_single_gpu".
        notes: List of human-readable notes about the configuration.
        config: Input parameters used to produce this plan.
    """

    weights_gb: float
    gradients_gb: float
    optimizer_gb: float
    activations_gb: float
    ema_gpu_gb: float
    total_gpu_gb: float
    verdict: Literal[
        "fits_single_h100",
        "fits_4x_h100_fsdp",
        "needs_cpu_offload",
        "infeasible_single_gpu",
    ]
    notes: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)


def compute_memory_plan(
    num_params: int = 12_000_000_000,
    batch_size: int = 1,
    resolution: int = 1024,
    use_bf16: bool = True,
    use_8bit_adam: bool = False,
    gradient_checkpointing: bool = True,
    ema_enabled: bool = False,
    ema_on_cpu: bool = True,
    distributed_strategy: str = "single",
    num_double_blocks: int = 19,
    num_single_blocks: int = 38,
    hidden_size: int = 3072,
) -> MemoryPlan:
    """Estimate GPU memory requirements for FLUX full fine-tuning.

    Uses the §5.5 memory budget analysis as calibration. All values are
    first-order approximations; actual usage can vary ±15% depending on
    CUDA version, attention implementation, and data types.

    Args:
        num_params: Number of trainable parameters (default 12B for FLUX.1-dev).
        batch_size: Training batch size per GPU.
        resolution: Image resolution in pixels (square assumed).
        use_bf16: If True, weights and gradients use bfloat16 (2 bytes/param).
            If False, float32 (4 bytes/param).
        use_8bit_adam: If True, optimizer states use 8-bit packing (~1 byte/param).
        gradient_checkpointing: If True, recompute activations; reduces peak
            activation memory by ~5x.
        ema_enabled: Whether EMA shadow weights are used.
        ema_on_cpu: If True (and ema_enabled), EMA shadow resides on CPU.
        distributed_strategy: One of "single" or "fsdp".
        num_double_blocks: Number of double (joint) transformer blocks.
        num_single_blocks: Number of single transformer blocks.
        hidden_size: Transformer hidden dimension.

    Returns:
        MemoryPlan with per-component GB breakdown and a hardware verdict.

    Example::

        plan = compute_memory_plan(use_8bit_adam=True, gradient_checkpointing=True)
        print(f"Total GPU: {plan.total_gpu_gb:.1f} GB — {plan.verdict}")
    """
    bytes_per_param = 2 if use_bf16 else 4

    # Weights
    weights_gb = (num_params * bytes_per_param) / (1024 ** 3)

    # Gradients (same dtype as weights)
    gradients_gb = (num_params * bytes_per_param) / (1024 ** 3)

    # Optimizer states
    if use_8bit_adam:
        # 8-bit packed: ~1 byte per param + ~15% overhead
        optimizer_gb = (num_params * 1.15) / (1024 ** 3)
    else:
        # Standard AdamW: 2 fp32 states (m, v) per param = 8 bytes/param
        optimizer_gb = (num_params * 8) / (1024 ** 3)

    # Activations
    # Patch tokens at resolution R: seq_len = (R//16)^2 (FLUX uses 8x VAE + 2x patch)
    seq_len = (resolution // 16) ** 2

    if gradient_checkpointing:
        # With full recompute AC: ~6 GB at seq=4096, scales with seq_len and batch
        base_activation_gb = 6.0
    else:
        # Without AC: ~30 GB at seq=4096, scales with seq_len and batch
        base_activation_gb = 30.0

    # Scale from reference (seq=4096, bs=1) to actual config
    ref_seq = 4096
    activation_scale = (seq_len / ref_seq) * batch_size
    activations_gb = base_activation_gb * activation_scale

    # EMA GPU footprint
    if ema_enabled and not ema_on_cpu:
        ema_gpu_gb = (num_params * 4) / (1024 ** 3)  # fp32 shadow
    else:
        ema_gpu_gb = 0.0

    # FSDP sharding: weights/grads/optimizer split across GPUs
    if distributed_strategy == "fsdp":
        # Assume 4-GPU shard — each GPU holds 1/4 of params
        shard_factor = 4.0
        weights_gb /= shard_factor
        gradients_gb /= shard_factor
        optimizer_gb /= shard_factor
        if not ema_on_cpu:
            ema_gpu_gb /= shard_factor

    total_gpu_gb = weights_gb + gradients_gb + optimizer_gb + activations_gb + ema_gpu_gb

    # Determine verdict
    notes: list[str] = []

    if distributed_strategy == "fsdp":
        if total_gpu_gb <= 80.0:
            verdict: Literal[
                "fits_single_h100",
                "fits_4x_h100_fsdp",
                "needs_cpu_offload",
                "infeasible_single_gpu",
            ] = "fits_4x_h100_fsdp"
            notes.append("4×80GB H100 FSDP config: each GPU holds sharded params.")
        else:
            verdict = "needs_cpu_offload"
            notes.append("FSDP with CPU offload recommended for this configuration.")
    else:
        if total_gpu_gb <= 68.0:
            verdict = "fits_single_h100"
            notes.append("Fits single 80GB H100 with ~12 GB headroom for buffers.")
        elif total_gpu_gb <= 80.0:
            verdict = "fits_single_h100"
            notes.append("Tight fit on single 80GB H100; monitor peak usage closely.")
        elif total_gpu_gb <= 320.0:
            verdict = "fits_4x_h100_fsdp"
            notes.append("Use --distributed-strategy fsdp with 4×80GB H100.")
        else:
            verdict = "infeasible_single_gpu"
            notes.append("Cannot fit on single GPU. Use FSDP or CPU offload.")

    if use_8bit_adam and distributed_strategy == "fsdp":
        notes.append(
            "WARNING: 8-bit Adam + FSDP is unsupported. This combination is "
            "rejected at trainer construction time."
        )

    if not gradient_checkpointing and total_gpu_gb > 80:
        notes.append(
            "Enable --gradient-checkpointing to reduce activation memory by ~5x."
        )

    if ema_enabled and not ema_on_cpu:
        notes.append(
            "EMA shadow weights reside on GPU. Use --ema-on-cpu to save "
            f"{(num_params * 4 / 1024**3):.1f} GB VRAM."
        )

    return MemoryPlan(
        weights_gb=round(weights_gb, 2),
        gradients_gb=round(gradients_gb, 2),
        optimizer_gb=round(optimizer_gb, 2),
        activations_gb=round(activations_gb, 2),
        ema_gpu_gb=round(ema_gpu_gb, 2),
        total_gpu_gb=round(total_gpu_gb, 2),
        verdict=verdict,
        notes=notes,
        config={
            "num_params": num_params,
            "batch_size": batch_size,
            "resolution": resolution,
            "use_bf16": use_bf16,
            "use_8bit_adam": use_8bit_adam,
            "gradient_checkpointing": gradient_checkpointing,
            "ema_enabled": ema_enabled,
            "ema_on_cpu": ema_on_cpu,
            "distributed_strategy": distributed_strategy,
        },
    )


def format_memory_plan(plan: MemoryPlan) -> str:
    """Format a MemoryPlan as a human-readable table.

    Args:
        plan: MemoryPlan produced by ``compute_memory_plan()``.

    Returns:
        Multi-line string table suitable for CLI output.
    """
    lines = [
        "=" * 60,
        "  FLUX Full Fine-Tune Memory Estimate",
        "=" * 60,
        f"  Weights (forward):        {plan.weights_gb:6.1f} GB",
        f"  Gradients:                {plan.gradients_gb:6.1f} GB",
        f"  Optimizer states:         {plan.optimizer_gb:6.1f} GB",
        f"  Activations (peak):       {plan.activations_gb:6.1f} GB",
        f"  EMA shadow (GPU):         {plan.ema_gpu_gb:6.1f} GB",
        "-" * 60,
        f"  Total GPU:                {plan.total_gpu_gb:6.1f} GB",
        "-" * 60,
        f"  Verdict: {plan.verdict}",
        "",
    ]
    if plan.notes:
        lines.append("  Notes:")
        for note in plan.notes:
            lines.append(f"    - {note}")
    lines.append("=" * 60)
    return "\n".join(lines)
