"""FSDP wrap policies and setup utilities for FLUX transformers.

Provides version-aware auto-wrap policies so that FSDP shards at the
transformer block boundary (joint blocks + single blocks), which gives
the best balance of memory efficiency and communication overhead.

Block class references (must match actual model construction):
- FLUX.1: FluxJointTransformerBlock, FluxSingleTransformerBlock
          (src.models.flux.components.attention)
- FLUX.2: Flux2TransformerBlock, Flux2SingleTransformerBlock
          (src.models.flux.v2.blocks)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.logging import get_logger

logger = get_logger(__name__)


def get_flux_auto_wrap_policy(version: str = "v1"):
    """Return FSDP ``ModuleWrapPolicy`` for FLUX joint and single transformer blocks.

    The wrap policy shards at each transformer block boundary so that:
    - Forward / backward passes for each block happen in isolation.
    - Peak memory for activations is bounded by one block at a time.

    Args:
        version: Model version — ``"v1"`` for FLUX.1 (dev/schnell/kontext)
            or ``"v2"`` for FLUX.2 (dev/klein).

    Returns:
        ``torch.distributed.fsdp.wrap.ModuleWrapPolicy`` covering the
        appropriate block classes for the given version.

    Raises:
        ValueError: If ``version`` is not ``"v1"`` or ``"v2"``.

    Example::

        policy = get_flux_auto_wrap_policy(version="v1")
        fsdp_model = FullyShardedDataParallel(model, auto_wrap_policy=policy)
    """
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    if version == "v1":
        from src.models.flux.components.attention import (
            FluxJointTransformerBlock,
            FluxSingleTransformerBlock,
        )
        wrap_cls = {FluxJointTransformerBlock, FluxSingleTransformerBlock}
        logger.info(
            "FSDP wrap policy: FLUX.1 — FluxJointTransformerBlock + FluxSingleTransformerBlock"
        )
    elif version == "v2":
        from src.models.flux.v2.blocks import (
            Flux2SingleTransformerBlock,
            Flux2TransformerBlock,
        )
        wrap_cls = {Flux2TransformerBlock, Flux2SingleTransformerBlock}
        logger.info(
            "FSDP wrap policy: FLUX.2 — Flux2TransformerBlock + Flux2SingleTransformerBlock"
        )
    else:
        raise ValueError(
            f"Unknown FLUX version '{version}' for FSDP wrap policy. "
            "Expected 'v1' or 'v2'."
        )

    return ModuleWrapPolicy(wrap_cls)


def setup_fsdp_model(
    model: nn.Module,
    version: str = "v1",
    cpu_offload: bool = False,
    mixed_precision_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Wrap a FLUX transformer module with FSDP.

    Applies the FLUX-specific auto-wrap policy and configures mixed precision
    and optional CPU offload.

    When combined with EMA, call ``EMAModel.update()`` inside a
    ``FullyShardedDataParallel.summon_full_params(model)`` context manager so
    that full (unsharded) parameter values are visible for the EMA accumulation.

    Args:
        model: The transformer module to wrap (e.g. ``Flux1Transformer``).
        version: FLUX version — ``"v1"`` or ``"v2"``.
        cpu_offload: If True, offload parameters to CPU when not in use.
            Slower but reduces GPU memory.
        mixed_precision_dtype: Param / reduce dtype for mixed precision
            (default ``torch.bfloat16``).

    Returns:
        FSDP-wrapped model.

    Raises:
        RuntimeError: If ``torch.distributed`` is not initialized.

    Example::

        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        fsdp_transformer = setup_fsdp_model(model.transformer, version="v1")
    """
    from torch.distributed.fsdp import (
        CPUOffload,
        FullyShardedDataParallel,
        MixedPrecision,
    )

    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before calling setup_fsdp_model(). "
            "Call torch.distributed.init_process_group() or use accelerate launch."
        )

    mp_policy = MixedPrecision(
        param_dtype=mixed_precision_dtype,
        reduce_dtype=mixed_precision_dtype,
        buffer_dtype=mixed_precision_dtype,
    )

    cpu_offload_cfg = CPUOffload(offload_params=cpu_offload)
    wrap_policy = get_flux_auto_wrap_policy(version=version)

    fsdp_model = FullyShardedDataParallel(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp_policy,
        cpu_offload=cpu_offload_cfg,
    )

    logger.info(
        f"FSDP model created: version={version}, "
        f"mixed_precision={mixed_precision_dtype}, cpu_offload={cpu_offload}"
    )
    return fsdp_model


def apply_fsdp_activation_checkpointing(fsdp_model: nn.Module, version: str = "v1") -> None:
    """Apply activation checkpointing to FSDP-wrapped transformer blocks.

    Must be called AFTER FSDP wrapping (not before), as FSDP changes the
    module hierarchy.

    Args:
        fsdp_model: FSDP-wrapped model returned by ``setup_fsdp_model()``.
        version: FLUX version — ``"v1"`` or ``"v2"``.
    """

    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )
    except ImportError:
        logger.warning(
            "torch.distributed.algorithms._checkpoint not available; "
            "skipping activation checkpointing"
        )
        return

    if version == "v1":
        from src.models.flux.components.attention import (
            FluxJointTransformerBlock,
            FluxSingleTransformerBlock,
        )

        def check_fn(m: nn.Module) -> bool:
            return isinstance(m, (FluxJointTransformerBlock, FluxSingleTransformerBlock))
    else:
        from src.models.flux.v2.blocks import (
            Flux2SingleTransformerBlock,
            Flux2TransformerBlock,
        )

        def check_fn(m: nn.Module) -> bool:
            return isinstance(m, (Flux2TransformerBlock, Flux2SingleTransformerBlock))

    apply_activation_checkpointing(
        fsdp_model,
        checkpoint_wrapper_fn=checkpoint_wrapper,
        check_fn=check_fn,
    )
    logger.info(f"Activation checkpointing applied to FSDP model (version={version})")
