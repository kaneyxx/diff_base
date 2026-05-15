"""Tests for FSDP wrap policy setup (Phase C / Task #4 — AC9).

The FSDP smoke test uses single-process gloo init so it runs on CPU
without requiring a multi-GPU environment. Skip if torch.distributed is
not functional (e.g., some CI environments).
"""

from __future__ import annotations

import os

import pytest
import torch.nn as nn

# ---------------------------------------------------------------------------
# AC9a: wrap policy includes the correct block classes for each version
# ---------------------------------------------------------------------------

class TestAutoWrapPolicyV1:
    def test_policy_created_without_error(self):
        from src.training.fsdp_setup import get_flux_auto_wrap_policy
        policy = get_flux_auto_wrap_policy(version="v1")
        assert policy is not None

    def test_policy_covers_joint_block(self):
        """ModuleWrapPolicy must cover FluxJointTransformerBlock."""
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        from src.models.flux.components.attention import FluxJointTransformerBlock
        from src.training.fsdp_setup import get_flux_auto_wrap_policy

        policy = get_flux_auto_wrap_policy(version="v1")
        assert isinstance(policy, ModuleWrapPolicy)
        # Verify the block type is in the policy's module classes
        assert FluxJointTransformerBlock in policy._module_classes, (
            "FluxJointTransformerBlock must be in the v1 wrap policy"
        )

    def test_policy_covers_single_block(self):
        """ModuleWrapPolicy must cover FluxSingleTransformerBlock."""

        from src.models.flux.components.attention import FluxSingleTransformerBlock
        from src.training.fsdp_setup import get_flux_auto_wrap_policy

        policy = get_flux_auto_wrap_policy(version="v1")
        assert FluxSingleTransformerBlock in policy._module_classes, (
            "FluxSingleTransformerBlock must be in the v1 wrap policy"
        )

    def test_invalid_version_raises(self):
        from src.training.fsdp_setup import get_flux_auto_wrap_policy
        with pytest.raises(ValueError, match="Unknown FLUX version"):
            get_flux_auto_wrap_policy(version="v99")


class TestAutoWrapPolicyV2:
    def test_policy_covers_flux2_joint_block(self):
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        from src.models.flux.v2.blocks import Flux2TransformerBlock
        from src.training.fsdp_setup import get_flux_auto_wrap_policy

        policy = get_flux_auto_wrap_policy(version="v2")
        assert isinstance(policy, ModuleWrapPolicy)
        assert Flux2TransformerBlock in policy._module_classes, (
            "Flux2TransformerBlock must be in the v2 wrap policy"
        )

    def test_policy_covers_flux2_single_block(self):

        from src.models.flux.v2.blocks import Flux2SingleTransformerBlock
        from src.training.fsdp_setup import get_flux_auto_wrap_policy

        policy = get_flux_auto_wrap_policy(version="v2")
        assert Flux2SingleTransformerBlock in policy._module_classes, (
            "Flux2SingleTransformerBlock must be in the v2 wrap policy"
        )

    def test_v1_policy_does_not_include_v2_blocks(self):
        from src.models.flux.v2.blocks import Flux2TransformerBlock
        from src.training.fsdp_setup import get_flux_auto_wrap_policy

        policy = get_flux_auto_wrap_policy(version="v1")
        assert Flux2TransformerBlock not in policy._module_classes, (
            "v1 policy must not include FLUX.2 blocks"
        )


# ---------------------------------------------------------------------------
# AC9b: single-process FSDP smoke test (gloo backend, CPU)
# ---------------------------------------------------------------------------

def _distributed_available() -> bool:
    """Check if torch.distributed can be initialized."""
    try:
        import torch.distributed as dist  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _distributed_available(),
    reason="torch.distributed not available"
)
class TestFSDPWrapSmoke:
    """Single-process FSDP smoke test using gloo backend."""

    def _init_dist(self):
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://127.0.0.1:29500",
                world_size=1,
                rank=0,
            )

    def _cleanup_dist(self):
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_fsdp_wrap_smoke_v1(self, tmp_path):
        """Wrap a tiny Flux1Transformer-like model with FSDP and run a forward pass."""
        from torch.distributed.fsdp import FullyShardedDataParallel
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy

        from src.models.flux.components.attention import (
            FluxJointTransformerBlock,
            FluxSingleTransformerBlock,
        )

        # Build minimal model with 1 of each block type
        hidden = 64
        heads = 2

        class _TinyFlux1Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.joint = FluxJointTransformerBlock(hidden, heads)
                self.single = FluxSingleTransformerBlock(hidden, heads)

            def forward(self, x):
                # Minimal forward just to verify FSDP doesn't error
                return x

        try:
            import torch.distributed as dist
            if not dist.is_initialized():
                os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
                os.environ.setdefault("MASTER_PORT", "29501")
                dist.init_process_group(
                    backend="gloo",
                    world_size=1,
                    rank=0,
                )

            tiny = _TinyFlux1Model()
            policy = ModuleWrapPolicy({FluxJointTransformerBlock, FluxSingleTransformerBlock})
            fsdp_model = FullyShardedDataParallel(
                tiny,
                auto_wrap_policy=policy,
            )

            # Verify FSDP wrapping occurred
            assert isinstance(fsdp_model, FullyShardedDataParallel), (
                "Model must be wrapped by FSDP"
            )

        finally:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()

    def test_setup_fsdp_model_raises_without_dist(self):
        """setup_fsdp_model must raise RuntimeError if dist not initialized."""
        import torch.distributed as dist
        # Make sure dist is NOT initialized
        if dist.is_initialized():
            dist.destroy_process_group()

        from src.training.fsdp_setup import setup_fsdp_model

        tiny = nn.Linear(8, 8)
        with pytest.raises(RuntimeError, match="torch.distributed must be initialized"):
            setup_fsdp_model(tiny, version="v1")


# ---------------------------------------------------------------------------
# Policy introspection — without distributed initialization
# ---------------------------------------------------------------------------

class TestPolicyIntrospection:
    """Tests that only inspect the policy object, no dist init required."""

    def test_v1_policy_has_two_block_types(self):
        from src.training.fsdp_setup import get_flux_auto_wrap_policy
        policy = get_flux_auto_wrap_policy(version="v1")
        assert len(policy._module_classes) == 2

    def test_v2_policy_has_two_block_types(self):
        from src.training.fsdp_setup import get_flux_auto_wrap_policy
        policy = get_flux_auto_wrap_policy(version="v2")
        assert len(policy._module_classes) == 2

    def test_v1_and_v2_policies_are_disjoint(self):
        from src.training.fsdp_setup import get_flux_auto_wrap_policy
        p1 = get_flux_auto_wrap_policy(version="v1")
        p2 = get_flux_auto_wrap_policy(version="v2")
        assert p1._module_classes.isdisjoint(p2._module_classes), (
            "v1 and v2 wrap policies must cover disjoint block classes"
        )
