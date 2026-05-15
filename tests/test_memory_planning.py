"""Tests for compute_memory_plan() (AC14).

AC14: compute_memory_plan() returns a MemoryPlan with:
- weights_gb matching §5.5 table within ±10%
- total_gpu_gb for target recipe ~68 GB (within ±10%)
- verdict "fits_single_h100" for the target recipe
- verdict "infeasible_single_gpu" for naive (no 8-bit, no AC) config
"""

from __future__ import annotations

from src.training.memory_planning import (
    MemoryPlan,
    compute_memory_plan,
    format_memory_plan,
)

# Reference values from §5.5 of the project plan
# FLUX.1-dev, 12B params, batch_size=1, resolution=1024

FLUX1_PARAMS = 12_000_000_000


class TestMemoryPlanStructure:
    def test_returns_memory_plan_instance(self):
        plan = compute_memory_plan()
        assert isinstance(plan, MemoryPlan)

    def test_has_all_required_fields(self):
        plan = compute_memory_plan()
        assert hasattr(plan, "weights_gb")
        assert hasattr(plan, "gradients_gb")
        assert hasattr(plan, "optimizer_gb")
        assert hasattr(plan, "activations_gb")
        assert hasattr(plan, "ema_gpu_gb")
        assert hasattr(plan, "total_gpu_gb")
        assert hasattr(plan, "verdict")
        assert hasattr(plan, "notes")
        assert hasattr(plan, "config")

    def test_all_gb_values_positive(self):
        plan = compute_memory_plan()
        assert plan.weights_gb > 0
        assert plan.gradients_gb > 0
        assert plan.optimizer_gb > 0
        assert plan.activations_gb > 0
        assert plan.total_gpu_gb > 0

    def test_total_equals_sum_of_components(self):
        plan = compute_memory_plan()
        expected = (
            plan.weights_gb + plan.gradients_gb + plan.optimizer_gb +
            plan.activations_gb + plan.ema_gpu_gb
        )
        assert abs(plan.total_gpu_gb - expected) < 0.1, (
            f"total_gpu_gb={plan.total_gpu_gb} does not match sum of components={expected:.2f}"
        )

    def test_verdict_is_valid_string(self):
        plan = compute_memory_plan()
        valid_verdicts = {
            "fits_single_h100",
            "fits_4x_h100_fsdp",
            "needs_cpu_offload",
            "infeasible_single_gpu",
        }
        assert plan.verdict in valid_verdicts, f"Invalid verdict: {plan.verdict!r}"

    def test_config_dict_populated(self):
        plan = compute_memory_plan(batch_size=2, resolution=512)
        assert plan.config["batch_size"] == 2
        assert plan.config["resolution"] == 512


# ---------------------------------------------------------------------------
# AC14: §5.5 table numeric checks (FLUX.1-dev target recipe)
# ---------------------------------------------------------------------------

class TestAC14ReferenceValues:
    """Match §5.5 plan table within ±10%."""

    def test_bf16_weights_approx_24gb(self):
        """§5.5: bf16 weights = 12B × 2 = 24 GB."""
        plan = compute_memory_plan(
            num_params=FLUX1_PARAMS,
            use_bf16=True,
        )
        expected = 24.0
        assert abs(plan.weights_gb - expected) / expected < 0.10, (
            f"bf16 weights: expected ~{expected} GB, got {plan.weights_gb:.1f} GB"
        )

    def test_bf16_gradients_approx_24gb(self):
        """§5.5: bf16 gradients = 12B × 2 = 24 GB."""
        plan = compute_memory_plan(
            num_params=FLUX1_PARAMS,
            use_bf16=True,
        )
        expected = 24.0
        assert abs(plan.gradients_gb - expected) / expected < 0.10, (
            f"bf16 gradients: expected ~{expected} GB, got {plan.gradients_gb:.1f} GB"
        )

    def test_8bit_optimizer_approx_12gb(self):
        """§5.5: 8-bit AdamW optimizer ≈ 12–14 GB for 12B params."""
        plan = compute_memory_plan(
            num_params=FLUX1_PARAMS,
            use_8bit_adam=True,
        )
        assert 10.0 <= plan.optimizer_gb <= 16.0, (
            f"8-bit optimizer: expected 10–16 GB, got {plan.optimizer_gb:.1f} GB"
        )

    def test_fp32_optimizer_approx_96gb(self):
        """§5.5 naïve: AdamW fp32 optimizer = 12B × 8 = 96 GB."""
        plan = compute_memory_plan(
            num_params=FLUX1_PARAMS,
            use_bf16=False,
            use_8bit_adam=False,
        )
        expected = 96.0
        assert abs(plan.optimizer_gb - expected) / expected < 0.10, (
            f"fp32 optimizer: expected ~{expected} GB, got {plan.optimizer_gb:.1f} GB"
        )

    def test_activations_with_checkpointing_approx_6gb(self):
        """§5.5: activations with AC ≈ 6 GB at res=1024, bs=1."""
        plan = compute_memory_plan(
            num_params=FLUX1_PARAMS,
            resolution=1024,
            batch_size=1,
            gradient_checkpointing=True,
        )
        expected = 6.0
        assert abs(plan.activations_gb - expected) / expected < 0.10, (
            f"activations with AC: expected ~{expected} GB, got {plan.activations_gb:.1f} GB"
        )

    def test_activations_without_checkpointing_approx_30gb(self):
        """§5.5: activations without AC ≈ 30 GB at res=1024, bs=1."""
        plan = compute_memory_plan(
            num_params=FLUX1_PARAMS,
            resolution=1024,
            batch_size=1,
            gradient_checkpointing=False,
        )
        expected = 30.0
        assert abs(plan.activations_gb - expected) / expected < 0.10, (
            f"activations without AC: expected ~{expected} GB, got {plan.activations_gb:.1f} GB"
        )

    def test_target_recipe_total_approx_68gb(self):
        """§5.5 target recipe total ≈ 68 GB (fits 80GB H100)."""
        plan = compute_memory_plan(
            num_params=FLUX1_PARAMS,
            batch_size=1,
            resolution=1024,
            use_bf16=True,
            use_8bit_adam=True,
            gradient_checkpointing=True,
            ema_enabled=True,
            ema_on_cpu=True,
            distributed_strategy="single",
        )
        expected = 68.0
        assert abs(plan.total_gpu_gb - expected) / expected < 0.10, (
            f"target recipe total: expected ~{expected} GB, got {plan.total_gpu_gb:.1f} GB"
        )

    def test_target_recipe_verdict_fits_h100(self):
        """Target recipe must return verdict 'fits_single_h100'."""
        plan = compute_memory_plan(
            num_params=FLUX1_PARAMS,
            batch_size=1,
            resolution=1024,
            use_bf16=True,
            use_8bit_adam=True,
            gradient_checkpointing=True,
            ema_enabled=True,
            ema_on_cpu=True,
        )
        assert plan.verdict == "fits_single_h100", (
            f"Target recipe should fit 80GB H100, got verdict={plan.verdict!r}"
        )

    def test_naive_config_infeasible(self):
        """§5.5 naïve: fp32 + AdamW fp32 + no AC => infeasible on single GPU."""
        plan = compute_memory_plan(
            num_params=FLUX1_PARAMS,
            batch_size=1,
            resolution=1024,
            use_bf16=False,
            use_8bit_adam=False,
            gradient_checkpointing=False,
            ema_enabled=False,
        )
        assert plan.verdict in ("infeasible_single_gpu", "fits_4x_h100_fsdp"), (
            f"Naïve config should be infeasible or require FSDP, got {plan.verdict!r}"
        )
        assert plan.total_gpu_gb > 100, (
            f"Naïve config should need >100 GB, got {plan.total_gpu_gb:.1f} GB"
        )

    def test_ema_on_gpu_adds_to_total(self):
        """EMA on GPU must add ~24 GB (fp32, 12B params) to total."""
        plan_cpu = compute_memory_plan(
            num_params=FLUX1_PARAMS,
            ema_enabled=True,
            ema_on_cpu=True,
        )
        plan_gpu = compute_memory_plan(
            num_params=FLUX1_PARAMS,
            ema_enabled=True,
            ema_on_cpu=False,
        )
        diff = plan_gpu.total_gpu_gb - plan_cpu.total_gpu_gb
        expected_diff = (FLUX1_PARAMS * 4) / (1024 ** 3)  # fp32 shadow
        assert abs(diff - expected_diff) / expected_diff < 0.10, (
            f"EMA GPU vs CPU diff: expected ~{expected_diff:.1f} GB, got {diff:.1f} GB"
        )
        assert plan_cpu.ema_gpu_gb == 0.0


# ---------------------------------------------------------------------------
# Verdict correctness
# ---------------------------------------------------------------------------

class TestVerdictLogic:
    def test_fsdp_strategy_returns_fsdp_verdict(self):
        plan = compute_memory_plan(
            num_params=FLUX1_PARAMS,
            distributed_strategy="fsdp",
            use_8bit_adam=True,
            gradient_checkpointing=True,
        )
        assert plan.verdict in ("fits_4x_h100_fsdp", "needs_cpu_offload"), (
            f"FSDP strategy should yield FSDP verdict, got {plan.verdict!r}"
        )

    def test_small_model_fits_h100(self):
        """A very small model (100M params) with AC must fit on a single H100."""
        plan = compute_memory_plan(
            num_params=100_000_000,
            resolution=256,
            use_bf16=True,
            use_8bit_adam=False,
            gradient_checkpointing=True,
        )
        assert plan.verdict == "fits_single_h100", (
            f"100M param model should fit H100, got {plan.verdict!r}"
        )
        assert plan.total_gpu_gb < 10.0

    def test_batch_scaling(self):
        """Larger batch size increases activation memory."""
        plan1 = compute_memory_plan(batch_size=1, resolution=512)
        plan4 = compute_memory_plan(batch_size=4, resolution=512)
        assert plan4.activations_gb > plan1.activations_gb, (
            "Larger batch should increase activation memory"
        )

    def test_resolution_scaling(self):
        """Higher resolution increases activation memory."""
        plan512 = compute_memory_plan(batch_size=1, resolution=512)
        plan1024 = compute_memory_plan(batch_size=1, resolution=1024)
        assert plan1024.activations_gb > plan512.activations_gb, (
            "Higher resolution should increase activation memory"
        )


# ---------------------------------------------------------------------------
# format_memory_plan
# ---------------------------------------------------------------------------

class TestFormatMemoryPlan:
    def test_format_contains_gb(self):
        plan = compute_memory_plan()
        text = format_memory_plan(plan)
        assert "GB" in text

    def test_format_contains_verdict(self):
        plan = compute_memory_plan()
        text = format_memory_plan(plan)
        assert "Verdict" in text or plan.verdict in text

    def test_format_contains_all_components(self):
        plan = compute_memory_plan()
        text = format_memory_plan(plan)
        assert "Weights" in text
        assert "Gradients" in text
        assert "Optimizer" in text
        assert "Activations" in text

    def test_format_returns_string(self):
        plan = compute_memory_plan()
        text = format_memory_plan(plan)
        assert isinstance(text, str)
        assert len(text) > 100
