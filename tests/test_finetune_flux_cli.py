"""Tests for scripts/finetune_flux.py CLI (AC10).

Covers:
- Argument parsing (required args, defaults, flag overrides)
- schnell rejection without --force-schnell
- --estimate-memory mode exits cleanly
- AC10: smoke test — 2 training steps with FLUX_TINY_OVERRIDE=1

The smoke test (test_cli_smoke_2_steps_with_tiny_model) invokes the CLI as
a subprocess with FLUX_TINY_OVERRIDE=1 which bypasses checkpoint loading,
text encoder, and VAE so it runs on CPU without real weights.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
CLI_PATH = REPO_ROOT / "scripts" / "finetune_flux.py"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
SYNTHETIC_DIR = FIXTURES_DIR / "synthetic_kontext"


# ---------------------------------------------------------------------------
# Conftest-style fixture: generate synthetic_kontext/ if absent
# ---------------------------------------------------------------------------

def _ensure_synthetic_kontext() -> Path:
    """Create tests/fixtures/synthetic_kontext/ with 4 paired 64x64 PNGs.

    Layout::

        synthetic_kontext/
          metadata.json        # 4 rows: {image, reference, caption}
          image_000.png        # 64x64 RGB target
          reference_000.png    # 64x64 RGB reference
          image_001.png
          reference_001.png
          image_002.png
          reference_002.png
          image_003.png
          reference_003.png
          caption_000.txt
          caption_001.txt
          caption_002.txt
          caption_003.txt

    Returns:
        Path to synthetic_kontext/ directory.
    """
    import numpy as np
    from PIL import Image

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

    metadata = []
    for i in range(4):
        # Deterministic random pixels so reruns are reproducible
        rng = np.random.default_rng(seed=i)
        target_arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        ref_arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

        target_name = f"image_{i:03d}.png"
        ref_name = f"reference_{i:03d}.png"
        caption = f"a synthetic test image number {i}"

        Image.fromarray(target_arr).save(SYNTHETIC_DIR / target_name)
        Image.fromarray(ref_arr).save(SYNTHETIC_DIR / ref_name)

        caption_path = SYNTHETIC_DIR / f"caption_{i:03d}.txt"
        caption_path.write_text(caption)

        metadata.append(
            {
                "image": target_name,
                "reference": ref_name,
                "caption": caption,
            }
        )

    (SYNTHETIC_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    return SYNTHETIC_DIR


@pytest.fixture(scope="session", autouse=True)
def synthetic_kontext_fixtures():
    """Session-scoped fixture: ensure synthetic dataset exists before any test."""
    _ensure_synthetic_kontext()


# ---------------------------------------------------------------------------
# Helper: run CLI as subprocess
# ---------------------------------------------------------------------------

def _run_cli(args: list[str], env_extra: dict | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run scripts/finetune_flux.py as a subprocess.

    Args:
        args: Argument list passed to the CLI (after the script path).
        env_extra: Extra environment variables to inject.
        timeout: Subprocess timeout in seconds.

    Returns:
        CompletedProcess with stdout, stderr, and returncode.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    if env_extra:
        env.update(env_extra)

    cmd = [sys.executable, str(CLI_PATH)] + args
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Argument parsing tests
# ---------------------------------------------------------------------------

class TestArgParsing:
    """Verify CLI argparse behavior without launching training."""

    def test_help_exits_zero(self):
        result = _run_cli(["--help"])
        assert result.returncode == 0
        assert "FLUX full fine-tuning" in result.stdout

    def test_help_epilog_documents_accelerate(self):
        result = _run_cli(["--help"])
        assert "accelerate launch" in result.stdout
        assert "fsdp" in result.stdout.lower()

    def test_print_config_only_exits_zero(self):
        result = _run_cli(
            [
                "--variant", "dev",
                "--pretrained-path", "/dummy",
                "--train-data", str(SYNTHETIC_DIR),
                "--output-dir", "/tmp/flux_test_out",
                "--print-config-only",
            ]
        )
        assert result.returncode == 0, result.stderr
        # Should print YAML — must contain key sections
        assert "training:" in result.stdout
        assert "model:" in result.stdout
        assert "hardware:" in result.stdout

    def test_print_config_variant_is_set(self):
        result = _run_cli(
            [
                "--variant", "dev",
                "--pretrained-path", "/dummy",
                "--train-data", str(SYNTHETIC_DIR),
                "--output-dir", "/tmp/flux_test_out",
                "--print-config-only",
            ]
        )
        assert "dev" in result.stdout

    def test_default_lr_is_1e6(self):
        result = _run_cli(
            [
                "--variant", "dev",
                "--pretrained-path", "/dummy",
                "--train-data", str(SYNTHETIC_DIR),
                "--output-dir", "/tmp/flux_test_out",
                "--print-config-only",
            ]
        )
        assert result.returncode == 0
        assert "1.0e-06" in result.stdout or "1e-06" in result.stdout or "0.000001" in result.stdout

    def test_no_ema_flag_disables_ema(self):
        result = _run_cli(
            [
                "--variant", "dev",
                "--pretrained-path", "/dummy",
                "--train-data", str(SYNTHETIC_DIR),
                "--output-dir", "/tmp/flux_test_out",
                "--no-ema",
                "--print-config-only",
            ]
        )
        assert result.returncode == 0
        # ema_decay: 0.0 means EMA disabled (flat key matches trainer contract)
        assert "ema_decay: 0.0" in result.stdout

    def test_ema_decay_enables_ema(self):
        result = _run_cli(
            [
                "--variant", "dev",
                "--pretrained-path", "/dummy",
                "--train-data", str(SYNTHETIC_DIR),
                "--output-dir", "/tmp/flux_test_out",
                "--ema-decay", "0.99",
                "--print-config-only",
            ]
        )
        assert result.returncode == 0
        assert "ema_decay: 0.99" in result.stdout

    def test_flux2_variant_sets_version_v2(self):
        # Use the un-distilled `-base` alias — the distilled klein-4b form
        # is refused without --force-distilled by the new umbrella guard.
        result = _run_cli(
            [
                "--variant", "flux2-klein-4b-base",
                "--pretrained-path", "/dummy",
                "--train-data", str(SYNTHETIC_DIR),
                "--output-dir", "/tmp/flux_test_out",
                "--print-config-only",
            ]
        )
        assert result.returncode == 0
        assert "v2" in result.stdout
        assert "klein-4b" in result.stdout

    def test_missing_required_args_exits_nonzero(self):
        # Missing --variant
        result = _run_cli(
            [
                "--pretrained-path", "/dummy",
                "--train-data", str(SYNTHETIC_DIR),
                "--output-dir", "/tmp/flux_test_out",
            ]
        )
        assert result.returncode != 0

    def test_invalid_variant_exits_nonzero(self):
        result = _run_cli(
            [
                "--variant", "does-not-exist",
                "--pretrained-path", "/dummy",
                "--train-data", str(SYNTHETIC_DIR),
                "--output-dir", "/tmp/flux_test_out",
            ]
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# schnell rejection tests
# ---------------------------------------------------------------------------

class TestSchnellRejection:
    """Verify that schnell is rejected unless --force-schnell is passed."""

    def test_cli_rejects_schnell_without_force(self):
        """AC10 support: schnell must fail without --force-schnell."""
        result = _run_cli(
            [
                "--variant", "schnell",
                "--pretrained-path", "/dummy",
                "--train-data", str(SYNTHETIC_DIR),
                "--output-dir", "/tmp/flux_schnell_test",
                "--max-steps", "1",
            ],
            env_extra={"FLUX_TINY_OVERRIDE": "1"},
        )
        assert result.returncode != 0
        stderr_or_out = result.stderr + result.stdout
        assert "schnell" in stderr_or_out.lower()
        assert "distilled" in stderr_or_out.lower() or "contraindicated" in stderr_or_out.lower()

    def test_cli_force_schnell_passes_guard(self):
        """--force-schnell bypasses the guard (training may still fail for other reasons)."""
        result = _run_cli(
            [
                "--variant", "schnell",
                "--pretrained-path", "/dummy",
                "--train-data", str(SYNTHETIC_DIR),
                "--output-dir", "/tmp/flux_schnell_force",
                "--force-schnell",
                "--print-config-only",
            ]
        )
        # With --print-config-only the guard is never hit (trainer not instantiated)
        # Just verify argparse accepts --force-schnell and prints config
        assert result.returncode == 0
        assert "schnell" in result.stdout


# ---------------------------------------------------------------------------
# --estimate-memory tests
# ---------------------------------------------------------------------------

class TestEstimateMemory:
    """Verify --estimate-memory prints plan table and exits cleanly."""

    def test_estimate_memory_exits_zero(self):
        """AC10 support: --estimate-memory must not crash."""
        result = _run_cli(
            [
                "--variant", "dev",
                "--pretrained-path", "/dummy",
                "--train-data", str(SYNTHETIC_DIR),
                "--output-dir", "/tmp/flux_mem_test",
                "--resolution", "1024",
                "--batch-size", "1",
                "--estimate-memory",
            ]
        )
        assert result.returncode == 0, result.stderr

    def test_estimate_memory_output_contains_verdict(self):
        result = _run_cli(
            [
                "--variant", "dev",
                "--pretrained-path", "/dummy",
                "--train-data", str(SYNTHETIC_DIR),
                "--output-dir", "/tmp/flux_mem_test",
                "--resolution", "1024",
                "--batch-size", "1",
                "--estimate-memory",
            ]
        )
        assert result.returncode == 0, result.stderr
        assert "verdict" in result.stdout.lower()

    def test_estimate_memory_output_contains_gb_values(self):
        result = _run_cli(
            [
                "--variant", "dev",
                "--pretrained-path", "/dummy",
                "--train-data", str(SYNTHETIC_DIR),
                "--output-dir", "/tmp/flux_mem_test",
                "--resolution", "1024",
                "--estimate-memory",
            ]
        )
        assert result.returncode == 0, result.stderr
        assert "GB" in result.stdout


# ---------------------------------------------------------------------------
# AC10: Smoke test — 2 training steps with FLUX_TINY_OVERRIDE=1
# ---------------------------------------------------------------------------

class TestCliSmokeWithTinyModel:
    """AC10: Full CLI smoke test using FLUX_TINY_OVERRIDE=1."""

    def test_cli_smoke_2_steps_with_tiny_model(self):
        """AC10: CLI completes 2 steps on tiny model without OOM.

        FLUX_TINY_OVERRIDE=1 causes the trainer to:
        - Skip from_bfl_checkpoint and instantiate tiny Flux1Transformer
          (hidden=64, num_layers=1, num_single_layers=1)
        - Skip text encoder (random pooled + seq embeddings of correct shape)
        - Skip VAE (identity encoder/decoder)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_cli(
                [
                    "--variant", "dev",
                    "--pretrained-path", "/dummy",
                    "--train-data", str(SYNTHETIC_DIR),
                    "--output-dir", tmpdir,
                    "--max-steps", "2",
                    "--batch-size", "1",
                    "--resolution", "256",
                    "--no-ema",
                    "--no-8bit-adam",
                    "--no-gradient-checkpointing",
                ],
                env_extra={"FLUX_TINY_OVERRIDE": "1"},
                timeout=180,
            )

        assert result.returncode == 0, (
            f"CLI smoke test failed.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    def test_cli_smoke_writes_output(self):
        """AC10 supplement: CLI writes at least config.yaml to output dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_cli(
                [
                    "--variant", "dev",
                    "--pretrained-path", "/dummy",
                    "--train-data", str(SYNTHETIC_DIR),
                    "--output-dir", tmpdir,
                    "--max-steps", "2",
                    "--batch-size", "1",
                    "--resolution", "256",
                    "--no-ema",
                    "--no-8bit-adam",
                    "--no-gradient-checkpointing",
                ],
                env_extra={"FLUX_TINY_OVERRIDE": "1"},
                timeout=180,
            )

            assert result.returncode == 0, result.stderr
            config_yaml = Path(tmpdir) / "config.yaml"
            assert config_yaml.exists(), "config.yaml not written to output dir"

    def test_cli_smoke_flux2_klein_dispatches_v2(self):
        """AC11 support: flux2-klein-4b-base variant routes through Flux2Model.

        Uses the un-distilled ``-base`` alias because the distilled klein-4b
        form is rejected without ``--force-distilled`` by the umbrella guard.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _run_cli(
                [
                    "--variant", "flux2-klein-4b-base",
                    "--pretrained-path", "/dummy",
                    "--train-data", str(SYNTHETIC_DIR),
                    "--output-dir", tmpdir,
                    "--max-steps", "2",
                    "--batch-size", "1",
                    "--resolution", "256",
                    "--no-ema",
                    "--no-8bit-adam",
                    "--no-gradient-checkpointing",
                ],
                env_extra={"FLUX_TINY_OVERRIDE": "1"},
                timeout=180,
            )

        assert result.returncode == 0, (
            f"flux2-klein-4b-base smoke test failed.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
