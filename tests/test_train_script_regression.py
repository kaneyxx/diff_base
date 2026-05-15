"""Regression smoke tests for scripts/finetune_flux.py (AC15).

AC15: All 7 trainer-method paths must execute 2 training steps without error
under FLUX_TINY_OVERRIDE=1 (no real weights, CPU-only, deterministic).

Parametrized cases:
  1. flux1-dev     — FluxFullFinetuneTrainer, variant=dev
  2. flux1-kontext — KontextFullFinetuneTrainer, variant=kontext
  3. flux1-schnell — FluxFullFinetuneTrainer, variant=schnell (--force-schnell)
  4. flux2-dev     — FluxFullFinetuneTrainer, variant=flux2-dev
  5. flux2-klein4b — FluxFullFinetuneTrainer, variant=flux2-klein-4b
  6. flux2-klein9b — FluxFullFinetuneTrainer, variant=flux2-klein-9b
  7. estimate-mem  — --estimate-memory path (no training, exits 0)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
CLI_PATH = REPO_ROOT / "scripts" / "finetune_flux.py"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
SYNTHETIC_DIR = FIXTURES_DIR / "synthetic_kontext"


# ---------------------------------------------------------------------------
# Fixture: synthetic dataset (reuse from test_finetune_flux_cli.py pattern)
# ---------------------------------------------------------------------------

def _ensure_synthetic_dataset() -> Path:
    """Create tests/fixtures/synthetic_kontext/ with 4 paired 64x64 PNGs if absent."""
    import numpy as np
    from PIL import Image

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = SYNTHETIC_DIR / "metadata.json"
    if meta_path.exists():
        return SYNTHETIC_DIR

    metadata = []
    for i in range(4):
        rng = np.random.default_rng(seed=i + 100)
        target_arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        ref_arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        target_name = f"image_{i:03d}.png"
        ref_name = f"reference_{i:03d}.png"
        caption = f"regression test image {i}"
        Image.fromarray(target_arr).save(SYNTHETIC_DIR / target_name)
        Image.fromarray(ref_arr).save(SYNTHETIC_DIR / ref_name)
        (SYNTHETIC_DIR / f"caption_{i:03d}.txt").write_text(caption)
        metadata.append({"image": target_name, "reference": ref_name, "caption": caption})

    meta_path.write_text(json.dumps(metadata, indent=2))
    return SYNTHETIC_DIR


@pytest.fixture(scope="session", autouse=True)
def _synthetic_dataset():
    _ensure_synthetic_dataset()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_cli(args: list[str], env_extra: dict | None = None, timeout: int = 180) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(CLI_PATH)] + args,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )


def _base_args(variant: str, output_dir: str, extra: list[str] | None = None) -> list[str]:
    args = [
        "--variant", variant,
        "--pretrained-path", "/dummy",
        "--train-data", str(SYNTHETIC_DIR),
        "--output-dir", output_dir,
        "--epochs", "1",
        "--max-steps", "2",
        "--batch-size", "1",
        "--resolution", "64",
        "--no-ema",
    ]
    if extra:
        args.extend(extra)
    return args


# ---------------------------------------------------------------------------
# AC15: parametrized 2-step smoke tests
# ---------------------------------------------------------------------------

_TINY_ENV = {"FLUX_TINY_OVERRIDE": "1"}

_SMOKE_CASES = [
    pytest.param(
        "dev", [], "flux1-dev",
        id="flux1-dev",
    ),
    pytest.param(
        "kontext", [], "flux1-kontext",
        id="flux1-kontext",
    ),
    pytest.param(
        "schnell", ["--force-schnell"], "flux1-schnell",
        id="flux1-schnell-forced",
    ),
    pytest.param(
        "flux2-dev", [], "flux2-dev",
        id="flux2-dev",
    ),
    # FLUX.2 klein full fine-tune smoke uses the un-distilled `-base` aliases —
    # the distilled forms are now refused unless --force-distilled is passed.
    pytest.param(
        "flux2-klein-4b-base", [], "flux2-klein-4b-base",
        id="flux2-klein-4b-base",
    ),
    pytest.param(
        "flux2-klein-9b-base", [], "flux2-klein-9b-base",
        id="flux2-klein-9b-base",
    ),
    # Keep the distilled override paths under test (mirrors flux1-schnell-forced).
    pytest.param(
        "flux2-klein-4b", ["--force-distilled"], "flux2-klein-4b-forced",
        id="flux2-klein-4b-forced",
    ),
    pytest.param(
        "flux2-klein-9b", ["--force-distilled"], "flux2-klein-9b-forced",
        id="flux2-klein-9b-forced",
    ),
]


@pytest.mark.parametrize("variant,extra_args,case_id", _SMOKE_CASES)
def test_trainer_smoke_2_steps(variant: str, extra_args: list[str], case_id: str, tmp_path: Path):
    """AC15: each trainer method must complete 2 steps with FLUX_TINY_OVERRIDE=1."""
    output_dir = str(tmp_path / case_id)
    result = _run_cli(
        _base_args(variant, output_dir, extra_args),
        env_extra=_TINY_ENV,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"[{case_id}] CLI exited with code {result.returncode}.\n"
        f"STDOUT:\n{result.stdout[-3000:]}\n"
        f"STDERR:\n{result.stderr[-3000:]}"
    )
    assert "step" in combined.lower() or "epoch" in combined.lower() or "loss" in combined.lower(), (
        f"[{case_id}] Expected training output in stdout/stderr but got:\n{combined[-2000:]}"
    )


def test_estimate_memory_all_variants():
    """AC15 (7th path): --estimate-memory exits 0 and prints verdict for each variant."""
    for variant in ("dev", "flux2-dev", "flux2-klein-4b-base"):
        result = _run_cli([
            "--variant", variant,
            "--pretrained-path", "/dummy",
            "--train-data", str(SYNTHETIC_DIR),
            "--output-dir", "/tmp/flux_reg_mem",
            "--estimate-memory",
        ])
        assert result.returncode == 0, (
            f"--estimate-memory failed for {variant}:\n{result.stderr}"
        )
        assert "verdict" in result.stdout.lower(), (
            f"--estimate-memory missing 'verdict' in output for {variant}"
        )
