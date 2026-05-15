# FLUX Full Fine-Tune Recipes

Practical recipes for running FLUX.1 and FLUX.2 full fine-tuning using
`scripts/finetune_flux.py`. All recipes assume the `diff_base` environment
is activated and dependencies installed.

---

## Hardware Tiers

| Tier | Hardware | Strategy | Memory config |
|------|----------|----------|---------------|
| A | 1× H100 80GB | single | bf16 + 8-bit Adam + gradient checkpoint + EMA on CPU |
| B | 4× H100 80GB | fsdp | bf16 + AdamW + gradient checkpoint |
| C | 8× A100 40GB | fsdp | bf16 + AdamW + gradient checkpoint + FSDP CPU offload |
| D | 1× A100 80GB | single | bf16 + 8-bit Adam + gradient checkpoint (no EMA) |

---

## Recipe 1 — Single 80GB H100 (Tier A, Recommended)

**Target:** FLUX.1-dev, 1024×1024, batch_size=1, gradient_accum=4.
**Memory:** ~68 GB GPU with EMA on CPU.

```bash
python scripts/finetune_flux.py \
  --variant dev \
  --pretrained-path /path/to/flux1-dev.safetensors \
  --train-data /path/to/dataset \
  --output-dir /path/to/outputs \
  --resolution 1024 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --lr 1e-6 \
  --epochs 30 \
  --warmup-steps 500 \
  --use-8bit-adam \
  --gradient-checkpointing \
  --ema-decay 0.99 \
  --ema-on-cpu \
  --bf16 \
  --guidance-value 1.0 \
  --flow-shift \
  --save-every-steps 500
```

**Memory breakdown:**

| Component | GB |
|---|---|
| Weights (bf16) | 24.0 |
| Gradients (bf16) | 24.0 |
| Optimizer (8-bit AdamW) | ~14.0 |
| Activations (with AC) | ~6.0 |
| EMA shadow (CPU) | 0.0 |
| **Total GPU** | **~68 GB** |

**Throughput:** ~0.3 steps/sec at 1024×1024 on H100 SXM5.

**Tips:**
- Use `--resolution 768` to reduce activation memory to ~3.5 GB, enabling
  batch_size=2 without OOM.
- Do not combine `--use-8bit-adam` with `--distributed-strategy fsdp` — this
  combination is rejected at startup.
- If you observe NaN losses after a few steps, reduce `--lr` to `5e-7`.

---

## Recipe 2 — Multi-GPU 4×80GB H100 FSDP (Tier B)

**Target:** FLUX.1-dev, 1024×1024, batch_size=4 (1 per GPU).
**Memory:** ~22 GB per GPU (FSDP shards weights, grads, and optimizer across 4 GPUs).

### Step 1 — Configure accelerate

```bash
accelerate config
# Select: multi-GPU, FSDP, 4 processes, bf16 mixed precision
```

Or use the provided template:

```bash
accelerate launch \
  --config_file configs/accelerate/multi_gpu_fsdp.yaml \
  scripts/finetune_flux.py \
    --variant dev \
    --pretrained-path /path/to/flux1-dev.safetensors \
    --train-data /path/to/dataset \
    --output-dir /path/to/outputs \
    --resolution 1024 \
    --batch-size 1 \
    --gradient-accumulation 1 \
    --lr 1e-6 \
    --epochs 30 \
    --distributed-strategy fsdp \
    --gradient-checkpointing \
    --ema-decay 0.99 \
    --ema-on-cpu \
    --bf16
```

**Memory per GPU (4-way FSDP):**

| Component | GB/GPU |
|---|---|
| Weights (bf16, sharded) | ~6.0 |
| Gradients (bf16, sharded) | ~6.0 |
| Optimizer (fp32, sharded) | ~24.0 |
| Activations (with AC) | ~6.0 |
| **Total per GPU** | **~42 GB** |

**Effective batch:** 4 (1 per GPU × 4 GPUs). Throughput scales ~3.5× vs Tier A.

**Notes:**
- EMA update in FSDP mode uses `summon_full_params` to gather all shards before
  computing the running average. This adds one all-gather per optimizer step.
- Checkpoints are always saved in full (unsharded) format. Loading on fewer GPUs
  than the training run works without conversion.

---

## Recipe 3 — Multi-GPU 8×40GB A100 with CPU Offload (Tier C)

**Target:** FLUX.1-dev, 1024×1024, larger dataset, tighter VRAM budget.

```bash
accelerate launch \
  --config_file configs/accelerate/multi_gpu_fsdp.yaml \
  scripts/finetune_flux.py \
    --variant dev \
    --pretrained-path /path/to/flux1-dev.safetensors \
    --train-data /path/to/dataset \
    --output-dir /path/to/outputs \
    --resolution 1024 \
    --batch-size 1 \
    --gradient-accumulation 2 \
    --lr 1e-6 \
    --epochs 30 \
    --distributed-strategy fsdp \
    --fsdp-cpu-offload \
    --gradient-checkpointing \
    --no-ema \
    --bf16
```

**Notes:**
- `--fsdp-cpu-offload` offloads FSDP parameters to CPU between forward/backward
  passes. This reduces per-GPU peak usage by ~15 GB at the cost of PCIe bandwidth.
- Disable EMA (`--no-ema`) to eliminate the `summon_full_params` overhead on 8 GPUs.
- Expect ~40% throughput reduction vs Tier B due to CPU offload overhead.

---

## Recipe 4 — FLUX.2 klein-4b Fine-Tune

FLUX.2 uses the same `finetune_flux.py` script with a different `--variant` flag.
The model dispatch is handled automatically by `prepare_training_inputs()`.

```bash
python scripts/finetune_flux.py \
  --variant flux2-klein-4b \
  --pretrained-path /path/to/flux2-klein-4b \
  --train-data /path/to/dataset \
  --output-dir /path/to/outputs \
  --resolution 1024 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --lr 1e-6 \
  --epochs 30 \
  --use-8bit-adam \
  --gradient-checkpointing \
  --ema-decay 0.99 \
  --ema-on-cpu \
  --bf16
```

**FLUX.2 differences from FLUX.1:**
- Latent channels: 32 (vs 16) → 128-channel patchified tokens (vs 64).
- Text encoder: Mistral-3 / Qwen3 (variant-dependent) with different vocab and
  max_length.
- 4D RoPE with `axes_dim=(32,32,32,32)` and `rope_theta=2000`.
- All bias tensors are `False` across blocks.
- `prepare_training_inputs()` handles these differences internally; the trainer
  code is identical.

---

## Exporting Fine-Tuned Weights

After training, export weights to BFL native format for use with the BFL inference
pipeline:

```python
from src.models.flux.v1.bfl_export import to_bfl_checkpoint

to_bfl_checkpoint(
    model.transformer,
    output_path="/path/to/flux1-finetune.safetensors",
)
```

Or use the CLI export flag:

```bash
python scripts/finetune_flux.py \
  --export-format bfl \
  --output-dir /path/to/outputs \
  ...
```

The exported `.safetensors` file uses BFL native key naming (`double_blocks.*`,
`single_blocks.*`, `img_in.*`) and can be loaded with the BFL inference pipeline
exactly as `flux1-dev.safetensors`.

---

## Memory Estimation

Use `--estimate-memory` to get a memory breakdown before committing to a training run:

```bash
python scripts/finetune_flux.py \
  --variant dev \
  --pretrained-path /dummy \
  --train-data /dummy \
  --output-dir /dummy \
  --batch-size 1 \
  --resolution 1024 \
  --use-8bit-adam \
  --gradient-checkpointing \
  --ema-decay 0.99 \
  --ema-on-cpu \
  --estimate-memory
```

Output example:
```
============================================================
  FLUX Full Fine-Tune Memory Estimate
============================================================
  Weights (forward):          24.0 GB
  Gradients:                  24.0 GB
  Optimizer states:           14.0 GB
  Activations (peak):          6.0 GB
  EMA shadow (GPU):            0.0 GB
------------------------------------------------------------
  Total GPU:                  68.0 GB
------------------------------------------------------------
  Verdict: fits_single_h100
============================================================
```

The estimate is a first approximation. Actual usage can vary ±15% depending on
attention implementation (Flash Attention reduces activation memory further),
CUDA version, and data types. Always run a 2-step smoke test before a long run:

```bash
FLUX_TINY_OVERRIDE=1 python scripts/finetune_flux.py \
  --variant dev \
  --pretrained-path /dummy \
  --train-data tests/fixtures/synthetic_kontext \
  --output-dir /tmp/smoke_test \
  --max-steps 2 \
  --batch-size 1 \
  --resolution 256
```

---

## Troubleshooting

### NaN losses

Cause: Learning rate too high, especially at high timestep values where velocity
magnitudes are large.

Fix:
1. Reduce `--lr` from `1e-6` to `5e-7`.
2. Verify `--flow-shift` is enabled (default ON). Without it, the timestep
   distribution is uniform and high-t samples dominate early training.
3. Check that `pred` and `target_velocity` are cast to fp32 before MSE — this is
   done internally by the trainer but verify if you have a custom training step.

### OOM during activation backward

Cause: Activation memory exceeds GPU capacity. Common at resolution > 1024 or
batch_size > 1.

Fix:
1. Enable `--gradient-checkpointing` (should be ON by default).
2. Reduce `--resolution` to 768 or 512.
3. Use FSDP (`--distributed-strategy fsdp`) to shard across GPUs.
4. Enable `--fsdp-cpu-offload` on tight hardware.

### EMA diverging from base model

Cause: EMA decay too high relative to learning rate, or EMA warmup missing.

Fix:
1. Use `--ema-decay 0.999` for longer runs (>10k steps), `0.99` for shorter runs.
2. The EMA only starts updating after the first optimizer step — this is correct
   behavior.
3. If base model loss drops but EMA-weighted inference quality stays poor, try
   `--ema-decay 0.9` for faster shadow tracking.

### Checkpoint resume across distributed strategies

Checkpoints are always saved with full (unsharded) state dicts, so you can resume
a single-GPU checkpoint on a multi-GPU FSDP run and vice versa:

```bash
python scripts/finetune_flux.py \
  --resume /path/to/checkpoint-1000 \
  --distributed-strategy fsdp \
  ...
```

The trainer restores `global_step` and `current_epoch` from `trainer_state.json`,
optimizer state from `optimizer.pt`, and EMA shadow from `ema.safetensors` (if
EMA was enabled in the original run).

### bitsandbytes install fails

`bitsandbytes` requires CUDA 11.8 or later and a compatible GPU driver.

```bash
pip install bitsandbytes>=0.43.0

# Verify installation:
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
```

If `bitsandbytes` cannot be installed, omit `--use-8bit-adam` and use standard
AdamW with FSDP (Tier B) or CPU optimizer offload instead.
