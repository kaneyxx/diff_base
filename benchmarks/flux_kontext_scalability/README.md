# FLUX.1-Kontext LoRA Scalability Benchmark

End-to-end benchmark of `KontextLoRATrainer` (FLUX.1-Kontext-dev base, 12 B-parameter DiT) under DDP across 1 / 2 / 4 A100 80 GB GPUs and three LoRA ranks (16 / 32 / 64). Result: **97 – 107 % scaling efficiency at 4 GPUs** across all three ranks. See `results/summary_table.md` for the full grid and `results/plots/` for the four figures.

---

## What is in this directory

| File | Purpose |
|---|---|
| `slurm_4gpu.sbatch` | SLURM job script (one node, 1/2/4 A100). Starts `gpu_metrics_recorder.sh` in the background, then calls `train_wrapper.sh`. |
| `train_wrapper.sh` | Inner launcher — fills `scal_lora_template.yaml` from env vars and invokes `accelerate launch` (DDP for 2/4 GPUs, plain `python` for 1 GPU). |
| `scal_lora_template.yaml` | Kontext LoRA config template; `__PLACEHOLDERS__` filled by `train_wrapper.sh`. |
| `submit_scal_sweep.sh` | Submit 1 / 2 / 4 GPU jobs independently so each starts as soon as the requested number of GPUs frees up. |
| `submit_rank_sweep.sh` | Outer loop — iterates over LoRA ranks (8 / 16 / 32 / 64) and calls `submit_scal_sweep.sh` for each. |
| `gpu_metrics_recorder.sh` | 1-Hz nvidia-smi sampler. Writes `results/gpu_metrics_<N>gpu_<jobid>.csv`. |
| `parse_metrics_csv.py` | Reduces a single GPU-metrics CSV to a one-line markdown row (peak mem, avg util). |
| `plot_scaling_results.py` | Generates the four PNG plots from the per-run CSVs + training logs. |
| `results/summary_table.md` | Headline table + full grid + interpretation. |
| `results/scaling_summary.csv` | Machine-readable per-config aggregate. |
| `results/plots/scaling_efficiency.png` | Per-rank speedup curves + per-config efficiency bars. |
| `results/plots/gpu_util_timeseries.png` | 15-s rolling-mean GPU utilisation over time. |
| `results/plots/memory_profile.png` | Peak vs average per-GPU memory per configuration. |
| `results/plots/power_profile.png` | 15-s rolling-mean per-GPU power draw over time. |

Per-job raw artefacts (1-Hz `gpu_metrics_*.csv` traces, SLURM `logs/scalability_*.{out,err}`, generated `_generated/*.yaml` configs) are reproducible from a single sweep submission and are gitignored.

---

## Reproduce on a 4 × A100 (or H100) 80 GB node

1. **Set environment variables for cluster paths.**
   ```bash
   export REPO_ROOT=/path/to/diff_base               # path to this repository
   export MODEL_HUB_ROOT=/path/to/huggingface        # HuggingFace cache root
   export DATA_ROOT=/path/to/your/dataset            # paired-dataset root (ORION-style splits)
   ```

2. **Adjust SLURM directives in `slurm_4gpu.sbatch`** for your cluster (`--partition`, `--account`, optional `--reservation`, environment activation block).

3. **Submit the default 1 / 2 / 4 GPU sweep at LoRA rank 16:**
   ```bash
   ./submit_scal_sweep.sh
   ```
   Or submit a different rank / batch size:
   ```bash
   LORA_RANK=64 LORA_ALPHA=64 BATCH_SIZE=1 ./submit_scal_sweep.sh
   ```
   Or run the full rank × GPU grid (12 jobs):
   ```bash
   ./submit_rank_sweep.sh
   ```

4. **After jobs complete, parse + plot:**
   ```bash
   # Quick one-line summary per CSV:
   python parse_metrics_csv.py results/gpu_metrics_4gpu_<JOBID>.csv 4

   # Regenerate the four PNG plots from all CSVs in results/:
   python plot_scaling_results.py
   ```

Expected wall-time per job: ~10 minutes per 150-step run on A100 80 GB (clean DDP startup + 50 warmup + 100 measured steps + GPU metrics flush).

---

## Reading the results

- **Use `results/summary_table.md` first** — it contains the headline 1/2/4 GPU table, the full 9-config grid, and four short interpretation bullets.
- The four PNG plots in `results/plots/` are publication-ready (publication DPI 150). They are derived from the 9 per-run `gpu_metrics_*gpu_*.csv` traces (gitignored) plus the corresponding training logs.
- The `results/scaling_summary.csv` aggregate is the machine-readable source for downstream re-plotting.

The benchmark intentionally separates per-GPU compute density (the GPU-utilisation column at 1-GPU) from multi-GPU communication overhead (the speedup-vs-ideal column at 2/4-GPU), so improvements to either dimension can be attributed cleanly.
