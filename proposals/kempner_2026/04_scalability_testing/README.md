# 04 Scalability Testing — Reproduction Guide

> NOTE: For the clean, submission-ready subset (sanitised scripts + final
> results + plots), see [submission/](submission/). The current directory
> keeps raw 1-Hz CSVs, per-job logs, and generated configs as audit trail.

This directory contains everything needed to run the 1/2/4-GPU scaling sweep
on HMS RC and produce the KempnerPulse telemetry CSVs required by the Kempner
Technical Readiness section.

## Prerequisites

Before running anything, confirm all four prerequisites are satisfied:

1. **4×A100 80GB reservation node** — local scalability testing uses a
   reservation node with 4× A100 80GB GPUs (the production target on the
   Kempner cluster is H100; see `../05_gpu_type.md`). Edit
   `slurm_4gpu.sbatch` and replace `YOUR_RESERVATION` with your actual
   reservation name. Adjust `--partition` / `--account` if you are not in
   the BAO lab (defaults: `gpu_yu` / `yu_ky98_contrib`).

2. **`dcgmi` on PATH on the GPU node** — KempnerPulse hard-fails (`exit 2`)
   if `dcgmi` is not available. This is a hard requirement: the Kempner spec
   mandates KempnerPulse with the DCGM backend. DCGM is supported on A100
   and H100 alike. Confirm `dcgmi --version` works on your reservation node
   before submitting the SLURM job.

3. **BAO data path exists** — the ORION pre-tiled corpus must be present at
   `/n/scratch/users/f/fas994/bao/data` with the standard layout
   (`CRC{NN}/HE/`, `CRC{NN}/tiles.h5`, `CRC{NN}/valid_coordinates.txt`).
   Training splits must exist at
   `/n/scratch/users/f/fas994/bao/dataset/training_splits/CD45_train.json`
   (note the `_train` suffix).

4. **Active codebase at `/n/scratch/users/f/fas994/diff_base`** — the
   full fine-tune trainer + Kontext support + **OrionDataset adapter** must
   be present (`src/data/orion_dataset.py`). Activate its venv before
   running: `source /n/scratch/users/f/fas994/diff_base/venv/bin/activate`
   (or use the Singularity container — see `02_frameworks_software.md`).

5. **FLUX.1-Kontext-dev base weights** — `train_wrapper.sh` reads the path
   from the `PRETRAINED` environment variable (defaults to
   `/n/scratch/users/f/fas994/weights/flux1-kontext-dev`). Either place the
   BFL safetensors there or export `PRETRAINED=...` to your existing copy
   before sbatch.

## Step-by-Step Reproduction

### 1. Install and verify KempnerPulse

Run this **on the GPU node** (inside an interactive job or before sbatch):

```bash
bash kempnerpulse_setup.sh
```

This script installs KempnerPulse via pip and performs a smoke-check that
confirms both KempnerPulse and the DCGM backend (`dcgmi`) are functional.
It **hard-fails with exit code 2** if `dcgmi` is not on PATH — do not
proceed until this exits 0.

### 2. Edit placeholder values in the SLURM script

```bash
# Open slurm_4gpu.sbatch and replace:
#   YOUR_RESERVATION  →  your actual A100 reservation name
#   gpu_yu / yu_ky98_contrib  →  adjust if not in the BAO lab
```

### 3. Pre-flight smoke test on 1 GPU (recommended)

Before submitting the full 1/2/4-GPU array, verify the training pipeline
end-to-end on a single GPU. This catches missing weights, dataset path
typos, and OOM issues without burning the reservation:

```bash
cd /n/scratch/users/f/fas994/diff_base
python scripts/finetune_flux.py \
  --variant kontext \
  --pretrained-path "${PRETRAINED:-/n/scratch/users/f/fas994/weights/flux1-kontext-dev}" \
  --train-data /n/scratch/users/f/fas994/bao/data \
  --dataset-type orion \
  --train-split /n/scratch/users/f/fas994/bao/dataset/training_splits/CD45_train.json \
  --output-dir /tmp/orion_preflight \
  --resolution 1024 \
  --batch-size 1 \
  --max-steps 2 \
  --gradient-checkpointing \
  --ema-on-cpu \
  --use-8bit-adam \
  --distributed-strategy single
```

Expect ~2 minutes wall-time on an A100 80GB and a printed "loss" line per
step. If this passes, you're clear to submit the array job.

### 4. Submit the SLURM array job

```bash
cd /n/scratch/users/f/fas994/diff_base/proposals/kempner_2026/04_scalability_testing
sbatch slurm_4gpu.sbatch
```

The `--array=1-3` directive launches three tasks:
- Task 1 → 1 GPU
- Task 2 → 2 GPUs
- Task 3 → 4 GPUs

Each task starts KempnerPulse in the background (1 Hz polling, all 34 columns),
runs `train_wrapper.sh` for 100 training steps, then stops KempnerPulse.

### 5. Locate the output CSVs

After the job completes, CSVs land in the `results/` subdirectory:

```
results/kempnerpulse_1gpu_<JOBID>.csv
results/kempnerpulse_2gpu_<JOBID>.csv
results/kempnerpulse_4gpu_<JOBID>.csv
```

SLURM logs are in `logs/scalability_<JOBID>_<TASKID>.{out,err}`.

### 6. Populate the summary table

For each CSV, run `parse_pulse_csv.py` to extract the two key metrics
(avg peak memory per GPU in GB, avg GPU utilization in %) and print a
markdown table row:

```bash
python parse_pulse_csv.py results/kempnerpulse_1gpu_<JOBID>.csv 1
python parse_pulse_csv.py results/kempnerpulse_2gpu_<JOBID>.csv 2
python parse_pulse_csv.py results/kempnerpulse_4gpu_<JOBID>.csv 4
```

Paste each printed row into `results/summary_table.md` to replace the
`_TBD_` placeholders.

## File Map

| File | Purpose |
|---|---|
| `kempnerpulse_setup.sh` | Install KempnerPulse; hard-fail if dcgmi missing |
| `slurm_4gpu.sbatch` | SLURM array job (1/2/4 GPU sweep) |
| `train_wrapper.sh` | Inner launcher: single-GPU or FSDP accelerate |
| `parse_pulse_csv.py` | Reduce CSV → one markdown table row |
| `results/` | CSVs land here post-run |
| `results/summary_table.md` | Template; fill from parse_pulse_csv.py output |
