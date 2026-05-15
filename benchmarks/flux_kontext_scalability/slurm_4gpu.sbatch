#!/usr/bin/env bash
# SLURM job for FLUX.1-Kontext LoRA scalability testing on a 4×A100 80GB node.
#
# Submit via submit_scal_sweep.sh (sets --gres + GPU_COUNT env per job so 1/2/4
# GPU runs queue independently and start as soon as the requested number of
# GPUs is free).
#
# Replace the partition / account / reservation lines below with values that
# match your cluster, or remove them and pass equivalents via sbatch CLI.
#SBATCH --job-name=flux1-kontext-scal
#SBATCH --partition=gpu                   # adjust for your cluster
#SBATCH --account=                        # adjust for your cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --output=logs/scalability_%j.out
#SBATCH --error=logs/scalability_%j.err

set -euo pipefail

# Submit_scal_sweep.sh cd's into this directory before sbatch, so SLURM_SUBMIT_DIR
# resolves to benchmarks/flux_kontext_scalability/.
BENCHMARK_DIR="${SLURM_SUBMIT_DIR}"
mkdir -p "${BENCHMARK_DIR}/logs" "${BENCHMARK_DIR}/results"

# Activate your project environment here (conda env, virtualenv, module load, ...).
# Example:
#   module load gcc/12.2.0 cuda/12.4.1
#   source /path/to/your/venv/bin/activate
test -x "$(command -v python)" || { echo "ERROR: python not on PATH"; exit 1; }

# GPU_COUNT is set by submit_scal_sweep.sh via --export=GPU_COUNT=N
GPU_COUNT="${GPU_COUNT:?GPU_COUNT must be set (1, 2, or 4) — submit via submit_scal_sweep.sh}"
case "${GPU_COUNT}" in
  1|2|4) : ;;
  *) echo "GPU_COUNT must be 1, 2, or 4 (got: ${GPU_COUNT})"; exit 1 ;;
esac
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))

# Background GPU metrics recorder — 1 Hz nvidia-smi sampler. Output CSV is
# consumed by plot_scaling_results.py and parse_metrics_csv.py.
METRICS_CSV="${BENCHMARK_DIR}/results/gpu_metrics_${GPU_COUNT}gpu_${SLURM_JOB_ID}.csv"
bash "${BENCHMARK_DIR}/gpu_metrics_recorder.sh" "${METRICS_CSV}" 1.0 &
METRICS_PID=$!
trap 'kill -TERM ${METRICS_PID} 2>/dev/null; wait ${METRICS_PID} 2>/dev/null || true' EXIT

bash "${BENCHMARK_DIR}/train_wrapper.sh" "${GPU_COUNT}"
