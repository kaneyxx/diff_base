#!/usr/bin/env bash
# Submit 1/2/4 GPU scalability tests as independent SLURM jobs so each starts
# as soon as the requested number of GPUs frees up (rather than waiting for
# all 4 GPUs to be simultaneously idle).
#
# Usage:
#   ./submit_scal_sweep.sh                        # all three GPU counts at default rank=16
#   ./submit_scal_sweep.sh 1 2                    # subset of GPU counts
#   LORA_RANK=64 LORA_ALPHA=64 ./submit_scal_sweep.sh
#   LORA_RANK=8  BATCH_SIZE=8 ./submit_scal_sweep.sh 1 2 4
#
# LoRA hyperparams travel through env → sbatch → train_wrapper.sh → template.
# Defaults: rank/alpha=16, batch_size=4, max_steps=150 (50 warmup + 100 measured).
set -euo pipefail

LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-${LORA_RANK}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_STEPS="${MAX_STEPS:-150}"

: "${REPO_ROOT:?set REPO_ROOT to the diff_base checkout}"
: "${MODEL_HUB_ROOT:?set MODEL_HUB_ROOT to your HuggingFace cache root}"
: "${DATA_ROOT:?set DATA_ROOT to your paired-dataset root}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH_FILE="${SCRIPT_DIR}/slurm_4gpu.sbatch"
# cd into SCRIPT_DIR so SLURM_SUBMIT_DIR resolves to this benchmark dir.
# slurm_4gpu.sbatch references ${SLURM_SUBMIT_DIR}/{train_wrapper.sh,gpu_metrics_recorder.sh}.
cd "${SCRIPT_DIR}"

if [[ $# -eq 0 ]]; then
  GPU_COUNTS=(1 2 4)
else
  GPU_COUNTS=("$@")
fi

test -f "${SBATCH_FILE}" || { echo "ERROR: sbatch not found at ${SBATCH_FILE}"; exit 1; }

echo "Submitting GPU scalability sweep:"
SUBMITTED=()
for N in "${GPU_COUNTS[@]}"; do
  case "${N}" in
    1|2|4) : ;;
    *) echo "Skipping invalid GPU_COUNT=${N} (must be 1, 2, or 4)"; continue ;;
  esac

  JOB_OUTPUT=$(sbatch --parsable \
    --gres="gpu:a100:${N}" \
    --export="ALL,GPU_COUNT=${N},LORA_RANK=${LORA_RANK},LORA_ALPHA=${LORA_ALPHA},BATCH_SIZE=${BATCH_SIZE},MAX_STEPS=${MAX_STEPS},REPO_ROOT=${REPO_ROOT},MODEL_HUB_ROOT=${MODEL_HUB_ROOT},DATA_ROOT=${DATA_ROOT}" \
    --job-name="flux1-kontext-scal-r${LORA_RANK}-${N}gpu" \
    "${SBATCH_FILE}")
  echo "  ${N}-GPU job submitted (rank=${LORA_RANK}): JobId=${JOB_OUTPUT}"
  SUBMITTED+=("${N}gpu_r${LORA_RANK}=${JOB_OUTPUT}")
done

echo ""
echo "Submitted ${#SUBMITTED[@]} jobs: ${SUBMITTED[*]}"
echo ""
echo "Monitor with:"
echo "  squeue -j $(IFS=,; echo "${SUBMITTED[*]##*=}") -o '%.12i %.6P %.20j %.2t %.10M %.10R'"
echo "  tail -f ./logs/scalability_<JOBID>.out"
