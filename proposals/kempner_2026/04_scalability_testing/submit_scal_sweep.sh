#!/usr/bin/env bash
# Submit 1/2/4 GPU scalability tests as 3 independent SLURM jobs.
# Each job requests only the GPUs it needs so they can start as soon as the
# corresponding number of GPUs frees up on the reservation node (rather than
# all 3 tasks blocking until all 4 GPUs are simultaneously idle).
#
# Usage:
#   ./submit_scal_sweep.sh                       # default rank=16, all 1/2/4 GPU
#   ./submit_scal_sweep.sh 1 2                   # subset of GPU counts
#   LORA_RANK=64 LORA_ALPHA=64 ./submit_scal_sweep.sh
#   LORA_RANK=8  BATCH_SIZE=8 ./submit_scal_sweep.sh 1 2 4
#
# LoRA hyperparams travel through env → sbatch → train_wrapper.sh → template.
# Default rank/alpha=16, batch=4, max_steps=150 (50 warmup + 100 measured).
set -euo pipefail

LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-${LORA_RANK}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_STEPS="${MAX_STEPS:-150}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH_FILE="${SCRIPT_DIR}/slurm_4gpu.sbatch"
# IMPORTANT: cd into SCRIPT_DIR before sbatch so that SLURM_SUBMIT_DIR matches
# the proposal dir. The sbatch references ${SLURM_SUBMIT_DIR}/train_wrapper.sh
# and ${SLURM_SUBMIT_DIR}/gpu_metrics_recorder.sh (SLURM does not copy
# anything other than the sbatch script itself to /var/spool/slurmd/jobN/).
cd "${SCRIPT_DIR}"

# Default to all three GPU counts when no args provided
if [[ $# -eq 0 ]]; then
  GPU_COUNTS=(1 2 4)
else
  GPU_COUNTS=("$@")
fi

# Pre-flight sanity checks
test -f "${SBATCH_FILE}" || { echo "ERROR: sbatch not found at ${SBATCH_FILE}"; exit 1; }
test -f /n/scratch/users/f/fas994/bao/dataset/training_splits/CD45_train.json || {
  echo "ERROR: ORION CD45 split missing"; exit 1; }
test -e /n/scratch/users/f/fas994/huggingface/hub/models--black-forest-labs--FLUX.1-Kontext-dev/snapshots/24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d || {
  echo "ERROR: FLUX.1-Kontext-dev HF snapshot missing"; exit 1; }

echo "Submitting GPU scalability sweep:"
SUBMITTED=()
for N in "${GPU_COUNTS[@]}"; do
  case "${N}" in
    1|2|4) : ;;
    *) echo "Skipping invalid GPU_COUNT=${N} (must be 1, 2, or 4)"; continue ;;
  esac

  JOB_OUTPUT=$(sbatch --parsable \
    --gres="gpu:a100:${N}" \
    --export="ALL,GPU_COUNT=${N},LORA_RANK=${LORA_RANK},LORA_ALPHA=${LORA_ALPHA},BATCH_SIZE=${BATCH_SIZE},MAX_STEPS=${MAX_STEPS}" \
    --job-name="gpu-scalability-test-r${LORA_RANK}-${N}gpu" \
    "${SBATCH_FILE}")
  JOB_ID="${JOB_OUTPUT}"
  echo "  ${N}-GPU job submitted (rank=${LORA_RANK}): JobId=${JOB_ID}"
  SUBMITTED+=("${N}gpu_r${LORA_RANK}=${JOB_ID}")
done

echo ""
echo "Submitted ${#SUBMITTED[@]} jobs: ${SUBMITTED[*]}"
echo ""
echo "Monitor with:"
echo "  squeue -j $(IFS=,; echo "${SUBMITTED[*]##*=}") -o '%.12i %.6P %.20j %.2t %.10M %.10R'"
echo ""
echo "Logs:"
echo "  tail -f $(dirname "$0")/logs/scalability_<JOBID>.out"
