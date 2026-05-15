#!/usr/bin/env bash
# LoRA scalability test wrapper.
# Generates a per-run YAML from scal_lora_template.yaml with rank / alpha /
# batch size / max_steps filled in, then launches scripts/train.py via
# accelerate (DDP for 2/4 GPUs, plain python for 1 GPU).
#
# Env vars (set by sbatch / submit_scal_sweep.sh):
#   GPU_COUNT     — 1 | 2 | 4   (required)
#   LORA_RANK     — default 16  (sweep dimension: 8, 16, 32, 64, …)
#   LORA_ALPHA    — default == LORA_RANK
#   BATCH_SIZE    — default 4   (per-GPU; total effective = BATCH_SIZE × GPU_COUNT)
#   MAX_STEPS     — default 150 (50 warmup + 100 steady-state)
set -euo pipefail

GPU_COUNT="${GPU_COUNT:-${1:?GPU_COUNT required (env or arg1)}}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-${LORA_RANK}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_STEPS="${MAX_STEPS:-150}"

REPO_ROOT="/n/scratch/users/f/fas994/diff_base"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/scal_lora_template.yaml"

# Sanity-check upstream paths (fail fast)
test -f "${TEMPLATE}" || { echo "ERROR: template missing: ${TEMPLATE}"; exit 1; }
test -f /n/scratch/users/f/fas994/bao/dataset/training_splits/CD45_train.json || {
  echo "ERROR: ORION CD45 split missing"; exit 1; }
test -e /n/scratch/users/f/fas994/huggingface/hub/models--black-forest-labs--FLUX.1-Kontext-dev/snapshots/24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d || {
  echo "ERROR: FLUX.1-Kontext-dev HF snapshot missing"; exit 1; }

EXP_NAME="scal_lora_r${LORA_RANK}_${GPU_COUNT}gpu_${SLURM_JOB_ID:-local}"
OUT_DIR="/n/scratch/users/f/fas994/outputs/${EXP_NAME}"
YAML_OUT="${SCRIPT_DIR}/_generated/${EXP_NAME}.yaml"
mkdir -p "${SCRIPT_DIR}/_generated" "${OUT_DIR}"

# Fill template via sed — heredoc kept the placeholders deliberately so we can
# track what was substituted by diffing the generated yaml against the template.
sed -e "s|__EXP_NAME__|${EXP_NAME}|g" \
    -e "s|__OUT_DIR__|${OUT_DIR}|g" \
    -e "s|__LORA_RANK__|${LORA_RANK}|g" \
    -e "s|__LORA_ALPHA__|${LORA_ALPHA}|g" \
    -e "s|__BATCH_SIZE__|${BATCH_SIZE}|g" \
    -e "s|__MAX_STEPS__|${MAX_STEPS}|g" \
    "${TEMPLATE}" > "${YAML_OUT}"

echo "Generated config: ${YAML_OUT}"
echo "  GPU_COUNT=${GPU_COUNT} LORA_RANK=${LORA_RANK} LORA_ALPHA=${LORA_ALPHA}"
echo "  BATCH_SIZE=${BATCH_SIZE} MAX_STEPS=${MAX_STEPS}"

if [[ "${GPU_COUNT}" == "1" ]]; then
  python "${REPO_ROOT}/scripts/train.py" --config "${YAML_OUT}"
else
  accelerate launch \
    --multi_gpu \
    --num_processes "${GPU_COUNT}" \
    --mixed_precision bf16 \
    "${REPO_ROOT}/scripts/train.py" --config "${YAML_OUT}"
fi
