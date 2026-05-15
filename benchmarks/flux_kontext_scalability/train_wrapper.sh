#!/usr/bin/env bash
# Inner launcher for the LoRA scalability benchmark.
#
# Generates a per-run YAML from scal_lora_template.yaml with rank / alpha /
# batch size / max_steps filled in, then launches scripts/train.py via
# accelerate (DDP for 2/4 GPUs, plain python for 1 GPU).
#
# Required env vars:
#   REPO_ROOT       — absolute path to the diff_base repository checkout
#   GPU_COUNT       — 1 | 2 | 4 (also accepted as positional arg 1)
#   MODEL_HUB_ROOT  — HuggingFace cache root (used by scal_lora_template.yaml)
#   DATA_ROOT       — paired-dataset root (e.g. ORION CD45 split)
#
# Optional env vars (with defaults):
#   LORA_RANK   — default 16   (sweep dimension: 8, 16, 32, 64, …)
#   LORA_ALPHA  — default == LORA_RANK
#   BATCH_SIZE  — default 4    (per-GPU; effective batch = BATCH_SIZE × GPU_COUNT)
#   MAX_STEPS   — default 150  (50 warmup + 100 measured)
set -euo pipefail

GPU_COUNT="${GPU_COUNT:-${1:?GPU_COUNT required (env or arg1)}}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-${LORA_RANK}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_STEPS="${MAX_STEPS:-150}"

: "${REPO_ROOT:?set REPO_ROOT to the diff_base checkout}"
: "${MODEL_HUB_ROOT:?set MODEL_HUB_ROOT to your HuggingFace cache root}"
: "${DATA_ROOT:?set DATA_ROOT to your paired-dataset root}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/scal_lora_template.yaml"
test -f "${TEMPLATE}" || { echo "ERROR: template missing: ${TEMPLATE}"; exit 1; }

EXP_NAME="scal_lora_r${LORA_RANK}_${GPU_COUNT}gpu_${SLURM_JOB_ID:-local}"
OUT_DIR="${REPO_ROOT}/outputs/${EXP_NAME}"
YAML_OUT="${SCRIPT_DIR}/_generated/${EXP_NAME}.yaml"
mkdir -p "${SCRIPT_DIR}/_generated" "${OUT_DIR}"

# Substitute placeholders in the template. Generated YAML is kept under
# _generated/ for audit (gitignored).
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
