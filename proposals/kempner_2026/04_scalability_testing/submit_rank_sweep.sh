#!/usr/bin/env bash
# Run a LoRA rank sweep × GPU count sweep on the reservation node.
# Useful for showing how trainable-param scale × GPU count interact in the
# Kempner proposal: rank=8 (light) vs rank=64 (compute-bound) reveals when
# multi-GPU scaling kicks in vs. when single-GPU is enough.
#
# Each rank submits 3 jobs (1/2/4 GPU), so a 4-rank sweep = 12 jobs total.
# Sequential between ranks (we wait for one rank's 3 jobs to complete before
# the next) so the reservation isn't oversubscribed.
#
# Usage:
#   ./submit_rank_sweep.sh                # default ranks: 8 16 32 64
#   ./submit_rank_sweep.sh 16 64          # subset of ranks
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

# Default rank set spans an order of magnitude in trainable param count.
if [[ $# -eq 0 ]]; then
  RANKS=(8 16 32 64)
else
  RANKS=("$@")
fi

ALL_SUBMITTED=()
for RANK in "${RANKS[@]}"; do
  ALPHA="${RANK}"   # convention: alpha = rank for proportional scaling
  echo ""
  echo "==== rank=${RANK} alpha=${ALPHA} ===="
  # bs grows inversely with rank to keep activations roughly constant
  case "${RANK}" in
    8)  BS=4 ;;   # rank=8  → low memory, larger bs
    16) BS=2 ;;   # rank=16 baseline (US-5 numbers)
    32) BS=2 ;;
    64) BS=1 ;;   # rank=64 → heavier LoRA layers, smaller bs
    *)  BS=1 ;;
  esac

  LORA_RANK="${RANK}" LORA_ALPHA="${ALPHA}" BATCH_SIZE="${BS}" \
    ./submit_scal_sweep.sh
done

echo ""
echo "All ranks submitted. Monitor with:"
echo "  squeue -u \$USER -t R,PD -o '%.10i %.34j %.2t %.10M %.16R'"
