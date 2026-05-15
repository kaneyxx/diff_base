# NOTE: This is the submission-ready copy. Replace ${REPO_ROOT} with your
#       cluster path (e.g. /home/you/diff_base) before launching. The dev
#       version with absolute paths lives in the parent dir.
#!/usr/bin/env bash
# Scalability testing — uses a local 4×A100 80GB reservation node on HMS RC.
# Production target is H100 (see 05_gpu_type.md). KempnerPulse is identical on
# both — DCGM backend supports A100 and H100. The numbers we collect here
# transfer as a lower-bound for H100 since A100 lacks FP8/Transformer Engine.
#SBATCH --job-name=fluxkontext-scal
#SBATCH --partition=gpu_yu                # BAO lab partition; adjust if different
#SBATCH --account=yu_ky98_contrib         # BAO lab account; adjust if different
#SBATCH --reservation=gpu_yu_res          # 4×A100 80GB reservation on compute-gc-17-252
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   # gpu_yu partition caps total CPUs at 20 per job
#SBATCH --mem=512G
#SBATCH --time=04:00:00
#SBATCH --output=${REPO_ROOT}/proposals/kempner_2026/04_scalability_testing/logs/scalability_%j.out
#SBATCH --error=${REPO_ROOT}/proposals/kempner_2026/04_scalability_testing/logs/scalability_%j.err
# NOTE: --gres is NOT pinned here; submit via submit_scal_sweep.sh which sets
# `--gres=gpu:a100:${GPU_COUNT}` and `--export=GPU_COUNT=...` per job, so each
# GPU-count test can independently queue/start as resources free up (instead of
# all 3 tasks waiting for all 4 GPUs to be idle simultaneously).

set -euo pipefail
# Absolute paths so paths don't depend on submit dir or worker cwd.
PROPOSAL_DIR="${REPO_ROOT}/proposals/kempner_2026/04_scalability_testing"
mkdir -p "${PROPOSAL_DIR}/logs" "${PROPOSAL_DIR}/results"

module load gcc/12.2.0 cuda/12.4.1 || true
# Python env: BAO lab conda env on /n/data2/. No venv; PATH-prepend is enough.
export PATH=/n/data2/hms/dbmi/kyu/lab/fas994/conda_env/kontext/bin:${PATH}
export HF_HOME=${MODEL_HUB_ROOT}
test -x "$(command -v python)" || { echo "ERROR: kontext python not on PATH"; exit 1; }

# GPU_COUNT is set by submit_scal_sweep.sh via --export=GPU_COUNT=N
GPU_COUNT="${GPU_COUNT:?GPU_COUNT must be set (1, 2, or 4) — submit via submit_scal_sweep.sh}"
case "${GPU_COUNT}" in
  1|2|4) : ;;
  *) echo "GPU_COUNT must be 1, 2, or 4 (got: ${GPU_COUNT})"; exit 1 ;;
esac

export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))

# Start KempnerPulse in background — 1 Hz, all 34 columns.
# Backend choice: dcgm requires `dcgmi` on PATH (best — 100ms+ resolution);
# falls back to prometheus if dcgm-exporter is up at localhost:9400; if
# neither is available, the proposal requirement (KempnerPulse mandatory) is
# not met — we hard-fail so we don't quietly produce a degraded report.
PULSE_CSV="${PROPOSAL_DIR}/results/kempnerpulse_${GPU_COUNT}gpu_${SLURM_JOB_ID}.csv"
if command -v dcgmi >/dev/null 2>&1; then
  echo "KempnerPulse backend = dcgm"
  kempnerpulse --backend dcgm --poll 1.0 --export all > "${PULSE_CSV}" &
  PULSE_PID=$!
elif curl -s --max-time 2 http://localhost:9400/metrics >/dev/null 2>&1; then
  echo "KempnerPulse backend = prometheus"
  kempnerpulse --backend prometheus --poll 1.0 --export all > "${PULSE_CSV}" &
  PULSE_PID=$!
else
  # HMS RC reservation does NOT have DCGM (verified 2026-05-14 via probe job
  # 40044599). Fall back to nvidia-smi based recorder. Proposal limitation
  # documented in cgroup_check.txt — DCGM-only metrics (tensor_active,
  # sm_occupancy, tc_hmma_pct …) are N/A but utilisation / memory / power /
  # clocks / temperature are captured at 1 Hz with KempnerPulse-aligned columns.
  echo "WARNING: dcgmi + dcgm-exporter both unavailable on $(hostname);" >&2
  echo "         falling back to nvidia-smi recorder. DCGM-only metrics will be N/A." >&2
  bash "${PROPOSAL_DIR}/gpu_metrics_recorder.sh" "${PULSE_CSV}" 1.0 &
  PULSE_PID=$!
fi
# Memory / cgroup notes:
#   - Smoke session on gpu_quad OOM-killed at 64 GB cgroup limit (A40 node).
#   - Reservation cgroup limit on gpu_yu_res (A100 80 GB): TBD — US-5 will
#     record the actual value via `sacctmgr show res gpu_yu_res`.
#   - text_encoder.cpu_offload is OFF in train_wrapper.sh because A100 80 GB
#     has enough VRAM to keep the text encoder on GPU for this sweep.
trap 'kill -TERM ${PULSE_PID} 2>/dev/null; wait ${PULSE_PID} 2>/dev/null || true' EXIT

bash "${PROPOSAL_DIR}/train_wrapper.sh" "${GPU_COUNT}"
