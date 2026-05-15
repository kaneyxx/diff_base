#!/usr/bin/env bash
# Install + verify KempnerPulse on the HMS RC GPU reservation node
# (currently A100 80GB; identical setup on H100 once we move to Kempner).
# KempnerPulse is REQUIRED by the Kempner proposal — this script HARD-FAILS
# if the DCGM backend is unavailable, so we don't quietly produce a degraded
# report.
set -euo pipefail

# 1. Pip install (in already-active venv)
pip install --upgrade kempnerpulse

# 2. Hard requirement: dcgmi must be on PATH on the GPU node
if ! command -v dcgmi >/dev/null 2>&1; then
  echo "ERROR: dcgmi not on PATH. KempnerPulse needs the DCGM backend to" >&2
  echo "       report tensor_active / sm_active / dram_active. Ask HMS RC" >&2
  echo "       to enable DCGM on the GPU partition before re-running."    >&2
  exit 2
fi

# 3. Smoke check — export 1 snapshot, all 34 columns
kempnerpulse --backend dcgm --once --export all > /tmp/kp_smoke.csv
head -2 /tmp/kp_smoke.csv

echo "OK — KempnerPulse + DCGM verified on $(hostname)"
