#!/usr/bin/env bash
# Background nvidia-smi GPU metrics recorder — 1 Hz CSV samples per GPU.
# Output is consumed by plot_scaling_results.py and parse_metrics_csv.py.
#
# Usage: ./gpu_metrics_recorder.sh <output_csv> <poll_seconds>
set -euo pipefail

OUT="${1:?output csv path required}"
POLL_S="${2:-1.0}"

HEADER="timestamp,gpu_id,model,gpu_util_pct,mem_used_mib,mem_total_mib,mem_used_pct,power_w,gpu_temp_c,mem_temp_c,sm_clock_mhz,mem_clock_mhz"

mkdir -p "$(dirname "${OUT}")"
echo "${HEADER}" > "${OUT}"

trap 'exit 0' TERM INT

while true; do
  TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,temperature.memory,clocks.current.graphics,clocks.current.memory \
    --format=csv,noheader,nounits 2>/dev/null | while IFS=, read -r idx name util mem_used mem_total power gpu_temp mem_temp sm_clk mem_clk; do
    idx=${idx// /}; name=${name## }; util=${util// /};
    mem_used=${mem_used// /}; mem_total=${mem_total// /};
    power=${power// /}; gpu_temp=${gpu_temp// /}; mem_temp=${mem_temp// /};
    sm_clk=${sm_clk// /}; mem_clk=${mem_clk// /}

    mem_pct="N/A"
    if [[ -n "${mem_total}" && "${mem_total}" != "0" ]]; then
      mem_pct=$(awk "BEGIN {printf \"%.2f\", (${mem_used}/${mem_total})*100}")
    fi

    echo "${TS},${idx},${name},${util},${mem_used},${mem_total},${mem_pct},${power},${gpu_temp},${mem_temp},${sm_clk},${mem_clk}"
  done >> "${OUT}"
  sleep "${POLL_S}"
done
