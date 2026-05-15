#!/usr/bin/env bash
# nvidia-smi based GPU metrics recorder. Writes a KempnerPulse-compatible CSV
# subset (the fields that nvidia-smi can supply without DCGM).
#
# Why this exists: KempnerPulse needs `dcgmi` or a dcgm-exporter Prometheus
# endpoint. Neither is available on HMS RC gpu_yu reservation node
# (verified by proposals/.../logs/dcgm_probe_40044599.out, 2026-05-14). For
# the Kempner proposal, we document this limitation and capture the metrics
# that nvidia-smi can supply at 1 Hz (utilisation, memory, power, clocks,
# temperature). DCGM-only fields (tensor_active, sm_occupancy, fp16/32_pipe,
# energy, tensor-core pct) are written as N/A and noted as gaps.
#
# Usage: ./gpu_metrics_recorder.sh <output_csv> <poll_hz>
set -euo pipefail

OUT="${1:?output csv path required}"
POLL_S="${2:-1.0}"

# KempnerPulse-aligned header. Fields nvidia-smi cannot provide carry "N/A".
HEADER="timestamp,gpu_id,model,real_util_pct,status,health,sm_active_pct,tensor_active_pct,dram_active_pct,gr_engine_active_pct,gpu_util_pct,mem_used_mib,mem_total_mib,mem_used_pct,power_w,gpu_temp_c,mem_temp_c,sm_occupancy_pct,fp16_pipe_pct,fp32_pipe_pct,fp64_pipe_pct,memcpy_util_pct,pcie_rx_bytes_s,pcie_tx_bytes_s,nvlink_gbps,sm_clock_mhz,mem_clock_mhz,pcie_replay_rate_s,energy_j,tc_hmma_pct,tc_imma_pct,tc_dfma_pct,tc_dmma_pct,tc_qmma_pct"

mkdir -p "$(dirname "${OUT}")"
echo "${HEADER}" > "${OUT}"

# Trap SIGTERM to flush + exit cleanly
trap 'exit 0' TERM INT

while true; do
  TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  # nvidia-smi --query-gpu fields; CSV stays compact, no units in output.
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu,temperature.memory,clocks.current.graphics,clocks.current.memory \
    --format=csv,noheader,nounits 2>/dev/null | while IFS=, read -r idx name util mem_used mem_total power gpu_temp mem_temp sm_clk mem_clk; do
    # Trim leading whitespace from comma-split fields
    idx=${idx// /}; name=${name## }; util=${util// /};
    mem_used=${mem_used// /}; mem_total=${mem_total// /};
    power=${power// /}; gpu_temp=${gpu_temp// /}; mem_temp=${mem_temp// /};
    sm_clk=${sm_clk// /}; mem_clk=${mem_clk// /}

    # Compute mem_used_pct
    mem_pct="N/A"
    if [[ -n "${mem_total}" && "${mem_total}" != "0" ]]; then
      mem_pct=$(awk "BEGIN {printf \"%.2f\", (${mem_used}/${mem_total})*100}")
    fi

    # status / health: assume HEALTHY when nvidia-smi responded
    # Field order matches KempnerPulse HEADER. DCGM-only fields → N/A.
    echo "${TS},${idx},${name},${util},RUNNING,HEALTHY,N/A,N/A,N/A,N/A,${util},${mem_used},${mem_total},${mem_pct},${power},${gpu_temp},${mem_temp},N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,${sm_clk},${mem_clk},N/A,N/A,N/A,N/A,N/A,N/A,N/A"
  done >> "${OUT}"
  sleep "${POLL_S}"
done
