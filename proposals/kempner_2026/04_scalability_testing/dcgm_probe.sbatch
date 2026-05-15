#!/usr/bin/env bash
# Tiny probe job — runs on the reservation node, checks whether dcgmi and
# dcgm-exporter (Prometheus on :9400) are available. Output answers whether
# KempnerPulse can use --backend dcgm or must fall back to prometheus.
# 1 GPU, 5 minutes, ~0 compute. Queues behind running surv_blt_r tasks.
#SBATCH --job-name=gpu-scalability-test-probe
#SBATCH --partition=gpu_yu
#SBATCH --account=yu_ky98_contrib
#SBATCH --reservation=gpu_yu_res
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/dcgm_probe_%j.out
#SBATCH --error=logs/dcgm_probe_%j.err

set +e   # don't fail-on-error — we WANT all probe outcomes captured
mkdir -p logs

echo "==== node ===="; hostname
echo "==== PATH ===="; echo "$PATH"
echo "==== dcgmi on PATH? ===="; command -v dcgmi || echo "NOT FOUND"
echo "==== dcgmi standard paths ===="; ls -l /usr/bin/dcgmi /usr/local/bin/dcgmi /opt/nvidia/dcgm/bin/dcgmi 2>&1
echo "==== module list dcgm ===="; module avail 2>&1 | grep -iE "dcgm|nvidia" || echo "no module match"
echo "==== dcgm-exporter on :9400 ===="; curl -s --max-time 2 http://localhost:9400/metrics 2>&1 | head -5
echo "==== nvidia-smi ===="; nvidia-smi --query-gpu=name,memory.total --format=csv
echo "==== gres in this job ===="; echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE   SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "DONE"
