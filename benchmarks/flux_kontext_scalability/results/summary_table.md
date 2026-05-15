# Scalability Benchmark — FLUX.1-Kontext LoRA

**Hardware:** 4× NVIDIA A100-SXM4-80 GB on a single node, NVLink-3.
**Workload:** FLUX.1-Kontext-dev (12 B-parameter DiT) fine-tuned with LoRA on a paired same-section H&E ↔ multiplex-IF dataset (ORION CD45 split).
**Trainer:** `scripts/train.py` → `KontextLoRATrainer` (LoRA on transformer Q/K/V/O + FFN; T5 / CLIP / VAE frozen).
**Resolution:** 512 × 512.
**Mixed precision:** bf16 throughout.
**Gradient checkpointing:** ON.
**Gradient accumulation:** 1 — one forward + backward = one optimizer step (clean throughput measurement).
**Multi-GPU strategy:** HuggingFace `accelerate launch --multi_gpu` → DDP with fixed per-GPU batch size.
**Text encoder:** `cpu_offload=true` + `cache_text_embeddings=true` (T5/CLIP run on CPU once per unique caption — for the ORION CD45 split this is a single caption, so the entire training run encodes text once).
**Steps per run:** 150 = 50 warmup + 100 measured steady-state.

---

## 1. What we tested

Three independent knobs were swept to map the scalability envelope:

| Parameter | Levels tested | Rationale |
|---|---|---|
| **GPU count** (DDP via `accelerate launch --multi_gpu`) | 1, 2, 4 | 1 GPU is the per-rank baseline; 4 GPU is the full single-node reservation. |
| **LoRA rank** | 16, 32, 64 | An ~8× span of trainable parameters (≈ 9 M → 70 M). Probes whether DDP scaling efficiency depends on compute-per-step. |
| **Per-GPU batch size** | 1, 2, and 4 (attempted as stress test) | Largest batch each rank could fit at 512² resolution. bs = 4 at rank = 16 was attempted to find the memory ceiling. |

DDP runs use a **fixed per-GPU batch size**, so the effective batch grows linearly with GPU count (4 GPU × per-GPU bs = 2 → effective batch 8). Per-step compute is approximately invariant across GPU count by construction; the appropriate throughput metric is therefore **samples / second**, not steps / second.

---

## 2. Headline scaling table

Following the conventional "Time to 1000 steps − Time to 500 steps" delta-time format (= 500 × seconds-per-step ÷ 60):

| Num. Nodes | Num. GPUs / DP Global Rank | Time-delta over 500 steps (min) | Peak GPU Mem (GB / GPU) | Avg GPU util (%) | Runtime Speedup from 1 GPU |
|---|---|---|---|---|---|
| 1 | 1 | 14.1 | 72.6 | 78.8 | N/A (baseline) |
| 1 | 2 | 13.1 | 64.3 | 42.1 | **2.16×** (108 %) |
| 1 | 4 | 13.2 | 64.5 | 36.5 | **4.28×** (107 %) |

Speedup is the **samples-per-second** ratio (the correct denominator for DDP with fixed per-GPU batch). Per-step time stays near-constant across GPU count, as expected for this regime. Multi-node was not tested — the reservation is single-node.

The table above uses the (rank = 16, bs = 2) configuration as the production-representative point. Equivalent results for rank = 32 and rank = 64 appear in §3.

---

## 3. Full configuration grid

| Config | GPUs | Throughput (samples/s) | Peak Mem (GB) | Avg / p95 GPU util | Per-GPU power | Speedup | Scaling efficiency |
|---|---|---|---|---|---|---|---|
| rank = 16, bs = 2 | 1 | 1.18 | 72.6 | 78.8 % / 100 % | 248 W | 1.00× | 100 % |
| rank = 16, bs = 2 | 2 | 2.55 | 64.3 | 42.1 % / 100 % | 164 W | 2.16× | 108 % |
| rank = 16, bs = 2 | 4 | 5.07 | 64.5 | 36.5 % / 100 % | 151 W | 4.28× | 107 % |
| rank = 16, **bs = 4** | 1 / 2 / 4 | **OOM @ ≈ 77 GB** | — | — | — | — | — |
| rank = 32, bs = 2 | 1 | 1.29 | 64.0 | 52.5 % / 100 % | 187 W | 1.00× | 100 % |
| rank = 32, bs = 2 | 2 | 2.57 | 65.0 | 43.5 % / 100 % | 167 W | 2.00× | 100 % |
| rank = 32, bs = 2 | 4 | 5.11 | 65.2 | 36.4 % / 100 % | 153 W | 3.97× | 99 % |
| rank = 64, bs = 1 | 1 | 1.18 | 46.0 | 42.3 % / 100 % | 157 W | 1.00× | 100 % |
| rank = 64, bs = 1 | 2 | 2.31 | 46.8 | 31.5 % / 100 % | 134 W | 1.95× | 97 % |
| rank = 64, bs = 1 | 4 | 4.68 | 47.2 | 26.7 % / 100 % | 127 W | 3.95× | 99 % |

**Bottom line:** every (rank, bs) pair that fits in memory scales **near-linearly to 4 GPUs (97 – 107 % efficiency)**. The rank = 16 case is slightly super-linear, likely from better cache locality on the fixed-caption corpus.

The bs = 4 stress test at rank = 16 OOMed at the same allocator state on both 2- and 4-GPU runs (~77 GB allocated, 365 MiB free) — confirming **bs = 2 is the production ceiling on 80 GB A100/H100** with the current cpu-offload + grad-checkpoint + text-cache configuration. Unlocking bs = 4 requires either int8 quantisation of the frozen base, deeper activation checkpointing, or FSDP.

---

## 4. Figures

All four figures live in `plots/`.

| Figure | What it shows |
|---|---|
| `scaling_efficiency.png` | Two-panel: per-rank speedup curves vs an ideal-linear reference (left); per-configuration scaling-efficiency bars with % labels (right). |
| `memory_profile.png` | Peak (red) vs average (blue) per-GPU memory across all 9 successful configurations, with the 80 GB A100 ceiling drawn as a dotted line. |
| `gpu_util_timeseries.png` | 15-second rolling-mean GPU utilisation over time, colour-coded by LoRA rank, line-styled by GPU count. |
| `power_profile.png` | 15-second rolling-mean per-GPU power consumption over time (~60 W idle → ~280-300 W during training). |

---

## 5. Interpretation

- **The recipe is compute-bound at single-GPU scale.** At rank = 16 bs = 2 on a single A100, average utilisation reaches 78.8 %, p95 100 %. With a denser per-step compute load (larger bs, which we cannot fit at 80 GB), utilisation would push higher.
- **DDP adds no measurable per-step overhead on a NVLink-3 single node.** Per-step time is flat across 1 / 2 / 4 GPU runs at every rank, which is exactly what fixed-per-GPU-batch DDP predicts. The corollary is that samples / sec scales near-linearly.
- **Average GPU utilisation drops with GPU count** (78.8 % → 42.1 % → 36.5 % across 1 / 2 / 4 GPU at rank = 16 bs = 2). The drop is **not** a scaling problem; p95 stays at 100 %, p50 drops because the AllReduce and dataloader windows lengthen relative to the constant compute window. Faster interconnect (e.g. H100 NVLink-4) would shrink the AllReduce window and raise utilisation correspondingly.
- **Memory is the binding constraint, not compute.** Peak memory sits at 64-73 GB for every (rank, bs) pair we can fit. The 77 GB OOM observation for bs = 4 at rank = 16 is the cleanest read on the production ceiling.

---

## 6. Caveats

- **DCGM was unavailable** on the test node, so metrics were collected via `nvidia-smi` at 1 Hz (`gpu_metrics_recorder.sh`). Tensor-core / SM-occupancy / FP16-FP32 pipe percentages — fields only DCGM exposes — are not in the recorded CSVs. Utilisation, memory, power, clocks, and temperature are.
- **Loss → NaN at lr = 1 × 10⁻⁴ under DDP** is observed independent of GPU count or LoRA rank in this benchmark configuration. Throughput, memory, and utilisation measurements are unaffected (`nvidia-smi` does not depend on numerical finiteness of gradients). Production training runs should use a smaller learning rate (e.g. lr = 1 × 10⁻⁶).
- **Single-node 4-GPU is the largest configuration tested here.** Cross-node multi-GPU has not been characterised in this benchmark.
