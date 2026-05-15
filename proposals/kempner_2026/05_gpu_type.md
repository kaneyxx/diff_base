GPU type requested for production: **H100 80GB (Hopper, SXM5)**.

Why H100 specifically:
  - 80 GB HBM3: required because a FLUX.1-Kontext full fine-tune at 1024² with
    EMA on CPU and 8-bit Adam needs ≈68 GB of resident VRAM (see
    `docs/full_finetune_recipes.md` and `src/training/memory_planning.py`).
    A100 40GB cannot hold the model + optimizer + activations; A100 80GB
    works but is ~30% slower in bf16 and lacks FP8.
  - FP8 Transformer Engine (H100-only): planned ablation for 1.5–2×
    throughput at matched quality. If FP8 turns out unstable for this
    workload, the baseline bf16 path still runs (no functional regression).
  - NVLink: required for FSDP all-gather efficiency on the 4-GPU node.

Scalability testing was performed on a **4× A100 80GB** reservation node
on HMS RC because that is what we currently have local reservation access
to. The same training stack (FLUX.1-Kontext + REPA-E VAE + FSDP wrap policy
on `FluxJointTransformerBlock` / `FluxSingleTransformerBlock`) is hardware
agnostic — moving to H100 changes only (a) per-step latency (faster), (b)
optional FP8 enablement, and (c) NVLink bandwidth headroom. The A100 80GB
numbers in `04_scalability_testing/results/summary_table.md` are therefore
a conservative lower bound for what we will see on Kempner H100s.

Fallback acceptance: A100 80GB is acceptable if H100 is unavailable;
throughput drops ~30% but training remains feasible at full quality.

Mixed precision: bfloat16 throughout (autocast); FP8 only for forward
matmuls inside the transformer blocks if Transformer Engine is enabled
(H100 only).
