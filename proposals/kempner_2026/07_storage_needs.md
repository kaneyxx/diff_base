Storage breakdown (rounded UP to nearest TB at the bottom):

| Item | Estimate | Notes |
|---|---|---|
| ORION pre-tiled corpus (CRC01..CRC35+, on disk) | 1.7 TB | Confirmed via `du -sh /n/scratch/users/f/fas994/bao/data/` |
| Cached REPA-E VAE latents for ORION (~10× compression on PNG side; H5 already compact) | 0.2 TB | One-time pre-compute to skip per-step VAE encoding |
| FLUX.1-Kontext-dev base weights (BFL safetensors) | 0.03 TB | One copy per cluster |
| REPA-E VAE weights | 0.01 TB | https://huggingface.co/REPA-E/e2e-flux-vae |
| T5-XXL + CLIP-L text encoders | 0.02 TB | Frozen, encoded outputs cached |
| Singularity image (.sif) | 0.05 TB | PyTorch 24.04 NGC base |
| Full-fine-tune checkpoints (5 × ~30 GB main + EMA) | 0.3 TB | Per-biomarker × top-of-range |
| LoRA adapters (13 biomarkers × ~0.5 GB) | 0.01 TB | Cheap to retain all |
| Logs / wandb cache | 0.05 TB | wandb offline + local TensorBoard |
| Ablation runs / extra biomarkers | 0.5 TB | Reserve |
| **Total** | **~2.9 TB → request 4 TB** | |

Output artifacts (rounded UP, separate from working storage):
- Final exported transformer checkpoints (BFL + diffusers, all biomarkers): ≤ 0.5 TB
- Sample/validation outputs (visual H&E vs predicted IF comparisons): ≤ 0.05 TB

Long-term storage: weights pushed to HF Hub upon publication; the ORION raw
tiles are openly available (Synapse) so we do not need to retain a permanent
private copy after the project ends.
