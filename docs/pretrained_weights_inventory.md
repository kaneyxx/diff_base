# Pretrained Weights Inventory

**HF cache root**: `/n/scratch/users/f/fas994/huggingface/` (`HF_HOME`).
**Last audited**: 2026-05-14.

## Current state

| Repo | Status | Size | Notes |
|---|---|---|---|
| `black-forest-labs/FLUX.1-Kontext-dev` | ✅ Complete | ~37 GB | 7 subdirs (`scheduler/text_encoder/text_encoder_2/tokenizer/tokenizer_2/transformer/vae`). Default `PRETRAINED` for the Kempner scalability test. |
| `REPA-E/e2e-flux-vae` | ❌ **Empty snapshot** | — | `.no_exist/` directory + lock files only — must re-download before any REPA-E run. |
| `t5-base`, `t5-small` | ⚠️ Empty | — | Not blockers (FLUX uses T5-XXL bundled in Kontext-dev, not these standalone repos). |
| `musk_weights/model.safetensors` | ℹ️ Standalone | ~1 GB | Pathology foundation model — unrelated to FLUX. |
| **FLUX.2 anything** | ❌ Absent | — | Plan to download `flux2-klein-4b-base` once Phase 2 starts. |

## Required downloads (in priority order)

1. **REPA-E VAE** (highest priority — blocks ORION + REPA-E pathway).
   ```
   scripts/download_pretrained_weights.sh repae
   ```
   Expected size: ~2 GB.

2. **FLUX.2 klein-4B-base** (Phase 2; queue when GPU lands).
   ```
   scripts/download_pretrained_weights.sh flux2-klein-4b-base
   ```
   Expected size: ~8 GB. Apache 2.0 license — preferred for any release.

3. **FLUX.2 klein-9B-base** (Phase 2 stretch goal).
   ```
   scripts/download_pretrained_weights.sh flux2-klein-9b-base
   ```
   Expected size: ~18 GB. Non-commercial license — research-only.

4. **FLUX.2 dev** (only if quality cap on klein is hit).
   ```
   scripts/download_pretrained_weights.sh flux2-dev
   ```
   Expected size: ~64 GB. 32B params; only viable for LoRA on multi-GPU.

## Recommended timing

| Trigger | Download |
|---|---|
| Anytime (no GPU needed for the download itself) | `repae` |
| Once a GPU allocation lands and the FLUX.1-Kontext + ORION smoke passes | `flux2-klein-4b-base` |
| If FLUX.2 klein-4B ablation underperforms FLUX.1-Kontext at the same parameter budget | `flux2-klein-9b-base` |
| If both klein variants underperform | `flux2-dev` (last resort — 64 GB) |

## Dry-run

To preview the exact commands without downloading anything:

```bash
scripts/download_pretrained_weights.sh all --dry-run
```

## Storage budget

Cluster scratch is at `/n/scratch/users/f/fas994/` with current usage ~133 GB
across the HF cache. The full inventory above (REPA-E + 3 FLUX.2 variants)
adds ~92 GB. Plan to either:
- Stage downloads serially (delete unused checkpoints between phases), or
- Request a scratch quota increase before Phase 2 starts.

See `.omc/plans/flux2_audit_and_virtual_staining.md` §1 for the originating
plan.
