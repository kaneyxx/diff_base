# FLUX.2 Audit & Virtual-Staining Feasibility

**Audit date**: 2026-05-14
**Reference**: <https://github.com/black-forest-labs/flux2>
**Originating plan**: `.omc/plans/flux2_audit_and_virtual_staining.md`

This document is a stable summary of the audit findings — read it before
adding new FLUX.2 features, downloading FLUX.2 weights, or proposing
FLUX.2 as the production virtual-staining backbone.

## TL;DR

| Topic | Verdict |
|---|---|
| Architecture | ✅ Complete (4D RoPE / shared modulation / SwiGLU / fused single-block proj / Kontext+Fill / FSDP) |
| Weight conversion | ✅ Round-trip BFL ↔ internal in `bfl_export.py` + `weight_mapping.py` (May 2026) |
| Distilled-variant guard | ✅ Single `--force-distilled` umbrella covers schnell + klein-{4b,9b} |
| Text encoder offline fallback | ✅ Honours `HF_HUB_OFFLINE`; wraps load errors with cache-path hint |
| Inference sampler / pipeline | ⚠️ Deferred — only the editing pipeline class exists; no text-to-image sampler yet |
| REPA-E VAE adapter | ⚠️ Deferred — REPA-E latent is 16-ch, FLUX.2 needs 32/128-ch |
| End-to-end Kontext / Fill tests | ✅ Added in `tests/test_flux2_editing.py` (May 2026) |
| Virtual staining: should we pivot now? | ❌ No — keep FLUX.1-Kontext for the Kempner submission; promote `flux2-klein-4b-base` in Phase 2 |

## 1. What is verified correct in `src/models/flux/v2/`

- **4D RoPE**: `transformer.py:48-50` `FLUX2_AXES_DIM=(32,32,32,32)`,
  `FLUX2_ROPE_THETA=2000`; 3D→4D zero-pad at `transformer.py:347-368`.
- **Shared `Flux2Modulation`**: three module instances (`double_stream_modulation_img/txt`,
  `single_stream_modulation`) at `transformer.py:221-230`, consumed by
  `Flux2TransformerBlock` (`blocks.py:326-390`) and the single block
  (`blocks.py:444-449`).
- **SwiGLU FF**: `blocks.py:24-56` (`Flux2SwiGLU`, `Flux2FeedForward`).
- **`bias=False` everywhere**: verified at `transformer.py:190, 203, 206, 212, 249`
  and `blocks.py:88-102, 222, 320, 425`.
- **Fused single-block QKV+MLP projection**: `blocks.py:195-232`
  (`Flux2ParallelSelfAttention.to_qkv_mlp_proj`).
- **Variant configs**: `transformer.py:VARIANT_CONFIGS` matches the BFL spec
  (dev 8/48, klein-4b 5/20, klein-9b 6/24 blocks; latent 32 / 128 / 128).
- **Text encoders**: `text_encoder.py:ENCODER_CONFIGS` — Mistral-3 for dev,
  Qwen3-4B / Qwen3-8B for klein; pooled dim 4096.
- **Kontext conditioning**: `conditioning.py:prepare_kontext_conditioning`
  with stream index 0.0=target / 1.0=reference; target-only slice now
  applied in `transformer.py:323-342, 416-419` (May 2026 fix).
- **Fill conditioning**: `conditioning.py:prepare_fill_conditioning` —
  channel-wise concat of `masked_image = image * (1 - mask)` plus mask
  channel; verified by `tests/test_flux2_editing.py:test_fill_mask_zeroed_outside_inpaint_region`.
- **FSDP**: `src/training/fsdp_setup.py:58-65` registers both block types.

## 2. What is still deferred

- **GAP-2 — Inference sampler / pipeline**. We have `flux2_editing_pipeline.py`
  but no general text-to-image sampler aligned with BFL's `flux2/sampling.py`.
  Required only when we want to produce sample images from a trained
  FLUX.2 checkpoint. Recommended approach: extend the existing flow-matching
  scheduler under `src/schedulers/`.
- **GAP-5 — REPA-E VAE adapter for FLUX.2**. REPA-E produces a 16-ch latent
  matched to FLUX.1; FLUX.2 expects 32/128 channels. Two options when this
  matters: (a) project 16→32/128 with a tiny conv head, or (b) accept the
  channel mismatch and skip REPA-E for FLUX.2 ablations.
- **GAP-7 — `FLUX_TINY_OVERRIDE` plumbing for v2**. v1 honours the env var
  to swap in tiny CPU stubs; v2 only does so via explicit config overrides
  (tests construct their own tiny models). Quality-of-life only.

## 3. Variant viability for virtual staining

ORION (Lin et al., *Cell* 2023) is a paired H&E ↔ multiplex-IF dataset.
Virtual staining = predicting an IF channel from an H&E patch. Both
FLUX.1-Kontext and FLUX.2 support this via Kontext-mode reference
conditioning, but the variants differ in cost / licensing:

| Variant | Params | License | Distilled | Full-FT viable? | Use case |
|---|---|---|---|---|---|
| `flux2-klein-4b-base` | 4 B | Apache 2.0 | No | ✅ on 1×A100 80GB (LoRA) / ≥4 GPU (full) | **Default upgrade target** |
| `flux2-klein-9b-base` | 9 B | NC | No | ✅ — higher quality cap | Research only |
| `flux2-dev` | 32 B | NC | No | ⚠️ LoRA only on multi-GPU | Last-resort ablation |
| `flux2-klein-4b` (distilled) | 4 B | Apache 2.0 | Yes | ❌ refused | Distillation removes score-matching capacity |
| `flux2-klein-9b` (distilled) | 9 B | NC | Yes | ❌ refused | Same as above |

The refusal is enforced at:
- `src/training/flux_full_finetune_trainer.py:DISTILLED_VARIANTS`
- `scripts/finetune_flux.py:DISTILLED_VARIANTS_CLI`

Both lists are kept in sync. The override flag is `--force-distilled`
(legacy `--force-schnell` still works but emits `DeprecationWarning`).

## 4. Why FLUX.2 isn't worth pivoting to for the Kempner grant

1. **VRAM blow-up**. Klein-4b LoRA ≈30 GB; klein-9b LoRA ≈50 GB; dev LoRA
   requires multi-GPU. FLUX.1-Kontext LoRA fits on a single A100 40GB.
2. **No production-validated inference path** in diff_base (GAP-2).
3. **REPA-E VAE channel mismatch** (GAP-5) — losing the REPA-E benefit
   that ORION currently relies on.
4. **Klein-9b license** is non-commercial; only klein-4b-base is publishable.
5. **No FLUX.2 weights downloaded** — would burn cluster bandwidth + storage.
6. **Narrative cost**: Kempner proposal already cites FLUX.1-Kontext;
   pivoting mid-submission wastes the existing scaffold.

## 5. Why FLUX.2 is the right Phase-2 upgrade target

1. **Native multi-reference editing** — useful for few-shot biomarker
   examples ("predict CD45 like patient CRC03, CRC07, CRC12").
2. **Better text encoders** (Mistral-3 24B / Qwen3 8B vs T5-XXL+CLIP-L)
   carry richer biomarker context.
3. **128-ch latents on klein** capture subtle channel variation better
   than FLUX.1's 16-ch latent (which REPA-E partly addresses).
4. **SwiGLU + shared modulation** — same or higher quality at fewer FLOPs.

## 6. References

- Plan: `.omc/plans/flux2_audit_and_virtual_staining.md`
- Pretrained-weight inventory: `docs/pretrained_weights_inventory.md`
- Download helper: `scripts/download_pretrained_weights.sh`
- FLUX.2 conversion: `src/models/flux/v2/{bfl_export.py,weight_mapping.py}`
- FLUX.2 tests: `tests/test_flux2_editing.py`, `tests/test_flux2_weight_loading.py`,
  `tests/test_flux2_text_encoder.py`
- Distilled refusal: `src/training/flux_full_finetune_trainer.py` (`DISTILLED_VARIANTS`)
- Lin et al., *Cell* 2023, ORION dataset.
- BFL FLUX.2 reference: <https://github.com/black-forest-labs/flux2>
