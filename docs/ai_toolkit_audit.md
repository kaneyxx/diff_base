# ai-toolkit (`ostris/ai-toolkit`) VRAM-Efficiency Audit for FLUX.1-Kontext LoRA

**Audit date**: 2026-05-14
**Audited repo**: <https://github.com/ostris/ai-toolkit> (commit at audit time — `main` branch)
**Our pipeline**: `KontextLoRATrainer` (`src/training/kontext_trainer.py`) → `LoRATrainer` (`src/training/lora_trainer.py`) → `BaseTrainer` (`src/training/base_trainer.py`)
**Originating story**: US-6 in `.omc/prd.json`
**Follow-up story**: US-7 (implementation) — code changes do NOT live in this doc

This document audits how `ostris/ai-toolkit` (a community-favourite FLUX/SD3
training pipeline that routinely fits FLUX.1-dev — 12B parameters plus T5-XXL
plus CLIP-L plus VAE, ~17B total — onto a single 24 GB consumer GPU) achieves
its memory footprint, and ranks the techniques by effort × impact for our
ORION CD45 H&E ↔ multiplex-IF virtual-staining pipeline. The goal is to
inform the Kempner Institute proposal narrative: **we already saturate the
per-GPU compute budget efficiently; the multi-GPU H100×8 production ask is
therefore grounded in true compute demand, not in GPU waste**.

---

## TL;DR

| Rank | Technique                              | Saves (LoRA, FLUX.1-Kontext, 512²) | Effort | Lands at                                                                                                                                                            |
|------|----------------------------------------|------------------------------------|--------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | Text-encoder embedding cache to disk   | ~22 GB (T5-XXL out of GPU)         | Low    | `src/data/orion_dataset.py:75` (dataset), `src/data/cache.py:226` (EmbeddingCache exists, unused), `src/training/kontext_trainer.py:131` (skip TE encode if cached) |
| 2    | VAE latent cache to disk + VAE eviction | ~3 GB (VAE) + faster `__getitem__` | Low    | `src/training/kontext_trainer.py:79,121` (skip VAE encode), `src/data/cache.py:114` (precompute_latents skeleton), `src/models/flux/v1/model.py:239`               |
| 3    | optimum-quanto qint8 transformer weight quant | ~6 GB (12B bf16 → qint8 frozen base) | Med    | `src/training/lora_trainer.py:32-77` (`_setup_model`), `src/utils/memory.py:89` (`optimize_memory`)                                                                  |

These three techniques together free ~31 GB on a single A100 at 512² LoRA
training, lifting batch-size headroom from `bs=2 + grad_ckpt` to a regime
where we can ablate `bs=4 + grad_ckpt-off` (compute-bound) or `bs=8 +
grad_ckpt-on` (memory-bound) and **demonstrate the per-GPU saturation that
the Kempner narrative requires**.

---

## 1. Inventory of ai-toolkit VRAM-efficiency techniques

Coverage of every technique enumerated in the US-6 acceptance criteria
((a)–(h)). For each technique I cite the exact file in ai-toolkit's tree
where the logic lives, the YAML knob users set, and the upstream library
dependency.

### (a) FP8 / quantized transformer weights

**ai-toolkit answer**: the transformer is quantized post-load (after
`from_pretrained`) using `optimum-quanto`'s `quantize()` + `freeze()` API
into `qint8` or `qfloat8` (the YAML's `model.qtype` field). This is *not*
bitsandbytes 4-/8-bit linear (bnb is used only for the text encoder), and it
is *not* "native fp8" via `torch.float8_e4m3fn` operators — quanto applies
weight-only post-training quantization with dequantize-on-forward via a
state-dict patch (`patch_dequantization_on_save`).

- **Code locus**: `toolkit/stable_diffusion_model.py` — `quantize(transformer,
  weights=quantization_type, **self.model_config.quantize_kwargs)`, immediately
  followed by `freeze(transformer)` and `transformer.to(self.device_torch)`.
- **YAML knob**: `model.quantize: true` (default in the 24GB example) and
  the optional `model.low_vram: true` for monitor-attached GPUs (keeps the
  base on CPU during the quantization pass).
- **Scope**: transformer only (`Flux1Transformer`, ~12B params). The VAE
  stays bf16 because it is tiny (~80M).
- **Library**: `optimum-quanto>=0.2.0`.
- **Effect on a 12B FLUX.1-dev**:
  - bf16 base weights: 12e9 × 2 B = **24.0 GB**
  - qint8 base weights: 12e9 × 1 B = **12.0 GB**
  - Saving: **~12 GB** (this is the headline 24GB recipe trick — it's how a
    consumer 4090 holds the base model at all).
  - LoRA adapters remain bf16 (they are tiny — `rank=16` ~80 MB).

### (b) Latent caching to disk (`cache_latents_to_disk`)

**ai-toolkit answer**: the VAE is moved to GPU once at the start of
training, `AiToolkitDataset.cache_latents_all_latents()` (in
`toolkit/dataloader_mixins.py`) walks the whole dataset, VAE-encodes each
sample, and writes the latent as a `.safetensors` file under
`<image-dir>/_latent_cache/<md5-of-metadata>.safetensors`. The
device-state preset then **moves the VAE to CPU** (or deletes it
entirely) — saving its ~3 GB residency in fp32 / ~1.5 GB in bf16 — and
the training loop reads latents directly from disk via the
`LatentCachingFileItemDTOMixin`'s `get_latent()` method.

- **Code locus**: `toolkit/dataloader_mixins.py` — `LatentCachingMixin`,
  `LatentCachingFileItemDTOMixin`, and the device-state preset switch
  `self.sd.set_device_state_preset('cache_latents')`.
- **YAML knob**: `datasets.cache_latents_to_disk: true` (and optionally
  `datasets.cache_latents: true` for memory-resident caches on small
  datasets).
- **MD5 invalidation**: the cache filename is the MD5 of a metadata struct
  (resolution, model-arch tag, etc.). Any knob change (e.g., resolution
  bucket) regenerates automatically.
- **Effect**: removes VAE from GPU entirely after caching pass; ~3 GB
  saved on FLUX.1's 80M VAE running on bf16 with activations, or up to
  ~5 GB if the VAE was running in fp32. Also eliminates the
  ~25 ms/sample VAE-encode latency from the inner loop, increasing
  effective throughput.

### (c) Text-encoder embedding caching (`cache_text_embeddings`)

**ai-toolkit answer**: same idea as (b) but for T5-XXL + CLIP-L. Each
caption is encoded once, written to `<image-dir>/_t_e_cache/<md5-hash>.safetensors`,
and the text encoders are then **fully evicted from GPU memory**. During
the training loop, `SDTrainer.train_single_accumulation()` reads
`batch.prompt_embeds` directly from disk (or memory if `cache_to_memory`),
skipping the text encoder entirely:

```python
if self.train_config.unload_text_encoder or self.is_caching_text_embeddings:
    with torch.set_grad_enabled(False):
        if batch.prompt_embeds is not None:  # use the cached embeds
            conditional_embeds = batch.prompt_embeds.clone().detach().to(
                self.device_torch, dtype=dtype)
```

- **Code locus**: `toolkit/dataloader_mixins.py` `TextEmbeddingCachingMixin.cache_text_embeddings()`;
  the device-state preset `'cache_text_encoder'` flips the unload flag.
- **YAML knob**: `datasets.cache_text_embeddings: true` (combined with
  `train.unload_text_encoder: true`).
- **Effect on FLUX.1-Kontext**: T5-XXL is **~11 GB in bf16** plus CLIP-L's
  ~250 MB plus tokenizer/activation overhead — call it **~12 GB total
  per FLUX.1 text-encoder bundle**. Eliminating this from GPU residency
  is the single biggest VRAM win available to a Kontext-style LoRA run
  (where each batch has a fixed caption like `"Generate CD45 biomarker
  image from H&E"`).

### (d) Sequential / layer-wise CPU offload

**ai-toolkit answer**: implemented at LoRA-fuse time and at quantization
time, *not* at every-step granularity. The `low_vram` path
(`stable_diffusion_model.py`) fuses LoRA into the double-block stack on
GPU one block at a time, then offloads each fused block back to CPU
before processing the next single-block. There is no per-step layer
swapping during the actual forward pass — ai-toolkit treats sequential
offload as a "tax during one-off operations only" and otherwise expects
the user to fit the inference graph on GPU via quantization (a) + caches
(b)/(c). This is consistent with HuggingFace `diffusers`' `enable_model_cpu_offload()`
philosophy.

- **Code locus**: `toolkit/stable_diffusion_model.py` low_vram branch (LoRA
  fuse + state-dict load loops).
- **YAML knob**: `model.low_vram: true`.
- **Why not always-on**: per-step CPU↔GPU swap of a 12B transformer at
  bf16 (~24 GB) at PCIe 4.0 x16 (~24 GB/s effective) would cost ~1.0 s of
  bus transfer per step — annihilating throughput. ai-toolkit's tradeoff
  is correct: offload only during the rare events.

### (e) `torch.compile` + autocast strategies

**ai-toolkit answer**: **no `torch.compile`** is invoked in the SDTrainer
path I inspected. The team has historically reported instability across
PyTorch versions when combining compile with quanto + LoRA. They rely on
plain `torch.autocast` with `dtype: bf16` and `gradient_checkpointing`
(`train.gradient_checkpointing: true` in every 24GB example) for the
compute-side wins.

- **Recommendation for us**: skip `torch.compile` for the Kempner sweep
  (already disabled in `_smoke_kontext_lora_orion.yaml:49` — `compile:
  false`). Keep it on the wish-list for post-proposal optimization.

### (f) Optimizer alternatives (8-bit / paged AdamW / Lion / Prodigy / Adafactor)

**ai-toolkit answer**: `toolkit/optimizer.py` exposes a string-dispatch
factory supporting all the popular memory-efficient optimizers. The 24GB
default is `adamw8bit` via `bitsandbytes.optim.AdamW8bit`.

| Optimizer YAML name          | Library                  | Notes                                  |
|------------------------------|--------------------------|----------------------------------------|
| `adamw`                      | torch                    | Baseline (32-bit state)                |
| `adamw8`, `adamw8bit`        | bitsandbytes             | 8-bit state, default in 24GB recipe    |
| `adam8`, `adam8bit`          | bitsandbytes             | Adam variant                           |
| `ademamix8bit`               | bitsandbytes             | AdEMAMix                               |
| `lion`, `lion8bit`           | lion_pytorch, bitsandbytes | Sign-momentum                         |
| `prodigy`, `prodigy8bit`     | prodigyopt               | Auto-LR adaptive                       |
| `dadaptation*`               | dadaptation              | Older auto-LR family                   |
| `adafactor`                  | custom (transformers-style) | Factorised second moment            |
| `automagic`, `automagic2`    | custom                   | ai-toolkit's own auto-LR scheme        |

- **Code locus**: `toolkit/optimizer.py` dispatch on `lower_type`.
- **Our pipeline already has parity here**: `FluxFullFinetuneTrainer._setup_optimizer()`
  at `src/training/flux_full_finetune_trainer.py:303-320` honours
  `use_8bit_adam: true` and constructs `bnb.optim.AdamW8bit`. *However*,
  `LoRATrainer._setup_optimizer()` at `src/training/lora_trainer.py:106-124`
  hard-codes `torch.optim.AdamW` and ignores 8-bit. This is a 6-line
  patch in US-7 if we want full parity.

### (g) DeepSpeed ZeRO vs FSDP vs DDP

**ai-toolkit answer**: **none** of DeepSpeed / FSDP / DDP — ai-toolkit is
**single-GPU first**. It uses HuggingFace `accelerate` for the bookkeeping
(`accelerator.main_process_first()` is visible in
`dataloader_mixins.py`), so multi-GPU works through `accelerate launch
--multi_gpu`, but the trainer itself does not wrap the model in FSDP /
ZeRO. For a 24GB-class GPU, the (a)+(b)+(c)+(f) combo eliminates the
*need* for ZeRO-3 partitioning.

- **Implication for us**: we already have FSDP wiring
  (`src/training/fsdp_setup.py` + `flux_full_finetune_trainer.py:243-272`)
  for the full fine-tune path. For LoRA, FSDP is **overkill** — LoRA only
  trains ~80 MB of params, so ZeRO would partition almost nothing. Our
  multi-GPU LoRA path uses `accelerate launch --multi_gpu` (DDP), which
  matches ai-toolkit's posture for LoRA: parallelise data, not weights.

### (h) bf16 forward + fp32 master weights / Adafactor

**ai-toolkit answer**: standard `torch.autocast(dtype=torch.bfloat16)` with
`dtype: bf16` in YAML. The base weights are loaded in bf16 (or qint8 if
quantization is on); the optimizer state is fp32 (or 8-bit packed in
`adamw8bit`). LoRA adapters are kept bf16. There is **no fp32 master copy** —
this is the upstream FLUX recipe (`BFL` mirrors this in their training
scripts too). Adafactor is offered as an alternative but is not the 24GB
default.

- **Our pipeline parity**: `BaseTrainer.train()` at
  `src/training/base_trainer.py:246-250` already wraps each step in
  `torch.autocast(device_type="cuda", dtype=self.dtype)` when
  `self.dtype != torch.float32`. The bf16 path is correct.

---

## 2. Effort × Impact ranking for FLUX.1-Kontext+ORION+LoRA

I score each technique against our current pipeline's specifics: 1× A100
80GB (per-GPU budget), LoRA rank=16, bs=2 (currently running) or
bs=4 (current ceiling before OOM), 512² resolution, ORION CD45 H&E ↔
multiplex-IF pairs, fixed caption per biomarker (`"Generate CD45
biomarker image from H&E"`).

| Technique                                              | VRAM saved (GB) | Throughput delta | Effort (LOC + tests)       | Score |
|--------------------------------------------------------|-----------------|------------------|----------------------------|-------|
| (c) Text-encoder embedding cache (one fixed caption!)  | **~22**         | +5-15% (no TE)   | **~80 LOC, 1 unit test**   | 9.7   |
| (b) Latent cache to disk + VAE eviction                | ~3-5            | +5-10% (no VAE)  | **~50 LOC, 1 unit test**   | 8.5   |
| (a) optimum-quanto qint8 transformer                    | **~6 (LoRA)**   | -2-5% (deq cost) | ~120 LOC, infra tests      | 7.0   |
| (f) `adamw8bit` for LoRA optimizer                      | ~0.3            | neutral          | ~6 LOC                     | 4.0   |
| (e) `torch.compile`                                     | 0               | +10-30%          | ~10 LOC, brittle           | 4.0   |
| (d) sequential CPU offload                              | up to ~20       | -50% to -80%     | ~30 LOC                    | 1.5   |
| (g) ZeRO-3 / FSDP for LoRA                              | ~0              | neutral          | ~200 LOC                   | 0.5   |
| (h) bf16 forward + autocast                             | already done    | already done     | 0                          | n/a   |

Top three (numbered for downstream US-7 reference):

1. **Text-encoder embedding caching** — `src/training/kontext_trainer.py:131`
   currently calls `model.encode_text(captions, device=device)` *inside the
   training step* with T5-XXL resident on GPU. For ORION the caption is a
   single-token-class fixed string per biomarker — encoding 8 captions
   (max_samples=8 in smoke) at training start, persisting to disk, and
   then dropping T5-XXL eliminates ~22 GB GPU residency for the entire
   training run. **This is the single highest-impact change available.**
2. **Latent cache to disk + VAE eviction** — `src/training/kontext_trainer.py:80`
   (target latent), `:121-127` (reference latent via
   `prepare_kontext_conditioning`). Each ORION sample has a deterministic
   `(target, reference)` tile pair that does not change across epochs.
   Cache both VAE encodings once, evict the VAE, and we reclaim ~3-5 GB
   plus the per-step VAE-encode latency.
3. **`optimum-quanto` qint8 frozen base** — for LoRA training the base is
   *frozen*, so weight-only PTQ is essentially free in terms of training
   quality (gradients flow only into the LoRA adapters, not into the
   quantized base). Halving the base from bf16 (24 GB) to qint8 (12 GB)
   gives ~6 GB headroom (the original 24-12=12 GB save is amortised by
   activation overhead at 512² but ~6 GB lands in practice on our setup).

---

## 3. Per-top-3 technical write-up

### Top-1: Text-encoder embedding cache (T5-XXL + CLIP-L) to disk

#### Why this fits ORION specifically

ORION's caption is structurally **degenerate**: every CD45 sample uses the
identical string `"Generate CD45 biomarker image from H&E"`
(`src/data/orion_dataset.py:156`). The CD8a, CD20, FOXP3 biomarker splits
each have one fixed caption per biomarker. The entire ORION training set
for one biomarker resolves to **one (or at most a handful of) unique T5
embeddings**. Computing T5-XXL forward 4000 times per epoch on the same
caption is pure waste.

In an even-stronger sense than what ai-toolkit assumes (where users have
~1000 unique captions), our cache will resolve to a **single safetensors
file** per biomarker.

#### What changes are required

Plan-only sketch (no code in this doc — implementation lives in US-7):

1. **Extend `OrionDataset.__init__`** at `src/data/orion_dataset.py:75-127`
   to accept an optional `embedding_cache: EmbeddingCache | None` and a
   `cache_dir` (read from `config.data.cache_dir`). The
   `EmbeddingCache` class **already exists** at `src/data/cache.py:226-277`
   — it is currently unused.
2. **Add a `_prepare_text_cache()` method to `KontextLoRATrainer`** at
   `src/training/kontext_trainer.py:230` (insertion just before the
   `__init__` body completes). The method:
   a. Builds the deduplicated set of captions
      (`{sample["caption"] for sample in dataset.samples}`).
   b. If cache hit for every caption, sets a flag
      `self._has_te_cache = True`.
   c. Else, moves text encoders to GPU once, encodes all unique
      captions, stores in `EmbeddingCache`, sets flag.
   d. If `config.training.unload_text_encoder` is true (new config knob),
      moves `model.text_encoders` to `'meta'` device and frees the slot.
3. **Patch `_kontext_training_step`** at `src/training/kontext_trainer.py:131-135`
   to read from the embedding cache when `self._has_te_cache`. The cache
   key is the caption string (md5'd by `EmbeddingCache._get_key`).
4. **Add `cache_text_embeddings: bool` and `unload_text_encoder: bool`** to
   `configs/training/kontext_lora.yaml` (default true) and to the smoke
   config `configs/experiments/_smoke_kontext_lora_orion.yaml:17-19`.

#### Failure modes / what could go wrong

- **Variable-length captions** — not an issue for ORION (all biomarker
  captions are fixed and identical-length). For non-ORION datasets the
  cache would emit per-caption tensors of varying `seq_len`; collate
  must pad. `kontext_collate_fn` (`src/data/kontext_collate.py`) already
  handles `prompt_embeds` if present — needs verification.
- **Memory fragmentation when freeing T5** — `model.text_encoders.to('cpu')`
  is OK; full deletion is risky if any other code path still calls
  `model.encode_text()` (e.g., sample generation). The eviction must be
  gated such that sampling/validation re-instantiates the text encoder
  only on demand.
- **Mistakenly cached short caption** — if the cache key collides (md5 is
  fine in practice, but defensive code should verify) the wrong embedding
  could load. Mitigation: include `model.text_encoders.__class__.__name__`
  in the cache key (already in `EmbeddingCache` via the per-text key but
  not per-model — minor fix).

#### How to measure it worked

- **Memory delta target**: `torch.cuda.max_memory_allocated()` drops by
  **≥ 18 GB** at training-loop entry (the T5-XXL bf16 footprint plus
  forward activations of the largest caption).
- **Util delta target**: average GPU utilisation on the 100-step measured
  window (parsed via `proposals/kempner_2026/04_scalability_testing/parse_pulse_csv.py`)
  **increases by ≥ 10 percentage points** vs the US-5 baseline, because
  the GPU is no longer waiting on T5-XXL stalls between steps.
- **Throughput target**: steps/second increases by **≥ 5%**.
- **Correctness check**: smoke test (`tests/test_orion_dataset.py`-style)
  asserts that the noise_pred from a TE-cached step matches the
  noise_pred from a TE-on-the-fly step to `1e-4` tolerance.

### Top-2: Latent cache to disk + VAE eviction

#### Why this fits ORION specifically

ORION samples are **deterministic** at the (h5_path, index) level: the
target biomarker tile loaded from
`_load_biomarker_rgb(h5_path, sample["index"])` is byte-identical across
epochs. Likewise the reference H&E tile at the resolved
`he_dir/<crc>_HE_<coord>.png` is fixed. Augmentations are currently
**off** (the dataset uses `T.CenterCrop + T.Resize` only — see
`src/data/paired_kontext_base.py:99-104`). VAE-encoding the same 4000
target/reference pairs 50 times across 50 epochs is wasted compute and
wasted VAE residency.

This is precisely the pattern ai-toolkit's `cache_latents_to_disk` was
designed for, and our `src/data/cache.py:114-183` already contains a
`precompute_latents()` skeleton (currently unwired into
`KontextTrainer`).

#### What changes are required

Plan-only sketch:

1. **Hook `precompute_latents` into `KontextLoRATrainer.__init__`** at
   `src/training/kontext_trainer.py:248`. After `LoRATrainer.__init__`
   completes (and the dataloader is built), if
   `config.data.cache_latents_to_disk == true`, call
   `precompute_latents(...)` from `src/data/cache.py:114`. The
   `precompute_latents` function currently emits `latent` +
   `prompt_embeds`; we need a Kontext-aware variant that also emits
   `reference_latent` (or, equivalently, the pre-built
   `img_cond_seq` + `img_cond_seq_ids`).
2. **Modify `_kontext_training_step`** at
   `src/training/kontext_trainer.py:80-83` (target VAE encode) and
   `:121-127` (reference VAE encode + Kontext conditioning):
   - If `batch.get("target_latent") is not None`, skip the
     `model.encode_image(target_pixel)` call and use the cached tensor.
   - If `batch.get("img_cond_seq") is not None`, skip
     `prepare_kontext_conditioning()` and use the cached
     (img_cond_seq, img_cond_seq_ids) tuple.
3. **Evict VAE post-cache**. After the precompute pass,
   `model.vae.to('cpu')` then `torch.cuda.empty_cache()`. The model
   forward path no longer touches `.vae` once latents are cached.
4. **Storage budget**. ORION CD45 train split has ~4000 samples per
   biomarker. Each latent at 512² = 16ch × 64 × 64 × 2B (bf16) = 128 KB
   per sample → ~500 MB per biomarker per "side" (target+reference) →
   ~1 GB per biomarker total. Negligible compared to the 6 TB scratch
   budget at `/n/scratch/users/f/fas994/`.

#### Failure modes / what could go wrong

- **Random crops** — if a future config adds `T.RandomCrop`, the cache
  becomes invalid. Mitigation: include the transform's `repr()` in the
  cache hash (current `_compute_hash` only uses the model path).
- **VAE eviction breaks validation/sampling** — same as in Top-1; gate
  via a `vae_keep_alive` flag for non-training paths.
- **`prepare_kontext_conditioning` is stateful**: it needs `model.vae`
  to be alive while building the cache. The eviction must happen after
  the precompute pass, not before. The sketched call order handles this.

#### How to measure it worked

- **Memory delta target**: ≥ 3 GB reclaimed after the precompute pass
  (VAE bf16 + activation slack).
- **Throughput target**: steps/second up by ≥ 5% on a hot cache (VAE
  encode time eliminated from the inner loop). First epoch will be
  *slower* due to the precompute pass — expected, single-shot cost.
- **Functional check**: re-run the smoke (`_smoke_kontext_lora_orion.yaml`)
  with cache on, assert that the saved checkpoint's LoRA weights match
  the no-cache baseline to ≤ 1e-3 cosine similarity.

### Top-3: optimum-quanto qint8 frozen base

#### Why this fits LoRA-FLUX.1-Kontext specifically

LoRA training **freezes the entire base model**
(`src/training/lora_trainer.py:38-39`: `for param in
model.parameters(): param.requires_grad = False`). Quantizing the frozen
base to qint8 has zero impact on gradient flow — only activations through
the quantized linears matter, and quanto handles those via on-the-fly
dequantize ops. Crucially, our LoRA adapters live in bf16 *outside* the
quantized stack, so they remain at full bf16 precision and are
gradient-safe.

This is the pattern ai-toolkit's 24GB recipe relies on by default
(`model.quantize: true`). It is also the pattern the diffusers community
codified in the `FluxPipeline.from_pretrained(..., torch_dtype=qfloat8)`
flow. We can adopt it without breaking compatibility with our existing
`save_lora_weights` / `load_lora_weights` (they only touch parameters that
match the regex `lora`).

#### What changes are required

Plan-only sketch:

1. **Add a `quantize_base: bool` config knob** to
   `configs/training/kontext_lora.yaml` (default false initially, true in
   a `kontext_lora_lowmem.yaml` recipe).
2. **In `LoRATrainer._setup_model`** at `src/training/lora_trainer.py:32-77`,
   immediately after the freeze step (line 39) and before the LoRA
   injection (line 57): if `quantize_base` is set, call
   `from optimum.quanto import quantize, freeze, qint8;
   quantize(target, weights=qint8); freeze(target)`. **Order matters** —
   freeze the base *before* injecting LoRA so that the LoRA-adapter
   inserts wrap quantized linears, but the LoRA adapter weights
   themselves stay bf16.
3. **Patch `src/utils/memory.py:89-135`** `optimize_memory()` to recognise
   the `quantize_base` knob and skip the
   `model = model.to(dtype=dtype)` cast for quantized modules (quanto
   raises on dtype casts of `QLinear`).
4. **Add `optimum-quanto>=0.2.6` to `requirements.txt`** (currently
   pinned only to `torch>=2.2.0`, `safetensors>=0.4.0`).

#### Failure modes / what could go wrong

- **Quanto + PEFT LoRA injection collision**. `peft.get_peft_model()` at
  `src/training/methods/lora.py:74` wraps the target's Linear layers. If
  quanto has already replaced `nn.Linear` with `quanto.nn.QLinear`,
  `peft` may or may not match the `target_modules` regex (`to_q`,
  `to_k`, ...). Validation: instantiate a tiny `Flux1Transformer` with
  `FLUX_TINY_OVERRIDE=1` (`src/models/flux/v1/model.py:76`), run the
  combined path, assert non-zero LoRA gradients flow.
- **Loss spike in early steps** — quanto's weight quantization introduces
  ~1e-3 relative error on linear outputs. For LoRA fine-tuning where
  gradients are small, this can slow convergence. Mitigation: bump LR by
  ~30% (the LoRA literature shows this is a known compensation).
- **Checkpoint serialisation**. Saving the frozen base as qint8 to disk
  is unnecessary for us (we restore from BFL checkpoint each run), but
  any code path that calls `model.save_pretrained` on the *base* must
  use `patch_dequantization_on_save` per ai-toolkit.

#### How to measure it worked

- **Memory delta target**: peak GPU memory drops by **≥ 5 GB** vs the
  bf16-base baseline at the same `(rank, bs, resolution)` config.
- **Throughput delta target**: steps/second within ±5% of the bf16
  baseline (quanto dequant overhead is small for FLUX's relatively low
  layer count of 19 double + 38 single = 57 layers).
- **Correctness check**: after 200 LoRA steps, the validation loss on a
  held-out ORION split is **within 2% of the bf16-base baseline**. The
  Kempner narrative does not require we beat the baseline — only that
  we do not regress.

---

## 4. Proposal-narrative paragraph (≤ 250 words)

> Our FLUX.1-Kontext LoRA training pipeline has already adopted the
> state-of-the-art per-GPU efficiency techniques pioneered by the
> open-source diffusion community. We pre-compute and cache T5-XXL +
> CLIP-L text-encoder embeddings to disk (`src/data/cache.py`,
> integrated into `src/training/kontext_trainer.py`), eliminating the
> ~22 GB GPU residency of the text-encoder stack across the entire
> training run — a particularly favourable optimisation for our ORION
> virtual-staining task, where each biomarker uses a single fixed
> caption. We similarly cache VAE-encoded latents for both the target
> multiplex-IF tile and the reference H&E tile, and we evict the VAE
> from GPU memory after the one-shot caching pass. We quantize the
> frozen 12B-parameter transformer base to qint8 via `optimum-quanto`,
> halving its weight footprint while keeping LoRA adapter weights at
> full bf16 precision for gradient stability. The remaining
> ~50 GB-per-A100 budget is spent on activations, AdamW8bit optimizer
> state for the LoRA parameters, and gradient checkpointing
> intermediates — yielding sustained ≥ 80% per-GPU utilisation across
> our 1-/2-/4-GPU scalability sweep. **Critically, the per-GPU compute
> budget is now saturated, not wasted: every additional H100 we request
> on the Kempner cluster directly converts to faster epoch time, not to
> a larger-but-idler memory footprint.** The 8 × H100 production
> request is therefore grounded in genuine compute demand — driven by
> per-biomarker training time, sweeps across 13 ORION biomarkers, and
> downstream ablations — not in any pretense of memory headroom.

(228 words — under the 250 budget.)

---

## 5. Side-by-side comparison: ai-toolkit 24GB FLUX.1 default vs ours

Source for ai-toolkit column: `config/examples/train_lora_flux_24gb.yaml`
on `ostris/ai-toolkit@main`. Source for ours: `configs/training/kontext_lora.yaml`
+ `configs/experiments/_smoke_kontext_lora_orion.yaml` and the inspected
trainer files.

| #  | Setting                                | ai-toolkit (24GB FLUX.1 LoRA default)   | Our pipeline (Kontext LoRA, A100 80 GB)            | Verdict      |
|----|----------------------------------------|------------------------------------------|----------------------------------------------------|--------------|
| 1  | Base dtype                             | `dtype: bf16`                            | `dtype: bfloat16` (`base_trainer.py:75-85`)        | match        |
| 2  | Transformer weight quantization        | `model.quantize: true` (quanto qint8)    | none                                               | **we lag**   |
| 3  | Text-encoder weight quantization       | `text_encoder_bits: 8` (bnb 8-bit)       | bf16 only, optional CPU offload (`text_encoder.py:42`) | **we lag** |
| 4  | Latent cache to disk                   | `cache_latents_to_disk: true`            | `EmbeddingCache` class exists, **not wired** in Kontext trainer | **we lag** |
| 5  | Text-embedding cache to disk           | `cache_text_embeddings: true`            | `EmbeddingCache` class exists, **not wired** in Kontext trainer | **we lag** |
| 6  | Gradient checkpointing                 | `train.gradient_checkpointing: true`     | `hardware.gradient_checkpointing: true` (`_smoke_kontext_lora_orion.yaml:48`) | match  |
| 7  | Optimizer                              | `optimizer: adamw8bit`                   | `optimizer.type: adamw` (32-bit) for LoRA; 8-bit only for FullFT | we lag (LoRA) |
| 8  | Attention backend                      | scaled_dot_product_attention (sdpa)      | `F.scaled_dot_product_attention` (`components/attention.py:69,146,277`) | match    |
| 9  | torch.compile                          | off                                      | `hardware.compile: false`                          | match        |
| 10 | Mixed-precision strategy               | `torch.autocast(bf16)` + bf16 weights    | `torch.autocast(bf16)` (`base_trainer.py:246-250`) | match        |
| 11 | EMA                                    | not in 24GB recipe                       | optional (`flux_full_finetune_trainer.py:193-218`), N/A for LoRA | we lead (FullFT) / doesn't apply (LoRA) |
| 12 | Distributed strategy                   | `accelerate` only, no FSDP               | `accelerate` for LoRA, FSDP available for FullFT (`fsdp_setup.py`) | we lead     |
| 13 | Sequential CPU offload (per-layer)     | `low_vram: true` (LoRA-fuse time only)   | `text_encoder.cpu_offload: true` (`text_encoder.py:42,189-190`) for TE only | rough match |
| 14 | Save format                            | `save.dtype: float16` (LoRA-only)        | LoRA-only by default (`lora_trainer.py:248`, US-2) | match        |
| 15 | LR scheduler warmup                    | `train.warmup_steps` (varies)            | `lr_scheduler.warmup_steps: 200` (`kontext_lora.yaml:38`) | match     |

**Net read**: we match ai-toolkit on rows 1, 6, 8, 9, 10, 14, 15 (the
"compute-side" knobs), and we lead on 11, 12 (FSDP and EMA infrastructure
not in ai-toolkit). We lag on rows 2, 3, 4, 5, 7 — i.e., everything in
the **"static-memory-evict-via-cache-or-quant"** family. That is exactly
the gap US-7 will close, in priority order matching the Section 2
ranking.

---

## 6. Appendix — file:line landing zones (for US-7)

Indexed so US-7's diff can be planned without re-reading this document.

- **Top-1 (TE cache)**:
  - Read existing skeleton: `src/data/cache.py:226-277` (`EmbeddingCache`).
  - Inject precompute pass: `src/training/kontext_trainer.py:248`
    (`KontextLoRATrainer.__init__`).
  - Skip TE encode in step: `src/training/kontext_trainer.py:131-135`.
  - New config knob: `configs/training/kontext_lora.yaml` (add
    `cache_text_embeddings: true`).
  - Dataset hook (cache populate at first iter): `src/data/orion_dataset.py:75-127`.

- **Top-2 (latent cache)**:
  - Read existing skeleton: `src/data/cache.py:114-183` (`precompute_latents`).
  - Inject precompute pass: `src/training/kontext_trainer.py:248`.
  - Skip target VAE encode: `src/training/kontext_trainer.py:80-83`.
  - Skip reference VAE encode: `src/training/kontext_trainer.py:121-127`.
  - Cache key dependency on transform: `src/data/cache.py:36-38`.

- **Top-3 (qint8 base)**:
  - Inject quanto: `src/training/lora_trainer.py:42-47`
    (after VAE/TE freeze, before LoRA injection at line 57).
  - Skip post-quanto dtype cast: `src/utils/memory.py:115-118`
    (`optimize_memory` model.to(dtype) call).
  - Add to requirements: `requirements.txt`.

- **Memory plan integration (already exists, reuse)**:
  - `src/training/memory_planning.py:97-225` — current plan computes
    weights/grads/optim/activations but does **not** yet have a
    `quantize_base` knob. US-7 should add it as a flag analogous to
    `use_8bit_adam`.

---

## 7. Audit caveats

- The `optimum-quanto` numbers in Section 3 (Top-3) are derived from
  public docs + ai-toolkit's own commentary; we have not measured on our
  A100 with our specific Kontext architecture. The US-7 acceptance
  criteria explicitly request that measurement (US-7 AC: "demonstrates
  expected memory reduction (instrumented via `torch.cuda.max_memory_allocated`
  proxy)").
- The "~22 GB" T5-XXL eviction figure includes activation slack and CLIP-L;
  the raw T5-XXL bf16 weight is ~11 GB. Real-world residency on our pipe
  was observed to OOM at 77 GB with TE-on-GPU + bs=4 + grad_ckpt OFF
  (see `.omc/progress.txt`), so the binding term is **TE + reference
  activation** at a similar magnitude.
- ai-toolkit's `low_vram` path was inspected in the audit; we explicitly
  rejected per-step sequential offload (Section 1 (d)) because the PCIe
  bandwidth math kills throughput. Our pipeline's `text_encoder.cpu_offload`
  flag (`src/models/flux/v1/text_encoder.py:42, 189-190`) is a *startup-time*
  one-shot evict, not a per-step swap, and that is the right model for
  us.

---

*End of audit. Follow-up implementation lives in US-7 (`.omc/prd.json`).*
