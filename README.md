# Diffusion Model Training Framework

A unified training framework for diffusion models supporting multiple architectures (SDXL, SD3.5, FLUX.1, FLUX.2) and training methods (LoRA, Full Fine-tune, DreamBooth, ControlNet, Textual Inversion).

## Features

- **Multiple Architectures**: SDXL, SD3.5 (Large, Large-Turbo, Medium), FLUX.1 (dev, schnell, kontext), and FLUX.2 (dev, klein-4B, klein-9B) support
- **Image Editing**: FLUX.1 Kontext (sequence-wise reference conditioning) and FLUX.2 (Kontext + Fill modes)
- **Training Methods**: LoRA, Full Fine-tuning, DreamBooth, ControlNet, Textual Inversion, Kontext LoRA / Full Fine-tune
- **Noise Schedulers**: DDPM, Euler, Flow Matching
- **Data Pipeline**: Aspect ratio bucketing, latent caching, flexible transforms, paired (target, reference) datasets for Kontext
- **Extensibility**: `block_hooks` and `register_conditioning_module` for downstream models (e.g. ControlNet on Kontext)
- **Configuration-Driven**: YAML configs with inheritance support
- **Memory Optimized**: Gradient checkpointing, mixed precision training
- **Checkpoint Management**: Save/load in safetensors format; FLUX.1 supports both BFL native and HuggingFace diffusers layouts

## Supported Models

| Model | Variants | Text Encoder | Latent Channels | Parameters |
|-------|----------|--------------|-----------------|------------|
| SDXL | base, refiner | CLIP-L + CLIP-G | 4 | ~2.6B |
| SD3.5 | Large | CLIP-L + CLIP-G + T5-XXL | 16 | ~8B |
| SD3.5 | Large-Turbo | CLIP-L + CLIP-G + T5-XXL | 16 | ~8B |
| SD3.5 | Medium | CLIP-L + CLIP-G + T5-XXL | 16 | ~2.5B |
| FLUX.1 | dev, schnell, kontext | T5-XXL + CLIP-L | 16 | ~12B |
| FLUX.2 | dev | Mistral-3 | 32 | ~32B |
| FLUX.2 | klein-4B | Qwen3-4B | 128 | ~4B |
| FLUX.2 | klein-9B | Qwen3-8B | 128 | ~9B |

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/diff_base.git
cd diff_base

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode (required for `from src.models...` imports)
pip install -e .
```

> **Important:** `pip install -e .` is required. Without it, Python cannot resolve `from src.models...` imports used throughout the codebase.

### Optional Dependencies

`requirements.txt` installs everything including dev tools and wandb. For a minimal install, you can use `pyproject.toml` optional groups instead:

```bash
# Minimal (training only, no wandb/pytest)
pip install -e .

# With dev tools (pytest, black, ruff)
pip install -e ".[dev]"

# With logging (wandb, tensorboard)
pip install -e ".[logging]"

# Everything
pip install -e ".[all]"
```

## Quick Start

### 1. Prepare Dataset

Create a dataset directory with images and captions:

```
dataset/
├── metadata.json
├── image_001.png
├── image_002.png
└── ...
```

`metadata.json` format:
```json
[
  {"image": "image_001.png", "caption": "a photo of a cat"},
  {"image": "image_002.png", "caption": "a scenic mountain landscape"}
]
```

Or use text files alongside images:
```
dataset/
├── image_001.png
├── image_001.txt  # Contains caption
└── ...
```

### 2. Create Experiment Config

```yaml
# configs/experiments/my_experiment.yaml
_base_:
  - ../models/sdxl.yaml
  - ../training/lora.yaml

experiment:
  name: "my-lora-training"
  output_dir: "./outputs/my_lora"
  seed: 42

model:
  pretrained_path: "/path/to/sdxl-base-1.0"

training:
  epochs: 50
  batch_size: 2
  gradient_accumulation: 4
  learning_rate: 1.0e-4

  lora:
    rank: 16
    alpha: 16

data:
  train_path: "/path/to/dataset"
  resolution: 1024
  cache_latents: true

hardware:
  mixed_precision: "bf16"
  gradient_checkpointing: true
```

### 3. Run Training

```bash
python scripts/train.py --config configs/experiments/my_experiment.yaml
```

### 4. Generate Images

```bash
python scripts/inference.py \
  --model_path /path/to/sdxl \
  --lora_path ./outputs/my_lora/checkpoint-1000 \
  --prompt "your prompt here" \
  --output ./generated.png
```

## Project Structure

```
diff_base/
├── configs/                 # YAML configuration files
│   ├── models/              # Model architecture configs
│   │   ├── sdxl.yaml
│   │   ├── sd3_large.yaml
│   │   ├── sd3_large_turbo.yaml
│   │   ├── sd3_medium.yaml
│   │   ├── flux.yaml        # Alias for flux1_dev
│   │   ├── flux1_dev.yaml
│   │   ├── flux1_schnell.yaml
│   │   ├── flux1_kontext.yaml
│   │   ├── flux2_dev.yaml
│   │   ├── flux2_klein_4b.yaml
│   │   └── flux2_klein_9b.yaml
│   ├── training/            # Training method configs
│   ├── experiments/         # Complete experiment configs
│   └── accelerate/          # Accelerate launcher configs (multi_gpu_fsdp, single_gpu_efficient)
├── src/
│   ├── models/              # Model definitions
│   │   ├── components/      # Shared components (attention, embeddings, etc.)
│   │   ├── sdxl/            # SDXL architecture
│   │   ├── sd3/             # SD3.5 architecture (MM-DiT)
│   │   │   └── components/  # SD3-specific layers (QKNorm, JointBlock, etc.)
│   │   └── flux/            # Flux architectures
│   │       ├── components/  # Flux-specific shared layers
│   │       ├── v1/          # FLUX.1 (dev, schnell, kontext)
│   │       │   ├── bfl_export.py    # Export to BFL native safetensors
│   │       │   ├── weight_mapping.py  # BFL ↔ internal key mapping
│   │       │   └── EXTENDING.md     # block_hooks + ControlNet extension guide
│   │       └── v2/          # FLUX.2 (dev, klein-4B, klein-9B)
│   │           ├── bfl_export.py    # Export to BFL native safetensors
│   │           └── weight_mapping.py  # BFL ↔ internal key mapping
│   ├── training/            # Training logic
│   │   ├── base_trainer.py
│   │   ├── lora_trainer.py
│   │   ├── full_finetune_trainer.py
│   │   ├── flux_full_finetune_trainer.py  # BFL flow-matching + EMA + 8-bit Adam + FSDP
│   │   ├── kontext_trainer.py             # KontextLoRA + KontextFullFinetune (paired)
│   │   ├── dreambooth_trainer.py
│   │   ├── controlnet_trainer.py
│   │   ├── ema.py                         # EMA shadow weight management
│   │   └── fsdp_setup.py
│   ├── data/                # Data pipeline
│   │   ├── paired_kontext_base.py         # Abstract base for paired datasets
│   │   ├── kontext_dataset.py             # metadata.json / *_target/_ref pairs
│   │   ├── orion_dataset.py               # BAO H5 + JSON splits adapter
│   │   ├── kontext_collate.py             # Shared paired-batch collation
│   │   ├── dataset.py / bucket.py / cache.py / transforms.py / collate.py
│   │   ├── dreambooth_dataset.py
│   │   └── __init__.py                    # KONTEXT_DATASET_REGISTRY + factories
│   ├── schedulers/          # Noise schedulers (DDPM, Euler, flow matching)
│   ├── inference/           # Inference pipelines
│   │   ├── flux1_editing_pipeline.py  # Flux1EditingPipeline (Kontext)
│   │   └── flux2_editing_pipeline.py  # Flux2EditingPipeline (Kontext + Fill)
│   └── utils/               # Utilities (config, checkpoint, logging)
├── scripts/                 # Training and inference entry points
│   ├── train.py                        # General training launcher
│   ├── inference.py                    # Inference script
│   ├── finetune_flux.py                # FLUX full fine-tune CLI (FSDP, EMA, 8-bit Adam)
│   └── download_pretrained_weights.sh  # Helper to fetch pretrained weights
├── tests/                   # Unit tests
├── requirements.txt
└── pyproject.toml
```

## Supported Training Methods

| Method | Description |
|--------|-------------|
| **LoRA** | Low-Rank Adaptation for efficient fine-tuning |
| **Full Fine-tune** | Train all model parameters |
| **DreamBooth** | Personalization with prior preservation |
| **ControlNet** | Add conditional control to generation |
| **Textual Inversion** | Learn new concepts via embeddings |
| **Kontext LoRA** | LoRA fine-tuning on paired (target, reference, caption) datasets for FLUX.1 Kontext |
| **Kontext Full Fine-tune** | Full-parameter fine-tuning for FLUX.1 Kontext |

## Using SD3.5 Models

SD3.5 uses the MM-DiT (Multimodal Diffusion Transformer) architecture with triple text encoders (CLIP-L, OpenCLIP-G, T5-XXL).

### SD3.5 Variants

```yaml
# configs/experiments/sd3_lora.yaml
_base_:
  - ../models/sd3_large.yaml  # or sd3_medium.yaml, sd3_large_turbo.yaml
  - ../training/lora.yaml

model:
  pretrained_path: "stabilityai/stable-diffusion-3.5-large"

training:
  epochs: 50
  batch_size: 1
  lora:
    rank: 16
```

### Programmatic Usage (SD3.5)

```python
from src.models import create_model
from omegaconf import OmegaConf

# Create SD3.5-Medium model
config = OmegaConf.create({
    'model': {
        'type': 'sd3',
        'variant': 'medium',  # or 'large', 'large-turbo'
    }
})
model = create_model(config)
```

## Using FLUX Models

### FLUX.1 (dev/schnell)

```yaml
# configs/experiments/flux1_lora.yaml
_base_:
  - ../models/flux1_dev.yaml  # or flux1_schnell.yaml, flux1_kontext.yaml
  - ../training/lora.yaml

model:
  pretrained_path: "black-forest-labs/FLUX.1-dev"

training:
  epochs: 50
  batch_size: 1
  lora:
    rank: 16
```

### FLUX.1 Kontext (Image Editing)

FLUX.1 Kontext shares the same architecture as `dev` (19 joint + 38 single blocks, T5-XXL + CLIP-L, 16 latent channels) — only the weights differ. Reference images are conditioned via sequence-wise concatenation with `stream` index 0.0 (target) vs 1.0 (reference), exactly matching the BFL official implementation.

**Inference:**

```python
from src.inference.flux1_editing_pipeline import Flux1EditingPipeline
from PIL import Image

pipe = Flux1EditingPipeline.from_pretrained(
    "/path/to/FLUX.1-Kontext-dev",
    variant="kontext",
    device="cuda",
    dtype="bfloat16",
)
result = pipe(
    prompt="Make it look like sunset",
    reference_image=Image.open("input.png"),
    height=1024,
    width=1024,
    num_inference_steps=28,
    guidance_scale=2.5,
)
result[0].save("output.png")
```

By default the reference image is snapped to the nearest of 17 BFL "preferred" resolution buckets; disable with `kontext_snap_resolution=False`.

**Weight loading** auto-detects format. Both BFL native (`flux1-kontext-dev.safetensors` from `black-forest-labs/FLUX.1-Kontext-dev`) and HuggingFace diffusers (`transformer/`, `vae/` subdirs) layouts work.

**Training (LoRA):**

```bash
python scripts/train.py --config configs/experiments/flux1_kontext_lora_example.yaml
```

```yaml
# configs/experiments/flux1_kontext_lora_example.yaml
_base_:
  - ../models/flux1_kontext.yaml
  - ../training/kontext_lora.yaml

model:
  pretrained_path: "black-forest-labs/FLUX.1-Kontext-dev"

data:
  train_path: "/path/to/kontext_dataset"  # metadata.json with target/reference/caption
  resolution: 1024

training:
  epochs: 30
  batch_size: 1
  lora:
    rank: 16
    target_modules: ["to_q", "to_k", "to_v", "to_out.0"]
```

The dataset must be paired. `metadata.json` format:
```json
[
  {"target": "edit_001.png", "reference": "src_001.png", "caption": "make it sunset"},
  {"target": "edit_002.png", "reference": "src_002.png", "caption": "add snow on the roof"}
]
```
Or directory layout: `{stem}_target.png`, `{stem}_ref.png`, `{stem}_caption.txt`.

Loss is computed only on target tokens (reference tokens are sliced from the transformer output before the loss step), structurally preventing reference-token gradient leakage.

**Extending:** the `block_hooks` parameter on `Flux1Transformer.forward()` and `register_conditioning_module()` helper allow building downstream models (e.g. ControlNet on Kontext) without forking the transformer. See `src/models/flux/v1/EXTENDING.md`.

#### Paired-image dataset architecture (Kontext + ORION + your own)

All Kontext-mode training (LoRA or full fine-tune) consumes paired
`(target_image, reference_image, caption)` triplets. The data pipeline is
factored as one **abstract base** plus per-corpus subclasses, so adding a
new paired dataset (BCI, ACROBAT, your-own) is a ~50-line change.

```
src/data/
├── paired_kontext_base.py   # PairedKontextDataset (abstract base)
├── kontext_dataset.py       # KontextDataset    — metadata.json / *_target.png pairs
├── orion_dataset.py         # OrionDataset      — BAO multi-patient H5 + JSON splits
├── kontext_collate.py       # kontext_collate_fn — shared batching
└── __init__.py              # KONTEXT_DATASET_REGISTRY + create_kontext_dataloader()
```

The base class defines a contract; subclasses implement two hooks:

```python
class PairedKontextDataset(Dataset, ABC):
    @abstractmethod
    def _discover_samples(self) -> list:
        """Enumerate samples on disk."""
    @abstractmethod
    def _load_pair(self, idx: int) -> tuple[Image.Image, Image.Image, str]:
        """Return (target_pil, reference_pil, caption)."""
```

CLI usage selects which paired dataset to load via `--dataset-type`:

```bash
# Generic Kontext (metadata.json / paired *_target/_ref images)
python scripts/finetune_flux.py --variant kontext --dataset-type kontext \
    --train-data /path/to/paired_dataset ...

# ORION (Lin et al. Cell 2023) — H&E ↔ multiplex IF biomarker translation
python scripts/finetune_flux.py --variant kontext --dataset-type orion \
    --train-data /path/to/bao/data \
    --train-split /path/to/bao/dataset/training_splits/CD45_train.json \
    --biomarker CD45  # auto-inferred from filename if omitted
```

| CLI flag | Config field | Description |
|---|---|---|
| `--dataset-type` | `data.dataset_type` | Registry key (`kontext`, `orion`, …) |
| `--train-data` | `data.train_path` | Dataset root (e.g. `/path/to/bao/data`) |
| `--train-split` | `data.train_split_path` | Only for `orion`: JSON split file path |
| `--biomarker` | `data.biomarker` | Only for `orion`: target biomarker name |

Registering a new paired dataset for, say, BCI:

```python
# src/data/bci_dataset.py
from .paired_kontext_base import PairedKontextDataset

class BCIDataset(PairedKontextDataset):
    def _discover_samples(self): ...
    def _load_pair(self, idx): ...

# src/data/__init__.py — one line:
from .bci_dataset import BCIDataset
KONTEXT_DATASET_REGISTRY["bci"] = BCIDataset
# or call register_kontext_dataset("bci", BCIDataset) at runtime
```

Once registered, `--dataset-type bci` is immediately available end-to-end
— `--dataset-type` is registry-validated, not argparse-validated, so no
CLI changes are needed.

### FLUX.2 (dev/klein)

```yaml
# configs/experiments/flux2_klein_lora.yaml
_base_:
  - ../models/flux2_klein_4b.yaml  # or flux2_dev.yaml, flux2_klein_9b.yaml
  - ../training/lora.yaml

model:
  pretrained_path: "black-forest-labs/FLUX.2-Klein-4B"

training:
  epochs: 50
  batch_size: 1
```

#### FLUX.2 status (May 2026)

The FLUX.2 path in `src/models/flux/v2/` is **architecturally complete**
(4D RoPE, shared `Flux2Modulation`, SwiGLU, fused `to_qkv_mlp_proj`,
Kontext + Fill conditioning, FSDP wrap policy, BFL↔internal weight
round-trip via `bfl_export.py` / `weight_mapping.py`) but the inference
sampler and REPA-E-VAE projection are still deferred. Full audit:
`docs/flux2_audit.md`.

**Variant viability for full fine-tuning** (the distilled-variant guard
enforces this — pass `--force-distilled` to override at your own risk):

| Variant | License | Distilled | Full-FT? | LoRA? | Recommended for… |
|---|---|---|---|---|---|
| `flux2-dev` | Non-commercial | No | ⚠️ 32B params, needs ≥4 GPU | ✅ | Ablation; research only |
| `flux2-klein-4b-base` | Apache 2.0 | No | ✅ | ✅ | **Default upgrade target** when promoting beyond FLUX.1-Kontext |
| `flux2-klein-9b-base` | Non-commercial | No | ✅ | ✅ | Higher quality ceiling; research only |
| `flux2-klein-4b` | Apache 2.0 | **Yes** | ❌ refused | ⚠️ allowed | Distilled — `--force-distilled` to override |
| `flux2-klein-9b` | Non-commercial | **Yes** | ❌ refused | ⚠️ allowed | Distilled — `--force-distilled` to override |

The CLI surface is unified:

```bash
python scripts/finetune_flux.py --variant flux2-klein-4b-base \
    --pretrained-path /path/to/FLUX.2-klein-4B-base \
    --train-data /path/to/dataset --dataset-type orion \
    --train-split /path/to/CD45_train.json \
    --output-dir ./outputs/flux2-vstain
```

The `--dataset-type orion` flag works identically across FLUX.1-Kontext and
FLUX.2 paths — they share the `PairedKontextDataset` contract. See the
"Paired-image dataset architecture" subsection above.

Weight format conversion is symmetric to FLUX.1:

```python
from src.models.flux.v2.bfl_export import to_bfl_checkpoint
from src.models.flux.v2.weight_mapping import load_flux2_checkpoint

to_bfl_checkpoint(transformer, "flux2-finetune.safetensors")
state_dict = load_flux2_checkpoint("flux2-finetune.safetensors", target="internal")
```

Required weights are documented in `docs/pretrained_weights_inventory.md`
(`scripts/download_pretrained_weights.sh` helper for `repae`,
`flux2-klein-4b-base`, etc.).

### Programmatic Usage

```python
from src.models import create_model
from omegaconf import OmegaConf

# Create FLUX.1-dev model
config = OmegaConf.create({
    'model': {
        'type': 'flux',
        'variant': 'flux1-dev',  # or flux1-schnell, flux2-dev, flux2-klein-4b, etc.
    }
})
model = create_model(config)
```

## Configuration System

Configs support inheritance via `_base_` key:

```yaml
_base_:
  - ../models/sdxl.yaml
  - ../training/lora.yaml

# Override specific values
training:
  epochs: 100
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.2.0 (required for `F.scaled_dot_product_attention` / Flash Attention support)
- CUDA-capable GPU with 24GB+ VRAM (recommended)
- CUDA >= 11.8 (for Flash Attention kernel acceleration)

## Running Tests

```bash
python -m pytest tests/ -v
```

## Documentation & Proposals

- [Full Fine-tune Recipes](docs/full_finetune_recipes.md) — single-GPU, multi-GPU FSDP, FLUX.2, and troubleshooting guides
- [FLUX.2 Architecture Audit](docs/flux2_audit.md) — alignment notes, deferred items, and implementation status
- [AI Toolkit Audit](docs/ai_toolkit_audit.md) — comparison against upstream ai-toolkit codebase
- [Pretrained Weights Inventory](docs/pretrained_weights_inventory.md) — required checkpoints and download paths
- [Kempner 2026 Technical Readiness](proposals/kempner_2026/README.md) — Kempner 2026 grant technical readiness package
- [Kempner 2026 Scalability Testing](proposals/kempner_2026/04_scalability_testing/submission/README.md) — proposal-ready scalability testing package

## License

MIT License

## Acknowledgements

- [Stability AI](https://stability.ai/) for SDXL
- [Black Forest Labs](https://blackforestlabs.ai/) for Flux
- [Hugging Face](https://huggingface.co/) for transformers and diffusers libraries
