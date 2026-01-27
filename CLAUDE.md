# Diffusion Model Training Framework

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run training
python scripts/train.py --config configs/experiments/sdxl_lora_example.yaml
```

---

## Project Overview

### Purpose
A unified training framework for diffusion models supporting multiple architectures and training methods. Designed for high-end hardware (48GB+ VRAM) with clean, modular architecture.

### Supported Models
| Model | Variants | Features |
|-------|----------|----------|
| SDXL | base-1.0 | UNet-based, DDPM scheduler |
| FLUX.1 | dev, schnell | DiT-based, flow matching, Kontext editing |
| FLUX.2 | dev, klein-4b, klein-9b | DiT-based, Kontext + Fill editing |
| SD3.5 | large, medium | MMDiT-based, rectified flow |

### Design Philosophy
- **Local-first initialization**: Define model structure in code, then load weights
- **Configuration-driven**: All training parameters via YAML files
- **Modular architecture**: Each component is independently swappable
- **Training method agnostic**: Same model code works for all approaches

---

## Directory Structure

```
diff_base/
├── CLAUDE.md                    # This file - project documentation
├── configs/                     # YAML configuration files
│   ├── models/                  # Model architecture configs
│   │   ├── sdxl.yaml
│   │   ├── flux.yaml
│   │   └── sd3.yaml
│   ├── training/                # Training method configs
│   │   ├── lora.yaml
│   │   ├── full_finetune.yaml
│   │   ├── dreambooth.yaml
│   │   └── controlnet.yaml
│   └── experiments/             # Complete experiment configs
│       └── sdxl_lora_example.yaml
├── src/
│   ├── __init__.py
│   ├── models/                  # Model definitions
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract base model class
│   │   ├── sdxl/                # SDXL implementation
│   │   ├── flux/                # FLUX implementations
│   │   │   ├── __init__.py
│   │   │   ├── v1/              # FLUX.1 (dev, schnell)
│   │   │   │   ├── model.py
│   │   │   │   ├── transformer.py
│   │   │   │   ├── vae.py
│   │   │   │   ├── text_encoder.py
│   │   │   │   └── conditioning.py  # Kontext support
│   │   │   ├── v2/              # FLUX.2 (dev, klein)
│   │   │   │   ├── model.py
│   │   │   │   ├── transformer.py
│   │   │   │   ├── vae.py
│   │   │   │   ├── text_encoder.py
│   │   │   │   └── conditioning.py  # Kontext + Fill support
│   │   │   └── components/      # Shared FLUX components
│   │   │       ├── attention.py
│   │   │       ├── embeddings.py
│   │   │       └── layers.py
│   │   ├── sd3/                 # SD3.5 implementation
│   │   └── components/          # Shared components
│   ├── training/                # Training logic
│   ├── data/                    # Data pipeline
│   ├── inference/               # Inference pipelines
│   ├── utils/                   # Utilities
│   └── schedulers/              # Noise schedulers
├── scripts/                     # Entry point scripts
├── tests/                       # Test suite
├── requirements.txt
└── pyproject.toml
```

---

## Model Implementations

### SDXL

UNet-based architecture with dual text encoders (CLIP-L + CLIP-G).

```yaml
model:
  type: "sdxl"
  unet:
    in_channels: 4
    model_channels: 320
    attention_resolutions: [4, 2, 1]
    transformer_depth: [1, 2, 10, 10]
    context_dim: 2048
  vae:
    latent_channels: 4
    scaling_factor: 0.13025
  text_encoder:
    clip_l: { hidden_size: 768, num_layers: 12 }
    clip_g: { hidden_size: 1280, num_layers: 32 }
  scheduler:
    type: "ddpm"
    prediction_type: "epsilon"
```

### FLUX.1

DiT (Diffusion Transformer) architecture with T5-XXL + CLIP-L text encoders.

**Variants:**
| Variant | Joint Blocks | Single Blocks | Guidance | Notes |
|---------|-------------|---------------|----------|-------|
| dev | 19 | 38 | Yes | Full model |
| schnell | 19 | 38 | No | Distilled |

**Image Editing Support:**
- **Kontext Mode**: Reference image editing via sequence-wise concatenation
- **Fill Mode**: NOT supported (use FLUX.2)

```yaml
model:
  type: "flux"
  variant: "dev"
  transformer:
    hidden_size: 3072
    num_attention_heads: 24
    num_layers: 19
    num_single_layers: 38
    in_channels: 64  # 16 latent channels * 4
    guidance_embeds: true
  vae:
    latent_channels: 16
    scaling_factor: 0.3611
    shift_factor: 0.1159
  text_encoder:
    t5: { hidden_size: 4096, num_layers: 24 }
    clip_l: { hidden_size: 768, num_layers: 12 }
  scheduler:
    type: "flow_matching"
```

### FLUX.2

Enhanced DiT architecture with QK normalization and Mistral/Qwen text encoders.

**Variants:**
| Variant | Joint Blocks | Single Blocks | Latent Channels | Text Encoder |
|---------|-------------|---------------|-----------------|--------------|
| dev | 8 | 48 | 32 | Mistral-3 |
| klein-4b | 5 | 20 | 128 | Qwen3-4B |
| klein-9b | 6 | 24 | 128 | Qwen3-8B |

**Image Editing Support:**
- **Kontext Mode**: Reference image editing via sequence-wise concatenation
- **Fill Mode**: Inpainting via channel-wise concatenation with masks

```yaml
model:
  type: "flux2"
  variant: "dev"
  transformer:
    hidden_size: 3072
    num_attention_heads: 24
    num_layers: 8
    num_single_layers: 48
    in_channels: 128  # 32 latent channels * 4
    qk_norm: true
    guidance_embeds: true
  vae:
    latent_channels: 32
    scaling_factor: 0.3611
    shift_factor: 0.1159
```

### SD3.5

MMDiT (Multimodal DiT) architecture with joint attention on image and text.

```yaml
model:
  type: "sd3"
  transformer:
    hidden_size: 1536
    num_attention_heads: 24
    num_layers: 24
    in_channels: 64  # 16 latent channels * 4
  vae:
    latent_channels: 16
    scaling_factor: 1.5305
  text_encoder:
    t5: { hidden_size: 4096 }
    clip_l: { hidden_size: 768 }
    clip_g: { hidden_size: 1280 }
```

---

## Image Editing (Kontext & Fill Modes)

### Position ID Format (4D)

FLUX models use 4D position IDs `[t, h, w, l]` for spatial-temporal encoding:

| Dimension | Description | Target Image | Reference Image |
|-----------|-------------|--------------|-----------------|
| t | Temporal/time offset | 0.0 | 1.0+ |
| h | Height coordinate | 0 to H-1 | 0 to H-1 |
| w | Width coordinate | 0 to W-1 | 0 to W-1 |
| l | Sequence index | 0 | 0 |

Position IDs have shape `[B, seq, 4]` and enable the model to distinguish
between target (generated) and reference (conditioning) images.

### Kontext Mode (Sequence-wise)

Reference images are concatenated along the sequence dimension:

```python
from src.models.flux.v2 import prepare_kontext_conditioning

# Encode reference images
img_cond_seq, img_cond_seq_ids = prepare_kontext_conditioning(
    reference_images=ref_imgs,  # [B, 3, H, W]
    vae=model.vae,
    device=device,
    dtype=dtype,
    time_offset=1.0,  # Reference uses t=1.0
)

# Pass to transformer
output = model.transformer(
    hidden_states=noisy_latent,
    timestep=timesteps,
    encoder_hidden_states=text_embeds,
    pooled_projections=pooled_embeds,
    img_cond_seq=img_cond_seq,          # [B, ref_seq, dim]
    img_cond_seq_ids=img_cond_seq_ids,  # [B, ref_seq, 4]
)
```

### Fill Mode (Channel-wise, FLUX.2 only)

Masked reference is concatenated along the channel dimension:

```python
from src.models.flux.v2 import prepare_fill_conditioning

# Prepare fill conditioning
img_cond = prepare_fill_conditioning(
    reference_image=ref_img,  # [B, 3, H, W]
    mask=mask,                 # [B, 1, H, W] where 1=inpaint
    vae=model.vae,
    device=device,
    dtype=dtype,
)

# Pass to transformer
output = model.transformer(
    hidden_states=noisy_latent,
    timestep=timesteps,
    encoder_hidden_states=text_embeds,
    pooled_projections=pooled_embeds,
    img_cond=img_cond,  # [B, seq, latent_dim + mask_dim]
)
```

---

## Alignment Notes

### VAE Encoding Order

FLUX VAE uses `(z - shift) * scale` order:

```python
# Correct order for FLUX VAE encoding
latent = (z - shift_factor) * scaling_factor

# NOT: (z * scaling_factor) - shift_factor
```

### VAE Scaling Factors

| Model | Scale Factor | Shift Factor |
|-------|-------------|--------------|
| FLUX.1/2 | 0.3611 | 0.1159 |
| SD3.5 | 1.5305 | - |
| SDXL | 0.13025 | - |

### Transformer Block Counts

| Model | Joint Blocks | Single Blocks |
|-------|-------------|---------------|
| FLUX.1 (all) | 19 | 38 |
| FLUX.2 dev | 8 | 48 |
| FLUX.2 klein-4b | 5 | 20 |
| FLUX.2 klein-9b | 6 | 24 |

### Text Encoder Configurations

| Model | Encoders | Pooled Dim |
|-------|----------|-----------|
| FLUX.1 | T5-XXL + CLIP-L | 768 |
| FLUX.2 dev | Mistral-3 | 4096 |
| FLUX.2 klein | Qwen3-4B/8B | 4096 |
| SD3.5 | T5 + CLIP-L + CLIP-G | 2048 |

---

## Training Methods

### LoRA

Low-Rank Adaptation for efficient fine-tuning:

```yaml
training:
  method: "lora"
  lora:
    rank: 32
    alpha: 32
    dropout: 0.0
    target_modules: ["to_q", "to_k", "to_v", "to_out.0"]
    train_text_encoder: false
```

### Full Fine-tuning

Train all model parameters:

```yaml
training:
  method: "full_finetune"
  optimizer:
    type: "adamw"
    lr: 1.0e-5
```

### DreamBooth

Personalization with prior preservation:

```yaml
training:
  method: "dreambooth"
  dreambooth:
    instance_prompt: "a photo of sks dog"
    class_prompt: "a photo of dog"
    num_class_images: 200
    prior_preservation_weight: 1.0
```

### ControlNet

Conditional control via auxiliary networks:

```yaml
training:
  method: "controlnet"
  controlnet:
    conditioning_type: "canny"
    conditioning_scale: 1.0
```

---

## Configuration System

### Experiment Config

```yaml
# configs/experiments/my_experiment.yaml
_base_:
  - ../models/flux.yaml
  - ../training/lora.yaml

experiment:
  name: "flux-lora-character"
  output_dir: "./outputs/flux_lora"
  seed: 42

model:
  pretrained_path: "/path/to/flux"
  dtype: "bfloat16"

training:
  epochs: 50
  batch_size: 2
  gradient_accumulation: 4
  learning_rate: 1.0e-4

data:
  train_path: "/path/to/dataset"
  resolution: 1024
  cache_latents: true

hardware:
  mixed_precision: "bf16"
  gradient_checkpointing: true
```

### Config Inheritance

Configs support `_base_` inheritance:

```python
from src.utils.config import load_config

config = load_config("configs/experiments/my_experiment.yaml")
# Automatically merges base configs
```

---

## Data Pipeline

### Dataset Format

Option 1: metadata.json
```json
[
  {"image": "image_001.png", "caption": "a photo of a cat"},
  {"image": "image_002.png", "caption": "a scenic landscape"}
]
```

Option 2: Paired files
```
dataset/
├── image_001.png
├── image_001.txt  # Contains caption
└── ...
```

### Aspect Ratio Bucketing

Multi-resolution training with aspect ratio preservation:

```yaml
data:
  resolution: 1024
  bucket_resolution_steps: 64  # Enable bucketing
  min_bucket_resolution: 512
  max_bucket_resolution: 2048
```

### Latent Caching

Pre-compute latents for faster training:

```yaml
data:
  cache_latents: true
  cache_dir: "./cache"
```

---

## Development Guidelines

### Code Style

- Python 3.10+
- Type hints required
- Google-style docstrings
- Formatting: Black + isort
- Linting: Ruff

### Design Patterns

1. **Configuration-driven**: All hyperparameters from YAML
2. **Factory pattern**: `create_model()`, `create_trainer()`
3. **Composition over inheritance**
4. **Single responsibility per module

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_flux2_editing.py -v

# Alignment verification
python -m pytest tests/test_alignment_verification.py -v
```

---

## Dependencies

```
# Core
torch>=2.2.0
torchvision>=0.17.0
safetensors>=0.4.0
accelerate>=0.25.0

# Configuration
omegaconf>=2.3.0

# Data
pillow>=10.0.0
numpy>=1.24.0

# Text encoders
transformers>=4.36.0
sentencepiece>=0.1.99

# Training
tqdm>=4.66.0
wandb>=0.16.0  # Optional

# Dev
pytest>=7.4.0
black>=23.0.0
ruff>=0.1.0
```
