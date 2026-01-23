# Diffusion Model Training Framework

A unified training framework for diffusion models supporting multiple architectures (SDXL, FLUX.1, FLUX.2) and training methods (LoRA, Full Fine-tune, DreamBooth, ControlNet, Textual Inversion).

## Features

- **Multiple Architectures**: SDXL, FLUX.1 (dev, schnell), and FLUX.2 (dev, klein-4B, klein-9B) support
- **Training Methods**: LoRA, Full Fine-tuning, DreamBooth, ControlNet, Textual Inversion
- **Noise Schedulers**: DDPM, Euler, Flow Matching
- **Data Pipeline**: Aspect ratio bucketing, latent caching, flexible transforms
- **Configuration-Driven**: YAML configs with inheritance support
- **Memory Optimized**: Gradient checkpointing, mixed precision training
- **Checkpoint Management**: Save/load in safetensors format

## Supported Models

| Model | Variants | Text Encoder | Latent Channels | Parameters |
|-------|----------|--------------|-----------------|------------|
| SDXL | base, refiner | CLIP-L + CLIP-G | 4 | ~2.6B |
| FLUX.1 | dev, schnell | T5-XXL + CLIP-L | 16 | ~12B |
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

# Install in development mode
pip install -e .
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
│   │   ├── flux.yaml        # Alias for flux1_dev
│   │   ├── flux1_dev.yaml
│   │   ├── flux1_schnell.yaml
│   │   ├── flux2_dev.yaml
│   │   ├── flux2_klein_4b.yaml
│   │   └── flux2_klein_9b.yaml
│   ├── training/            # Training method configs
│   └── experiments/         # Complete experiment configs
├── src/
│   ├── models/              # Model definitions
│   │   ├── components/      # Shared components (attention, embeddings, etc.)
│   │   ├── sdxl/            # SDXL architecture
│   │   └── flux/            # Flux architectures
│   │       ├── components/  # Flux-specific shared layers
│   │       ├── v1/          # FLUX.1 (dev, schnell)
│   │       └── v2/          # FLUX.2 (dev, klein-4B, klein-9B)
│   ├── training/            # Training logic
│   │   └── methods/         # LoRA, ControlNet implementations
│   ├── data/                # Data pipeline
│   ├── schedulers/          # Noise schedulers
│   ├── inference/           # Inference pipeline
│   └── utils/               # Utilities (config, checkpoint, logging)
├── scripts/                 # Training and inference scripts
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

## Using FLUX Models

### FLUX.1 (dev/schnell)

```yaml
# configs/experiments/flux1_lora.yaml
_base_:
  - ../models/flux1_dev.yaml  # or flux1_schnell.yaml
  - ../training/lora.yaml

model:
  pretrained_path: "black-forest-labs/FLUX.1-dev"

training:
  epochs: 50
  batch_size: 1
  lora:
    rank: 16
```

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
- PyTorch >= 2.2.0
- CUDA-capable GPU with 24GB+ VRAM (recommended)

## Running Tests

```bash
# Run all tests
python run_tests.py

# Or use pytest
python -m pytest tests/ -v
```

## License

MIT License

## Acknowledgements

- [Stability AI](https://stability.ai/) for SDXL
- [Black Forest Labs](https://blackforestlabs.ai/) for Flux
- [Hugging Face](https://huggingface.co/) for transformers and diffusers libraries
