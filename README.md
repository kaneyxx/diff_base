# Diffusion Model Training Framework

A unified training framework for diffusion models supporting multiple architectures (SDXL, Flux) and training methods (LoRA, Full Fine-tune, DreamBooth, ControlNet, Textual Inversion).

## Features

- **Multiple Architectures**: SDXL and Flux (DiT) support
- **Training Methods**: LoRA, Full Fine-tuning, DreamBooth, ControlNet, Textual Inversion
- **Noise Schedulers**: DDPM, Euler, Flow Matching
- **Data Pipeline**: Aspect ratio bucketing, latent caching, flexible transforms
- **Configuration-Driven**: YAML configs with inheritance support
- **Memory Optimized**: Gradient checkpointing, mixed precision training
- **Checkpoint Management**: Save/load in safetensors format

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
│   ├── training/            # Training method configs
│   └── experiments/         # Complete experiment configs
├── src/
│   ├── models/              # Model definitions
│   │   ├── components/      # Shared components (attention, embeddings, etc.)
│   │   ├── sdxl/            # SDXL architecture
│   │   └── flux/            # Flux DiT architecture
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
