GPU framework: PyTorch 2.4 with CUDA 12.4 and Hugging Face Accelerate 0.30
for FSDP-based multi-GPU training. Mixed-precision uses bfloat16 with optional
FP8 (Transformer Engine 1.6) on H100. Diffusion training scaffolding lives in
our diff_base repo (PyTorch + safetensors); no proprietary forks.

Core ML libraries:
- transformers 4.40 (text encoders: T5-XXL, CLIP-L)
- diffusers 0.27 (VAE + flow-matching reference)
- bitsandbytes 0.43 (8-bit AdamW for ≥30% optimizer-state savings)
- peft 0.10 (kept for ablation, not used in full fine-tune)
- xformers 0.0.26 / FlashAttention-3 (memory-efficient attention)

Image / data: pillow, numpy, openslide-python, tifffile (whole-slide reading),
albumentations (geometric augmentations), torchvision.

Config / orchestration: omegaconf, hydra (planned), wandb 0.16 (experiment
tracking). Testing: pytest, ruff.

Containerisation: Singularity / Apptainer 1.3 from a pinned Dockerfile (PyTorch
24.04 NGC base image), distributed as `.sif` for HMS RC; Conda env file kept
as fallback. KempnerPulse 0.x is invoked from within the container for GPU
telemetry.
