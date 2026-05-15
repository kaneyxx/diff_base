External APIs / AI services used:

1. Hugging Face Hub
   - Function: download FLUX.1-Kontext-dev pre-trained weights once, plus
     T5-XXL / CLIP-L text encoders (one-time, mirrored to /n/scratch).
   - Frequency: 1× per cluster (cached locally; not called per training step).
   - Auth: read-only HF token in `${HOME}/.cache/huggingface`.

2. KempnerPulse (local CLI, no network call)
   - Used for GPU telemetry; reads DCGM counters via dcgmi.

3. wandb.ai (optional)
   - Function: experiment tracking only; metric uploads in batches of ≤5 KB.
   - Frequency: ≤ 60 calls / run (every N steps + checkpoint events).
   - Disable for sensitive runs via WANDB_MODE=offline.

No OpenAI / Anthropic / Google AI API calls. All inference stays on-cluster.
No PHI is uploaded externally; data security level 1–2 only.
