Deliverables at award conclusion:

1. Codebase
   - Public release of the virtual-staining branch of diff_base
     (Apache 2.0), with reproducible config files and SLURM scripts.

2. Model weights
   - Fine-tuned FLUX.1-Kontext-vstain checkpoints in BFL safetensors and
     HF diffusers format, hosted on Hugging Face Hub.

3. Dataset preprocessing pipeline
   - Scripts for ACROBAT / BCI tiling, paired registration, latent caching,
     released alongside the code.

4. Documentation
   - Model card (limitations, intended use, evaluation results), README,
     and a Jupyter notebook demonstrating inference on a held-out WSI.

5. Manuscript
   - Submission to a venue such as Nature Communications, MICCAI, or
     Medical Image Analysis describing the virtual staining method and
     scaling results.

6. Evaluation suite
   - Quantitative metrics (SSIM, FID, expert-rated panel scores) and the
     evaluation harness used to compute them.
