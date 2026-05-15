- Primary repo: https://github.com/kaneyxx/diff_base
  (fork-based virtual-staining work will live in branch `vstain/*` or a new repo)
- Status: PRIVATE — collaborator `kempner-compute` will be added before submission
- Subdirectories of interest for reviewers:
    src/models/flux/v1/             Flux1Transformer (Kontext supported)
    src/training/flux_full_finetune_trainer.py   Full FT trainer w/ EMA + 8-bit AdamW + FSDP
    scripts/finetune_flux.py        CLI entry point
    configs/accelerate/             Single-GPU + multi-GPU FSDP recipes
    docs/full_finetune_recipes.md   Memory plans + recipes
