# Kempner Technical Readiness — Submission Package (May 2026)

This directory holds the working files for the Technical Readiness section of
the Kempner Institute proposal. **Not git-tracked** (see top-level .gitignore).

## To assemble for submission
1. Run `04_scalability_testing/slurm_4gpu.sbatch` on a 4×A100 80GB reservation node (production target is H100; A100 chosen for local availability — see `05_gpu_type.md`).
2. After completion, run `parse_pulse_csv.py` for each result CSV and paste
   rows into `04_scalability_testing/results/summary_table.md`.
3. Fill placeholders in `08_technical_expertise.md`.
4. Choose 8 vs 16 GPU request based on observed scaling efficiency.
5. Concatenate sections 1..9 into `final_proposal.md` for upload to Quickbase.

## Reproducibility
All code paths reference `${HOME}/diff_base` and `/n/scratch/users/f/fas994/`.
Replace placeholders (account, reservation, dataset paths) before launching.
