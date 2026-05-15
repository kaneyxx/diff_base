# Data Requirements

The authoritative dataset list for this proposal is in [`dataset_table.md`](./dataset_table.md).

## Primary Corpus

**ORION** (Lin et al., *Cell* 186, 363–381, 2023) is the primary training corpus for this project.
It is paired H&E + multiplex immunofluorescence (CyCIF) data for colorectal cancer (CRC) patients,
already on the cluster at `/n/scratch/users/f/fas994/bao/data`. See `dataset_table.md` for the
full schema, on-disk path structure, and data-loader conventions.

## Public Benchmarks

The public benchmarks listed in `dataset_table.md` (BCI, HER2 Contest 2016, ACROBAT 2022, MIST)
are included **for cross-corpus evaluation only** — they are not used for training. They allow
reviewers to assess generalization of the trained models to out-of-distribution stain types.

## Inventory Files

- [`orion_inventory.txt`](./orion_inventory.txt) — `du -sh` output confirming total on-disk size.
- [`orion_patients.txt`](./orion_patients.txt) — one CRC patient ID per line (CRC01..CRC40+).
