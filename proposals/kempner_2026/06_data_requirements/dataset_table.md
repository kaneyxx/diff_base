## Primary training corpus

**ORION** — Multiplexed 3D atlas of state transitions and immune interaction in
colorectal cancer (Lin et al., *Cell* 186, 363–381, 2023).

- Paired modalities: **H&E + multiplex IF (CyCIF, 13 biomarker channels per
  patient: CD45, CD31, CD68, CD4, FOXP3, CD8a, CD45RO, CD20, PD-L1, CD3e,
  CD163, PD-1, Ki67)**.
- Cohort: 40 colorectal-cancer patients (CRC01..CRC40, including CRC33_01/CRC33_02).
- License: open access via Sage Synapse / Sorger lab; data security level 1.
- **Local copy already on cluster** (Kempner is not paying for storage of the
  raw WSIs):
  - Path: `/n/scratch/users/f/fas994/bao/data/CRC{NN}/`
  - Pre-tiled at 1024² PNG: `CRC{NN}/HE/CRC{NN}_HE_{x}_{y}.png`
  - Multiplex H5 with R/G/B channels: `CRC{NN}/tiles.h5`
    - R = biomarker-specific signal
    - G = Pan-CK (epithelium)
    - B = Hoechst (nuclei)
  - Coordinate index: `CRC{NN}/valid_coordinates.txt`
  - Total on-disk: **1.7 TB** (sample CRC01 ≈ 43 GB).
- Data loader convention (reused from existing BAO repo, will be ported to the
  proposed full-fine-tune workflow):
  - `bao/src/data/multi_patient_dataset.py::MultiPatientBiomarkerDataset`
  - JSON `training_splits` referencing `(crc_id, h5_path, he_dir, index,
    coordinate, sample_type)` for balanced positive/negative sampling.

## Public benchmarks for cross-corpus evaluation (no training)

| Dataset | Source | Paired stains | Approx. size | License / Security | URL |
|---|---|---|---|---|---|
| **BCI** | Liu et al., CVPRW 2022 | H&E + HER2 IHC | ~50 GB tiles (9,648 pairs @ 1024²) | Research-only, level 1 | https://bupt-ai-cz.github.io/BCI/ |
| **HER2 Contest 2016** | University of Warwick | H&E + HER2 IHC (86 WSIs) | ~150 GB | CC BY 4.0, level 1 | https://warwick.ac.uk/fac/cross_fac/tia/data/her2contest/ |
| **ACROBAT 2022** | Karolinska + Wieslander et al., Sci. Data 2024 | H&E + HER2 / ER / PgR / KI67 (4 IHC) | ~5 TB raw / ~120 GB curated tiles | CC BY-NC 4.0, level 1 | https://acrobat.grand-challenge.org/ |
| **MIST (MultIple STain)** | Wang et al., MICCAI 2023 | H&E + multiplex IF (4 stains) | ~80 GB | Research-only, level 1 | https://github.com/lhaof/MIST |

These are **not used for training** — included only to benchmark trained
models on out-of-distribution stains and to satisfy reviewer questions about
generalization breadth.

## Prior art / comparable methods

These works define the state of the art our method is benchmarked against.
They are **models**, not redistributable training corpora — their training
data (e.g. Providence) is private and therefore unsuitable for level 1–2 use.

| Method | Source | Input → Output | Code / model |
|---|---|---|---|
| **GigaTIME** | Microsoft / Providence, *Cell* (Dec 2025) | H&E WSI → 21-channel virtual mIF; trained on Providence (40 M paired cells, private) | Code: aka.ms/gigatime-code · Weights: HF `prov-gigatime/GigaTIME` |
| **Path2Space** | (preprint, citation TBD by user) | H&E → spatial molecular signal | TBD |
| **Rivenson et al.** | Nature Biomed. Eng. 2019 | Auto-fluorescence → virtual H&E (foundational virtual staining work) | Public supplementary code |
| **BCI baselines** | Liu et al., CVPRW 2022 | H&E patch → HER2 IHC patch (CycleGAN / pix2pix baselines) | https://bupt-ai-cz.github.io/BCI/ |

Differentiation: our work uses a **flow-matching DiT (FLUX.1-Kontext)** rather
than CNN-based GAN/diffusion baselines, leveraging the Kontext sequence-wise
reference-image conditioning mechanism. We rely on **publicly redistributable
paired datasets only** (ACROBAT, BCI, HER2 Contest), keeping data security
strictly at level 1.
