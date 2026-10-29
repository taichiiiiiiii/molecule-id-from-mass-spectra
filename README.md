# Molecule ID From Mass Spectra

**Kaggle competition:** Envela CASMI 2026 — Molecule ID From Mass Spectra

- URL: https://www.kaggle.com/competitions/envela-CASMI26-molecule-id-mass-spectra
- Start date: 2026-09-14
- Entry deadline: 2026-12-07
- Final submission deadline: 2026-12-14
- Prize pool: $50,000 (1st $16,000 / 2nd $12,000 / 3rd $9,000 / 4th $7,000 / 5th $6,000)

## Task

Given LC-MS/MS tandem mass spectra, predict the 2D chemical structure (SMILES)
of each molecule. Each `molecule_id` may have 1–16 spectra (median 3) acquired
at different collision energies / adducts — aggregate the evidence into a
single ranked list of candidate structures per molecule.

Predictions are scored by **MRR@25**:

```
MRR@25 = (1 / U) * sum_u ( 1 / rank_u )
```

A molecule scores 0 if none of its 25 guesses is correct; the first correct
guess counts only. A correct guess at rank 1 scores 1.0, rank 2 scores 0.5,
rank 25 scores 0.04.

## Data

- `train.parquet` — ~2.5M MS/MS spectra covering ~275k unique structures.
  One row per spectrum (with ground-truth structure).
- `test.parquet` — the spectra to identify (~1,500 spectra, ~400 molecules,
  1–16 spectra per molecule). Structure not given.
- `sample_submission.csv` — a valid submission, used as the format reference.

### Test set notes
- Acquired on a Bruker timsTOF.
- Monoisotopic masses 157–1,159 Da (median 348).
- Three novelty classes (tier 1 in-public-library → tier 3 novel, de novo):
  which molecule belongs to which class is hidden.

## Repository layout

```
.
├── README.md
├── config.toml
├── .gitignore
└── scripts/
    └── download.sh          # download competition data (Kaggle CLI)
```

## Quick start

1. Install the Kaggle CLI and make sure it is authenticated:
   ```bash
   kaggle auth print-access-token
   ```
2. Download the competition data (first time requires a one-time
   subscription). From this folder run:
   ```bash
   bash scripts/download.sh
   ```
   The data will land in `data/` as `train.parquet`, `test.parquet`,
   `sample_submission.csv`, and a metadata JSON.
3. Build a submission notebook (the competition requires submissions to be
   made through Notebooks, with `submission.csv` naming).

## Notes / rules

- Submissions must be made through Kaggle Notebooks.
- Notebook run-time cap: CPU/GPU <= 9 hours per commit.
- Internet access must be **disabled** for commits.
- External/pre-trained data and models are allowed and must be public/ freely
  available.
- A valid submission must be named `submission.csv`, with columns
  `molecule_id,smiles` (up to 25 semicolon-separated guesses per row,
  best guess first, one row per molecule).
