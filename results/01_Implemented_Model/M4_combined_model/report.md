# M4 — illustrative scenario report

**M2 + M3 combined (proposed)** — data in this folder are **synthetic** (see `../SCENARIO_NOTICE.md`).

## Files

| File | Meaning |
|------|---------|
| `summary.csv` | Pooled metrics per reference-scale dataset |
| `fold_metrics.csv` | 5-fold CV-style fold scores |
| `predictions.csv` | Labels + class probabilities (source for CM / ROC) |
| `figures/` | Plots generated from the CSVs above |

## Interpretation

Confusion matrices and ROC curves are computed from `predictions.csv`. Accuracy in `summary.csv`
is the arithmetic match to those predictions—not a re-run of SEED-IV or MAHNOB-HCI.
