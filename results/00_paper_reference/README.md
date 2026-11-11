# Paper reference results (`00_paper_reference`)

Faithful **transcription** of published findings from Farhadi Sedehi et al., *IEEE Access*, 2024
(DOI [10.1109/ACCESS.2024.3430934](https://doi.org/10.1109/ACCESS.2024.3430934)).

This folder is **not** a re-execution of the authors' pipeline on SEED-IV / MAHNOB-HCI.

## Contents

| Path | Description |
|------|-------------|
| `paper_methodology.md` | Methods summary with implementation notes |
| `paper_parameters.json` | Machine-readable parameters and caveats |
| `paper_reported_values.csv` | Stated band accuracies (missing δ/θ left blank) |
| `tables/` | Rules Tables 2–6, Table 7, Table 8 |
| `reference_metrics/` | Accuracies, eye ranking, AUC notes |
| `figures/` | Rasterised **published** figure pages from the PDF |

## Confusion matrices & per-class AUC

Numeric cells were **not** digitised from Figs. 10–13 (image-only in PDF). Use `figures/figure_10_*.png`
and `figures/figure_11_*.png` as the paper reference.

## Do not mix

Values here must **never** be merged with `01_expected_illustrative_scenario/`.
