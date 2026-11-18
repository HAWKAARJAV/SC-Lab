# Final research paper (illustrative M0–M4 study)

Self-contained Springer/JSC-style research paper. It does **not** replace `manuscript/main.tex`.

| File | Role |
|------|------|
| `main.tex` | Master document |
| `sections/body.tex` | Sections I–IX |
| `appendix/appendices.tex` | Published rules, Table 7, gate formulas |
| `references.tex` | Bibliography copied from `manuscript/parts/bib.tex` |
| `build_architecture.py` | Generates the publication architecture in `figures/` |
| `main.pdf` / `final_research_paper.pdf` | Compiled 155 × 235 mm journal output |

Figures are loaded from `docs/architecture/`, `docs/fuzzy_rulebook/`, and
`results/01_Implemented_Model/` via `\graphicspath`. Numbers in the results tables were taken
from the synthetic scenario `summary.csv`, `band_metrics.csv`, `learned_band_weights.csv`,
`gate_quality.csv` and `per_class_metrics.csv` files. Published accuracies come from
`results/00_paper_reference/paper_reported_values.csv`.

## Compile

```bash
cd manuscript/final_report
python3 build_architecture.py
tectonic main.tex
cp main.pdf final_research_paper.pdf
```

`pdflatex` is not required when Tectonic is installed.

Before submission, replace all author, affiliation, date, funding, conflict-of-interest and
author-contribution placeholders in `main.tex`.
