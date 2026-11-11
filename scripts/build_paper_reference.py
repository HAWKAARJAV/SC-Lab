#!/usr/bin/env python3
"""Build results/00_paper_reference/ from gfer/paper_reference_data.py and the IEEE Access PDF.

    .venv/bin/python scripts/build_paper_reference.py
    PAPER_PDF=/path/to/paper.pdf .venv/bin/python scripts/build_paper_reference.py

Does not modify other results/ folders.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gfer.paper_figures_method import figure_A4_membership, figure_A5_rule_grid  # noqa: E402
from gfer.paper_reference_data import (  # noqa: E402
    CLASSES,
    PAPER,
    RULE_SETS,
    TABLE_7_RULES,
    TABLE_8_COMPARISON,
)

OUT = os.path.join(ROOT, "results", "00_paper_reference")
DEFAULT_PDF = os.path.join(ROOT, "Recognition_of_emotional_categories_using_mined_fu (1).pdf")
FIG = os.path.join(OUT, "figures")
SRC = os.path.join(OUT, "source_pages")
TAB = os.path.join(OUT, "tables")
MET = os.path.join(OUT, "reference_metrics")

# Printed page number → 0-based index; crop = union of embedded image rects above header/logo.
FIGURE_SPECS = {
    "figure_07_eye_mrmr_ranking.png": (9, (149, 150, 151)),
    "figure_08_sugeno_fis_architecture.png": (10, (154,)),
    "figure_09_ga_convergence.png": (10, (155,)),
    "figure_10_seed_iv_confusion_matrices.png": (13, (162, 163, 164, 165, 166)),
    "figure_11_seed_iv_roc_curves.png": (14, (169, 170, 171, 172, 173)),
    "figure_12_mahnob_confusion_matrices.png": (15, (176, 177, 178, 179, 180)),
    "figure_13_mahnob_roc_curves.png": (16, (183, 184, 185, 186, 187)),
}


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _union_rect(page, xrefs):
    import fitz
    rects = []
    for xref in xrefs:
        for r in page.get_image_rects(xref):
            rects.append(r)
    if not rects:
        return page.rect
    x0 = min(r.x0 for r in rects)
    y0 = min(r.y0 for r in rects)
    x1 = max(r.x1 for r in rects)
    y1 = max(r.y1 for r in rects)
    return fitz.Rect(x0, y0, x1, y1)


def export_pdf_figures(pdf_path: str) -> list[str]:
    """Crop individual paper figures; save full pages under source_pages/."""
    try:
        import fitz  # pymupdf
    except ImportError:
        print("ERROR: pymupdf required — pip install pymupdf")
        return []
    if not os.path.exists(pdf_path):
        print(f"ERROR: paper PDF not found: {pdf_path}")
        print("Set PAPER_PDF or pass --pdf")
        return []
    os.makedirs(SRC, exist_ok=True)
    doc = fitz.open(pdf_path)
    pages_saved = set()
    written = []
    mat = fitz.Matrix(3, 3)
    for name, (printed_page, xrefs) in FIGURE_SPECS.items():
        pidx = printed_page - 1
        page = doc[pidx]
        if printed_page not in pages_saved:
            full = page.get_pixmap(matrix=mat)
            full.save(os.path.join(SRC, f"page_{printed_page:02d}.png"))
            pages_saved.add(printed_page)
        clip = _union_rect(page, xrefs)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        out = os.path.join(FIG, name)
        pix.save(out)
        written.append(out)
    doc.close()
    figure_A4_membership(os.path.join(FIG, "figure_A4_fuzzy_membership_functions.png"))
    figure_A5_rule_grid(os.path.join(FIG, "figure_A5_fuzzy_rule_distribution.png"))
    written.extend([
        os.path.join(FIG, "figure_A4_fuzzy_membership_functions.png"),
        os.path.join(FIG, "figure_A5_fuzzy_rule_distribution.png"),
    ])
    return written


def write_figure_manifest(figures_written: list[str]) -> None:
    rows = []
    for name, (pg, _) in FIGURE_SPECS.items():
        rows.append({
            "id": name.replace("figure_", "").replace(".png", ""),
            "path": f"figures/{name}",
            "status": "generated_cropped" if os.path.exists(os.path.join(FIG, name)) else "missing",
            "printed_page": pg,
            "provenance": "published_paper_pdf_crop",
            "data_source": "n/a (image transcription)",
        })
    for aid, fn in [("A4", "figure_A4_fuzzy_membership_functions.png"), ("A5", "figure_A5_fuzzy_rule_distribution.png")]:
        rows.append({
            "id": aid,
            "path": f"figures/{fn}",
            "status": "generated",
            "printed_page": "",
            "provenance": "derived_methodology_diagram",
            "data_source": "gfer/paper_figures_method.py",
        })
    with open(os.path.join(OUT, "FIGURES_MANIFEST.md"), "w", encoding="utf-8") as f:
        f.write("# Paper reference figure manifest\n\n")
        f.write("| ID | Path | Page | Status | Provenance |\n|---|---|---:|---|---|\n")
        for r in rows:
            f.write(f"| {r['id']} | `{r['path']}` | {r['printed_page']} | {r['status']} | {r['provenance']} |\n")
        f.write(f"\nFull pages: `source_pages/page_XX.png`. PDF: configured at build time.\n")
        f.write(f"\nGenerated files this run: {len(figures_written)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=os.environ.get("PAPER_PDF", DEFAULT_PDF))
    a = ap.parse_args()
    for d in (OUT, FIG, SRC, TAB, MET):
        os.makedirs(d, exist_ok=True)

    with open(os.path.join(OUT, "paper_parameters.json"), "w", encoding="utf-8") as f:
        json.dump({"paper": PAPER, "classes_order": CLASSES, "provenance": "transcription"}, f, indent=2)

  # reported accuracies
    acc_rows = []
    for ds in ("SEED-IV", "MAHNOB-HCI"):
        key = "SEED-IV" if ds == "SEED-IV" else "MAHNOB-HCI"
        for band, val in PAPER["reported_accuracy_percent"][key].items():
            if band == "note_theta_delta":
                continue
            acc_rows.append({
                "dataset": ds,
                "frequency_band": band,
                "accuracy_percent": "" if val is None else val,
                "reported_in_paper": "yes" if val is not None else "no_numeric_value",
                "source": "Sec. IV results text",
            })
    write_csv(os.path.join(OUT, "paper_reported_values.csv"), acc_rows,
              ["dataset", "frequency_band", "accuracy_percent", "reported_in_paper", "source"])

    write_csv(os.path.join(MET, "seed_iv_accuracy.csv"),
              [r for r in acc_rows if r["dataset"] == "SEED-IV"],
              ["dataset", "frequency_band", "accuracy_percent", "reported_in_paper", "source"])
    write_csv(os.path.join(MET, "mahnob_accuracy.csv"),
              [r for r in acc_rows if r["dataset"] == "MAHNOB-HCI"],
              ["dataset", "frequency_band", "accuracy_percent", "reported_in_paper", "source"])

    eye_rows = [
        {"rank": i + 1, "feature": feat,
         "mrmr_score_note": "top bar Fig. 7 ~0.0157272 (SEED-IV) / ~0.0202258 (MAHNOB) if rank=1 else not_given",
         "source": "Fig. 7 caption order (p. 10)"}
        for i, feat in enumerate(PAPER["eye_features_order"])
    ]
    write_csv(os.path.join(MET, "eye_mrmr_ranking.csv"), eye_rows,
              ["rank", "feature", "mrmr_score_note", "source"])

    band_table = {"delta": 2, "theta": 3, "alpha": 4, "beta": 5, "gamma": 6}
    for band, rules in RULE_SETS.items():
        tno = band_table[band]
        rows = [{"rule_id": rid, "if_part": iff, "then_class": then, "source": f"TABLE {tno} (Set {chr(64+tno-1)})"}
                for rid, iff, then in rules]
        write_csv(os.path.join(TAB, f"table_{tno:02d}_{band}_rules.csv"),
                  rows, ["rule_id", "if_part", "then_class", "source"])

    write_csv(os.path.join(TAB, "table_07_rule_coverage_accuracy.csv"), TABLE_7_RULES,
              ["emotion", "rule", "coverage_pct", "accuracy_pct"])
    write_csv(os.path.join(TAB, "table_08_previous_research_comparison.csv"), TABLE_8_COMPARISON,
              ["ref", "dataset", "method", "evaluation", "classes", "accuracy"])

    auc_note = [
        {"dataset": "SEED-IV", "note": PAPER["auc_summary_approximate"]["SEED-IV"], "per_class_numeric": "not_transcribed"},
        {"dataset": "MAHNOB-HCI", "note": PAPER["auc_summary_approximate"]["MAHNOB-HCI"], "per_class_numeric": "not_transcribed"},
    ]
    write_csv(os.path.join(MET, "seed_iv_auc.csv"), auc_note, ["dataset", "note", "per_class_numeric"])
    write_csv(os.path.join(MET, "mahnob_auc.csv"), auc_note, ["dataset", "note", "per_class_numeric"])

    readme = """# Paper reference results (`00_paper_reference`)

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
"""
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    meth = """# Original paper methodology (reference summary)

**Source:** Farhadi Sedehi et al., IEEE Access, 2024. Transcribed for teaching/lab reference.

## Eye tracking (8 features / 4 s)

Mean and SD of fixation duration; fixation frequency; max fixation duration; mean and SD of
pupil X and Y. mRMR ranks features; **mean pupil diameter Y** is the gate feature (Fig. 7).

## EEG

Butterworth band-pass, notch, downsampling to 128 Hz; MAHNOB-HCI uses common average referencing.

## Frequency bands (as stated in paper)

| Band | Hz |
|------|-----|
| Delta | 1–4 |
| Theta | 4–8 |
| Alpha | 8–13 |
| Beta | 13–25 |
| Gamma | 25–60 |

**Repository note:** `gfer/signal.py` uses 4-level db4 DWT at 128 Hz → dyadic bands differ slightly
(see `docs/doc.md`).

## Entropies

SampEn, PermEn, SymbEn, PhEn (m=5, τ=2). 62×4 features (SEED-IV) or 32×4 (MAHNOB-HCI).
mRMR selects **PhEn-FP2, SampEn-T8, PermEn-T7** for the S-FIS.

## S-FIS + GA

3 inputs × 3 levels → 27 initial rules; GA population 27; alpha-band training stops at generation
450 with patience 10 (Fig. 9).

## Evaluation

5-fold CV; accuracy, TPR, FNR, AUC (segment-level in the original paper).
"""
    with open(os.path.join(OUT, "paper_methodology.md"), "w", encoding="utf-8") as f:
        f.write(meth)

    written = export_pdf_figures(a.pdf)
    write_figure_manifest(written)
    print("wrote", OUT, f"({len(written)} figures)")


if __name__ == "__main__":
    main()
