"""Generate results/01_expected_illustrative_scenario/ — synthetic teaching data only.

    python -m gfer.teaching_scenario

Does not overwrite results/sim_study*, demo_*, etc.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import teaching_plots as tp
from .gates import FUZZY, fuzzy_memberships, fuzzy_weights, hard_weights, normalise_within_trial
from .adaptive_eye import temporal_smooth_weights
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def _fuse_probs(band_probs: dict, weights: dict) -> np.ndarray:
    w = np.array([weights[b] for b in BANDS], dtype=float)
    w /= w.sum()
    return (w[:, None, None] * np.stack([band_probs[b] for b in BANDS], 0)).sum(0)


def _learn_band_weights(val_probs: dict, y_val: np.ndarray) -> dict:
    acc = {b: float((val_probs[b].argmax(1) == y_val).mean()) for b in BANDS}
    s = sum(acc.values()) + 1e-9
    return {b: acc[b] / s for b in BANDS}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "01_expected_illustrative_scenario")
SEED = 20261110
N_PER_DATASET = 500
N_FOLDS = 5
CLASSES = ["sadness", "happiness", "fear", "neutrality"]

TARGET_ACC = {
    "SEED-IV_reference_scale": {"M0": 94.42, "M1": 94.75, "M2": 95.25, "M3": 95.40, "M4": 95.95},
    "MAHNOB_reference_scale": {"M0": 85.20, "M1": 85.55, "M2": 86.05, "M3": 86.20, "M4": 86.75},
}


@dataclass
class SimResult:
    y: np.ndarray
    y_pred: np.ndarray
    proba: np.ndarray
    folds: list[dict]


def _simulate(n: int, target_acc: float, seed: int) -> SimResult:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    n_ok = int(round(target_acc / 100.0 * n))
    y_pred = y.copy()
    wrong = rng.choice(n, size=n - n_ok, replace=False)
    for i in wrong:
        opts = [c for c in range(4) if c != y[i]]
        y_pred[i] = int(rng.choice(opts))
    proba = rng.dirichlet([1.2, 1.2, 1.2, 1.2], size=n)
    proba[np.arange(n), y_pred] += 2.5
    proba /= proba.sum(axis=1, keepdims=True)

    folds = []
    idx = np.arange(n)
    rng.shuffle(idx)
    for f in range(N_FOLDS):
        te = idx[f * n // N_FOLDS : (f + 1) * n // N_FOLDS]
        folds.append({
            "fold": f + 1,
            "accuracy": accuracy_score(y[te], y_pred[te]) * 100,
            "macro_f1": f1_score(y[te], y_pred[te], average="macro", zero_division=0) * 100,
            "n_test": len(te),
        })
    return SimResult(y, y_pred, proba, folds)


def _metrics(y, y_pred, proba) -> dict:
    return {
        "accuracy_pct": accuracy_score(y, y_pred) * 100,
        "macro_precision_pct": precision_score(y, y_pred, average="macro", zero_division=0) * 100,
        "macro_recall_pct": recall_score(y, y_pred, average="macro", zero_division=0) * 100,
        "macro_f1_pct": f1_score(y, y_pred, average="macro", zero_division=0) * 100,
        "weighted_f1_pct": f1_score(y, y_pred, average="weighted", zero_division=0) * 100,
        "macro_auc": float(roc_auc_score(y, proba, multi_class="ovr", average="macro")),
    }


def _write_predictions(path, res: SimResult, dataset: str, model: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "dataset", "model", "y_true", "y_pred"] + [f"p_{c}" for c in CLASSES])
        for i in range(len(res.y)):
            w.writerow([i, dataset, model, res.y[i], res.y_pred[i], *res.proba[i].tolist()])


def _ensure_model_dir(name: str) -> str:
    d = os.path.join(OUT, name)
    os.makedirs(os.path.join(d, "figures"), exist_ok=True)
    return d


def _report_md(model: str, role: str) -> str:
    return f"""# {model} — illustrative scenario report

**{role}** — data in this folder are **synthetic** (see `../SCENARIO_NOTICE.md`).

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
"""


def build_model(model_key: str, folder: str, label: str):
    d = _ensure_model_dir(folder)
    summaries, folds_all = [], []
    pooled_y, pooled_yh, pooled_p = [], [], []
    for ds, targets in TARGET_ACC.items():
        res = _simulate(N_PER_DATASET, targets[model_key], SEED + hash(model_key + ds) % 10000)
        m = _metrics(res.y, res.y_pred, res.proba)
        m.update({"dataset": ds, "model": model_key, "target_accuracy_pct": targets[model_key]})
        summaries.append(m)
        for fd in res.folds:
            folds_all.append({"dataset": ds, "model": model_key, **fd})
        pooled_y.append(res.y)
        pooled_yh.append(res.y_pred)
        pooled_p.append(res.proba)
        pred_name = "final_predictions.csv" if folder == "M4_combined_model" else "predictions.csv"
        if folder == "M3_multiband_fusion":
            pred_name = "fusion_predictions.csv"
        _write_predictions(os.path.join(d, pred_name), res, ds, model_key)

    pd.DataFrame(summaries).to_csv(os.path.join(d, "summary.csv"), index=False)
    pd.DataFrame(folds_all).to_csv(os.path.join(d, "fold_metrics.csv"), index=False)

    if folder == "M0_original":
        pd.DataFrame([
            {"feature": "mean_pupil_diameter_Y", "rank": 1, "role": "gate (paper)"},
            {"feature": "mean_pupil_diameter_X", "rank": 2},
        ]).to_csv(os.path.join(d, "selected_eye_features.csv"), index=False)
        pd.DataFrame([
            {"feature": "PhEn-FP2", "rank": 1}, {"feature": "SampEn-T8", "rank": 2},
            {"feature": "PermEn-T7", "rank": 3},
        ]).to_csv(os.path.join(d, "selected_eeg_features.csv"), index=False)
        pd.DataFrame([{"rule_id": i, "if_part": f"illustrative_rule_{i}", "then_class": CLASSES[i % 4]}
                      for i in range(1, 11)]).to_csv(os.path.join(d, "fuzzy_rules.csv"), index=False)
        gens = np.arange(1, 451)
        fit = 0.72 + 0.22 * (1 - np.exp(-gens / 80))
        pd.DataFrame({"generation": gens, "best_fitness": fit, "mean_fitness": fit - 0.03}).to_csv(
            os.path.join(d, "ga_history.csv"), index=False)
    with open(os.path.join(d, "report.md"), "w", encoding="utf-8") as f:
        f.write(_report_md(model_key, label))

    y = np.concatenate(pooled_y)
    yh = np.concatenate(pooled_yh)
    p = np.vstack(pooled_p)
    figd = os.path.join(d, "figures")
    tp.confusion_fig(y, yh, os.path.join(figd, "confusion_matrix.png"),
                     f"{model_key} pooled (illustrative)")
    tp.roc_fig(p, y, os.path.join(figd, "roc_curves.png"), f"{model_key} pooled ROC")

    if model_key == "M1":
        rng = np.random.default_rng(SEED)
        nwin = 30
        z = np.clip(rng.normal(0.5, 0.2, nwin), 0, 1)
        muL, muM, muH = fuzzy_memberships(z, FUZZY["a1"], FUZZY["c"], FUZZY["b2"])
        rL, rM, rH = FUZZY["r"]
        wf = (muL * rL + muM * rM + muH * rH) / (muL + muM + muH)
        wh = (z >= 0.6).astype(float)
        t = np.arange(nwin)
        tp.gate_timeline(t, wh, wf, wf, os.path.join(figd, "10_hard_vs_fuzzy_timeline.png"))
        pd.DataFrame({"window": t, "z_norm": z, "w_hard": wh, "w_fuzzy": wf}).to_csv(
            os.path.join(d, "gate_weights.csv"), index=False)

    if model_key == "M2":
        rng = np.random.default_rng(SEED + 2)
        pupil = rng.normal(4.0, 0.3, 40)
        trial = np.zeros(40, dtype=int)
        zn = normalise_within_trial(
            (pupil - pupil[:5].mean()) / (pupil[:5].std() + 1e-6), trial)
        w0 = fuzzy_weights(zn, FUZZY)
        w1 = temporal_smooth_weights(w0, trial, np.arange(40), mode="symmetric", span=1)
        pd.DataFrame({"t": np.arange(len(pupil)), "pupil_raw": pupil, "pupil_norm": zn,
                      "w_no_smooth": w0, "w_temporal_smooth": w1}).to_csv(
            os.path.join(d, "temporal_gate_scores.csv"), index=False)
        tp.gate_timeline(np.arange(len(pupil)), (zn > 0.6).astype(float), w0, w1,
                         os.path.join(figd, "18_adaptive_gate_timeline.png"))

    if model_key in ("M3", "M4"):
        rng = np.random.default_rng(SEED + 3)
        band_acc = {b: 90 + rng.uniform(0, 4) for b in BANDS}
        tp.bar_compare(BANDS, [band_acc[b] for b in BANDS],
                       os.path.join(figd, "21_individual_band_accuracy.png"),
                       "Per-band illustrative accuracy (%)")
        val_probs = {b: _simulate(200, band_acc[b], SEED + i).proba for i, b in enumerate(BANDS)}
        yv = np.concatenate([_simulate(200, 94, SEED).y for _ in BANDS])[:200]
        wlearn = _learn_band_weights(val_probs, yv)
        pd.DataFrame([{"band": b, "weight": wlearn[b]} for b in BANDS]).to_csv(
            os.path.join(d, "learned_band_weights.csv"), index=False)
        fused = _fuse_probs(val_probs, wlearn)
        yf = yv[: len(fused)]
        yhat = fused.argmax(axis=1)
        tp.confusion_fig(yf, yhat, os.path.join(figd, "26_fused_confusion_matrix.png"),
                         "Fused band probabilities")


def comparison_figure():
    figdir = os.path.join(OUT, "comparison_figures")
    os.makedirs(figdir, exist_ok=True)
    for ds in TARGET_ACC:
        labels = list(TARGET_ACC[ds].keys())
        vals = [TARGET_ACC[ds][k] for k in labels]
        tp.bar_compare(labels, vals, os.path.join(figdir, f"model_ladder_{ds}.png"),
                       f"Illustrative accuracy ladder — {ds}")


def write_root_docs():
    os.makedirs(OUT, exist_ok=True)
    config = {
        "purpose": "illustrative_teaching_scenario",
        "random_seed": SEED,
        "n_samples_per_dataset": N_PER_DATASET,
        "n_folds": N_FOLDS,
        "class_names": CLASSES,
        "target_accuracy_percent": TARGET_ACC,
        "not_experimental": True,
    }
    with open(os.path.join(OUT, "scenario_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    notice = """# SCENARIO NOTICE — ILLUSTRATIVE DATA ONLY

Everything under `01_expected_illustrative_scenario/` is **synthetic teaching material**.

- Targets (e.g. M4 → 95.95 % on a *reference scale*) are **pedagogical anchors**, not SEED-IV runs.
- Predictions are generated so that **confusion matrices, ROC, and summary CSVs agree mathematically**.
- This package must **never** be cited as experimental validation of M1–M4.

Regenerate: `python -m gfer.teaching_scenario`
"""
    with open(os.path.join(OUT, "SCENARIO_NOTICE.md"), "w", encoding="utf-8") as f:
        f.write(notice)

    readme = """# Results packages

| Folder | Provenance |
|--------|------------|
| `00_paper_reference/` | Transcription of the IEEE Access 2024 baseline paper |
| `01_expected_illustrative_scenario/` | **Synthetic** M0–M4 teaching scenario |

Do not mix CSVs or figures between the two.

## Regenerate

```bash
.venv/bin/python scripts/build_paper_reference.py
.venv/bin/python -m gfer.teaching_scenario
```
"""
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write("""# Illustrative expected-results scenario (M0–M4)

See `SCENARIO_NOTICE.md` and `scenario_config.json`.

| Model | Folder |
|-------|--------|
| M0 Original hard gate | `M0_original/` |
| M1 Fuzzy eye gate (existing) | `M1_fuzzy_gate/` |
| M2 Adaptive temporal gate | `M2_adaptive_temporal_gate/` |
| M3 Multi-band fusion | `M3_multiband_fusion/` |
| M4 Combined | `M4_combined_model/` |

Shared comparison charts: `comparison_figures/`
""")


def main():
    write_root_docs()
    build_model("M0", "M0_original", "Original hard-gated baseline (reference-scale labels)")
    build_model("M1", "M1_fuzzy_gate", "Existing fuzzy eye gate improvement")
    build_model("M2", "M2_adaptive_temporal_gate", "Adaptive temporal fuzzy eye gate (proposed)")
    build_model("M3", "M3_multiband_fusion", "Multi-band weighted EEG fusion (proposed)")
    build_model("M4", "M4_combined_model", "M2 + M3 combined (proposed)")
    comparison_figure()
    _write_figure_manifest()
    shutil_copy_scenario_generator()
    print("wrote", OUT)


def _write_figure_manifest():
    manifest = """# Figure manifest (illustrative scenario)

Figures are generated from CSVs in each model folder. Additional methodology panels
reuse `results/00_paper_reference/figures/` for the **published baseline** (not M1–M4 runs).

| ID | Path | Status |
|----|------|--------|
| A1–A9 | See `00_paper_reference/figures/` | Published paper raster |
| B10 | `M1_fuzzy_gate/figures/10_hard_vs_fuzzy_timeline.png` | Generated |
| B11–B14 | See `M1_fuzzy_gate/figures/` + comparison_figures | Partial / extend in code |
| C15–C20 | `M2_adaptive_temporal_gate/figures/` | C18 generated |
| D21–D27 | `M3_multiband_fusion/figures/` | D21,D26 generated |
| E28–E33 | `M4_combined_model/figures/` | Confusion + ROC generated |
"""
    with open(os.path.join(OUT, "FIGURES_MANIFEST.md"), "w", encoding="utf-8") as f:
        f.write(manifest)


def shutil_copy_scenario_generator():
    """Expose scenario_generator.py at package root (symlink content)."""
    src = os.path.abspath(__file__)
    dst = os.path.join(OUT, "scenario_generator.py")
    with open(src, encoding="utf-8") as f:
        body = f.read()
    with open(dst, "w", encoding="utf-8") as f:
        f.write('"""Entry point — run: python scenario_generator.py from this folder."""\n')
        f.write("import runpy\n")
        f.write("runpy.run_module('gfer.teaching_scenario', run_name='__main__')\n")


if __name__ == "__main__":
    main()
