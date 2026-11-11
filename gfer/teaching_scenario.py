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

from . import teaching_plots as tp
from .adaptive_eye import temporal_smooth_weights
from .gates import FUZZY, fuzzy_weights, hard_weights, normalise_within_trial
from .teaching_data import (
    BANDS,
    CLASSES,
    band_accuracy_with_gate,
    build_fusion_dataset,
    emotion_head_from_fusion,
    fusion_accuracy_for_bands,
    fuse_band_probs,
    gate_ablation_band_table,
    noise_gate_sweep,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "01_expected_illustrative_scenario")
SEED = 20261110
N_PER_DATASET = 500
N_FOLDS = 5

TARGET_ACC = {
    "SEED-IV_reference_scale": {"M0": 94.42, "M1": 94.75, "M2": 95.25, "M3": 95.40, "M4": 95.95},
    "MAHNOB_reference_scale": {"M0": 85.20, "M1": 85.55, "M2": 86.05, "M3": 86.20, "M4": 86.75},
}

# A1–E33 manifest entries written at end
FIGURE_REGISTRY: list[dict] = []


def _reg(fig_id: str, name: str, path: str, status: str, source: str, provenance: str):
    FIGURE_REGISTRY.append({
        "id": fig_id, "name": name, "path": path, "status": status,
        "data_source": source, "provenance": provenance,
    })


def _dataset_seed(ds: str, extra: int = 0) -> int:
    return SEED + (hash(ds) % 10_000) + extra


@dataclass
class Pooled:
    y: np.ndarray
    y_pred: np.ndarray
    proba: np.ndarray
    folds: list[dict]


def _ensure_model_dir(name: str) -> str:
    d = os.path.join(OUT, name)
    os.makedirs(os.path.join(d, "figures"), exist_ok=True)
    return d


def _write_predictions(path, y, y_pred, proba, dataset: str, model: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "dataset", "model", "y_true", "y_pred"] + [f"p_{c}" for c in CLASSES])
        for i in range(len(y)):
            w.writerow([i, dataset, model, int(y[i]), int(y_pred[i]), *proba[i].tolist()])


def _write_band_probs(path, dataset: str, y, band_probs: np.ndarray):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["sample_id", "dataset", "y_true"]
        for b in BANDS:
            header += [f"p_{b}_{c}" for c in CLASSES]
        w.writerow(header)
        for i in range(len(y)):
            row = [i, dataset, int(y[i])]
            for b in range(len(BANDS)):
                row.extend(band_probs[i, b].tolist())
            w.writerow(row)


def _metrics_from_arrays(y, y_pred, proba) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    return {
        "accuracy_pct": accuracy_score(y, y_pred) * 100,
        "macro_precision_pct": precision_score(y, y_pred, average="macro", zero_division=0) * 100,
        "macro_recall_pct": recall_score(y, y_pred, average="macro", zero_division=0) * 100,
        "macro_f1_pct": f1_score(y, y_pred, average="macro", zero_division=0) * 100,
        "weighted_f1_pct": f1_score(y, y_pred, average="weighted", zero_division=0) * 100,
        "macro_auc": float(roc_auc_score(y, proba, multi_class="ovr", average="macro")),
    }


def build_classifier_models():
    """M0–M2: same synthetic labels; emotion metrics from consistent fusion head."""
    worlds = {}
    for ds in TARGET_ACC:
        worlds[ds] = build_fusion_dataset(N_PER_DATASET, TARGET_ACC[ds]["M3"], _dataset_seed(ds))

    for model_key, folder, label in [
        ("M0", "M0_original", "Original hard-gated baseline"),
        ("M1", "M1_fuzzy_gate", "Fuzzy eye gate"),
        ("M2", "M2_adaptive_temporal_gate", "Adaptive temporal fuzzy gate"),
    ]:
        d = _ensure_model_dir(folder)
        summaries, folds_all = [], []
        py, pyh, pp = [], [], []
        pred_rows = []
        for ds in TARGET_ACC:
            world = worlds[ds]
            target = TARGET_ACC[ds][model_key]
            proba, y_pred = emotion_head_from_fusion(world, target, _dataset_seed(ds, ord(model_key[1]) * 17))
            m = _metrics_from_arrays(world["y"], y_pred, proba)
            m.update({"dataset": ds, "model": model_key, "target_accuracy_pct": target})
            summaries.append(m)
            for fd in world["folds"]:
                folds_all.append({"dataset": ds, "model": model_key, **fd})
            py.append(world["y"])
            pyh.append(y_pred)
            pp.append(proba)
            for i in range(len(world["y"])):
                pred_rows.append(
                    [i, ds, model_key, int(world["y"][i]), int(y_pred[i]), *proba[i].tolist()]
                )
        with open(os.path.join(d, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sample_id", "dataset", "model", "y_true", "y_pred"] + [f"p_{c}" for c in CLASSES])
            w.writerows(pred_rows)

        pd.DataFrame(summaries).to_csv(os.path.join(d, "summary.csv"), index=False)
        pd.DataFrame(folds_all).to_csv(os.path.join(d, "fold_metrics.csv"), index=False)
        figd = os.path.join(d, "figures")
        y = np.concatenate(py)
        yh = np.concatenate(pyh)
        p = np.vstack(pp)
        tp.confusion_fig(y, yh, os.path.join(figd, "confusion_matrix.png"), f"{model_key} pooled (illustrative)")
        tp.roc_fig(p, y, os.path.join(figd, "roc_curves.png"), f"{model_key} pooled ROC")

        world0 = worlds["SEED-IV_reference_scale"]
        g = world0["gates"]
        t = np.arange(min(60, N_PER_DATASET))

        if model_key == "M0":
            pd.DataFrame([
                {"feature": "mean_pupil_diameter_Y", "rank": 1, "role": "gate (paper)"},
                {"feature": "mean_pupil_diameter_X", "rank": 2},
            ]).to_csv(os.path.join(d, "selected_eye_features.csv"), index=False)
            pd.DataFrame([
                {"feature": "PhEn-FP2", "rank": 1}, {"feature": "SampEn-T8", "rank": 2},
                {"feature": "PermEn-T7", "rank": 3},
            ]).to_csv(os.path.join(d, "selected_eeg_features.csv"), index=False)
            gens = np.arange(1, 451)
            fit = 0.72 + 0.22 * (1 - np.exp(-gens / 80))
            pd.DataFrame({"generation": gens, "best_fitness": fit, "mean_fitness": fit - 0.03}).to_csv(
                os.path.join(d, "ga_history.csv"), index=False)

        if model_key == "M1":
            pd.DataFrame({
                "window": t,
                "z_norm": g["pupil_norm"][: len(t)],
                "w_hard": g["w_hard"][: len(t)],
                "w_fuzzy": g["w_fuzzy"][: len(t)],
            }).to_csv(os.path.join(d, "gate_weights.csv"), index=False)
            tp.gate_timeline(
                t, g["w_hard"][: len(t)], g["w_fuzzy"][: len(t)], g["w_adaptive"][: len(t)],
                os.path.join(figd, "10_hard_vs_fuzzy_timeline.png"),
                "B10 — Hard vs fuzzy gate (synthetic excerpt)",
            )
            _reg("B10", "Hard vs fuzzy timeline", f"{folder}/figures/10_hard_vs_fuzzy_timeline.png",
                 "generated", "gate_weights.csv", "synthetic_teaching")
            tp.membership_m1(os.path.join(figd, "11_fuzzy_membership_functions.png"))
            _reg("B11", "M1 membership functions", f"{folder}/figures/11_fuzzy_membership_functions.png",
                 "generated", "n/a (FUZZY params)", "synthetic_teaching")
            wf = g["w_fuzzy"]
            tp.gate_weight_histogram(wf, os.path.join(figd, "12_gate_weight_distribution.png"))
            pd.DataFrame({"w_fuzzy": wf}).describe().to_csv(os.path.join(d, "gate_weight_summary.csv"))
            _reg("B12", "Gate weight histogram", f"{folder}/figures/12_gate_weight_distribution.png",
                 "generated", "gate_weights.csv", "synthetic_teaching")
            abl = gate_ablation_band_table(world0["band_probs"], world0["y"], g)
            tp.gate_ablation_bands(
                BANDS,
                {k: [abl[k][b] for b in BANDS] for k in abl},
                os.path.join(figd, "13_gate_ablation_by_band.png"),
            )
            pd.DataFrame(abl).to_csv(os.path.join(d, "gate_ablation_by_band.csv"))
            _reg("B13", "Gate ablation by band", f"{folder}/figures/13_gate_ablation_by_band.png",
                 "generated", "gate_ablation_by_band.csv", "synthetic_teaching")
            gq = {k: world0["gate_quality"][k] for k in ("none", "hard", "fuzzy")}
            tp.gate_quality_bars(
                gq, os.path.join(figd, "14_gate_precision_recall_f1.png"),
                "B14 — Gate selection precision / recall / F1",
            )
            _reg("B14", "Gate P/R/F1", f"{folder}/figures/14_gate_precision_recall_f1.png",
                 "generated", "synthetic episode mask", "synthetic_teaching")

        if model_key == "M2":
            w0 = fuzzy_weights(g["pupil_norm"], FUZZY)
            trial = np.zeros(len(g["pupil_raw"]), dtype=int)
            w1 = temporal_smooth_weights(w0, trial, np.arange(len(w0)), mode="symmetric", span=1)
            pd.DataFrame({
                "t": np.arange(len(t)),
                "pupil_raw": g["pupil_raw"][: len(t)],
                "pupil_norm": g["pupil_norm"][: len(t)],
                "w_no_smooth": w0[: len(t)],
                "w_temporal_smooth": w1[: len(t)],
                "w_adaptive": g["w_adaptive"][: len(t)],
            }).to_csv(os.path.join(d, "temporal_gate_scores.csv"), index=False)
            tp.pupil_raw_norm(
                np.arange(len(t)), g["pupil_raw"][: len(t)], g["pupil_norm"][: len(t)],
                os.path.join(figd, "15_raw_vs_normalised_pupil.png"),
            )
            _reg("C15", "Raw vs normalised pupil", f"{folder}/figures/15_raw_vs_normalised_pupil.png",
                 "generated", "temporal_gate_scores.csv", "synthetic_teaching")
            tp.smoothing_effect(
                np.arange(len(t)), w0[: len(t)], w1[: len(t)],
                os.path.join(figd, "16_temporal_smoothing.png"),
            )
            _reg("C16", "Temporal smoothing", f"{folder}/figures/16_temporal_smoothing.png",
                 "generated", "temporal_gate_scores.csv", "synthetic_teaching")
            tp.adaptive_timeline_panel(
                np.arange(len(t)),
                g["pupil_raw"][: len(t)],
                g["pupil_norm"][: len(t)],
                w1[: len(t)],
                g["w_adaptive"][: len(t)],
                os.path.join(figd, "17_adaptive_gate_components.png"),
            )
            _reg("C17", "Adaptive gate components", f"{folder}/figures/17_adaptive_gate_components.png",
                 "generated", "temporal_gate_scores.csv", "synthetic_teaching")
            tp.gate_timeline(
                np.arange(len(t)),
                g["w_hard"][: len(t)],
                g["w_fuzzy"][: len(t)],
                g["w_adaptive"][: len(t)],
                os.path.join(figd, "18_hard_fuzzy_adaptive_comparison.png"),
                "C18 — Hard vs fuzzy vs adaptive on same series",
            )
            _reg("C18", "Gate mechanism comparison", f"{folder}/figures/18_hard_fuzzy_adaptive_comparison.png",
                 "generated", "temporal_gate_scores.csv", "synthetic_teaching")
            tp.gate_quality_extended(world0["gate_quality"], os.path.join(figd, "19_gate_quality_comparison.png"))
            _reg("C19", "Gate quality", f"{folder}/figures/19_gate_quality_comparison.png",
                 "generated", "episode mask in teaching_data", "synthetic_teaching")
            sigmas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
            _, gf1, emo = noise_gate_sweep(N_PER_DATASET, _dataset_seed(ds="noise"), sigmas)
            tp.noise_sensitivity_dual(sigmas, gf1, emo, os.path.join(figd, "20_noise_sensitivity.png"))
            _reg("C20", "Noise sensitivity", f"{folder}/figures/20_noise_sensitivity.png",
                 "generated", "noise_gate_sweep()", "synthetic_teaching")

        with open(os.path.join(d, "report.md"), "w", encoding="utf-8") as f:
            f.write(f"# {model_key}\n\n{label}. Synthetic data — see ../SCENARIO_NOTICE.md.\n")


def build_fusion_model(folder: str, model_key: str, pred_file: str, *, m3_panel: bool, m4_panel: bool):
    d = _ensure_model_dir(folder)
    summaries, folds_all = [], []
    py, pyh, pp = [], [], []
    world_ref = None
    pred_rows: list = []
    band_rows: list = []
    weight_rows: list = []
    for ds in TARGET_ACC:
        target = TARGET_ACC[ds][model_key]
        world = build_fusion_dataset(N_PER_DATASET, target, _dataset_seed(ds, 90 + ord(model_key[1])))
        if world_ref is None:
            world_ref = world
        y, y_pred, proba = world["y"], world["y_pred"], world["proba_fused"]
        for i in range(len(y)):
            pred_rows.append([i, ds, model_key, int(y[i]), int(y_pred[i]), *proba[i].tolist()])
        for i in range(len(y)):
            row = [i, ds, int(y[i])]
            for b in range(len(BANDS)):
                row.extend(world["band_probs"][i, b].tolist())
            band_rows.append(row)
        for i, b in enumerate(BANDS):
            weight_rows.append({
                "dataset": ds,
                "band": b,
                "weight_equal": world["weights_equal"][i],
                "weight_learned": world["weights_learned"][i],
            })

        m = world["metrics"]
        m.update({"dataset": ds, "model": model_key, "target_accuracy_pct": target})
        summaries.append(m)
        for fd in world["folds"]:
            folds_all.append({"dataset": ds, "model": model_key, **fd})
        py.append(y)
        pyh.append(y_pred)
        pp.append(proba)

    with open(os.path.join(d, pred_file), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "dataset", "model", "y_true", "y_pred"] + [f"p_{c}" for c in CLASSES])
        w.writerows(pred_rows)
    with open(os.path.join(d, "band_probabilities.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["sample_id", "dataset", "y_true"] + [f"p_{b}_{c}" for b in BANDS for c in CLASSES]
        w.writerow(header)
        w.writerows(band_rows)
    pd.DataFrame(weight_rows).to_csv(os.path.join(d, "learned_band_weights.csv"), index=False)

    pd.DataFrame(summaries).to_csv(os.path.join(d, "summary.csv"), index=False)
    pd.DataFrame(folds_all).to_csv(os.path.join(d, "fold_metrics.csv"), index=False)
    figd = os.path.join(d, "figures")
    y = np.concatenate(py)
    yh = np.concatenate(pyh)
    p = np.vstack(pp)

    if m3_panel:
        w = world_ref["weights_learned"]
        per_band = [world_ref["per_band_acc"][b] for b in BANDS]
        tp.bar_compare(BANDS, per_band, os.path.join(figd, "21_individual_band_accuracy.png"),
                       "D21 — Per-band argmax accuracy (synthetic)")
        _reg("D21", "Individual band accuracy", f"{folder}/figures/21_individual_band_accuracy.png",
             "generated", "band_probabilities.csv", "synthetic_teaching")
        tp.band_weights_bar(BANDS, w.tolist(), os.path.join(figd, "22_learned_band_weights.png"))
        _reg("D22", "Learned band weights", f"{folder}/figures/22_learned_band_weights.png",
             "generated", "learned_band_weights.csv", "synthetic_teaching")
        y0 = world_ref["y"]
        bp = world_ref["band_probs"]
        acc_alpha = fusion_accuracy_for_bands(bp, y0, ["alpha"], np.array([1.0]))
        acc_multi = float((world_ref["proba_fused"].argmax(1) == y0).mean() * 100)
        tp.fusion_strategy_compare(
            ["Alpha only", "5-band fused (learned)"],
            [acc_alpha, acc_multi],
            os.path.join(figd, "23_alpha_vs_multiband.png"),
            "D23 — Alpha-only vs multiband fusion",
        )
        _reg("D23", "Alpha vs multiband", f"{folder}/figures/23_alpha_vs_multiband.png",
             "generated", "band_probabilities.csv", "synthetic_teaching")
        two = fusion_accuracy_for_bands(bp, y0, ["alpha", "beta"], world_ref["weights_learned"])
        three = fusion_accuracy_for_bands(bp, y0, ["alpha", "beta", "gamma"], world_ref["weights_learned"])
        tp.fusion_strategy_compare(
            ["α+β", "α+β+γ", "All 5"],
            [two, three, acc_multi],
            os.path.join(figd, "24_fusion_band_count.png"),
            "D24 — Two- vs three- vs five-band fusion",
        )
        _reg("D24", "Band-count fusion", f"{folder}/figures/24_fusion_band_count.png",
             "generated", "band_probabilities.csv", "synthetic_teaching")
        acc_eq = float((world_ref["y_pred_equal"] == y0).mean() * 100)
        acc_lr = float((world_ref["y_pred"] == y0).mean() * 100)
        tp.fusion_strategy_compare(
            ["Equal weights", "Validation-learned"],
            [acc_eq, acc_lr],
            os.path.join(figd, "25_equal_vs_learned_fusion.png"),
            "D25 — Equal vs learned fusion weights",
        )
        _reg("D25", "Equal vs learned", f"{folder}/figures/25_equal_vs_learned_fusion.png",
             "generated", "learned_band_weights.csv", "synthetic_teaching")
        tp.confusion_fig(y, yh, os.path.join(figd, "26_fused_confusion_matrix.png"),
                         "D26 — M3 fused predictions (illustrative)")
        _reg("D26", "Fused confusion matrix", f"{folder}/figures/26_fused_confusion_matrix.png",
             "generated", pred_file, "synthetic_teaching")
        tp.roc_fig(p, y, os.path.join(figd, "27_fused_roc_curves.png"), "D27 — M3 fused one-vs-rest ROC")
        _reg("D27", "Fused ROC", f"{folder}/figures/27_fused_roc_curves.png",
             "generated", pred_file, "synthetic_teaching")

    tp.confusion_fig(y, yh, os.path.join(figd, "confusion_matrix.png"), f"{model_key} pooled CM")
    tp.roc_fig(p, y, os.path.join(figd, "roc_curves.png"), f"{model_key} pooled ROC")

    if m4_panel:
        g = world_ref["gates"]
        tt = np.arange(min(80, len(g["w_adaptive"])))
        tp.m4_gate_timeline(tt, g["w_adaptive"][: len(tt)], os.path.join(figd, "29_m4_adaptive_gate.png"))
        _reg("E29", "M4 adaptive gate", f"{folder}/figures/29_m4_adaptive_gate.png",
             "generated", "band_probabilities.csv + gates", "synthetic_teaching")
        mean_bp = world_ref["band_probs"][: len(tt)].mean(axis=0)
        w = world_ref["weights_learned"]
        tp.m4_band_contribution(BANDS, w, mean_bp, os.path.join(figd, "30_m4_band_contributions.png"))
        _reg("E30", "M4 band contributions", f"{folder}/figures/30_m4_band_contributions.png",
             "generated", "learned_band_weights.csv", "synthetic_teaching")
        tp.architecture_e28(os.path.join(figd, "28_combined_architecture.png"))
        _reg("E28", "Combined architecture", f"{folder}/figures/28_combined_architecture.png",
             "generated", "n/a (diagram)", "synthetic_teaching")
        tp.classwise_grouped(y, yh, os.path.join(figd, "33_classwise_precision_recall_f1.png"))
        _reg("E33", "Class-wise metrics", f"{folder}/figures/33_classwise_precision_recall_f1.png",
             "generated", "final_predictions.csv", "synthetic_teaching")
        # E31/E32 alias pooled CM and ROC with explicit IDs
        tp.confusion_fig(y, yh, os.path.join(figd, "31_final_confusion_matrix.png"),
                         "E31 — M4 final confusion (from final_predictions.csv)")
        tp.roc_fig(p, y, os.path.join(figd, "32_final_roc_curves.png"), "E32 — M4 final ROC")
        _reg("E31", "Final CM", f"{folder}/figures/31_final_confusion_matrix.png",
             "generated", "final_predictions.csv", "synthetic_teaching")
        _reg("E32", "Final ROC", f"{folder}/figures/32_final_roc_curves.png",
             "generated", "final_predictions.csv", "synthetic_teaching")


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


def _paper_figure_registry():
    """A1–A9: published / derived methodology figures in 00_paper_reference."""
    base = "00_paper_reference/figures"
    mapping = [
        ("A1", "Paper Fig. 7 eye mRMR", f"{base}/figure_07_eye_mrmr_ranking.png", "eye_mrmr_ranking.csv"),
        ("A2", "Paper Fig. 8 Sugeno FIS", f"{base}/figure_08_sugeno_fis_architecture.png", "n/a"),
        ("A3", "Paper Fig. 9 GA convergence", f"{base}/figure_09_ga_convergence.png", "n/a"),
        ("A4", "Derived triangular MFs", f"{base}/figure_A4_fuzzy_membership_functions.png", "paper_figures_method.py"),
        ("A5", "Derived rule space", f"{base}/figure_A5_fuzzy_rule_distribution.png", "paper_figures_method.py"),
        ("A6", "Paper Fig. 10 SEED-IV CM", f"{base}/figure_10_seed_iv_confusion_matrices.png", "n/a"),
        ("A7", "Paper Fig. 11 SEED-IV ROC", f"{base}/figure_11_seed_iv_roc_curves.png", "n/a"),
        ("A8", "Paper Fig. 12 MAHNOB CM", f"{base}/figure_12_mahnob_confusion_matrices.png", "n/a"),
        ("A9", "Paper Fig. 13 MAHNOB ROC", f"{base}/figure_13_mahnob_roc_curves.png", "n/a"),
    ]
    for fid, name, path, src in mapping:
        full = os.path.join(ROOT, "results", path)
        status = "generated" if os.path.isfile(full) else "missing"
        _reg(fid, name, path, status, src, "published_paper_or_derived_methodology")


def _write_figure_manifest():
    _paper_figure_registry()
    lines = [
        "# Figure manifest (illustrative + paper reference)",
        "",
        "**Provenance:** A1–A9 → `00_paper_reference/` (published crops + derived A4–A5). "
        "B10–E33 → `01_expected_illustrative_scenario/` (synthetic teaching data).",
        "",
        "| ID | Name | Path | Status | Data source | Provenance |",
        "|----|------|------|--------|-------------|------------|",
    ]
    for r in sorted(FIGURE_REGISTRY, key=lambda x: x["id"]):
        lines.append(
            f"| {r['id']} | {r['name']} | `{r['path']}` | {r['status']} | {r['data_source']} | {r['provenance']} |"
        )
    with open(os.path.join(OUT, "FIGURES_MANIFEST.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def shutil_copy_scenario_generator():
    src = os.path.abspath(__file__)
    dst = os.path.join(OUT, "scenario_generator.py")
    with open(dst, "w", encoding="utf-8") as f:
        f.write('"""Entry point — run: python scenario_generator.py from this folder."""\n')
        f.write("import runpy\n")
        f.write("runpy.run_module('gfer.teaching_scenario', run_name='__main__')\n")


def main():
    FIGURE_REGISTRY.clear()
    write_root_docs()
    build_classifier_models()
    build_fusion_model("M3_multiband_fusion", "M3", "fusion_predictions.csv", m3_panel=True, m4_panel=False)
    build_fusion_model("M4_combined_model", "M4", "final_predictions.csv", m3_panel=False, m4_panel=True)
    comparison_figure()
    _write_figure_manifest()
    shutil_copy_scenario_generator()
    print("wrote", OUT, f"({len(FIGURE_REGISTRY)} manifest entries)")


if __name__ == "__main__":
    main()
