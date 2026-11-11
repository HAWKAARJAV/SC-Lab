"""Generate results/01_expected_illustrative_scenario/ — synthetic teaching data only."""
from __future__ import annotations

import csv
import json
import os

import numpy as np
import pandas as pd

from . import teaching_plots as tp
from .teaching_data import (
    BANDS,
    CLASSES,
    TARGET_REFERENCE,
    build_teaching_world,
    fuse_band_probs,
    fusion_accuracy_for_bands,
    gate_ablation_band_table,
    noise_gate_sweep,
    stable_seed,
    verify_fusion,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "01_expected_illustrative_scenario")
MASTER_SEED = 20261110
N_PER_DATASET = 500
FIGURE_REGISTRY: list[dict] = []


def _reg(fig_id, name, path, status, source, provenance):
    FIGURE_REGISTRY.append({"id": fig_id, "name": name, "path": path, "status": status,
                            "data_source": source, "provenance": provenance})


def _ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def _write_preds(path, world: TeachingWorld, model, y_pred, proba):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "subject_id", "trial_id", "window_id", "dataset", "model",
                    "y_true", "y_pred"] + [f"p_{c}" for c in CLASSES])
        for i in range(world.n):
            w.writerow([i, world.subject_id[i], world.trial_id[i], world.window_id[i],
                        world.dataset, model, int(world.y[i]), int(y_pred[i]), *proba[i].tolist()])


def _summary_row(world, model, out, ref):
    m = out.metrics_pooled
    return {
        "dataset": world.dataset,
        "model": model,
        "reference_accuracy_pct": ref,
        "achieved_accuracy_pct": m["accuracy_pct"],
        "delta_vs_reference_pct": m["accuracy_pct"] - ref,
        **m,
    }


def _save_gate_quality(path, gq):
    rows = []
    for k, v in gq.items():
        rows.append({"gate_policy": k, **v})
    pd.DataFrame(rows).to_csv(path, index=False)


def _save_eye_tables(model_dir, world: TeachingWorld, n_excerpt=80):
    pd.DataFrame(world.pupil_baselines.values()).to_csv(
        os.path.join(model_dir, "pupil_baselines.csv"), index=False)
    n = min(n_excerpt, world.n)
    pd.DataFrame({
        "sample_id": np.arange(n),
        "subject_id": world.subject_id[:n],
        "trial_id": world.trial_id[:n],
        "window_id": world.window_id[:n],
        "pupil_raw": world.pupil_raw[:n],
        "pupil_norm": world.pupil_norm[:n],
    }).to_csv(os.path.join(model_dir, "normalised_eye_features.csv"), index=False)
    pd.DataFrame({
        "sample_id": np.arange(n),
        "subject_id": world.subject_id[:n],
        "trial_id": world.trial_id[:n],
        "window_id": world.window_id[:n],
        "pupil_norm": world.pupil_norm[:n],
        "w_no_smooth": world.w_fuzzy[:n],
        "w_temporal_smooth": world.w_smooth[:n],
        "smooth_alpha": 0.25,
        "smooth_beta": 0.50,
        "smooth_gamma": 0.25,
    }).to_csv(os.path.join(model_dir, "temporal_gate_scores.csv"), index=False)
    pd.DataFrame({
        "sample_id": np.arange(n),
        "w_hard": world.w_hard[:n],
        "w_fuzzy": world.w_fuzzy[:n],
        "w_adaptive": world.w_adaptive[:n],
    }).to_csv(os.path.join(model_dir, "gate_weights.csv"), index=False)


def _save_band_probs(path, world, model, band_probs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        hdr = ["sample_id", "dataset", "model", "y_true"] + [f"p_{b}_{c}" for b in BANDS for c in CLASSES]
        w.writerow(hdr)
        for i in range(world.n):
            row = [i, world.dataset, model, int(world.y[i])]
            for b in range(5):
                row.extend(band_probs[i, b].tolist())
            w.writerow(row)


def build_classifier(model: str, folder: str, worlds: dict):
    d = _ensure(os.path.join(OUT, folder))
    figd = _ensure(os.path.join(d, "figures"))
    summaries, folds, preds_y, preds_yh, preds_p, pred_path = [], [], [], [], [], os.path.join(d, "predictions.csv")
    first_written = False
    for ds, world in worlds.items():
        out = world.classifier_outputs(model)
        ref = TARGET_REFERENCE[ds][model]
        summaries.append(_summary_row(world, model, out, ref))
        folds.extend(out.folds)
        preds_y.append(world.y)
        preds_yh.append(out.y_pred)
        preds_p.append(out.proba)
        mode = "w" if not first_written else "a"
        with open(pred_path, mode, newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not first_written:
                w.writerow(["sample_id", "subject_id", "trial_id", "window_id", "dataset", "model",
                            "y_true", "y_pred"] + [f"p_{c}" for c in CLASSES])
                first_written = True
            for i in range(world.n):
                w.writerow([i, world.subject_id[i], world.trial_id[i], world.window_id[i],
                            world.dataset, model, int(world.y[i]), int(out.y_pred[i]), *out.proba[i].tolist()])

        if model == "M0" and ds == "SEED-IV_reference_scale":
            pd.DataFrame([
                {"feature": "mean_pupil_diameter_Y", "rank": 1}, {"feature": "mean_pupil_diameter_X", "rank": 2},
            ]).to_csv(os.path.join(d, "selected_eye_features.csv"), index=False)
            pd.DataFrame([
                {"feature": "PhEn-FP2", "rank": 1}, {"feature": "SampEn-T8", "rank": 2}, {"feature": "PermEn-T7", "rank": 3},
            ]).to_csv(os.path.join(d, "selected_eeg_features.csv"), index=False)
            gens = np.arange(1, 451)
            fit = 0.72 + 0.22 * (1 - np.exp(-gens / 80))
            pd.DataFrame({"generation": gens, "best_fitness": fit, "mean_fitness": fit - 0.03}).to_csv(
                os.path.join(d, "ga_history.csv"), index=False)
            pd.DataFrame([{"rule_id": i, "if_part": f"illustrative_rule_{i}", "then_class": CLASSES[i % 4]}
                          for i in range(1, 11)]).to_csv(os.path.join(d, "fuzzy_rules.csv"), index=False)

        if model == "M1" and ds == "SEED-IV_reference_scale":
            pd.DataFrame([{"rule_id": i, "if_part": f"Illustrative antecedent {i}", "then_class": CLASSES[i % 4]}
                          for i in range(1, 11)]).to_csv(os.path.join(d, "fuzzy_rules.csv"), index=False)
            _save_eye_tables(d, world, 60)
            _save_gate_quality(os.path.join(d, "gate_quality.csv"), world.gate_quality_table())
            t = np.arange(60)
            tp.gate_timeline(t, world.w_hard[t], world.w_fuzzy[t], world.w_adaptive[t],
                             os.path.join(figd, "10_hard_vs_fuzzy_timeline.png"),
                             "B10 — Hard vs fuzzy gate (synthetic excerpt)")
            tp.membership_m1(os.path.join(figd, "11_fuzzy_membership_functions.png"))
            tp.gate_weight_histogram(world.w_fuzzy, os.path.join(figd, "12_gate_weight_distribution.png"))
            abl = gate_ablation_band_table(out.band_probs, world.y, world.w_hard, world.w_fuzzy)
            tp.gate_ablation_bands(BANDS, {k: [abl[k][b] for b in BANDS] for k in abl},
                                   os.path.join(figd, "13_gate_ablation_by_band.png"))
            gq = world.gate_quality_table()
            tp.gate_quality_bars({k: gq[k] for k in ("none", "hard", "fuzzy")},
                                 os.path.join(figd, "14_gate_precision_recall_f1.png"),
                                 "B14 — Gate selection precision / recall / F1")
            _reg("B10", "Hard vs fuzzy timeline", f"{folder}/figures/10_hard_vs_fuzzy_timeline.png",
                 "generated", "gate_weights.csv", "synthetic_teaching")
            _reg("B11", "M1 membership", f"{folder}/figures/11_fuzzy_membership_functions.png",
                 "generated", "FUZZY params", "synthetic_teaching")
            _reg("B12", "Weight histogram", f"{folder}/figures/12_gate_weight_distribution.png",
                 "generated", "gate_weights.csv", "synthetic_teaching")
            _reg("B13", "Gate ablation", f"{folder}/figures/13_gate_ablation_by_band.png",
                 "generated", "band_probabilities", "synthetic_teaching")
            _reg("B14", "Gate P/R/F1", f"{folder}/figures/14_gate_precision_recall_f1.png",
                 "generated", "gate_quality.csv", "synthetic_teaching")

        if model == "M2" and ds == "SEED-IV_reference_scale":
            _save_eye_tables(d, world)
            _save_gate_quality(os.path.join(d, "gate_quality.csv"), world.gate_quality_table())
            t = np.arange(60)
            tp.pupil_raw_norm(t, world.pupil_raw[t], world.pupil_norm[t],
                              os.path.join(figd, "15_raw_vs_normalised_pupil.png"))
            tp.smoothing_effect(t, world.w_fuzzy[t], world.w_smooth[t],
                              os.path.join(figd, "16_temporal_smoothing.png"))
            tp.adaptive_timeline_panel(t, world.pupil_raw[t], world.pupil_norm[t],
                                       world.w_smooth[t], world.w_adaptive[t],
                                       os.path.join(figd, "17_adaptive_gate_components.png"))
            tp.gate_timeline(t, world.w_hard[t], world.w_fuzzy[t], world.w_adaptive[t],
                             os.path.join(figd, "18_hard_fuzzy_adaptive_comparison.png"),
                             "C18 — Hard vs fuzzy vs adaptive")
            tp.gate_quality_extended(world.gate_quality_table(),
                                     os.path.join(figd, "19_gate_quality_comparison.png"))
            sigmas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
            _, gf1, emo = noise_gate_sweep(world, sigmas)
            tp.noise_sensitivity_dual(sigmas, gf1, emo, os.path.join(figd, "20_noise_sensitivity.png"))
            for cid, name in [("C15", "15_raw_vs_normalised_pupil.png"), ("C16", "16_temporal_smoothing.png"),
                              ("C17", "17_adaptive_gate_components.png"), ("C18", "18_hard_fuzzy_adaptive_comparison.png"),
                              ("C19", "19_gate_quality_comparison.png"), ("C20", "20_noise_sensitivity.png")]:
                _reg(cid, cid, f"{folder}/figures/{name}", "generated", "eye/gate CSVs", "synthetic_teaching")

    pd.DataFrame(summaries).to_csv(os.path.join(d, "summary.csv"), index=False)
    pd.DataFrame(folds).to_csv(os.path.join(d, "fold_metrics.csv"), index=False)
    y = np.concatenate(preds_y)
    yh = np.concatenate(preds_yh)
    p = np.vstack(preds_p)
    tp.confusion_fig(y, yh, os.path.join(figd, "confusion_matrix.png"), f"{model} pooled CM")
    tp.roc_fig(p, y, os.path.join(figd, "roc_curves.png"), f"{model} pooled ROC")
    if model == "M0":
        from sklearn.metrics import precision_recall_fscore_support
        p, r, f1, sup = precision_recall_fscore_support(y, yh, labels=range(4), zero_division=0)
        pd.DataFrame([{"class": CLASSES[c], "precision_pct": 100 * p[c], "recall_pct": 100 * r[c],
                       "f1_pct": 100 * f1[c], "support": int(sup[c])} for c in range(4)]).to_csv(
            os.path.join(d, "per_class_metrics.csv"), index=False)


def build_fusion(model: str, folder: str, worlds: dict, *, m3_figs: bool, m4_figs: bool):
    d = _ensure(os.path.join(OUT, folder))
    figd = _ensure(os.path.join(d, "figures"))
    summaries, folds, py, pyh, pp = [], [], [], [], []
    world_ref = None
    weight_rows, band_metric_rows = [], []
    pred_rows, band_rows = [], []
    for ds, world in worlds.items():
        out = world.fusion_outputs(model)
        ref = TARGET_REFERENCE[ds][model]
        summaries.append(_summary_row(world, model, out, ref))
        folds.extend(out.folds)
        py.append(world.y)
        pyh.append(out.y_pred)
        pp.append(out.proba)
        verify_fusion(out.proba, out.band_probs, out.weights_learned)
        for i in range(world.n):
            pred_rows.append([i, world.subject_id[i], world.trial_id[i], world.window_id[i],
                              world.dataset, model, int(world.y[i]), int(out.y_pred[i]), *out.proba[i].tolist()])
        for i in range(world.n):
            row = [i, world.dataset, model, int(world.y[i])]
            for b in range(5):
                row.extend(out.band_probs[i, b].tolist())
            band_rows.append(row)
        for i, b in enumerate(BANDS):
            weight_rows.append({
                "dataset": ds, "band": b,
                "weight_equal": out.weights_equal[i],
                "weight_learned": out.weights_learned[i],
            })
        for b in BANDS:
            bi = BANDS.index(b)
            pred = out.band_probs[:, bi].argmax(1)
            band_metric_rows.append({
                "dataset": ds, "band": b,
                "accuracy_pct": (pred == world.y).mean() * 100,
            })
        if world_ref is None:
            world_ref = (world, out)
    pred_hdr = ["sample_id", "subject_id", "trial_id", "window_id", "dataset", "model",
                "y_true", "y_pred"] + [f"p_{c}" for c in CLASSES]
    for fname in (["fusion_predictions.csv", "final_predictions.csv"] if model == "M4" else ["fusion_predictions.csv"]):
        with open(os.path.join(d, fname), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(pred_hdr)
            w.writerows(pred_rows)
    with open(os.path.join(d, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(pred_hdr)
        w.writerows(pred_rows)
    with open(os.path.join(d, "band_probabilities.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "dataset", "model", "y_true"] + [f"p_{b}_{c}" for b in BANDS for c in CLASSES])
        w.writerows(band_rows)
    pd.DataFrame(weight_rows).to_csv(os.path.join(d, "learned_band_weights.csv"), index=False)
    pd.DataFrame(band_metric_rows).to_csv(os.path.join(d, "band_metrics.csv"), index=False)
    pred_name = "final_predictions.csv" if model == "M4" else "fusion_predictions.csv"
    pd.DataFrame(summaries).to_csv(os.path.join(d, "summary.csv"), index=False)
    pd.DataFrame(folds).to_csv(os.path.join(d, "fold_metrics.csv"), index=False)
    y, yh, p = np.concatenate(py), np.concatenate(pyh), np.vstack(pp)
    w0, o0 = world_ref
    if m3_figs:
        per_band = [o0.per_band_acc[b] for b in BANDS]
        tp.bar_compare(BANDS, per_band, os.path.join(figd, "21_individual_band_accuracy.png"),
                       "D21 — Per-band argmax accuracy")
        tp.band_weights_bar(BANDS, o0.weights_learned.tolist(), os.path.join(figd, "22_learned_band_weights.png"))
        acc_a = fusion_accuracy_for_bands(o0.band_probs, w0.y, ["alpha"], o0.weights_learned)
        acc_m = float((o0.proba.argmax(1) == w0.y).mean() * 100)
        tp.fusion_strategy_compare(["Alpha only", "5-band learned fusion"], [acc_a, acc_m],
                                   os.path.join(figd, "23_alpha_vs_multiband.png"), "D23 — Alpha vs multiband")
        two = fusion_accuracy_for_bands(o0.band_probs, w0.y, ["alpha", "beta"], o0.weights_learned)
        three = fusion_accuracy_for_bands(o0.band_probs, w0.y, ["alpha", "beta", "gamma"], o0.weights_learned)
        tp.fusion_strategy_compare(["α+β", "α+β+γ", "All 5"], [two, three, acc_m],
                                   os.path.join(figd, "24_fusion_band_count.png"), "D24 — Band-count fusion")
        eq = float((fuse_band_probs(o0.band_probs, o0.weights_equal).argmax(1) == w0.y).mean() * 100)
        tp.fusion_strategy_compare(["Equal weights", "Learned weights"], [eq, acc_m],
                                   os.path.join(figd, "25_equal_vs_learned_fusion.png"), "D25 — Fusion weights")
        tp.confusion_fig(y, yh, os.path.join(figd, "26_fused_confusion_matrix.png"), "D26 — M3 fused CM")
        tp.roc_fig(p, y, os.path.join(figd, "27_fused_roc_curves.png"), "D27 — M3 fused ROC")
        for i in range(21, 28):
            _reg(f"D{i}", f"D{i}", f"{folder}/figures/", "generated", pred_name, "synthetic_teaching")

    tp.confusion_fig(y, yh, os.path.join(figd, "confusion_matrix.png"), f"{model} CM")
    tp.roc_fig(p, y, os.path.join(figd, "roc_curves.png"), f"{model} ROC")
    if m4_figs:
        _save_eye_tables(d, w0, 80)
        _save_gate_quality(os.path.join(d, "gate_quality.csv"), w0.gate_quality_table())
        tt = np.arange(80)
        tp.architecture_e28(os.path.join(figd, "28_combined_architecture.png"))
        tp.m4_gate_timeline(tt, w0.w_adaptive[tt], os.path.join(figd, "29_m4_adaptive_gate.png"))
        mean_bp = o0.band_probs[:80].mean(axis=0)
        tp.m4_band_contribution(BANDS, o0.weights_learned, mean_bp,
                                os.path.join(figd, "30_m4_band_contributions.png"))
        tp.confusion_fig(y, yh, os.path.join(figd, "31_final_confusion_matrix.png"), "E31 — M4 final CM")
        tp.roc_fig(p, y, os.path.join(figd, "32_final_roc_curves.png"), "E32 — M4 final ROC")
        tp.classwise_grouped(y, yh, os.path.join(figd, "33_classwise_precision_recall_f1.png"))
        pd.DataFrame(o0.metrics_per_class).to_csv(os.path.join(d, "per_class_metrics.csv"), index=False)
        for eid, nm in [("E28", "28_combined_architecture.png"), ("E29", "29_m4_adaptive_gate.png"),
                        ("E30", "30_m4_band_contributions.png"), ("E31", "31_final_confusion_matrix.png"),
                        ("E32", "32_final_roc_curves.png"), ("E33", "33_classwise_precision_recall_f1.png")]:
            _reg(eid, eid, f"{folder}/figures/{nm}", "generated", "final_predictions.csv", "synthetic_teaching")


def write_docs():
    os.makedirs(OUT, exist_ok=True)
    seeds = {ds: stable_seed(MASTER_SEED, ds) for ds in TARGET_REFERENCE}
    cfg = {
        "purpose": "illustrative_teaching_scenario",
        "master_seed": MASTER_SEED,
        "dataset_seeds": seeds,
        "seed_derivation": "hashlib.sha256 over master:dataset:labels",
        "n_samples_per_dataset": N_PER_DATASET,
        "class_names": CLASSES,
        "reference_accuracy_percent": TARGET_REFERENCE,
        "not_experimental": True,
        "generator_assumptions": "See SCENARIO_ASSUMPTIONS.md",
    }
    with open(os.path.join(OUT, "scenario_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    assumptions = """# Illustrative scenario assumptions (synthetic)

1. **Shared windows**: All models M0–M4 use identical subject/trial/window IDs and ground-truth labels.
2. **Eye gate**: Pupil is generated with subject baselines (train-masked medians); episodes modulate diameter.
3. **Band classifiers**: Each band has distinct logit scale, noise, and confusion partners (not 100% separable).
4. **Gate coupling**: Gate weight scales class logit separation (weak gate → probabilities closer to uniform).
5. **M0/M1/M2**: Alpha-band softmax after hard / fuzzy / adaptive gating (other bands logged in `band_probs`).
6. **M3/M4**: Hard (M3) or adaptive (M4) gating on band logits; fusion weights learned on validation indices only.
7. **Fusion**: `P_final(c) = Σ_b w_b P_b(c)` — saved probabilities are never post-edited for target accuracy.
8. **Reference targets** in `summary.csv` are paper-scale anchors; `achieved_*` may differ.
"""
    with open(os.path.join(OUT, "SCENARIO_ASSUMPTIONS.md"), "w") as f:
        f.write(assumptions)


def _paper_registry():
    base = "00_paper_reference/figures"
    for fid, fn in [
        ("A1", "figure_07_eye_mrmr_ranking.png"), ("A2", "figure_08_sugeno_fis_architecture.png"),
        ("A3", "figure_09_ga_convergence.png"), ("A4", "figure_A4_fuzzy_membership_functions.png"),
        ("A5", "figure_A5_fuzzy_rule_distribution.png"), ("A6", "figure_10_seed_iv_confusion_matrices.png"),
        ("A7", "figure_11_seed_iv_roc_curves.png"), ("A8", "figure_12_mahnob_confusion_matrices.png"),
        ("A9", "figure_13_mahnob_roc_curves.png"),
    ]:
        p = os.path.join(ROOT, "results", base, fn.split("/")[-1] if "/" not in fn else fn)
        _reg(fid, fn, f"{base}/{os.path.basename(fn)}",
             "generated" if os.path.isfile(p) else "missing", "paper PDF", "published_or_derived")


def _manifest():
    _paper_registry()
    lines = ["# Figure manifest", "", "| ID | Name | Path | Status | Data source | Provenance |",
             "|----|------|------|--------|-------------|------------|"]
    for r in sorted(FIGURE_REGISTRY, key=lambda x: x["id"]):
        lines.append(f"| {r['id']} | {r['name']} | `{r['path']}` | {r['status']} | {r['data_source']} | {r['provenance']} |")
    with open(os.path.join(OUT, "FIGURES_MANIFEST.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    global FIGURE_REGISTRY
    FIGURE_REGISTRY = []
    write_docs()
    worlds = {ds: build_teaching_world(ds, N_PER_DATASET, stable_seed(MASTER_SEED, ds))
              for ds in TARGET_REFERENCE}
    for m, folder in [("M0", "M0_original"), ("M1", "M1_fuzzy_gate"), ("M2", "M2_adaptive_temporal_gate")]:
        build_classifier(m, folder, worlds)
    build_fusion("M3", "M3_multiband_fusion", worlds, m3_figs=True, m4_figs=False)
    build_fusion("M4", "M4_combined_model", worlds, m3_figs=False, m4_figs=True)
    _ensure(os.path.join(OUT, "comparison_figures"))
    achieved = {ds: {} for ds in TARGET_REFERENCE}
    for m, folder in [("M0", "M0_original"), ("M1", "M1_fuzzy_gate"), ("M2", "M2_adaptive_temporal_gate"),
                      ("M3", "M3_multiband_fusion"), ("M4", "M4_combined_model")]:
        sm = pd.read_csv(os.path.join(OUT, folder, "summary.csv"))
        for _, row in sm.iterrows():
            achieved[row["dataset"]][m] = row["achieved_accuracy_pct"]
    for ds in TARGET_REFERENCE:
        lbls = ["M0", "M1", "M2", "M3", "M4"]
        tp.bar_compare(lbls, [achieved[ds][m] for m in lbls],
                       os.path.join(OUT, "comparison_figures", f"model_ladder_{ds}.png"),
                       f"Achieved synthetic accuracy — {ds}")
    with open(os.path.join(OUT, "SCENARIO_NOTICE.md"), "w") as f:
        f.write("# ILLUSTRATIVE SYNTHETIC DATA ONLY\n\nSee `SCENARIO_ASSUMPTIONS.md` and `scenario_config.json`.\n")
    _manifest()
    with open(os.path.join(OUT, "scenario_generator.py"), "w") as f:
        f.write("import runpy\nrunpy.run_module('gfer.teaching_scenario', run_name='__main__')\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
