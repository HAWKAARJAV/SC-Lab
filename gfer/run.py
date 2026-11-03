"""End-to-end pipeline:  python -m gfer.run --data data/demo --out results/demo

1. load trials -> 2. per-window eye + EEG entropy features (cached)
3. rank eye features with mRMR, gate EEG windows on the top one
4. per band and per number of fuzzy levels: mRMR -> S-FIS (lookup table) -> GA -> CV metrics
5. ablations: no gate; shuffled labels (sanity control)
6. write results.json, summary.csv, report.md and figures
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from . import plots
from .config import Config
from .data import load_dataset
from .evaluate import apply_gate, roc_points, run_band, select_gate_feature
from .fuzzy import LEVEL_NAMES  # noqa: F401  (re-exported for report readers)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items() if k not in ("models", "pooled_proba", "pooled_y")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/demo")
    ap.add_argument("--dataset", default="npz", help="loader: npz (demo), ...")
    ap.add_argument("--out", default="results/demo")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--protocol", default=None, choices=["paper", "trial", "loso"])
    ap.add_argument("--levels", default=None, help="comma list, e.g. 2,3,4")
    ap.add_argument("--bands", default=None, help="comma list, default all five")
    ap.add_argument("--gate-mode", default=None, choices=["quantile", "max_ratio", "none"])
    ap.add_argument("--gate-feature", default=None,
                    help="'auto' (top mRMR eye feature) or a name, e.g. pupil_y_mean (paper)")
    ap.add_argument("--gate-quantile", type=float, default=None)
    ap.add_argument("--features", default=None,
                    help="'auto' (mRMR) or comma list e.g. PhEn-FP2,SampEn-T8,PermEn-T7")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    a = parse_args()
    cfg = Config(seed=a.seed)
    if a.protocol: cfg.protocol = a.protocol
    if a.levels: cfg.n_levels = [int(v) for v in a.levels.split(",")]
    if a.bands: cfg.bands = a.bands.split(",")
    if a.gate_mode: cfg.gate_mode = a.gate_mode
    if a.gate_feature: cfg.gate_feature = a.gate_feature
    if a.gate_quantile is not None: cfg.gate_quantile = a.gate_quantile
    if a.features and a.features != "auto": cfg.eeg_features = a.features.split(",")
    os.makedirs(a.out, exist_ok=True)
    fig_dir = os.path.join(a.out, "figures"); os.makedirs(fig_dir, exist_ok=True)
    t0 = time.time()

    # 1-2 ----------------------------------------------------------------
    trials, meta = load_dataset(a.dataset, a.data)
    classes = meta["class_names"]; C = len(classes)
    print(f"Loaded {len(trials)} trials, {len(set(t.subject for t in trials))} subjects, "
          f"{len(trials[0].ch_names)} channels, classes={classes}", flush=True)
    F = extract_all_cached(trials, cfg, a.cache)
    print(f"Features: {F['eeg'].shape[0]} windows x {F['eeg'].shape[1:]} (band, entropy, channel)  "
          f"[{time.time() - t0:.0f}s]", flush=True)

    # 3 ------------------------------------------------------------------
    gate_name, ranking, _ = select_gate_feature(F, cfg)
    keep = apply_gate(F, cfg, gate_name)
    print(f"Gate feature: {gate_name}; kept {keep.sum()}/{len(keep)} windows", flush=True)
    plots.eye_ranking(ranking, os.path.join(fig_dir, "eye_mrmr_ranking.png"))
    gate_check = _gate_check(F, keep, a.data, cfg, fig_dir, gate_name, trials)

    # 4 ------------------------------------------------------------------
    rng = np.random.default_rng(cfg.seed)
    results = {}
    for L in cfg.n_levels:
        for band in cfg.bands:
            r = run_band(F, keep, band, cfg, C, L, rng)
            results[(L, band)] = r
            s = r["summary"]
            print(f"  levels={L} {band:5s}  acc={s['accuracy']['mean']:.3f}±{s['accuracy']['std']:.3f}  "
                  f"F1={s['f1_macro']['mean']:.3f}  AUC={s['auc_macro']['mean']:.3f}  "
                  f"(lookup-only acc={s['acc_lookup_only']['mean']:.3f})  [{time.time() - t0:.0f}s]",
                  flush=True)
    best_L = max(cfg.n_levels, key=lambda L: np.mean([results[(L, b)]["summary"]["accuracy"]["mean"]
                                                        for b in cfg.bands]))
    print(f"Best number of fuzzy levels: {best_L}", flush=True)

    # 5 ------------------------------------------------------------------
    ablation = {}
    if not a.no_ablation:
        all_win = np.ones_like(keep)
        for band in cfg.bands:
            r = run_band(F, all_win, band, cfg, C, best_L, rng)
            ablation[("no_gate", band)] = r["summary"]
        Fs = dict(F); Fs["label"] = _shuffle_within_subject(F["label"], F["trial"], F["subject"], rng)
        best_band = max(cfg.bands, key=lambda b: results[(best_L, b)]["summary"]["accuracy"]["mean"])
        ablation[("shuffled_labels", best_band)] = run_band(Fs, keep, best_band, cfg, C, best_L, rng)["summary"]
        print("Ablations done", flush=True)

    # 6 ------------------------------------------------------------------
    _write_outputs(a, cfg, meta, classes, F, keep, gate_name, ranking, gate_check,
                   results, best_L, ablation, fig_dir, time.time() - t0)
    print(f"Done in {time.time() - t0:.0f}s -> {a.out}", flush=True)


def extract_all_cached(trials, cfg, cache):
    from .features import extract_all
    return extract_all(trials, cfg, cache)


def _shuffle_within_subject(labels, trial, subject, rng):
    """Permute trial labels within each subject (keeps windows of a trial together)."""
    out = labels.copy()
    for s in np.unique(subject):
        tids = np.unique(trial[subject == s])
        tl = np.array([labels[trial == t][0] for t in tids])
        perm = rng.permutation(tl)
        for t, l in zip(tids, perm):
            out[trial == t] = l
    return out


def _gate_check(F, keep, data_root, cfg, fig_dir, gate_name, trials):
    """On the synthetic data we know where the emotional episodes are: measure how well the
    gate finds them and plot one example trial."""
    gt_path = os.path.join(data_root, "ground_truth_episodes.json")
    gt = json.load(open(gt_path)) if os.path.exists(gt_path) else None
    ti = 0
    rows = F["trial"] == ti
    vals = F["eye"][rows, F["eye_names"].index(gate_name)]
    times = (F["window"][rows] + 0.5) * cfg.window_s
    eps = gt[f"{trials[ti].subject}/{trials[ti].trial_id}"]["episodes"] if gt else None
    plots.gate_example(times, vals, keep[rows], eps, os.path.join(fig_dir, "gate_example.png"), gate_name)
    if not gt:
        return None
    in_ep = np.zeros(len(keep), bool)
    for i in range(len(keep)):
        t = trials[F["trial"][i]]
        a0, b0 = F["window"][i] * cfg.window_s, (F["window"][i] + 1) * cfg.window_s
        for a, b in gt[f"{t.subject}/{t.trial_id}"]["episodes"]:
            if min(b, b0) - max(a, a0) >= cfg.window_s / 2:
                in_ep[i] = True
    return {"episode_rate_all_windows": float(in_ep.mean()),
            "episode_rate_kept_windows": float(in_ep[keep].mean()),
            "episode_recall": float(keep[in_ep].mean()) if in_ep.any() else None}


def _write_outputs(a, cfg, meta, classes, F, keep, gate_name, ranking, gate_check,
                   results, best_L, ablation, fig_dir, elapsed):
    bands = cfg.bands
    # summary.csv
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["levels", "band", "acc_mean", "acc_std", "f1_macro", "precision_macro",
                    "recall_macro(TPR)", "auc_macro", "acc_lookup_only", "n_samples"]
                   + [f"TPR_{c}" for c in classes] + [f"FNR_{c}" for c in classes]
                   + [f"AUC_{c}" for c in classes])
        for (L, b), r in results.items():
            s = r["summary"]
            w.writerow([L, b, f"{s['accuracy']['mean']:.4f}", f"{s['accuracy']['std']:.4f}",
                        f"{s['f1_macro']['mean']:.4f}", f"{s['precision_macro']['mean']:.4f}",
                        f"{s['recall_macro']['mean']:.4f}", f"{s['auc_macro']['mean']:.4f}",
                        f"{s['acc_lookup_only']['mean']:.4f}", s["n_samples"]]
                       + [f"{v:.4f}" for v in s["tpr_per_class"]]
                       + [f"{v:.4f}" for v in s["fnr_per_class"]]
                       + [f"{v:.4f}" for v in s["auc_per_class"]])

    # figures for the chosen number of levels
    cms, rocs = {}, {}
    for b in bands:
        r = results[(best_L, b)]
        cms[b] = np.sum([f["confusion"] for f in r["folds"]], axis=0)
        rocs[b] = []
        for c in range(len(classes)):
            fpr, tpr = roc_points(r["pooled_y"], r["pooled_proba"], c)
            rocs[b].append((fpr, tpr, r["summary"]["auc_per_class"][c]))
    tag = f"{meta.get('dataset', 'data')}, {cfg.protocol} protocol, {best_L} fuzzy levels"
    plots.confusion_grid(cms, classes, os.path.join(fig_dir, "confusion_matrices.png"),
                         f"Confusion matrices, all folds pooled ({tag})")
    plots.roc_grid(rocs, classes, os.path.join(fig_dir, "roc_curves.png"), f"ROC curves ({tag})")
    best_band = max(bands, key=lambda b: results[(best_L, b)]["summary"]["accuracy"]["mean"])
    rb = results[(best_L, best_band)]
    bf = rb["summary"]["best_fold"]
    model = rb["models"][bf]
    plots.ga_curve(model["history"], os.path.join(fig_dir, f"ga_curve_{best_band}.png"),
                   f"GA tuning of fuzzy rules, {best_band} band (fold {bf + 1})")
    plots.membership_functions(model["fis"], model["Xtr"], os.path.join(fig_dir, f"membership_{best_band}.png"))
    if ablation:
        rows = []
        for b in bands:
            d = {"eye-gated": results[(best_L, b)]["summary"]["accuracy"]["mean"]}
            if ("no_gate", b) in ablation:
                d["no gate"] = ablation[("no_gate", b)]["accuracy"]["mean"]
            rows.append((b, d))
        plots.accuracy_bars(rows, os.path.join(fig_dir, "gate_ablation.png"),
                            "Effect of the eye gate", chance=1 / len(classes))

    # rules (top-10 per band and best rule per class) from each band's best fold
    rule_tables = {}
    for b in bands:
        r = results[(best_L, b)]
        m = r["models"][r["summary"]["best_fold"]]
        fis = m["fis"]
        cov, acc = fis.rule_stats(m["Xtr"], m["ytr"])
        importance = cov * acc * fis.weight
        order = np.argsort(-importance)
        rule_tables[b] = [{"rule": fis.rule_text(i, classes), "class": classes[fis.cons[i]],
                           "coverage": float(cov[i]), "accuracy": float(acc[i]),
                           "weight": float(fis.weight[i])} for i in order]

    out = {
        "dataset": meta, "config": cfg.to_dict(), "elapsed_s": elapsed,
        "n_windows_total": int(len(keep)), "n_windows_gated": int(keep.sum()),
        "gate_feature": gate_name, "eye_mrmr_ranking": ranking, "gate_check": gate_check,
        "best_levels": best_L, "best_band": best_band,
        "results": {f"levels{L}_{b}": _jsonable(r) for (L, b), r in results.items()},
        "ablation": {f"{k}_{b}": _jsonable(v) for (k, b), v in ablation.items()},
        "rules": rule_tables,
    }
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(_jsonable(out), f, indent=1)
    _write_report(a, cfg, meta, classes, out, results, best_L, ablation, rule_tables)


def _pm(s):
    return f"{100 * s['mean']:.2f} ± {100 * s['std']:.2f}"


def _write_report(a, cfg, meta, classes, out, results, best_L, ablation, rule_tables):
    L = []
    L.append(f"# Results: eye-gated S-FIS-GA emotion recognition\n")
    if "SYNTHETIC" in meta.get("note", "").upper():
        L.append("> **Synthetic demo data.** These numbers only show that the pipeline runs and that each "
                 "stage does what it should. They are not emotion-recognition results and must not be "
                 "compared with the paper.\n")
    L.append(f"- Dataset: `{a.data}` ({meta.get('dataset')}), classes: {', '.join(classes)}")
    L.append(f"- Windows: {out['n_windows_total']} total, {out['n_windows_gated']} kept by the eye gate "
             f"(`{out['gate_feature']}`, mode `{cfg.gate_mode}`)")
    L.append(f"- Protocol: `{cfg.protocol}`, {cfg.n_folds} folds; features: "
             f"{'mRMR on each training fold' if cfg.eeg_features == 'auto' else cfg.eeg_features}")
    L.append(f"- Runtime: {out['elapsed_s']:.0f} s\n")

    L.append("## Eye-feature ranking (mRMR)\n")
    L.append("| rank | feature | score |\n|---|---|---|")
    for i, (n, s) in enumerate(out["eye_mrmr_ranking"], 1):
        L.append(f"| {i} | {n} | {s:.4f} |")
    if out["gate_check"]:
        g = out["gate_check"]
        L.append(f"\nGate check against the known synthetic episodes: {100 * g['episode_rate_all_windows']:.1f}% "
                 f"of all windows are inside an episode vs {100 * g['episode_rate_kept_windows']:.1f}% of the "
                 f"windows the gate keeps (episode recall {100 * g['episode_recall']:.1f}%).")

    L.append("\n## Accuracy by band and number of fuzzy levels (mean ± std over folds, %)\n")
    L.append("| band | " + " | ".join(f"{k} levels" for k in cfg.n_levels) + " |")
    L.append("|---|" + "---|" * len(cfg.n_levels))
    for b in cfg.bands:
        L.append(f"| {b} | " + " | ".join(_pm(results[(k, b)]["summary"]["accuracy"]) for k in cfg.n_levels) + " |")
    L.append(f"\nBest number of levels: **{best_L}** (the paper also found 3 best).\n")

    L.append(f"## Full metrics, {best_L} levels\n")
    L.append("| band | accuracy | macro F1 | macro precision | macro recall (TPR) | macro AUC | lookup table only (no GA) |")
    L.append("|---|---|---|---|---|---|---|")
    for b in cfg.bands:
        s = results[(best_L, b)]["summary"]
        L.append(f"| {b} | {_pm(s['accuracy'])} | {_pm(s['f1_macro'])} | {_pm(s['precision_macro'])} | "
                 f"{_pm(s['recall_macro'])} | {s['auc_macro']['mean']:.3f} | {_pm(s['acc_lookup_only'])} |")
    L.append("\nPer-class (mean over folds):\n")
    L.append("| band | metric | " + " | ".join(classes) + " |")
    L.append("|---|---|" + "---|" * len(classes))
    for b in cfg.bands:
        s = results[(best_L, b)]["summary"]
        for k, lab in [("tpr_per_class", "TPR"), ("fnr_per_class", "FNR"), ("f1_per_class", "F1"),
                       ("auc_per_class", "AUC")]:
            L.append(f"| {b} | {lab} | " + " | ".join(f"{v:.3f}" for v in s[k]) + " |")

    if ablation:
        L.append("\n## Ablations\n")
        L.append("| band | eye-gated acc | no-gate acc |\n|---|---|---|")
        for b in cfg.bands:
            if ("no_gate", b) in ablation:
                L.append(f"| {b} | {_pm(results[(best_L, b)]['summary']['accuracy'])} | "
                         f"{_pm(ablation[('no_gate', b)]['accuracy'])} |")
        for (k, b), s in ablation.items():
            if k == "shuffled_labels":
                L.append(f"\nShuffled-label control ({b} band): accuracy {_pm(s['accuracy'])} "
                         f"(should be near chance, {100 / len(classes):.0f}%).")

    L.append(f"\n## Mined fuzzy rules ({best_L} levels, best fold of each band, ranked by coverage × accuracy × weight)\n")
    for b in cfg.bands:
        L.append(f"### {b}\n")
        L.append("| # | rule | coverage | accuracy |\n|---|---|---|---|")
        for i, r in enumerate(rule_tables[b][:cfg.n_top_rules], 1):
            L.append(f"| {i} | {r['rule']} | {100 * r['coverage']:.0f}% | {100 * r['accuracy']:.1f}% |")
        L.append("")
    L.append("### Best rule per class (all bands)\n")
    L.append("| class | band | rule | coverage | accuracy |\n|---|---|---|---|---|")
    for c in classes:
        cands = [(b, r) for b in cfg.bands for r in rule_tables[b] if r["class"] == c]
        if cands:
            b, r = max(cands, key=lambda br: br[1]["coverage"] * br[1]["accuracy"] * br[1]["weight"])
            L.append(f"| {c} | {b} | {r['rule']} | {100 * r['coverage']:.0f}% | {100 * r['accuracy']:.1f}% |")
        else:
            L.append(f"| {c} | - | (no rule) | - | - |")

    L.append("\n## Figures\n")
    for fn in sorted(os.listdir(os.path.join(a.out, "figures"))):
        L.append(f"![{fn}](figures/{fn})")
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
