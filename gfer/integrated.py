"""Integrated M4: Adaptive Temporal Fuzzy Eye Gate + multi-band probability fusion.

    python -m gfer.integrated --data data/demo --experiment five_learned --gate adaptive

Uses ``adaptive_eye.compute_gate_weights`` and ``fusion.run_fusion_fold`` (not illustrative logic).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .adaptive_eye import compute_gate_weights
from .compare import load_features, trial_folds
from .config import Config
from .fusion import FUSION_EXPERIMENTS, run_fusion_fold, run_single_band_fold


def parse_args():
    ap = argparse.ArgumentParser(description="M4 integrated adaptive gate + band fusion")
    ap.add_argument("--data", default="data/demo")
    ap.add_argument("--out", default="results/m4_integrated")
    ap.add_argument("--protocol", default="trial", choices=["trial", "loso"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", default="adaptive",
                    choices=["adaptive", "adaptive_nosmooth", "adaptive_causal", "fuzzy", "hard"])
    ap.add_argument("--experiment", default="five_learned",
                    choices=list(FUSION_EXPERIMENTS.keys()))
    ap.add_argument("--gate-feature", default="pupil_y_mean")
    return ap.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = Config(seed=a.seed, protocol=a.protocol)
    t0 = time.time()
    F = load_features(a.data, cfg)
    C = len(F["meta"]["class_names"])
    rng = np.random.default_rng(a.seed)
    y, trial = F["label"], F["trial"]
    g = a.gate_feature
    v = F["eye"][:, F["eye_names"].index(g)].astype(float)
    bands = FUSION_EXPERIMENTS[a.experiment]
    fusion_mode = "learned" if a.experiment.endswith("_learned") or "learned" in a.experiment else "equal"

    fold_rows = []
    for fi, (tr_t, te_t) in enumerate(trial_folds(F, a.protocol, a.seed)):
        tr_win = np.isin(trial, tr_t)
        te_win = np.isin(trial, te_t)
        w_all = compute_gate_weights(v, F["subject"], trial, F["window"], tr_win, a.gate)
        if len(bands) == 1:
            out = run_single_band_fold(F, bands[0], tr_win, te_win, w_all, cfg, C, rng)
        else:
            out = run_fusion_fold(F, bands, fusion_mode, tr_win, te_win, w_all, cfg, C, rng)
        acc = np.mean([p == t for _, p, t in out["trial_preds"]])
        row = {"fold": fi, "accuracy": float(acc), "n_test_trials": len(out["trial_preds"])}
        if "fusion_weights" in out:
            row["fusion_weights"] = out["fusion_weights"]
        fold_rows.append(row)

    summary = {
        "model": "M4",
        "gate": a.gate,
        "experiment": a.experiment,
        "bands": list(bands),
        "fusion_mode": fusion_mode,
        "protocol": a.protocol,
        "gate_feature": g,
        "accuracy_mean": float(np.mean([r["accuracy"] for r in fold_rows])),
        "accuracy_std": float(np.std([r["accuracy"] for r in fold_rows])),
        "elapsed_s": time.time() - t0,
        "folds": fold_rows,
    }
    with open(os.path.join(a.out, "m4_results.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"M4 {a.experiment} gate={a.gate}  acc={summary['accuracy_mean']:.3f}"
          f"±{summary['accuracy_std']:.3f}  -> {a.out}/m4_results.json")


if __name__ == "__main__":
    main()
