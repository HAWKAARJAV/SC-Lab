"""Controlled comparison of eye-tracking gates on SYNTHETIC data with known episodes.

    python -m gfer.compare --out results/sim_study --datasets data/demo,data/demo_easy

Variants (only the gate differs; everything else is shared):
    none   : w = 1 for every window                       (A1, no-gate reference)
    hard   : w = 1[z~ >= within-trial 60th percentile]     (M0, baseline hard gate)
    fuzzy  : zero-order Sugeno gate, Low/Medium/High        (M1, proposed)
    linear : w = max(r_L, z~)                               (A4, non-fuzzy soft gate)

z~ is the gate feature normalised inside each trial with robust (P5, P95) scaling (label-free).
Window weights enter (i) weighted mRMR, (ii) weighted entropy discretisation, (iii) weighted
Wang-Mendel rule degrees, (iv) weighted GA fitness, (v) weighted trial-level aggregation.
With binary weights every step reduces to the computation on retained windows only.
Evaluation is trial-level (one prediction per 60-s trial) under trial-disjoint stratified
5-fold CV and leave-one-subject-out CV; all label-dependent steps are fitted on training folds.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from .adaptive_eye import GATE_VARIANTS as ADAPTIVE_GATE_VARIANTS, compute_gate_weights
from .config import Config
from .data import load_npz
from .evaluate import band_matrix, zscore_by_subject
from .features import extract_all
from .fuzzy import SugenoFIS
from .ga import tune_rules
from .gates import FUZZY, HARD_Q, fuzzy_weights, gate_weights, normalise_within_trial
from .mrmr import mrmr, mrmr_weighted

VARIANTS = ["none", "hard", "fuzzy", "linear", "adaptive", "adaptive_nosmooth", "adaptive_causal"]
# second round (parameters fixed before any result was seen; appended so seeds of the above are unchanged)
EXTRA_VARIANTS = ["fuzzy_sharp", "fuzzy_tuned", "hard_tuned"]
ALL_VARIANTS = VARIANTS + EXTRA_VARIANTS
FUZZY_SHARP = dict(a1=0.4, c=0.6, b2=0.8, r=(0.0, 0.3, 1.0))
FUZZY_GRID = [FUZZY, FUZZY_SHARP,
              dict(a1=0.3, c=0.6, b2=0.9, r=(0.0, 0.2, 1.0)),
              dict(a1=0.5, c=0.7, b2=0.9, r=(0.0, 0.2, 1.0))]
HARD_GRID = [0.4, 0.5, 0.6, 0.7]

_CACHE = {}


# ------------------------------------------------------------------ data
def load_features(root, cfg):
    if root not in _CACHE:
        trials, meta = load_npz(root)
        F = extract_all(trials, cfg, "cache")
        gt = json.load(open(os.path.join(root, "ground_truth_episodes.json")))
        in_ep = np.zeros(len(F["label"]), bool)
        for i in range(len(in_ep)):
            t = trials[F["trial"][i]]
            a0, b0 = F["window"][i] * cfg.window_s, (F["window"][i] + 1) * cfg.window_s
            for a, b in gt[f"{t.subject}/{t.trial_id}"]["episodes"]:
                if min(b, b0) - max(a, a0) >= cfg.window_s / 2:
                    in_ep[i] = True
        F["in_episode"] = in_ep
        F["meta"] = meta
        _CACHE[root] = F
    return _CACHE[root]


# ------------------------------------------------------------------ gates (see gates.py; M2 via compute_gate_weights)
def _window_weights(v, trial, window, subjects, train_win, variant):
    if variant in ADAPTIVE_GATE_VARIANTS and variant not in ("none", "hard", "fuzzy", "linear"):
        return compute_gate_weights(v, subjects, trial, window, train_win, variant)
    z = normalise_within_trial(v, trial)
    if variant == "fuzzy_sharp":
        return fuzzy_weights(z, FUZZY_SHARP)
    if variant == "hard_tuned":
        raise RuntimeError("hard_tuned handled in run_task")
    return gate_weights(z, trial, variant)


# ------------------------------------------------------------------ folds (trial level)
def trial_folds(F, protocol, seed, n_folds=5):
    tids = np.unique(F["trial"])
    tlab = np.array([F["label"][F["trial"] == t][0] for t in tids])
    tsub = np.array([F["subject"][F["trial"] == t][0] for t in tids])
    if protocol == "trial":
        sp = StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(tids, tlab)
    elif protocol == "loso":
        sp = LeaveOneGroupOut().split(tids, tlab, tsub)
    else:
        raise ValueError(protocol)
    return [(tids[a], tids[b]) for a, b in sp]


# ------------------------------------------------------------------ one fit
def fit_predict(X, y, trial, tr, te, w, cfg, C, names, rng):
    """Weighted mRMR -> weighted S-FIS -> weighted GA on `tr`; weighted trial vote on `te`."""
    wtr = w[tr]
    sel = mrmr_weighted(X[tr], y[tr], wtr, k=cfg.n_selected, n_bins=cfg.mrmr_bins)
    Xtr, Xte = X[tr][:, sel], X[te][:, sel]
    fis = SugenoFIS.fit_mfs(Xtr, y[tr], 3, C, [names[i] for i in sel], sw=wtr)
    fis.wang_mendel(Xtr, y[tr], sw=wtr)
    fis, hist = tune_rules(fis, Xtr, y[tr], cfg, rng, sw=wtr)
    P = fis.predict_proba(Xte)
    wte, tte, yte = w[te], trial[te], y[te]
    preds = []
    for t in np.unique(tte):
        m = tte == t
        ww = wte[m]
        S = (ww[:, None] * P[m]).sum(0) / max(ww.sum(), 1e-12)
        preds.append((int(t), int(yte[m][0]), int(S.argmax())))
    info = {"features": [names[i] for i in sel], "n_rules": int(len(fis.ant)), "ga_gen": len(hist),
            "train_eff_windows": float(wtr.sum()), "train_windows": int(tr.sum()),
            "window_acc_unweighted": float((P.argmax(1) == yte).mean())}
    return preds, info


# ------------------------------------------------------------------ one task
def run_task(task):
    root, protocol, seed, variant, band, gate_feat = task[:6]
    noise = task[6] if len(task) > 6 else 0.0
    cfg = Config(seed=seed)
    F = load_features(root, cfg)
    C = len(F["meta"]["class_names"])
    rng = np.random.default_rng(seed * 1000 + ALL_VARIANTS.index(variant) * 10 + cfg.bands.index(band))
    X, names = band_matrix(F, band)
    X = np.nan_to_num(zscore_by_subject(X, F["subject"]))
    y, trial = F["label"], F["trial"]
    folds_out, t0 = [], time.time()
    for fi, (tr_t, te_t) in enumerate(trial_folds(F, protocol, seed)):
        tr, te = np.isin(trial, tr_t), np.isin(trial, te_t)
        # gate feature: mRMR over eye features on TRAINING windows only (or fixed)
        if gate_feat == "auto":
            eye = zscore_by_subject(F["eye"], F["subject"])
            g = F["eye_names"][int(mrmr(eye[tr], y[tr], k=1)[0][0])]
        else:
            g = gate_feat
        v = F["eye"][:, F["eye_names"].index(g)].astype(float)
        if noise > 0:   # gate-signal corruption: Gaussian noise, SD = noise x within-subject SD
            nr = np.random.default_rng(10_000 + seed)
            sd = np.array([np.nanstd(v[F["subject"] == s_]) for s_ in F["subject"]])
            v = v + nr.standard_normal(len(v)) * noise * sd
        chosen = None
        if variant in ("fuzzy_tuned", "hard_tuned"):
            z = normalise_within_trial(v, trial)
            # nested selection: inner stratified 3-fold CV over TRAINING trials only
            grid = FUZZY_GRID if variant == "fuzzy_tuned" else HARD_GRID
            make = (lambda P: fuzzy_weights(z, P)) if variant == "fuzzy_tuned" else (lambda q: hard_weights(z, trial, q))
            tl = np.array([y[trial == t][0] for t in tr_t])
            inner = list(StratifiedKFold(3, shuffle=True, random_state=seed + 100).split(tr_t, tl))
            scores = []
            for gi, P in enumerate(grid):
                wg = make(P); acc = []
                for ia, ib in inner:
                    itr, ite = np.isin(trial, tr_t[ia]), np.isin(trial, tr_t[ib])
                    pr, _ = fit_predict(X, y, trial, itr, ite, wg, cfg, C, names,
                                        np.random.default_rng(seed * 7 + gi))
                    acc.append(np.mean([a == b for _, a, b in pr]))
                scores.append(float(np.mean(acc)))
            chosen = int(np.argmax(scores))
            w = make(grid[chosen])
        else:
            w = _window_weights(v, trial, F["window"], F["subject"], tr, variant)
        preds, info = fit_predict(X, y, trial, tr, te, w, cfg, C, names, rng)
        info.update({"fold": fi, "gate_feature": g, "trial_preds": preds, "tuned_choice": chosen})
        folds_out.append(info)
    return {"root": root, "protocol": protocol, "seed": seed, "variant": variant, "band": band, "noise": noise,
            "gate_feature_mode": gate_feat, "elapsed_s": time.time() - t0, "folds": folds_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="data/demo,data/demo_easy")
    ap.add_argument("--protocols", default="trial,loso")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--gate-feature", default="auto")
    ap.add_argument("--out", default="results/sim_study")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--noise", default="0", help="comma list of gate-noise levels (x within-subject SD)")
    ap.add_argument("--variants", default=",".join(VARIANTS), help="comma list from " + ",".join(ALL_VARIANTS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = Config()
    for root in a.datasets.split(","):                 # warm the feature cache once
        load_features(root, cfg)
    tasks = [(r, p, int(s), v, b, a.gate_feature, float(nz))
             for nz in a.noise.split(",")
             for r in a.datasets.split(",") for p in a.protocols.split(",")
             for s in a.seeds.split(",") for v in a.variants.split(",") for b in cfg.bands]
    print(f"{len(tasks)} tasks on {a.jobs} processes", flush=True)
    out, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for i, r in enumerate(ex.map(run_task, tasks), 1):
            out.append(r)
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} [{time.time() - t0:.0f}s]", flush=True)
                with open(os.path.join(a.out, "raw.json"), "w") as f:
                    json.dump(out, f)
    with open(os.path.join(a.out, "raw.json"), "w") as f:
        json.dump(out, f)
    print(f"Done in {time.time() - t0:.0f}s -> {a.out}/raw.json", flush=True)


if __name__ == "__main__":
    main()
