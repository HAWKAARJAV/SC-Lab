"""Figures + numbers for the ZuCo report (M0, M1, M2, M1+temporal).

    cd "<repo>" && python3 manuscript/zuco_report/make_figures.py

Reads the raw predictions of the three ZuCo runs, data/zuco_npz and the feature cache; writes
manuscript/zuco_report/figures/*.pdf|png and manuscript/zuco_report/stats.json.
"""
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from gfer.compare import _window_weights, dataset_config, load_features   # noqa: E402
from gfer.evaluate import zscore_by_subject                                # noqa: E402

OUT = os.path.join(ROOT, "manuscript", "zuco_report")
FIG = os.path.join(OUT, "figures")
DATA = "data/zuco_npz"
RAWS = ["results/zuco_final_m0m1/raw.json",            # none, hard, fuzzy, fuzzy_trialnorm
        "results/zuco_m2/m2_only/raw.json",            # adaptive, adaptive_nosmooth, adaptive_causal
        "results/zuco_combo/combo_only/raw.json"]      # fuzzy_smooth
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
ALL = ["none", "hard", "fuzzy", "fuzzy_trialnorm", "adaptive", "adaptive_nosmooth",
       "adaptive_causal", "fuzzy_smooth"]
MAIN = ["none", "hard", "fuzzy", "adaptive", "fuzzy_smooth"]
PROPOSED = ["fuzzy", "adaptive", "adaptive_nosmooth", "adaptive_causal", "fuzzy_smooth", "fuzzy_trialnorm"]
LAB = {"none": "No gate", "hard": "M0 hard", "fuzzy": "M1 fuzzy", "fuzzy_trialnorm": "M1 v1",
       "adaptive": "M2 adaptive temporal", "adaptive_nosmooth": "M2 adaptive only",
       "adaptive_causal": "M2 causal", "fuzzy_smooth": "M1 + temporal"}
COL = {"none": "#9e9e9e", "hard": "#3b6ea8", "fuzzy": "#d9772b", "fuzzy_trialnorm": "#f0c08f",
       "adaptive": "#2a9d6f", "adaptive_nosmooth": "#8fd1b5", "adaptive_causal": "#5c7d6f",
       "fuzzy_smooth": "#a04f8f"}
PROT = {"trial": "5-fold (sentence-disjoint)", "loso": "Leave-one-subject-out"}


def preds(r):
    p = np.array([q for f in r["folds"] for q in f["trial_preds"]])
    return p[np.argsort(p[:, 0])]


def logreg_ceiling(F):
    """Strong non-fuzzy reference: L2 logistic regression on sentence-mean features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    E = F["eeg"].reshape(len(F["label"]), -1)
    E = zscore_by_subject(E[:, E.std(0) > 1e-9], F["subject"])
    Y = zscore_by_subject(np.nan_to_num(F["eye"]), F["subject"])
    t = F["trial"]; ids = np.unique(t)
    agg = lambda A: np.nan_to_num(np.array([np.nanmean(A[t == i], 0) for i in ids]))
    Xe, Xy = agg(E), agg(Y)
    y = np.array([F["label"][t == i][0] for i in ids]); g = np.array([F["subject"][t == i][0] for i in ids])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced"))
    out = {}
    for name, X in [("eeg", Xe), ("eye", Xy), ("eeg_eye", np.c_[Xe, Xy])]:
        for prot, cv, kw in [("trial", StratifiedKFold(5, shuffle=True, random_state=0), {}),
                             ("loso", LeaveOneGroupOut(), {"groups": g})]:
            p = cross_val_predict(clf, X, y, cv=cv, **kw)
            out[f"{name}|{prot}"] = {"acc": float(np.mean(p == y)), "bal_acc": float(balanced_accuracy_score(y, p))}
    out["n_eeg_features"] = int(Xe.shape[1])
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import wilcoxon
    from sklearn.metrics import balanced_accuracy_score, f1_score
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    os.makedirs(FIG, exist_ok=True)
    R = [r for p in RAWS for r in json.load(open(p))]
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    idx = {(r["protocol"], r["variant"], r["band"], r["seed"]): r for r in R}
    seeds = sorted({r["seed"] for r in R})
    S = {"variants": {v: LAB[v] for v in ALL}}

    # ---------------- dataset statistics
    subj_of, lab, dur = [], [], []
    for s in sorted(d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d))):
        for fn in sorted(f for f in os.listdir(os.path.join(DATA, s)) if f.endswith(".npz")):
            z = np.load(os.path.join(DATA, s, fn))
            subj_of.append(s); lab.append(int(z["label"])); dur.append(z["eeg"].shape[1] / 128.0)
    subj_of, lab, dur = np.array(subj_of), np.array(lab), np.array(dur)
    subs = sorted(set(subj_of))
    S["n_trials"] = int(len(lab)); S["n_subjects"] = len(subs)
    S["class_counts"] = dict(zip(meta["class_names"], np.bincount(lab).tolist()))
    S["trials_per_subject"] = {s: int((subj_of == s).sum()) for s in subs}
    S["duration_s"] = {"min": float(dur.min()), "median": float(np.median(dur)), "max": float(dur.max())}
    F = load_features(DATA, dataset_config(DATA))
    nwin = np.bincount(F["trial"])
    S["n_windows"] = int(len(F["label"]))
    S["windows_per_trial"] = {"min": int(nwin.min()), "median": float(np.median(nwin)), "max": int(nwin.max())}
    S["gate_feature_counts"] = dict(Counter(f["gate_feature"] for r in R for f in r["folds"]))

    # ---------------- gate behaviour
    g = Counter(f["gate_feature"] for r in R for f in r["folds"]).most_common(1)[0][0]
    v = F["eye"][:, F["eye_names"].index(g)].astype(float)
    W = {k: _window_weights(v, F["trial"], F["subject"], k, F["window"]) for k in ALL if k != "none"}
    S["gate"] = {"feature": g}
    for k, w in W.items():
        S["gate"][k] = {"mean": float(w.mean()),
                        "extreme_frac": float(np.mean((w <= 0.1001) | (w >= 0.999))),
                        "corr_M0": float(np.corrcoef(w, W["hard"])[0, 1]),
                        "corr_M1": float(np.corrcoef(w, W["fuzzy"])[0, 1])}
    panels = ["hard", "fuzzy_trialnorm", "fuzzy", "adaptive", "fuzzy_smooth"]
    fig, axs = plt.subplots(1, len(panels), figsize=(7.0, 1.75), sharey=True)
    for ax, k in zip(axs, panels):
        w = W[k]
        ax.hist(w, bins=np.linspace(0, 1, 21), color=COL[k], edgecolor="white", weights=np.ones(len(w)) / len(w))
        ax.set_title(LAB[k], fontsize=7.5); ax.set_xlabel("weight", fontsize=7)
    axs[0].set_ylabel("share of windows")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "gate_weights.pdf")); fig.savefig(os.path.join(FIG, "gate_weights.png"), dpi=180)
    plt.close(fig)

    # ---------------- per-cell results
    def acc(p, v_, b, s):
        q = preds(idx[(p, v_, b, s)]); return float(np.mean(q[:, 1] == q[:, 2]))
    res = {}
    for p in PROT:
        for v_ in ALL:
            for b in BANDS:
                a, ba, f1 = [], [], []
                for s in seeds:
                    q = preds(idx[(p, v_, b, s)])
                    a.append(np.mean(q[:, 1] == q[:, 2])); ba.append(balanced_accuracy_score(q[:, 1], q[:, 2]))
                    f1.append(f1_score(q[:, 1], q[:, 2], average="macro"))
                res[f"{p}|{v_}|{b}"] = {"acc": float(np.mean(a)), "acc_sd": float(np.std(a)),
                                       "bal_acc": float(np.mean(ba)), "f1": float(np.mean(f1))}
            res[f"{p}|{v_}|mean"] = {"acc": float(np.mean([res[f"{p}|{v_}|{b}"]["acc"] for b in BANDS]))}
    S["results"] = res

    # ---------------- proposed - M0
    agg = {}
    for p in PROT:
        for v_ in PROPOSED:
            d = np.array([[acc(p, v_, b, s) - acc(p, "hard", b, s) for s in seeds] for b in BANDS])
            e = {"mean_pts": float(100 * d.mean()), "wins": int((d > 0).sum()), "n": int(d.size),
                 "min": float(100 * d.min()), "max": float(100 * d.max())}
            if p == "loso":       # per-subject effect, averaged over bands and seeds; Wilcoxon over subjects
                ps = {}
                for s_ in subs:
                    dd = []
                    for b in BANDS:
                        for s in seeds:
                            q0, q1 = preds(idx[(p, "hard", b, s)]), preds(idx[(p, v_, b, s)])
                            m = subj_of[q0[:, 0]] == s_
                            dd.append(100 * ((q1[m, 1] == q1[m, 2]).mean() - (q0[m, 1] == q0[m, 2]).mean()))
                    ps[s_] = float(np.mean(dd))
                e["per_subject"] = ps
                e["subjects_helped"] = int(sum(x > 0 for x in ps.values()))
                e["wilcoxon_p_subjects"] = float(wilcoxon(list(ps.values())).pvalue)
            agg[f"{p}|{v_}"] = e
    S["vs_M0"] = agg

    # ---------------- main bar figure
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    xw = 0.16
    for ax, p in zip(axs, PROT):
        for j, v_ in enumerate(MAIN):
            m = [100 * res[f"{p}|{v_}|{b}"]["acc"] for b in BANDS]
            sd = [100 * res[f"{p}|{v_}|{b}"]["acc_sd"] for b in BANDS]
            ax.bar(np.arange(5) + (j - 2) * xw, m, xw, yerr=sd, color=COL[v_], label=LAB[v_],
                   capsize=1.2, error_kw={"lw": 0.5})
        ax.axhline(100 / 3, color="k", ls=":", lw=0.9)
        ax.set_xticks(range(5)); ax.set_xticklabels(BANDS); ax.set_ylim(28, 37); ax.set_title(PROT[p])
    axs[0].set_ylabel("sentence-level accuracy (%)")
    axs[0].text(-0.45, 33.55, "chance", fontsize=6.5, ha="left")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(FIG, "results_bars.pdf")); fig.savefig(os.path.join(FIG, "results_bars.png"), dpi=180)
    plt.close(fig)

    # ---------------- proposed - M0 figure (per band x seed, per protocol)
    show = ["fuzzy", "adaptive", "fuzzy_smooth"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, p in zip(axs, PROT):
        for j, v_ in enumerate(show):
            for bi, b in enumerate(BANDS):
                d = [100 * (acc(p, v_, b, s) - acc(p, "hard", b, s)) for s in seeds]
                ax.scatter(np.full(len(d), bi + (j - 1) * 0.22), d, s=11, color=COL[v_],
                           label=LAB[v_] if bi == 0 else None)
        ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(5)); ax.set_xticklabels(BANDS)
        ax.set_title(PROT[p])
    axs[0].set_ylabel("accuracy minus M0 (pts)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(FIG, "minus_m0.pdf")); fig.savefig(os.path.join(FIG, "minus_m0.png"), dpi=180)
    plt.close(fig)

    # ---------------- prediction shares + rule-base size
    sh = {}
    for p in PROT:
        for v_ in ALL:
            q = np.concatenate([preds(idx[(p, v_, b, s)])[:, 2] for b in BANDS for s in seeds])
            nr = np.mean([f["n_rules"] for b in BANDS for s in seeds for f in idx[(p, v_, b, s)]["folds"]])
            sh[f"{p}|{v_}"] = {"pred_share": (np.bincount(q, minlength=3) / len(q)).round(3).tolist(),
                               "mean_rules": float(nr)}
    S["prediction_shares"] = sh
    S["logreg_ceiling"] = logreg_ceiling(F)
    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(S, f, indent=1)
    print("ok")


if __name__ == "__main__":
    main()
