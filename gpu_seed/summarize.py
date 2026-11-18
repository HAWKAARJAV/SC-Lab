"""Summarise a gpu_seed run:  python -m gpu_seed.summarize gpu_seed/results/full"""
import csv
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest, wilcoxon

NAMES = {"none": "No gate", "hard": "M0 hard", "fuzzy": "M1 fuzzy", "adaptive": "M2 adaptive temporal",
         "adaptive_nosmooth": "M2 adaptive only", "adaptive_causal": "M2 causal", "fuzzy_smooth": "M1+temporal",
         "ref_logreg": "Ref: logistic regr."}
BANDS = ["delta", "theta", "alpha", "beta", "gamma", "all"]
PROT = {"within": "per-person (subject-dependent), 5-fold over each subject's trials",
        "trial": "5-fold over trials", "loso": "leave-one-subject-out",
        "paper": "paper protocol: 5-fold over pooled 4-s windows (optimistic)"}


def _acc(rec):
    if rec["protocol"] == "paper":
        return float(np.mean([f["window_acc"] for f in rec["folds"]]))
    p = np.array([q for f in rec["folds"] for q in f["trial_preds"]])
    return float(np.mean(p[:, 1] == p[:, 2]))


def _preds(rec):
    p = np.array([q for f in rec["folds"] for q in f["trial_preds"]])
    return p[np.argsort(p[:, 0])]


def summarize(out):
    R = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, "tasks", "*.json")))]
    if not R:
        print("no finished tasks yet"); return
    meta = json.load(open(os.path.join(out, "trials.json")))
    subj = np.array(meta["subject"])
    C = len(meta["class_names"])
    idx = {(r["protocol"], r["variant"], r["band"], r["seed"]): r for r in R}
    prots = [p for p in PROT if any(r["protocol"] == p for r in R)]
    variants = [v for v in NAMES if any(r["variant"] == v for r in R)]
    bands = [b for b in BANDS if any(r["band"] == b for r in R)]
    seeds = sorted({r["seed"] for r in R})
    n_trials = len(meta["trial_ids"])
    L = [f"# SEED-IV: M0 vs M1 vs M2 (GPU run `{os.path.basename(out)}`, preset "
         f"`{meta.get('config', {}).get('preset', 'paper')}`)\n",
         f"- {n_trials} trials, {len(set(subj))} subjects, classes {', '.join(meta['class_names'])} "
         f"(chance {100 / C:.1f}%); device {meta.get('device')}",
         f"- finished tasks: {len(R)}; seeds {seeds}\n"]
    rows = []
    for p in prots:
        L.append(f"## {PROT[p]}\n")
        L.append("Accuracy %, mean ± SD over seeds. `*` = above chance (binomial p < 0.05, trial protocols).\n")
        L.append("| band | " + " | ".join(NAMES[v] for v in variants) + " |")
        L.append("|---|" + "---|" * len(variants))
        cell = {}
        for b in bands:
            line = f"| {b} |"
            for v in variants:
                a = [_acc(idx[(p, v, b, s)]) for s in seeds if (p, v, b, s) in idx]
                if not a:
                    line += " – |"; continue
                cell[(v, b)] = a
                star = ""
                if p != "paper":
                    k = int(round(np.mean(a) * n_trials))
                    star = "*" if binomtest(k, n_trials, 1 / C, alternative="greater").pvalue < 0.05 else ""
                line += f" {100 * np.mean(a):.1f} ± {100 * np.std(a):.1f}{star} |"
                rows.append([p, b, v, f"{np.mean(a):.4f}", f"{np.std(a):.4f}", len(a)])
            L.append(line)
        means = {v: np.mean([np.mean(cell[(v, b)]) for b in bands if (v, b) in cell]) for v in variants
                 if any((v, b) in cell for b in bands)}
        L.append("| **mean** |" + "".join(f" **{100 * means[v]:.1f}** |" if v in means else " – |" for v in variants))
        if "hard" in variants:
            L.append(f"\n**Each gate minus M0** ({p}; over band × seed runs)\n")
            L.append("| model | mean Δ (pts) | runs ahead | McNemar p (pooled, seed 0) | Wilcoxon p (subjects) |")
            L.append("|---|---|---|---|---|")
            for v in [v for v in variants if v not in ("none", "hard", "ref_logreg")]:
                d = [_acc(idx[(p, v, b, s)]) - _acc(idx[(p, "hard", b, s)])
                     for b in bands for s in seeds if (p, v, b, s) in idx and (p, "hard", b, s) in idx]
                if not d:
                    continue
                mc, wx = "–", "–"
                if p != "paper":
                    n01 = n10 = 0
                    ds = {}
                    for b in bands:
                        k0, k1 = (p, "hard", b, seeds[0]), (p, v, b, seeds[0])
                        if k0 in idx and k1 in idx:
                            q0, q1 = _preds(idx[k0]), _preds(idx[k1])
                            c0, c1 = q0[:, 1] == q0[:, 2], q1[:, 1] == q1[:, 2]
                            n01 += int((c0 & ~c1).sum()); n10 += int((~c0 & c1).sum())
                            for s_ in np.unique(subj):
                                m = subj[q0[:, 0]] == s_
                                ds.setdefault(s_, []).append(c1[m].mean() - c0[m].mean())
                    mc = f"{binomtest(n10, n01 + n10, 0.5).pvalue:.3f} ({n10} vs {n01})" if n01 + n10 else "1"
                    dd = [np.mean(x) for x in ds.values()]
                    if p == "loso" and len(dd) > 1 and np.any(np.array(dd) != 0):
                        wx = f"{wilcoxon(dd).pvalue:.3f} ({sum(x > 0 for x in dd)}/{len(dd)} helped)"
                L.append(f"| {NAMES[v]} | {100 * np.mean(d):+.2f} | {sum(x > 0 for x in d)}/{len(d)} | {mc} | {wx} |")
        L.append("")
    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["protocol", "band", "variant", "acc_mean", "acc_sd_over_seeds", "n_seeds"])
        w.writerows(rows)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        col = {"none": "#9e9e9e", "hard": "#3b6ea8", "fuzzy": "#d9772b", "adaptive": "#2a9d6f",
               "adaptive_nosmooth": "#8fd1b5", "adaptive_causal": "#5c7d6f", "fuzzy_smooth": "#a04f8f",
               "ref_logreg": "#444444"}
        fig, axs = plt.subplots(1, len(prots), figsize=(4.2 * len(prots), 3.2), squeeze=False)
        for ax, p in zip(axs[0], prots):
            xw = 0.8 / len(variants)
            for j, v in enumerate(variants):
                m = [100 * float(r[3]) for r in rows if r[0] == p and r[2] == v]
                bb = [r[1] for r in rows if r[0] == p and r[2] == v]
                ax.bar(np.array([bands.index(x) for x in bb]) + (j - (len(variants) - 1) / 2) * xw, m, xw,
                       color=col[v], label=NAMES[v])
            ax.axhline(100 / C, color="k", ls=":", lw=1)
            ax.set_xticks(range(len(bands))); ax.set_xticklabels(bands); ax.set_title(p)
            ax.set_ylabel("accuracy (%)")
        axs[0][0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out, "accuracy.png"), dpi=150)
    except Exception as e:                                       # figure is optional
        print("figure skipped:", e)
    try:
        print("\n".join(L))
    except UnicodeEncodeError:                             # old Windows consoles
        print("\n".join(L).encode("ascii", "replace").decode())


if __name__ == "__main__":
    summarize(sys.argv[1])
