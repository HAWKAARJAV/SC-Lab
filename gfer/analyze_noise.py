"""Gate-noise robustness analysis (SYNTHETIC data).

    python -m gfer.analyze_noise --out manuscript/generated
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from .analyze_sim import DNAME, VNAME, load
from .compare import VARIANTS, gate_weights, load_features, normalise_within_trial
from .config import Config


def gate_auc(root, noise, seeds=(0, 1, 2)):
    F = load_features(root, Config())
    v0 = F["eye"][:, F["eye_names"].index("pupil_y_mean")].astype(float)
    out = []
    for seed in seeds:
        v = v0.copy()
        if noise > 0:
            nr = np.random.default_rng(10_000 + seed)
            sd = np.array([np.nanstd(v0[F["subject"] == s]) for s in F["subject"]])
            v = v + nr.standard_normal(len(v)) * noise * sd
        z = normalise_within_trial(v, F["trial"])
        wh = gate_weights(z, F["trial"], "hard")
        out.append((roc_auc_score(F["in_episode"], z), F["in_episode"][wh > 0].mean()))
    return float(np.mean([o[0] for o in out])), float(np.mean([o[1] for o in out]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results/sim_study_pupilY/raw.json")
    ap.add_argument("--noise", default="results/sim_study_noise/raw.json")
    ap.add_argument("--out", default="manuscript/generated")
    a = ap.parse_args()
    base = {(k[0], k[1], k[2], k[3], k[4], 0.0): v for k, v in load(a.base).items() if k[1] == "trial"}
    R = json.load(open(a.noise))
    nz = {}
    for r in R:
        preds = [p for f in r["folds"] for p in f["trial_preds"]]
        acc = float(np.mean([p[1] == p[2] for p in preds]))
        nz[(r["root"], r["protocol"], r["seed"], r["variant"], r["band"], float(r["noise"]))] = {"m": {"acc": acc}}
    allr = {**base, **nz}
    levels = sorted({k[5] for k in allr})
    roots = ["data/demo", "data/demo_easy"]
    res = {}
    L = [r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
         r"Data & Noise $\kappa$ & Gate AUC & No gate & M0 hard & M1 fuzzy & A4 linear & $\Delta$(M1$-$M0) & $p$\\", r"\midrule"]
    for root in roots:
        for i, lv in enumerate(levels):
            accs = {}
            for v in VARIANTS:
                vals = [allr[k]["m"]["acc"] * 100 for k in allr if k[0] == root and k[3] == v and k[5] == lv]
                accs[v] = (np.mean(vals), np.std(vals))
            keys = sorted(k for k in allr if k[0] == root and k[3] == "fuzzy" and k[5] == lv)
            d = np.array([allr[k]["m"]["acc"] - allr[(k[0], k[1], k[2], "hard", k[4], k[5])]["m"]["acc"] for k in keys]) * 100
            p = wilcoxon(d).pvalue if np.any(np.abs(d) > 1e-9) else 1.0
            auc, hprec = gate_auc(root, lv)
            res[f"{DNAME[root]}|{lv}"] = {"auc": auc, "hard_prec": hprec, **{v: list(accs[v]) for v in VARIANTS},
                                          "d_mean": float(d.mean()), "d_median": float(np.median(d)), "p": float(p)}
            L.append((DNAME[root] if i == 0 else "") + f" & {lv:g} & {auc:.3f} & " +
                     " & ".join(f"{accs[v][0]:.1f}" for v in VARIANTS) + f" & {d.mean():+.1f} & {p:.3f}\\\\")
        if root != roots[-1]:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.out, "tab_noise.tex"), "w").write("\n".join(L) + "\n")

    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.3), sharey=True)
    sty = {"none": ("0.6", ":", "s"), "hard": ("0.35", "--", "^"), "fuzzy": ("k", "-", "o"), "linear": ("0.5", "-.", "d")}
    for ax, root in zip(axs, roots):
        for v in VARIANTS:
            m = [res[f"{DNAME[root]}|{lv}"][v][0] for lv in levels]
            c, ls, mk = sty[v]
            ax.plot(levels, m, ls=ls, marker=mk, color=c, ms=4, lw=1.1, label=VNAME[v])
        ax.axhline(25, color="k", lw=0.6, ls=":")
        ax.set_xlabel(r"gate noise $\kappa$ (× within-subject SD)")
        ax.set_title(f"{DNAME[root]} (synthetic)", fontsize=9)
        ax.set_xticks(levels)
    axs[0].set_ylabel("Mean trial accuracy (%)")
    axs[1].legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_noise.pdf"), bbox_inches="tight"); plt.close(fig)
    json.dump(res, open(os.path.join(a.out, "noise_summary.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
