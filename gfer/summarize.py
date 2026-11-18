"""Summarise a compare.py run (M0 hard gate vs M1 fuzzy gate) on real data.

    python -m gfer.summarize --raw results/zuco_m0m1/raw.json --data data/zuco_npz

Writes summary.csv, report.md and m0_vs_m1.png next to raw.json.
Trial-level accuracy / macro-F1, mean ± SD over seeds; binomial test against chance (1/C);
M1 - M0: paired Wilcoxon over subjects (LOSO) and exact McNemar on pooled trial predictions.
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import f1_score

NAMES = {"none": "No gate", "hard": "M0 hard gate", "fuzzy": "M1 fuzzy gate",
         "fuzzy_trialnorm": "M1 (per-sentence scaling, v1)",
         "adaptive": "M2 adaptive temporal", "adaptive_nosmooth": "M2 adaptive only",
         "adaptive_causal": "M2 causal", "fuzzy_smooth": "M1+temporal"}
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


def preds_of(r):
    p = np.array([q for f in r["folds"] for q in f["trial_preds"]])   # (trial, y, yhat)
    return p[np.argsort(p[:, 0])]


def mcnemar(y, a, b):
    ca, cb = a == y, b == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--proposed", default="fuzzy", help="variant compared against M0 (hard)")
    ap.add_argument("--variants", default=None, help="comma list of variants to report (default all)")
    a = ap.parse_args()
    P, PN = a.proposed, NAMES[a.proposed].split()[0]
    out = os.path.dirname(a.raw)
    R = json.load(open(a.raw))
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    C = len(meta["class_names"])
    # trial index -> subject (same ordering as load_npz: sorted subjects, sorted files)
    subj_of = []
    for s in sorted(d for d in os.listdir(a.data) if os.path.isdir(os.path.join(a.data, d))):
        subj_of += [s] * len([f for f in os.listdir(os.path.join(a.data, s)) if f.endswith(".npz")])
    subj_of = np.array(subj_of)

    protocols = sorted({r["protocol"] for r in R})
    variants = [v for v in NAMES if any(r["variant"] == v for r in R)
                and (a.variants is None or v in a.variants.split(","))]
    bands = [b for b in BANDS if any(r["band"] == b for r in R)]
    seeds = sorted({r["seed"] for r in R})
    idx = {(r["protocol"], r["variant"], r["band"], r["seed"]): r for r in R}

    rows, L = [], []
    L.append(f"# M0 vs {PN} on {meta.get('dataset')}\n")
    L.append(f"> {meta.get('note', '')}\n")
    n_trials = len(preds_of(R[0]))
    L.append(f"- Trials (sentences): {n_trials}; subjects: {len(set(subj_of))}; classes: "
             f"{', '.join(meta['class_names'])} (chance = {100 / C:.1f}%)")
    L.append(f"- Window {meta.get('window_s', 4)} s; seeds {seeds}; protocols {protocols}\n")
    for prot in protocols:
        L.append(f"## Protocol `{prot}` — trial-level accuracy %, mean ± SD over seeds\n")
        L.append("| band | " + " | ".join(NAMES[v] for v in variants) + f" | {PN} − M0 | McNemar p (seed 0) |"
                 + (" Wilcoxon p (subjects) |" if prot == "loso" else ""))
        L.append("|---|" + "---|" * (len(variants) + 2 + (prot == "loso")))
        for b in bands:
            cells, acc_mean = [], {}
            for v in variants:
                accs, f1s, ps = [], [], []
                for s in seeds:
                    r = idx.get((prot, v, b, s))
                    if r is None:
                        continue
                    p = preds_of(r)
                    k = int((p[:, 1] == p[:, 2]).sum())
                    accs.append(k / len(p)); f1s.append(f1_score(p[:, 1], p[:, 2], average="macro"))
                    ps.append(binomtest(k, len(p), 1 / C, alternative="greater").pvalue)
                acc_mean[v] = np.mean(accs)
                star = "*" if np.median(ps) < 0.05 else ""
                cells.append(f"{100 * np.mean(accs):.1f} ± {100 * np.std(accs):.1f}{star}")
                rows.append([prot, b, v, f"{np.mean(accs):.4f}", f"{np.std(accs):.4f}",
                             f"{np.mean(f1s):.4f}", f"{np.median(ps):.4g}", len(accs)])
            line = f"| {b} | " + " | ".join(cells)
            if "hard" in acc_mean and P in acc_mean:
                p0, p1 = preds_of(idx[(prot, "hard", b, seeds[0])]), preds_of(idx[(prot, P, b, seeds[0])])
                n01, n10, pm = mcnemar(p0[:, 1], p0[:, 2], p1[:, 2])
                line += f" | {100 * (acc_mean[P] - acc_mean['hard']):+.1f} | {pm:.3f} ({n10} vs {n01})"
                if prot == "loso":
                    ds = []
                    for s_ in sorted(set(subj_of)):
                        m = subj_of[p0[:, 0]] == s_
                        ds.append((p1[m, 1] == p1[m, 2]).mean() - (p0[m, 1] == p0[m, 2]).mean())
                    pw = wilcoxon(ds).pvalue if np.any(np.array(ds) != 0) else 1.0
                    line += f" | {pw:.3f} |"
                else:
                    line += " |"
            L.append(line)
        L.append("\n`*` = above chance, one-sided binomial p < 0.05 (median over seeds). "
                 f"McNemar column: trials {PN} got right and M0 wrong vs the reverse.\n")
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protocol", "band", "variant", "acc_mean", "acc_sd_over_seeds", "macro_f1",
                    "p_above_chance", "n_seeds"])
        w.writerows(rows)
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(protocols), figsize=(5.2 * len(protocols), 3.4), squeeze=False)
    cols = {"none": "#b0b0b0", "hard": "#4c72b0", "fuzzy": "#dd8452", "fuzzy_trialnorm": "#f2c49b",
            "adaptive": "#2a9d6f", "adaptive_nosmooth": "#8fd1b5", "adaptive_causal": "#5c7d6f", "fuzzy_smooth": "#a04f8f"}
    for ax, prot in zip(axs[0], protocols):
        xw = 0.8 / len(variants)
        for j, v in enumerate(variants):
            m = [100 * float(r[3]) for r in rows if r[0] == prot and r[2] == v]
            sd = [100 * float(r[4]) for r in rows if r[0] == prot and r[2] == v]
            ax.bar(np.arange(len(bands)) + (j - (len(variants) - 1) / 2) * xw, m, xw, yerr=sd,
                   color=cols[v], label=NAMES[v], capsize=2)
        ax.axhline(100 / C, color="k", ls=":", lw=1, label="chance")
        ax.set_xticks(range(len(bands))); ax.set_xticklabels(bands)
        ax.set_ylabel("trial accuracy (%)"); ax.set_title(f"{meta.get('dataset')} — {prot}")
        lo = min(100 / C, min(100 * float(r[3]) for r in rows)) - 8
        ax.set_ylim(max(0, lo), None)
    axs[0][0].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out, f"m0_vs_{PN.lower()}.png"), dpi=160)
    print("\n".join(L))


if __name__ == "__main__":
    main()
