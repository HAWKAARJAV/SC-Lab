"""Turn results/sim_study_*/raw.json into LaTeX tables, figures and statistics.

    python -m gfer.analyze_sim --out manuscript/generated

Every number written to the manuscript comes from this script (no manual transcription).
All results are on SYNTHETIC data (gfer/demo.py).
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from .compare import VARIANTS, gate_weights, load_features, normalise_within_trial
from .config import Config
from .evaluate import zscore_by_subject
from .mrmr import mrmr

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
VNAME = {"none": "No gate (A1)", "hard": "Hard gate, M0", "fuzzy": "Fuzzy gate, M1",
         "linear": "Linear soft gate (A4)"}
DNAME = {"data/demo": "SIM-H", "data/demo_easy": "SIM-E"}
PNAME = {"trial": "trial-disjoint 5-fold", "loso": "LOSO"}
CLASSES = ["sad", "happy", "fear", "neutral"]


def trial_metrics(preds):
    y = np.array([p[1] for p in preds]); yh = np.array([p[2] for p in preds])
    lab = list(range(4))
    return {"acc": float((y == yh).mean()),
            "bacc": float(recall_score(y, yh, average="macro", labels=lab, zero_division=0)),
            "prec": float(precision_score(y, yh, average="macro", labels=lab, zero_division=0)),
            "rec": float(recall_score(y, yh, average="macro", labels=lab, zero_division=0)),
            "f1": float(f1_score(y, yh, average="macro", labels=lab, zero_division=0))}


def load(path):
    R = json.load(open(path))
    idx = {}
    for r in R:
        preds = [p for f in r["folds"] for p in f["trial_preds"]]
        idx[(r["root"], r["protocol"], r["seed"], r["variant"], r["band"])] = {
            "preds": preds, "m": trial_metrics(preds), "elapsed": r["elapsed_s"],
            "eff": np.mean([f["train_eff_windows"] / f["train_windows"] for f in r["folds"]]),
            "gates": [f["gate_feature"] for f in r["folds"]],
            "feats": [f["features"] for f in r["folds"]]}
    return idx


def agg(idx, root, prot, var, band, key="acc"):
    v = [idx[k]["m"][key] for k in idx if k[0] == root and k[1] == prot and k[3] == var and k[4] == band]
    return np.mean(v) * 100, np.std(v) * 100, len(v)


def fmt(m, s):
    return f"{m:.1f} $\\pm$ {s:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", default="results/sim_study_autogate/raw.json")
    ap.add_argument("--pupil", default="results/sim_study_pupilY/raw.json")
    ap.add_argument("--out", default="manuscript/generated")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = {"auto": load(a.auto), "pupil": load(a.pupil)}
    roots = ["data/demo", "data/demo_easy"]
    summary = {}

    # ---------------- Table: per-band accuracy, all variants (both gate features, trial protocol + LOSO)
    for gk, gname in [("pupil", "pupil-Y"), ("auto", "mRMR-selected")]:
        idx = S[gk]
        for prot in ["trial", "loso"]:
            L = [r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
                 r"Data & Configuration & Delta & Theta & Alpha & Beta & Gamma\\", r"\midrule"]
            for root in roots:
                for i, v in enumerate(VARIANTS):
                    cells = []
                    best = {b: max(agg(idx, root, prot, vv, b)[0] for vv in VARIANTS) for b in BANDS}
                    for b in BANDS:
                        m, s, _ = agg(idx, root, prot, v, b)
                        c = fmt(m, s)
                        cells.append(r"\textbf{" + c + "}" if abs(m - best[b]) < 1e-9 else c)
                        summary[f"{gk}|{prot}|{DNAME[root]}|{v}|{b}|acc"] = [m, s]
                    L.append((DNAME[root] if i == 0 else "") + f" & {VNAME[v]} & " + " & ".join(cells) + r"\\")
                if root != roots[-1]:
                    L.append(r"\midrule")
            L += [r"\bottomrule", r"\end{tabular}"]
            open(os.path.join(a.out, f"tab_bands_{gk}_{prot}.tex"), "w").write("\n".join(L) + "\n")

    # ---------------- Table: macro metrics averaged over bands (pupil gate, trial protocol)
    for gk in ["pupil", "auto"]:
        idx = S[gk]
        for prot in ["trial", "loso"]:
            L = [r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
                 r"Data & Configuration & Accuracy & Bal.\ acc. & Macro P & Macro R & Macro F1\\", r"\midrule"]
            for root in roots:
                for i, v in enumerate(VARIANTS):
                    row = []
                    for key in ["acc", "bacc", "prec", "rec", "f1"]:
                        vals = [idx[k]["m"][key] for k in idx if k[0] == root and k[1] == prot and k[3] == v]
                        row.append(fmt(np.mean(vals) * 100, np.std(vals) * 100))
                        summary[f"{gk}|{prot}|{DNAME[root]}|{v}|allbands|{key}"] = [np.mean(vals) * 100, np.std(vals) * 100]
                    L.append((DNAME[root] if i == 0 else "") + f" & {VNAME[v]} & " + " & ".join(row) + r"\\")
                if root != roots[-1]:
                    L.append(r"\midrule")
            L += [r"\bottomrule", r"\end{tabular}"]
            open(os.path.join(a.out, f"tab_macro_{gk}_{prot}.tex"), "w").write("\n".join(L) + "\n")

    # ---------------- Statistics: M1 vs M0, M1 vs A4, M1 vs A1 (paired over seed x band)
    L = [r"\begin{tabular}{@{}lllcccc@{}}", r"\toprule",
         r"Gate feature & Protocol & Data & Comparison & Median $\Delta$acc (pp) & W / wins--ties--losses & $p$\\", r"\midrule"]
    stats = {}
    for gk, gname in [("pupil", "pupil-Y"), ("auto", "mRMR")]:
        idx = S[gk]
        for prot in ["trial", "loso"]:
            for root in roots:
                for other in ["hard", "linear", "none"]:
                    keys = sorted(k for k in idx if k[0] == root and k[1] == prot and k[3] == "fuzzy")
                    d = np.array([idx[k]["m"]["acc"] - idx[(k[0], k[1], k[2], other, k[4])]["m"]["acc"] for k in keys]) * 100
                    w_, t_, l_ = int((d > 1e-9).sum()), int((np.abs(d) <= 1e-9).sum()), int((d < -1e-9).sum())
                    try:
                        W, p = wilcoxon(d) if np.any(np.abs(d) > 1e-9) else (np.nan, 1.0)
                    except ValueError:
                        W, p = np.nan, 1.0
                    stats[f"{gk}|{prot}|{DNAME[root]}|M1-{other}"] = {"median": float(np.median(d)), "mean": float(d.mean()),
                                                                        "W": float(W) if np.isfinite(W) else None, "p": float(p),
                                                                        "wins": w_, "ties": t_, "losses": l_, "n": len(d)}
                    L.append(f"{gname} & {PNAME[prot]} & {DNAME[root]} & M1 vs {VNAME[other].split(',')[0].replace(' (A1)','').replace(' (A4)','')} & "
                             f"{np.median(d):+.1f} & {W if np.isfinite(W) else 0:.0f} / {w_}--{t_}--{l_} & {p:.3f}\\\\")
            L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    open(os.path.join(a.out, "tab_stats.tex"), "w").write("\n".join(L) + "\n")
    summary["stats"] = stats

    # McNemar (exact) for the best band of SIM-E, pupil gate, trial protocol, M1 vs M0, pooled over seeds
    idx = S["pupil"]
    mc = {}
    for root in roots:
        bb = max(BANDS, key=lambda b: agg(idx, root, "trial", "fuzzy", b)[0])
        b01 = b10 = 0
        for s in range(3):
            p0 = {t: (y, yh) for t, y, yh in idx[(root, "trial", s, "hard", bb)]["preds"]}
            p1 = {t: (y, yh) for t, y, yh in idx[(root, "trial", s, "fuzzy", bb)]["preds"]}
            for t in p0:
                c0, c1 = p0[t][0] == p0[t][1], p1[t][0] == p1[t][1]
                b01 += int(c0 and not c1); b10 += int(c1 and not c0)
        p = binomtest(b10, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
        mc[DNAME[root]] = {"band": bb, "M1_only_correct": b10, "M0_only_correct": b01, "p_exact": float(p)}
    summary["mcnemar"] = mc

    # ---------------- Gate quality vs known episodes
    cfg = Config()
    L = [r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
         r"Data & Gate feature & Episode rate & AUC($\tilde z$) & Hard: prec./recall & Fuzzy: w-prec. & Linear: w-prec.\\", r"\midrule"]
    gq = {}
    for root in roots:
        F = load_features(root, cfg)
        ep = F["in_episode"]
        eye = zscore_by_subject(F["eye"], F["subject"])
        auto_g = F["eye_names"][int(mrmr(eye, F["label"], k=1)[0][0])]
        for g in ["pupil_y_mean", auto_g]:
            z = normalise_within_trial(F["eye"][:, F["eye_names"].index(g)], F["trial"])
            wh = gate_weights(z, F["trial"], "hard"); wf = gate_weights(z, F["trial"], "fuzzy")
            wl = gate_weights(z, F["trial"], "linear")
            auc = roc_auc_score(ep, z)
            hp, hr = ep[wh > 0].mean(), (wh[ep] > 0).mean()
            fp = (wf * ep).sum() / wf.sum(); lp = (wl * ep).sum() / wl.sum()
            gq[f"{DNAME[root]}|{g}"] = {"episode_rate": float(ep.mean()), "auc": float(auc), "hard_prec": float(hp),
                                        "hard_recall": float(hr), "fuzzy_wprec": float(fp), "linear_wprec": float(lp),
                                        "fuzzy_mean_w_in": float(wf[ep].mean()), "fuzzy_mean_w_out": float(wf[~ep].mean())}
            gn = g.replace("_", r"\_")
            L.append(f"{DNAME[root]} & \\texttt{{{gn}}} & {100*ep.mean():.1f}\\% & {auc:.3f} & {100*hp:.1f}\\% / {100*hr:.1f}\\% & "
                     f"{100*fp:.1f}\\% & {100*lp:.1f}\\%\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.out, "tab_gate_quality.tex"), "w").write("\n".join(L) + "\n")
    summary["gate_quality"] = gq

    # gate feature chosen by mRMR across folds (auto run)
    from collections import Counter
    summary["auto_gate_choices"] = {DNAME[r]: dict(Counter(g for k, v in S["auto"].items() if k[0] == r for g in v["gates"]))
                                    for r in roots}

    # ---------------- Cost
    cost = {}
    for v in VARIANTS:
        e = [S[g][k]["elapsed"] for g in S for k in S[g] if k[3] == v]
        f = [S[g][k]["eff"] for g in S for k in S[g] if k[3] == v]
        cost[v] = {"elapsed_mean_s": float(np.mean(e)), "eff_train_fraction": float(np.mean(f))}
    summary["cost"] = cost
    L = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
         r"Configuration & Effective training windows (\%) & Time per band, all folds (s)\\", r"\midrule"]
    for v in VARIANTS:
        L.append(f"{VNAME[v]} & {100*cost[v]['eff_train_fraction']:.1f} & {cost[v]['elapsed_mean_s']:.1f}\\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.out, "tab_cost.tex"), "w").write("\n".join(L) + "\n")

    # ---------------- Figures
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    cols = {"none": "0.75", "hard": "0.45", "fuzzy": "0.1", "linear": "0.6"}
    hatch = {"none": "", "hard": "//", "fuzzy": "", "linear": ".."}
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, root in zip(axs, roots):
        x = np.arange(len(BANDS))
        for i, v in enumerate(VARIANTS):
            ms = [agg(S["pupil"], root, "trial", v, b) for b in BANDS]
            ax.bar(x + (i - 1.5) * 0.2, [m for m, _, _ in ms], 0.2, yerr=[s for _, s, _ in ms], color=cols[v],
                   hatch=hatch[v], edgecolor="k", lw=0.4, capsize=1.5, error_kw={"lw": 0.6}, label=VNAME[v])
        ax.axhline(25, color="k", ls=":", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([b.capitalize() for b in BANDS])
        ax.set_title(f"{DNAME[root]} (synthetic)", fontsize=9)
    axs[0].set_ylabel("Trial-level accuracy (%)")
    axs[1].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_bands.pdf"), bbox_inches="tight"); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.5))
    root = "data/demo_easy"
    bb = mc["SIM-E"]["band"]
    for ax, v in zip(axs, ["hard", "fuzzy"]):
        y, yh = [], []
        for s in range(3):
            for _, t, p in S["pupil"][(root, "trial", s, v, bb)]["preds"]:
                y.append(t); yh.append(p)
        cm = confusion_matrix(y, yh, labels=range(4))
        ax.imshow(cm, cmap="Greys")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="w" if cm[i, j] > cm.max() / 2 else "k")
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels(CLASSES, fontsize=7); ax.set_yticklabels(CLASSES, fontsize=7)
        ax.set_xlabel("Predicted"); ax.set_title(VNAME[v], fontsize=8)
        summary[f"cm|SIM-E|{bb}|{v}"] = cm.tolist()
    axs[0].set_ylabel("True")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_confusion.pdf"), bbox_inches="tight"); plt.close(fig)

    # gate trace on one synthetic trial
    F = load_features("data/demo_easy", cfg)
    t0 = 0
    m = F["trial"] == t0
    z = normalise_within_trial(F["eye"][:, F["eye_names"].index("pupil_y_mean")], F["trial"])[m]
    tt = (F["window"][m] + 0.5) * cfg.window_s
    wh = gate_weights(z, np.zeros(len(z)), "hard"); wf = gate_weights(z, np.zeros(len(z)), "fuzzy")
    fig, ax = plt.subplots(figsize=(6.0, 1.9))
    for i, e in enumerate(F["in_episode"][m]):
        if e:
            ax.axvspan(tt[i] - 2, tt[i] + 2, color="0.88", lw=0)
    ax.step(tt, wh, where="mid", color="0.5", lw=1.3, label="hard gate (M0)")
    ax.plot(tt, wf, "k-o", ms=3, lw=1.2, label="fuzzy gate (M1)")
    ax.plot(tt, z, ":", color="k", lw=0.9, label=r"normalised pupil-Y $\tilde z$")
    ax.set_xlabel("Time in trial (s)"); ax.set_ylabel("weight"); ax.set_ylim(-0.05, 1.1)
    ax.legend(frameon=False, fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_gate_trace.pdf"), bbox_inches="tight"); plt.close(fig)

    # ---------------- team's earlier baseline runs (window-level, original protocol)
    prev = {}
    for name in ["demo_paper_protocol", "demo_trial_protocol", "demo_easy_paper_protocol", "demo_easy_pupilY_gate"]:
        p = os.path.join("results", name, "summary.csv")
        if os.path.exists(p):
            rows = [r for r in csv.DictReader(open(p)) if r["levels"] == "3"]
            prev[name] = {r["band"]: [float(r["acc_mean"]) * 100, float(r["acc_std"]) * 100, float(r["f1_macro"]) * 100] for r in rows}
    summary["previous_runs"] = prev
    L = [r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
         r"Run & Gate feature / protocol & Delta & Theta & Alpha & Beta & Gamma\\", r"\midrule"]
    lab = {"demo_paper_protocol": ("SIM-H", r"mRMR (\texttt{fix\_dur\_mean}) / window 5-fold"),
           "demo_trial_protocol": ("SIM-H", r"mRMR (\texttt{fix\_dur\_mean}) / trial-grouped"),
           "demo_easy_paper_protocol": ("SIM-E", r"mRMR (\texttt{fix\_dur\_mean}) / window 5-fold"),
           "demo_easy_pupilY_gate": ("SIM-E", r"\texttt{pupil\_y\_mean} / window 5-fold")}
    for n, d in prev.items():
        L.append(f"{lab[n][0]} & {lab[n][1]} & " + " & ".join(fmt(d[b][0], d[b][1]) for b in BANDS) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.out, "tab_previous.tex"), "w").write("\n".join(L) + "\n")

    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k in ("stats", "mcnemar", "gate_quality", "cost", "auto_gate_choices")}, indent=1, default=float))


if __name__ == "__main__":
    main()
