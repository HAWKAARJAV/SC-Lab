"""Figures + numbers for the final project report (ZuCo and SEED-IV).

    cd "<repo>" && python3 manuscript/final_project_report/make_figures.py

Inputs: gpu_seed/results/improved/ (SEED-IV, GPU run), manuscript/zuco_report/stats.json (ZuCo).
Outputs: figures/*.pdf|png and seed_stats.json next to this file.
"""
import collections
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG = os.path.join(HERE, "figures")
SEED = os.path.join(ROOT, "gpu_seed", "results", "improved")
ZUCO = os.path.join(ROOT, "manuscript", "zuco_report", "stats.json")

# fixed categorical order (validated palette, light mode); the no-gate reference is neutral grey
COL = {"none": "#8a8a85", "hard": "#2a78d6", "fuzzy": "#eb6834", "adaptive": "#1baf7a",
       "ref_logreg": "#eda100", "fuzzy_smooth": "#e87ba4"}
LAB = {"none": "No gate", "hard": "M0 hard (paper)", "fuzzy": "M1 fuzzy", "adaptive": "M2 adaptive temporal",
       "ref_logreg": "Reference: logistic regression", "fuzzy_smooth": "M1 + temporal"}
INK, INK2, GRID = "#1f1f1e", "#5f5e5a", "#e6e5e1"


def style(plt):
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlesize": 8.5, "axes.titleweight": "bold"})


def seed_stats():
    meta = json.load(open(os.path.join(SEED, "trials.json")))
    subj = np.array(meta["subject"])
    R = {}
    for p in glob.glob(os.path.join(SEED, "tasks", "*.json")):
        r = json.load(open(p))
        R[(r["protocol"], r["variant"], r["seed"])] = r
    S = {"n_trials": len(meta["trial_ids"]), "n_subjects": len(set(subj)), "classes": meta["class_names"],
         "acc": {}, "persubject": {}, "recall": {}, "feature_mix": {}, "rules": {}}
    for (p, v, s), r in R.items():
        if p == "paper":
            a = float(np.mean([f["window_acc"] for f in r["folds"]]))
        else:
            q = np.array([x for f in r["folds"] for x in f["trial_preds"]])
            a = float(np.mean(q[:, 1] == q[:, 2]))
        S["acc"].setdefault(f"{p}|{v}", []).append(a)
    for p in ["within", "trial", "loso"]:
        for v in ["none", "hard", "fuzzy", "adaptive", "ref_logreg"]:
            cm = np.zeros((4, 4)); ps = collections.defaultdict(list); mix = collections.Counter(); rules = []
            for s in (0, 1, 2):
                r = R[(p, v, s)]
                q = np.array([x for f in r["folds"] for x in f["trial_preds"]])
                for _, yt, yp in q:
                    cm[yt, yp] += 1
                for s_ in np.unique(subj):
                    m = subj[q[:, 0]] == s_
                    ps[s_].append(float((q[m, 1] == q[m, 2]).mean()))
                for f in r["folds"]:
                    rules.append(f["n_rules"])
                    if isinstance(f["features"], list):
                        for x in f["features"]:
                            mix["eye" if x.startswith("eye:") else ("DE" if ":DE-" in x else "entropy")] += 1
            S["recall"][f"{p}|{v}"] = (cm.diagonal() / cm.sum(1)).tolist()
            S["persubject"][f"{p}|{v}"] = {k: float(np.mean(x)) for k, x in ps.items()}
            tot = sum(mix.values())
            S["feature_mix"][f"{p}|{v}"] = {k: n / tot for k, n in mix.items()} if tot else {}
            S["rules"][f"{p}|{v}"] = float(np.mean(rules))
    S["acc_mean"] = {k: float(np.mean(v)) for k, v in S["acc"].items()}
    S["acc_sd"] = {k: float(np.std(v)) for k, v in S["acc"].items()}
    gate = collections.Counter(f["gate_feature"] for k, r in R.items() for f in r["folds"]
                               if k[1] in ("hard", "fuzzy", "adaptive"))
    S["gate_feature"] = gate.most_common(3)
    return S


def box(ax, x, y, w, h, text, fc, ec=INK2, fs=6.8, bold=False):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            weight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=0, shrinkB=0))


def fig_architecture(plt):
    EYE, EEG, FIS, GATE = "#fbe6dc", "#e3edf8", "#e2f3ea", "#fff1cc"
    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    ax.set_xlim(0, 24); ax.set_ylim(0, 10.5); ax.axis("off")
    ax.text(0, 10.1, "Original pipeline (Farhadi Sedehi et al., 2024): the eye gate selects EEG windows; "
            "our models change only the gate", fontsize=7.5, color=INK, weight="bold")
    # eye branch
    box(ax, 0.0, 7.2, 3.6, 1.9, "Eye tracking\n4-s windows\n(pupil, fixations)", EYE)
    box(ax, 4.4, 7.2, 3.0, 1.9, "mRMR ranks\neye features\n(training folds)", EYE)
    box(ax, 8.2, 7.2, 5.6, 1.9, "EYE GATE\nweight w(t) per window\nM0 / M1 / M2", GATE, ec="#c98500", bold=True)
    arrow(ax, 3.6, 8.15, 4.4, 8.15); arrow(ax, 7.4, 8.15, 8.2, 8.15)
    # EEG branch
    xs = [0.0, 4.4, 8.2, 12.6, 16.6, 20.4]
    ws = [3.6, 3.0, 3.6, 3.2, 3.0, 3.6]
    tx = ["EEG\nband-pass 1–60 Hz\nnotch, 128 Hz", "db4 wavelet\n5 bands\nδ θ α β γ",
          "4 entropies per\nwindow & channel\nSampEn PermEn\nSymbEn PhEn", "mRMR selects\n3 features\n(weighted by w)",
          "Sugeno FIS\nWang–Mendel\nrules (27 cells)", "Genetic algorithm\ntunes rules →\nemotion per\ntrial (vote)"]
    fcs = [EEG, EEG, EEG, EEG, FIS, FIS]
    for x, w, t, fc in zip(xs, ws, tx, fcs):
        box(ax, x, 2.6, w, 2.6, t, fc)
    for i in range(5):
        arrow(ax, xs[i] + ws[i], 3.9, xs[i + 1], 3.9)
    # gate acts on the classifier stages
    arrow(ax, 11.0, 7.2, 14.2, 5.2)
    arrow(ax, 12.4, 7.2, 18.1, 5.2)
    arrow(ax, 13.8, 7.6, 22.2, 5.2)
    ax.text(14.6, 8.15, "w(t) weights feature selection, membership\nfunctions, rule degrees, GA fitness and the\ntrial vote",
            fontsize=6.4, color=INK2, ha="left", va="center")
    ax.text(0, 1.2, "M0 (paper): w = 1 for the top 40% of a trial's windows by pupil size, else 0 (windows discarded).\n"
            "M1 / M2 (ours): every window is kept with a graded weight w ∈ [0.1, 1] from a fuzzy Low/Medium/High rule base.",
            fontsize=6.6, color=INK, va="center", linespacing=1.4)
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(FIG, "architecture.pdf")); fig.savefig(os.path.join(FIG, "architecture.png"), dpi=200)
    plt.close(fig)


def fig_gates(plt):
    """Weight as a function of the scaled pupil value z for the three gates."""
    import sys
    sys.path.insert(0, ROOT)
    from gfer.gates import fuzzy_memberships, fuzzy_weights
    z = np.linspace(0, 1, 401)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.0))
    # M0
    axs[0].plot(z, (z >= 0.6).astype(float), color=COL["hard"], lw=2, drawstyle="steps-post")
    axs[0].set_title("M0 hard gate (paper)")
    axs[0].text(0.03, 0.55, "keep the top 40%\nof each trial;\ndrop the rest", fontsize=6.5, color=INK2)
    # M1 memberships + output
    muL, muM, muH = fuzzy_memberships(z)
    for mu, n in [(muL, "Low"), (muM, "Medium"), (muH, "High")]:
        axs[1].plot(z, mu, color="#b9b8b2", lw=1)
        axs[1].text(z[np.argmax(mu)], 1.04, n, fontsize=6.2, ha="center", color=INK2)
    axs[1].plot(z, fuzzy_weights(z), color=COL["fuzzy"], lw=2)
    axs[1].set_title("M1 fuzzy gate", pad=12)
    axs[1].text(0.02, 0.62, "rules → 0.1 / 0.5 / 1.0\nper-person scaling", fontsize=6.3, color=INK2)
    # M2: per-person breakpoints for two example people + smoothing note
    from gfer.gates import FUZZY
    for (a1, c, b2), ls in [((0.15, 0.35, 0.6), "-"), ((0.4, 0.65, 0.85), "--")]:
        mL, mM, mH = fuzzy_memberships(z, a1, c, b2)
        r = FUZZY["r"]
        w = (mL * r[0] + mM * r[1] + mH * r[2]) / np.maximum(mL + mM + mH, 1e-12)
        axs[2].plot(z + 0.0, w, color=COL["adaptive"], lw=2, ls=ls)
    axs[2].set_title("M2 adaptive temporal gate")
    axs[2].text(0.98, 0.06, "two example\npeople", fontsize=6.0, color=INK2, ha="right")
    for ax in axs:
        ax.set_xlabel("pupil value z (scaled)"); ax.set_ylim(-0.05, 1.12)
        ax.set_yticks([0, 0.5, 1]); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axs[0].set_ylabel("window weight w")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "gates.pdf")); fig.savefig(os.path.join(FIG, "gates.png"), dpi=200)
    plt.close(fig)


def bars(ax, groups, models, val, err, chance, ymax, labels_on=True):
    n = len(models)
    width = 0.8 / n
    for j, m in enumerate(models):
        xs = np.arange(len(groups)) + (j - (n - 1) / 2) * width
        v = [val(g, m) for g in groups]
        e = [err(g, m) for g in groups]
        ax.bar(xs, v, width * 0.9, yerr=e, color=COL[m], label=LAB[m], capsize=1.5,
               error_kw={"lw": 0.6, "ecolor": INK2})
        if labels_on:
            for x, y in zip(xs, v):
                ax.text(x, y + max(e) + 0.8, f"{y:.0f}", ha="center", va="bottom", fontsize=5.6, color=INK)
    ax.axhline(chance, color=INK, ls=":", lw=0.9)
    ax.set_ylim(0, ymax); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)


def fig_seed(plt, S):
    prots = ["within", "trial", "loso", "paper"]
    names = {"within": "Per-person\n(unseen clips)", "trial": "5-fold over trials\n(all people)",
             "loso": "Leave-one-\nsubject-out", "paper": "Paper protocol\n(window split, leaky)"}
    models = ["none", "hard", "fuzzy", "adaptive", "ref_logreg"]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    bars(ax, prots, models, lambda g, m: 100 * S["acc_mean"][f"{g}|{m}"],
         lambda g, m: 100 * S["acc_sd"][f"{g}|{m}"], 25, 108)
    ax.set_xticks(range(len(prots))); ax.set_xticklabels([names[p] for p in prots])
    ax.set_ylabel("accuracy (%)")
    ax.text(0.5, 26.5, "chance 25%", fontsize=6, color=INK2, ha="center")
    ax.axhline(94.42, color="#e34948", lw=0.9, ls="--")
    ax.text(-0.45, 95.6, "paper's reported 94.4% (α band, window split)", fontsize=6, color="#b8302f", ha="left")
    ax.legend(fontsize=6.4, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "seed_results.pdf")); fig.savefig(os.path.join(FIG, "seed_results.png"), dpi=200)
    plt.close(fig)

    # per-person accuracy (within protocol): one dot per subject, connected per model
    models = ["none", "hard", "fuzzy", "adaptive", "ref_logreg"]
    subs = sorted(S["persubject"]["within|none"])
    fig, ax = plt.subplots(figsize=(7.2, 2.3))
    rng = np.random.default_rng(0)
    for j, m in enumerate(models):
        vals = np.array([100 * S["persubject"][f"within|{m}"][s] for s in subs])
        ax.scatter(j + rng.uniform(-0.12, 0.12, len(vals)), vals, s=14, color=COL[m], edgecolor="white", lw=0.5, zorder=3)
        ax.plot([j - 0.25, j + 0.25], [vals.mean()] * 2, color=INK, lw=1.4, zorder=4)
        ax.text(j + 0.3, vals.mean(), f"{vals.mean():.1f}", va="center", fontsize=6.3, color=INK)
    ax.axhline(25, color=INK, ls=":", lw=0.9)
    ax.set_xticks(range(len(models))); ax.set_xticklabels([LAB[m] for m in models], fontsize=6.6)
    ax.set_ylabel("per-person accuracy (%)"); ax.set_ylim(20, 95)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "seed_persubject.pdf")); fig.savefig(os.path.join(FIG, "seed_persubject.png"), dpi=200)
    plt.close(fig)


def fig_zuco(plt):
    Z = json.load(open(ZUCO))
    models = ["none", "hard", "fuzzy", "adaptive", "fuzzy_smooth"]
    prots = ["trial", "loso"]
    names = {"trial": "5-fold over sentences", "loso": "Leave-one-subject-out"}
    fig, ax = plt.subplots(figsize=(4.6, 2.5))
    bars(ax, prots, models, lambda g, m: 100 * Z["results"][f"{g}|{m}|mean"]["acc"], lambda g, m: 0.0, 33.3, 40,
         labels_on=False)
    ax.set_xticks(range(2)); ax.set_xticklabels([names[p] for p in prots])
    ax.set_ylabel("accuracy (%), mean of 5 bands")
    ax.text(0.5, 31.6, "chance 33.3%", fontsize=6, color=INK2, ha="center")
    ax.axhline(100 * Z["logreg_ceiling"]["eeg_eye|loso"]["acc"], color=COL["ref_logreg"], lw=0.9, ls="--")
    ax.text(-0.45, 36.4, "logistic-regression ceiling ≈ 35%", fontsize=6, color="#9a6a00")
    ax.legend(fontsize=6.1, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "zuco_results.pdf")); fig.savefig(os.path.join(FIG, "zuco_results.png"), dpi=200)
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    os.makedirs(FIG, exist_ok=True)
    S = seed_stats()
    with open(os.path.join(HERE, "seed_stats.json"), "w", encoding="utf-8") as f:
        json.dump(S, f, indent=1)
    fig_architecture(plt)
    fig_gates(plt)
    fig_seed(plt, S)
    fig_zuco(plt)
    for k in sorted(S["acc_mean"]):
        print(f"{k:22s} {100 * S['acc_mean'][k]:.1f} ± {100 * S['acc_sd'][k]:.1f}")
    print("gate feature:", S["gate_feature"])


if __name__ == "__main__":
    main()
