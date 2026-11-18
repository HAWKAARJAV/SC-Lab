"""Build every figure and generated table used by manuscript/springer_paper/main.tex.

All numbers are read from the CSV files in results/ and docs/fuzzy_rulebook/;
nothing is typed in by hand. Run from the repository root:

    .venv/bin/python scripts/build_springer_paper.py [--mined <gfer.run output dir>]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from sklearn.metrics import auc, confusion_matrix, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "01_Implemented_Model"
REF = ROOT / "results" / "00_paper_reference"
RULES = ROOT / "docs" / "fuzzy_rulebook"
OUT = ROOT / "manuscript" / "springer_paper"
FIG = OUT / "figures"
GEN = OUT / "generated"

MODELS = {
    "M0": "M0_original",
    "M1": "M1_fuzzy_gate",
    "M2": "M2_adaptive_temporal_gate",
    "M3": "M3_multiband_fusion",
    "M4": "M4_combined_model",
}
DATASETS = {"SEED-IV_reference_scale": "SIM-SEED", "MAHNOB_reference_scale": "SIM-MAHNOB"}
CLASSES = ["sadness", "happiness", "fear", "neutrality"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
BAND_SYM = {"delta": r"$\delta$", "theta": r"$\theta$", "alpha": r"$\alpha$", "beta": r"$\beta$", "gamma": r"$\gamma$"}

# Reference palette (dataviz skill, light mode) -- categorical slots in fixed order.
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
CAT = [C1, C2, C3, C4, C5]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=220)
    plt.close(fig)


def f2(x, d=2):
    return f"{x:.{d}f}"


# --------------------------------------------------------------------------- data
def load():
    summ = pd.concat([pd.read_csv(RES / d / "summary.csv") for d in MODELS.values()], ignore_index=True)
    folds = pd.concat([pd.read_csv(RES / d / "fold_metrics.csv").assign(
        dataset=lambda x: np.where(np.arange(len(x)) < len(x) // 2, "SEED-IV_reference_scale", "MAHNOB_reference_scale"))
        for d in MODELS.values()], ignore_index=True)
    preds = {m: pd.read_csv(RES / d / "predictions.csv") for m, d in MODELS.items()}
    return summ, folds, preds


# ------------------------------------------------------------------ architecture
def box(ax, x, y, w, h, text, fc, ec=INK2, fs=7.6, weight="normal", tag=None, tagc=None, lw=0.9, ha="center"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.2", fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    tx = x + w / 2 if ha == "center" else x + 1.6
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=INK, weight=weight, linespacing=1.25)
    if tag:
        ax.text(x + w - 0.8, y + h - 0.7, tag, ha="right", va="top", fontsize=6.3, color="white", weight="bold",
                bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.25", fc=tagc, ec="none"))
    return p


def arrow(ax, x0, y0, x1, y1, color=INK2, lw=1.0, style="-|>", ls="-", rad=0.0):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=8, color=color, lw=lw,
                        linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def fig_architecture():
    fig = plt.figure(figsize=(7.2, 8.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 120)
    ax.axis("off")

    EYE, EYE_E = "#fbe3d8", C2
    EEG, EEG_E = "#dde9f8", C1
    FIS, FUS, OUTC = "#dff3ea", "#efe6f6", "#fff1cc"
    T_M1, T_M2, T_M3 = "#b8501f", "#6250d6", "#0f7a55"

    # lanes
    ax.add_patch(FancyBboxPatch((1, 42.5), 37, 75.5, boxstyle="round,pad=0,rounding_size=2", fc="#fdf6f2", ec=EYE_E, lw=0.8, ls="--"))
    ax.add_patch(FancyBboxPatch((41, 42.5), 58, 75.5, boxstyle="round,pad=0,rounding_size=2", fc="#f4f8fd", ec=EEG_E, lw=0.8, ls="--"))
    ax.text(3, 116.2, "A. Eye-tracking gate branch", fontsize=8.6, weight="bold", color="#9a3f14", va="center")
    ax.text(43, 116.2, "B. EEG fuzzy-classification branch", fontsize=8.6, weight="bold", color="#1d58a0", va="center")

    # inputs
    box(ax, 4, 104.5, 31, 8.5, "Eye tracker (60 Hz)\npupil diameter X/Y, fixations", "white", EYE_E, weight="bold")
    box(ax, 44, 104.5, 52, 8.5, "EEG (200 Hz)\n62 ch (SEED-IV montage) / 32 ch (MAHNOB-HCI montage)", "white", EEG_E, weight="bold")

    # eye branch
    ey = [94.5, 85, 75.5, 65, 54.5]
    eh = 7.6
    box(ax, 4, ey[0], 31, eh, "4-s windows; 8 eye features\n(fixation stats, pupil mean/std X,Y)", EYE, EYE_E)
    box(ax, 4, ey[1], 31, eh, "mRMR ranking of eye features\n→ gate signal: mean pupil diameter (Y)", EYE, EYE_E)
    box(ax, 4, ey[2], 31, eh, "Subject baseline + adaptive MF cuts\n(training windows only) → $\\tilde z(t)\\in[0,1]$", EYE, EYE_E,
        tag="M2", tagc=T_M2)
    box(ax, 4, ey[3], 31, eh + 1.4, "Fuzzy eye gate (Sugeno, 3 rules)\nLow→0.1, Medium→0.5, High→1.0\n$w(t)=\\Sigma\\mu_i r_i/\\Sigma\\mu_i$",
        EYE, EYE_E, tag="M1", tagc=T_M1)
    box(ax, 4, ey[4], 31, eh, "Temporal smoothing within trial\n(symmetric window, span = 1)", EYE, EYE_E, tag="M2", tagc=T_M2)
    for a, b in [(104.5, ey[0] + eh), (ey[0], ey[1] + eh), (ey[1], ey[2] + eh), (ey[2], ey[3] + eh + 1.4), (ey[3], ey[4] + eh)]:
        arrow(ax, 19.5, a, 19.5, b)
    # gate weight output
    box(ax, 9, 45.3, 21, 5.6, "gate weight $w(t)\\in[0.1,1]$", "white", EYE_E, weight="bold")
    arrow(ax, 19.5, ey[4], 19.5, 50.9)
    ax.text(19.5, 41.4, "M0 baseline: hard gate, keep window if\n$\\tilde z(t)\\geq Q_{0.6}$ of its trial (weight 0/1)",
            ha="center", va="top", fontsize=6.6, color=INK2, style="italic")

    # eeg branch
    bx, bw = 44, 52
    by = [94.5, 85, 75.5]
    box(ax, bx, by[0], bw, eh, "Pre-processing: 4th-order Butterworth 1–60 Hz, 50 Hz notch,\ncommon average reference, resample to 128 Hz, 4-s windows", EEG, EEG_E)
    box(ax, bx, by[1], bw, eh, "Daubechies-4 DWT (4 levels) → five sub-bands\n$\\delta$ 0–4, $\\theta$ 4–8, $\\alpha$ 8–16, $\\beta$ 16–32, $\\gamma$ 32–64 Hz", EEG, EEG_E)
    box(ax, bx, by[2], bw, eh, "Entropy features per channel: SampEn, PermEn, SymbEn, PhEn\n(248 candidates for 62 ch, 128 for 32 ch)", EEG, EEG_E)
    arrow(ax, 70, 104.5, 70, by[0] + eh)
    arrow(ax, 70, by[0], 70, by[1] + eh)
    arrow(ax, 70, by[1], 70, by[2] + eh)

    # per-band S-FIS
    box(ax, bx, 66.5, bw, 6.2, "Gate-weighted mRMR → 3 inputs per band (e.g. PhEn-FP2, SampEn-T8, PermEn-T7)", EEG, EEG_E, fs=7.2)
    arrow(ax, 70, by[2], 70, 72.7)
    ax.add_patch(FancyBboxPatch((bx, 44.2), bw, 19.3, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec="#0f7a55", lw=0.9))
    ax.text(bx + bw / 2, 61.6, "Five parallel Sugeno FIS, GA-refined rule bases", ha="center", va="center", fontsize=7.6, weight="bold")
    cw = 9.0
    gap = (bw - 5 * cw) / 6
    centers = []
    for i, b in enumerate(BANDS):
        x = bx + gap + i * (cw + gap)
        box(ax, x, 47.5, cw, 11.2, f"{BAND_SYM[b]}\nS-FIS\n3×3 MF\n≤27 rules", FIS, "#0f7a55", fs=6.9)
        centers.append(x + cw / 2)
        arrow(ax, x + cw / 2, 59.9, x + cw / 2, 58.7, lw=0.7)
    ax.text(bx + bw / 2, 45.6, "Wang–Mendel initialisation; GA: pop 27, tournament 3, elite 2, $p_c$=0.8, patience 10",
            ha="center", va="center", fontsize=6.4, color=INK2)

    # gate into FIS
    ax.plot([centers[0], centers[-1]], [59.9, 59.9], color=INK2, lw=0.7)
    arrow(ax, 70, 66.5, 70, 63.5)
    arrow(ax, 30, 48.1, bx - 0.2, 53, color=C2, lw=1.4, rad=-0.1)
    ax.text(34.6, 50.6, "sample\nweight", fontsize=6.4, color="#9a3f14", ha="center", va="bottom")

    # fusion
    box(ax, 18, 25, 64, 11.5,
        "Late multi-band fusion:  $P(c)=\\Sigma_{b}\\, w_b\\, P_b(c)$,  $w_b\\geq0$,  $\\Sigma_b\\, w_b=1$\n"
        "$w_b$ learned by coordinate search on an inner validation split of each training fold",
        FUS, "#6b4f8a", fs=7.6, tag="M3", tagc=T_M3)
    arrow(ax, 70, 44.2, 62, 36.5, lw=1.0)
    ax.text(67.5, 39.2, "band posteriors $P_\\delta(c),\\dots,P_\\gamma(c)$", fontsize=7, color=INK2)

    box(ax, 30, 11.5, 40, 7.5, "Decision: $\\hat c=\\arg\\max_c P(c)$", OUTC, "#a07000", fs=8, weight="bold")
    arrow(ax, 50, 25, 50, 19)
    for i, (c, col) in enumerate(zip(["Sadness", "Happiness", "Fear", "Neutrality"], CAT)):
        x = 14 + i * 18.5
        box(ax, x, 2, 16, 5.5, c, "white", col, fs=7.6, lw=1.1)
        arrow(ax, 50, 11.5, x + 8, 7.5, lw=0.6)

    # legend of model variants
    ax.text(85.5, 22.5, "Variants", fontsize=7.2, weight="bold", ha="left")
    for j, (t, c, s) in enumerate([("M1", T_M1, "fuzzy gate"), ("M2", T_M2, "adaptive + smooth"), ("M3", T_M3, "band fusion"),
                                   ("M4", INK, "M2 + M3")]):
        yy = 19.4 - j * 3.2
        ax.text(85.5, yy, t, fontsize=6.3, color="white", weight="bold", va="center",
                bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.25", fc=c, ec="none"))
        ax.text(89.6, yy, s, fontsize=6.4, va="center", color=INK2)
    save(fig, "fig01_architecture")


def fig_sfis():
    FIS, FUS = "#dff3ea", "#efe6f6"
    fig = plt.figure(figsize=(7.2, 3.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")
    names = ["PhEn-FP2", "SampEn-T8", "PermEn-T7"]
    xs = np.linspace(0, 1, 200)
    for i, n in enumerate(names):
        ia = fig.add_axes([0.04, 0.70 - i * 0.29, 0.17, 0.19])
        for lab, (a, b, c), col in [("L", (-0.5, 0, 0.5), C1), ("M", (0, 0.5, 1), C3), ("H", (0.5, 1, 1.5), C2)]:
            mu = np.clip(np.minimum((xs - a) / (b - a), (c - xs) / (c - b)), 0, 1)
            ia.plot(xs, mu, color=col, lw=1.4)
            ia.text(b if lab != "H" else 0.93, 1.07, lab, color=INK, fontsize=7, ha="center")
        ia.set_ylim(0, 1.25)
        ia.set_yticks([0, 1])
        ia.set_xticks([0, 0.5, 1])
        ia.tick_params(labelsize=6, length=2)
        ia.grid(False)
        ia.set_title(n, fontsize=7.5, pad=6, loc="left")
    ax.text(12.5, 47.5, "Inputs: 3 entropy features,\n3 triangular MFs each", ha="center", fontsize=7.5, weight="bold", va="top")
    box(ax, 30, 22, 26, 22, "Rule base\n$3^3=27$ antecedent combinations\nIF $x_1$ is $A_1$ AND $x_2$ is $A_2$\nAND $x_3$ is $A_3$ THEN class $k$\n"
        "firing $\\tau_r=\\prod_j\\mu_{A_j}(x_j)$", FIS, "#0f7a55", fs=7.2)
    for i in range(3):
        arrow(ax, 24.5, 39 - i * 14.5, 30, 33 - i * 0.5)
    box(ax, 62, 22, 18, 22, "Class score\n$s_k=\\dfrac{\\sum_{r:k_r=k}\\tau_r\\,\\omega_r}{\\sum_r\\tau_r}$\n\nsoftmax → $P(c)$", FUS, "#6b4f8a", fs=7.2)
    arrow(ax, 56, 33, 62, 33)
    for i, (c, col) in enumerate(zip(["Sad", "Happy", "Fear", "Neutral"], CAT)):
        box(ax, 85, 38 - i * 5.6, 13, 4.4, c, "white", col, fs=7.2, lw=1.1)
        arrow(ax, 80, 33, 85, 40.2 - i * 5.6, lw=0.6)
    box(ax, 30, 2, 50, 14, "Genetic algorithm (rule mining)\nchromosome = active-rule mask + antecedent levels; population 27\n"
        "fitness = accuracy + 0.2·coverage − 0.01·(rules used); tournament 3, elite 2,\n"
        "uniform crossover 0.8, mutation 0.05; stop after 10 stalled generations",
        "#fff6e0", "#a07000", fs=6.9)
    arrow(ax, 50, 16, 43, 22, color="#a07000", rad=0.25)
    arrow(ax, 43, 22, 50, 16, color="#a07000", rad=0.25, style="-")
    ax.text(38.5, 18.2, "refine", fontsize=6.5, color="#a07000")
    save(fig, "fig02_sfis_ga")


def fig_eye_gate():
    z = np.linspace(0, 1, 400)
    a1, c, b2 = 0.2, 0.5, 0.8
    muL = np.clip((c - z) / (c - a1), 0, 1)
    muH = np.clip((z - c) / (b2 - c), 0, 1)
    muM = np.clip(np.minimum((z - a1) / (c - a1), (b2 - z) / (b2 - c)), 0, 1)
    w = (muL * .1 + muM * .5 + muH * 1.0) / np.maximum(muL + muM + muH, 1e-12)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.3))
    for mu, lab, col in [(muL, "Low", C1), (muM, "Medium", C3), (muH, "High", C2)]:
        axs[0].plot(z, mu, color=col, lw=2, label=lab)
    axs[0].set_xlabel(r"normalised pupil feature $\tilde z$")
    axs[0].set_ylabel(r"membership $\mu$")
    axs[0].legend(loc="center right", fontsize=7)
    axs[0].set_title("(a) Gate memberships", loc="left")
    axs[1].plot(z, w, color=C1, lw=2, label="fuzzy gate (M1)")
    axs[1].plot(z, (z >= 0.6).astype(float), color=INK2, lw=1.4, ls="--", label=r"hard gate (M0, $Q_{0.6}$)")
    axs[1].set_xlabel(r"normalised pupil feature $\tilde z$")
    axs[1].set_ylabel(r"window weight $w$")
    axs[1].legend(loc="upper left", fontsize=7)
    axs[1].set_title("(b) Defuzzified weight", loc="left")
    axs[1].set_ylim(-0.03, 1.08)
    fig.tight_layout()
    save(fig, "fig03_eye_gate")


def fig_gate_traces():
    g = pd.read_csv(RES / "M1_fuzzy_gate" / "gate_weights.csv")
    t = pd.read_csv(RES / "M1_fuzzy_gate" / "temporal_gate_scores.csv")
    fig, axs = plt.subplots(2, 1, figsize=(6.6, 3.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.4]))
    axs[0].plot(t.sample_id, t.pupil_norm, color=INK2, lw=1.2, marker="o", ms=2.5)
    axs[0].axhline(0.6, color=MUTED, lw=0.8, ls=":")
    axs[0].set_ylabel(r"$\tilde z(t)$")
    axs[0].set_title("(a) Normalised pupil feature", loc="left")
    axs[1].step(g.sample_id, g.w_hard, where="mid", color=INK2, lw=1.2, ls="--", label="hard (M0)")
    axs[1].plot(g.sample_id, g.w_fuzzy, color=C1, lw=1.6, label="fuzzy (M1)")
    axs[1].plot(g.sample_id, g.w_adaptive, color=C2, lw=1.6, label="adaptive + smoothed (M2)")
    axs[1].set_ylabel("gate weight")
    axs[1].set_xlabel("window index")
    axs[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=7)
    axs[1].set_ylim(-0.05, 1.1)
    fig.tight_layout()
    save(fig, "fig04_gate_traces")


def fig_accuracy(summ, folds):
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=False)
    for ax, metric, ttl in [(axs[0], "accuracy_pct", "(a) Accuracy (%)"), (axs[1], "macro_f1_pct", "(b) Macro-F1 (%)")]:
        x = np.arange(5)
        for j, (d, lab) in enumerate(DATASETS.items()):
            s = summ[summ.dataset == d].set_index("model").loc[list(MODELS)]
            fm = "accuracy_pct" if metric == "accuracy_pct" else "macro_f1_pct"
            sd = [folds[(folds.model == m) & (folds.dataset == d)][fm].std(ddof=1) for m in MODELS]
            bars = ax.bar(x + (j - 0.5) * 0.36, s[metric], 0.34, color=[C1, C2][j], label=lab, yerr=sd,
                          error_kw=dict(ecolor=INK2, lw=0.8, capsize=2), edgecolor="white", linewidth=0.8)
            for b, v in zip(bars, s[metric]):
                ax.text(b.get_x() + b.get_width() / 2, 3, f"{v:.1f}", ha="center", va="bottom", fontsize=6.2,
                        color="white", rotation=90, weight="bold")
        ax.set_xticks(x, list(MODELS))
        ax.set_ylim(0, 100)
        ax.set_title(ttl, loc="left")
    axs[0].legend(loc="lower center", bbox_to_anchor=(1.1, 1.08), ncol=2)
    fig.tight_layout()
    save(fig, "fig05_model_comparison")


def fig_bands():
    bm = pd.read_csv(RES / "M3_multiband_fusion" / "band_metrics.csv")
    lw = pd.read_csv(RES / "M3_multiband_fusion" / "learned_band_weights.csv")
    summ3 = pd.read_csv(RES / "M3_multiband_fusion" / "summary.csv").set_index("dataset")
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
    x = np.arange(5)
    for j, (d, lab) in enumerate(DATASETS.items()):
        v = bm[bm.dataset == d].set_index("band").loc[BANDS].accuracy_pct
        axs[0].bar(x + (j - 0.5) * 0.36, v, 0.34, color=[C1, C2][j], label=lab, edgecolor="white", linewidth=0.8)
        axs[0].axhline(summ3.loc[d, "accuracy_pct"], color=[C1, C2][j], lw=1, ls="--")
        w = lw[lw.dataset == d].set_index("band").loc[BANDS].weight_learned
        axs[1].bar(x + (j - 0.5) * 0.36, w, 0.34, color=[C1, C2][j], label=lab, edgecolor="white", linewidth=0.8)
    axs[0].axhline(25, color=MUTED, lw=0.8, ls=":")
    axs[0].text(4.55, 26.5, "chance", fontsize=6.5, color=MUTED, ha="right")
    axs[0].text(-0.45, summ3.iloc[0].accuracy_pct + 1.5, "fused (M3)", fontsize=6.5, color=INK2)
    axs[1].axhline(0.2, color=MUTED, lw=0.8, ls=":")
    axs[1].text(4.55, 0.207, "equal weight", fontsize=6.5, color=MUTED, ha="right")
    for ax in axs:
        ax.set_xticks(x, [BAND_SYM[b] for b in BANDS])
    axs[0].set_ylim(0, 100)
    axs[0].set_title("(a) Single-band accuracy (%)", loc="left")
    axs[1].set_title(r"(b) Learned fusion weight $w_b$", loc="left")
    axs[1].legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    save(fig, "fig06_bands_fusion")


def fig_gate_quality():
    gq = pd.read_csv(RES / "M1_fuzzy_gate" / "gate_quality.csv")
    fig, ax = plt.subplots(figsize=(6.0, 2.3))
    x = np.arange(len(gq))
    for j, (col, lab) in enumerate([("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("episode_auc", "Episode AUC")]):
        ax.bar(x + (j - 1.5) * 0.2, gq[col], 0.19, color=CAT[j], label=lab, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x, [p.capitalize() + f"\n({r:.0f}% kept)" for p, r in zip(gq.gate_policy, gq.windows_retained_pct)])
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save(fig, "fig07_gate_quality")


def fig_cm_roc(preds, model="M4"):
    p = preds[model]
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.9))
    short = ["Sad", "Happy", "Fear", "Neutral"]
    for ax, (d, lab) in zip(axs, DATASETS.items()):
        q = p[p.dataset == d]
        cm = confusion_matrix(q.y_true, q.y_pred, labels=range(4))
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for k in range(4):
                ax.text(k, i, f"{cm[i, k]}\n{100 * cmn[i, k]:.1f}%", ha="center", va="center", fontsize=6.6,
                        color="white" if cmn[i, k] > 0.55 else INK)
        ax.set_xticks(range(4), short)
        ax.set_yticks(range(4), short)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.grid(False)
        ax.set_title(f"{lab} ({model}, acc. {100 * (q.y_true == q.y_pred).mean():.1f}%)", loc="left")
    fig.tight_layout()
    save(fig, f"fig08_confusion_{model}")

    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.7), sharey=True)
    for ax, (d, lab) in zip(axs, DATASETS.items()):
        q = p[p.dataset == d]
        for k, (c, col) in enumerate(zip(CLASSES, CAT)):
            fpr, tpr, _ = roc_curve((q.y_true == k).astype(int), q[f"p_{c}"])
            ax.plot(fpr, tpr, color=col, lw=1.5, label=f"{short[k]} (AUC {auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("false positive rate")
        ax.set_title(f"{lab} ({model})", loc="left")
        ax.legend(loc="lower right", fontsize=6.6)
    axs[0].set_ylabel("true positive rate")
    fig.tight_layout()
    save(fig, f"fig09_roc_{model}")


# ------------------------------------------------------------------- fuzzy rules
LEVEL_ABBR = {"low": "L", "medium": "M", "high": "H"}


def parse_rule(text):
    out = {"PhEn-FP2": "--", "SampEn-T8": "--", "PermEn-T7": "--"}
    for feat, lvl in re.findall(r"(PhEn-FP2|SampEn-T8|PermEn-T7|Perm-FP2)\s+is\s+(low|medium|high)", text, flags=re.I):
        key = "PermEn-T7" if feat == "Perm-FP2" else feat
        out[key] = lvl.lower() + ("$^{\\dagger}$" if feat == "Perm-FP2" else "")
    return out


def fig_rules():
    r = pd.read_csv(RULES / "01_published_eeg_emotion_rules" / "all_published_rules.csv")
    emo_col = dict(zip(CLASSES, CAT))
    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    for i, b in enumerate(BANDS):
        rb = r[r.frequency_band == b].sort_values("rule_rank")
        for _, row in rb.iterrows():
            k = int(row.rule_rank) - 1
            a = parse_rule(row.if_text_original)
            code = "".join(LEVEL_ABBR.get(v.split("$")[0], "-") for v in a.values())
            ax.add_patch(Rectangle((k, 4 - i), 0.94, 0.9, fc=emo_col[row.consequent_emotion], ec="white", alpha=0.85))
            ax.text(k + 0.47, 4 - i + 0.45, code, ha="center", va="center", fontsize=7.4, color=INK, weight="bold",
                    family="monospace")
    ax.set_xlim(-0.05, 10)
    ax.set_ylim(-0.05, 5)
    ax.set_xticks(np.arange(10) + 0.47, [f"R{k}" for k in range(1, 11)])
    ax.set_yticks(np.arange(5) + 0.45, [BAND_SYM[b] for b in BANDS[::-1]])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    hs = [Rectangle((0, 0), 1, 1, fc=c, alpha=0.85) for c in CAT[:4]]
    ax.legend(hs, ["THEN sadness", "THEN happiness", "THEN fear", "THEN neutrality"], ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), fontsize=7)
    ax.set_xlabel("rule rank within band (cell code = level of PhEn-FP2 · SampEn-T8 · PermEn-T7; – = absent)", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig10_rule_map")


# ------------------------------------------------------------------ tex tables
def tex_main_table(summ, folds):
    rows = []
    for d, lab in DATASETS.items():
        for m in MODELS:
            s = summ[(summ.dataset == d) & (summ.model == m)].iloc[0]
            fa = folds[(folds.dataset == d) & (folds.model == m)].accuracy_pct
            rows.append((lab, m, s, fa.mean(), fa.std(ddof=1)))
    best = {lab: max(r[2].accuracy_pct for r in rows if r[0] == lab) for lab in DATASETS.values()}
    lines = []
    prev = None
    for lab, m, s, fm, fsd in rows:
        if prev and prev != lab:
            lines.append(r"\midrule")
        b = (lambda t: r"\textbf{" + t + "}") if s.accuracy_pct == best[lab] else (lambda t: t)
        lines.append(" & ".join([lab if prev != lab else "", m, b(f2(s.accuracy_pct)), f"{fm:.1f} $\\pm$ {fsd:.1f}",
                                 f2(s.balanced_accuracy_pct), f2(s.macro_precision_pct), f2(s.macro_recall_pct),
                                 f2(s.macro_f1_pct), f"{s.macro_auc:.3f}"]) + r" \\")
        prev = lab
    (GEN / "tab_main_results.tex").write_text("\n".join(lines) + "\n")


def tex_per_class(preds):
    lines = []
    for m in ["M0", "M4"]:
        p = preds[m]
        for d, lab in DATASETS.items():
            q = p[p.dataset == d]
            cells = []
            for k in range(4):
                tp = ((q.y_pred == k) & (q.y_true == k)).sum()
                pr = 100 * tp / max((q.y_pred == k).sum(), 1)
                rc = 100 * tp / max((q.y_true == k).sum(), 1)
                f1 = 2 * pr * rc / max(pr + rc, 1e-9)
                fpr, tpr, _ = roc_curve((q.y_true == k).astype(int), q[f"p_{CLASSES[k]}"])
                cells.append(f"{pr:.1f}/{rc:.1f}/{f1:.1f}/{auc(fpr, tpr):.3f}")
            lines.append(f"{m} & {lab} & " + " & ".join(cells) + r" \\")
        if m == "M0":
            lines.append(r"\midrule")
    (GEN / "tab_per_class.tex").write_text("\n".join(lines) + "\n")


def mcnemar_exact(a_ok, b_ok):
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return b, c, min(p, 1.0)


def tex_mcnemar(preds):
    pairs = [("M1", "M0"), ("M2", "M0"), ("M3", "M0"), ("M4", "M0"), ("M4", "M3"), ("M4", "M2")]
    lines, stats = [], {}
    for a, b in pairs:
        cells = []
        for d, lab in DATASETS.items():
            pa = preds[a][preds[a].dataset == d].sort_values("sample_id")
            pb = preds[b][preds[b].dataset == d].sort_values("sample_id")
            assert (pa.y_true.values == pb.y_true.values).all()
            ok_a = (pa.y_pred.values == pa.y_true.values)
            ok_b = (pb.y_pred.values == pb.y_true.values)
            n10, n01, pv = mcnemar_exact(ok_a, ok_b)
            dlt = 100 * (ok_a.mean() - ok_b.mean())
            ps = "$<$0.001" if pv < 1e-3 else f"{pv:.3f}"
            cells.append(f"{dlt:+.1f} & {n10}/{n01} & {ps}")
            stats[f"{a}-{b}-{lab}"] = pv
        lines.append(f"{a} vs {b} & " + " & ".join(cells) + r" \\")
    (GEN / "tab_mcnemar.tex").write_text("\n".join(lines) + "\n")
    return stats


def tex_bands_weights():
    bm = pd.read_csv(RES / "M3_multiband_fusion" / "band_metrics.csv")
    lw = pd.read_csv(RES / "M3_multiband_fusion" / "learned_band_weights.csv")
    bp = pd.read_csv(RES / "M3_multiband_fusion" / "band_probabilities.csv")
    p3 = pd.read_csv(RES / "M3_multiband_fusion" / "predictions.csv")
    lines, extra = [], {}
    for d, lab in DATASETS.items():
        acc = bm[bm.dataset == d].set_index("band").loc[BANDS].accuracy_pct
        w = lw[lw.dataset == d].set_index("band").loc[BANDS].weight_learned
        q = bp[bp.dataset == d]
        P = np.stack([q[[f"p_{b}_{c}" for c in CLASSES]].values for b in BANDS])
        eq = (P.mean(0).argmax(1) == q.y_true.values).mean() * 100
        lw_acc = (np.tensordot(w.values, P, 1).argmax(1) == q.y_true.values).mean() * 100
        m3 = p3[p3.dataset == d]
        m3acc = (m3.y_true == m3.y_pred).mean() * 100
        extra[lab] = dict(equal=eq, global_w=lw_acc, m3=m3acc)
        lines.append(f"{lab} & accuracy (\\%) & " + " & ".join(f"{v:.1f}" for v in acc) + f" & {eq:.1f} & {m3acc:.1f}" + r" \\")
        lines.append(f" & weight $w_b$ & " + " & ".join(f"{v:.3f}" for v in w) + r" & 0.200 & -- \\")
    (GEN / "tab_bands.tex").write_text("\n".join(lines) + "\n")
    return extra


def tex_gate_quality():
    gq = pd.read_csv(RES / "M1_fuzzy_gate" / "gate_quality.csv")
    names = {"none": "No gate", "hard": "Hard (M0)", "fuzzy": "Fuzzy (M1)", "adaptive": "Adaptive temporal (M2)"}
    lines = [f"{names[r.gate_policy]} & {r.precision:.3f} & {r.recall:.3f} & {r.f1:.3f} & {r.episode_auc:.3f} & {r.windows_retained_pct:.1f}"
             + r" \\" for r in gq.itertuples()]
    (GEN / "tab_gate_quality.tex").write_text("\n".join(lines) + "\n")


def tex_published_rules():
    r = pd.read_csv(RULES / "01_published_eeg_emotion_rules" / "all_published_rules.csv")
    lines = []
    for i, b in enumerate(BANDS):
        rb = r[r.frequency_band == b].sort_values("rule_rank")
        for j, row in enumerate(rb.itertuples()):
            a = parse_rule(row.if_text_original)
            first = (f"\\multirow{{10}}{{*}}{{{BAND_SYM[b]}}}" if j == 0 else "")
            lines.append(f"{first} & {row.rule_id} & {a['PhEn-FP2']} & {a['SampEn-T8']} & {a['PermEn-T7']} & {row.consequent_emotion}" + r" \\")
        if i < 4:
            lines.append(r"\midrule")
    (GEN / "tab_published_rules.tex").write_text("\n".join(lines) + "\n")
    t7 = pd.read_csv(RULES / "02_original_rule_performance" / "table7_rule_coverage_accuracy.csv")
    l7 = []
    for row in t7.itertuples():
        txt = row.rule_description.replace("If ", "")
        l7.append(f"{row.rule_rank} & {row.emotion} & {txt} & {row.coverage_pct} & {row.rule_accuracy_pct:.2f}" + r" \\")
    (GEN / "tab_rule_coverage.tex").write_text("\n".join(l7) + "\n")


# ----------------------------------------------------- mined rules (real GA run)
def mined(run_dir: Path | None):
    if run_dir is None or not (run_dir / "results.json").exists():
        return None
    res = json.loads((run_dir / "results.json").read_text())
    lines = []
    for b in ["alpha", "beta"]:
        for i, r in enumerate(res["rules"][b][:10]):
            m = re.match(r"If (.*) then class is (\w+)", r["rule"])
            ante = m.group(1).replace(" and ", " $\\wedge$ ").replace(" is ", " = ")
            first = f"\\multirow{{10}}{{*}}{{{BAND_SYM[b]}}}" if i == 0 else ""
            lines.append(f"{first} & {i + 1} & {ante} & {m.group(2)} & {100 * r['coverage']:.0f} & {100 * r['accuracy']:.1f}" + r" \\")
        if b == "alpha":
            lines.append(r"\midrule")
    (GEN / "tab_mined_rules.tex").write_text("\n".join(lines) + "\n")
    lines = []
    for b in BANDS:
        rr = res["results"][f"levels3_{b}"]
        s, fl = rr["summary"], rr["folds"]
        lines.append(" & ".join([BAND_SYM[b], f"{100 * s['accuracy']['mean']:.1f} $\\pm$ {100 * s['accuracy']['std']:.1f}",
                                 f"{100 * s['acc_lookup_only']['mean']:.1f}",
                                 f"{100 * s['f1_macro']['mean']:.1f}", f"{s['auc_macro']['mean']:.3f}",
                                 f"{np.mean([fo['n_rules_lookup'] for fo in fl]):.1f}",
                                 f"{np.mean([fo['n_rules_ga'] for fo in fl]):.1f}",
                                 f"{np.mean([fo['ga_generations'] for fo in fl]):.0f}"]) + r" \\")
    (GEN / "tab_mined_summary.tex").write_text("\n".join(lines) + "\n")
    meta = dict(windows=res["n_windows_total"], kept=res["n_windows_gated"], gate=res["gate_feature"],
                elapsed=res["elapsed_s"], gate_check=res["gate_check"])
    (GEN / "mined_meta.json").write_text(json.dumps(meta, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", type=Path, default=None)
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    summ, folds, preds = load()
    # integrity: summary accuracy must equal accuracy recomputed from predictions
    for m, p in preds.items():
        for d in DATASETS:
            q = p[p.dataset == d]
            s = summ[(summ.model == m) & (summ.dataset == d)].accuracy_pct.iloc[0]
            assert abs(100 * (q.y_true == q.y_pred).mean() - s) < 1e-6, (m, d)
    # Drawn here: architecture, S-FIS/GA schematic, model comparison (the committed
    # model-ladder images disagree with summary.csv) and the rule map. All other
    # figures in the paper are the result images under results/01_Implemented_Model.
    fig_architecture()
    fig_sfis()
    fig_accuracy(summ, folds)
    fig_rules()
    tex_main_table(summ, folds)
    tex_per_class(preds)
    st = tex_mcnemar(preds)
    ex = tex_bands_weights()
    tex_gate_quality()
    tex_published_rules()
    mined(a.mined)
    print(json.dumps({"mcnemar_p": st, "fusion": ex}, indent=1))


if __name__ == "__main__":
    main()
