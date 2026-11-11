"""Publication-style plots for the illustrative teaching scenario (from saved CSVs only)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, precision_recall_fscore_support, roc_curve

from .gates import FUZZY, fuzzy_memberships

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CLASSES = ["sadness", "happiness", "fear", "neutrality"]
DPI = 300


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "text.color": INK, "axes.labelsize": 9, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.5,
    })


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def confusion_fig(y, yh, path, title):
    _style()
    cm = confusion_matrix(y, yh, labels=range(4))
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=INK, fontsize=8)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, loc="left", fontsize=10)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    _save(fig, path)


def roc_fig(proba, y, path, title):
    _style()
    fig, ax = plt.subplots(figsize=(4.4, 4))
    for c, name in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve((y == c).astype(int), proba[:, c])
        ax.plot(fpr, tpr, color=SERIES[c], lw=2, label=f"{name} AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    _save(fig, path)


def bar_compare(labels, values, path, title, ylabel="Accuracy (%)"):
    _style()
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.95), 3.4))
    x = np.arange(len(labels))
    ax.bar(x, values, color=SERIES[2], width=0.65)
    for xi, v in zip(x, values):
        ax.text(xi, v + 0.35, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10)
    _save(fig, path)


def gate_timeline(t, w_hard, w_fuzzy, w_adapt, path, title="Gate weight timeline (synthetic)"):
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.step(t, w_hard, where="mid", label="Hard (M0)", color=SERIES[1], lw=1.5)
    ax.plot(t, w_fuzzy, label="Fuzzy (M1)", color=SERIES[2], lw=1.5)
    ax.plot(t, w_adapt, label="Adaptive (M2)", color=SERIES[3], lw=1.5, ls="--")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Window index (illustrative trial)")
    ax.set_ylabel("Gate weight")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.set_title(title, loc="left")
    _save(fig, path)


def membership_m1(path):
    """B11 — M1 fuzzy membership + gate output weights."""
    _style()
    xs = np.linspace(0, 1, 400)
    muL, muM, muH = fuzzy_memberships(xs, FUZZY["a1"], FUZZY["c"], FUZZY["b2"])
    rL, rM, rH = FUZZY["r"]
    wf = (muL * rL + muM * rM + muH * rH) / np.maximum(muL + muM + muH, 1e-12)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(xs, muL, lw=2, label="μ Low", color=SERIES[0])
    ax.plot(xs, muM, lw=2, label="μ Medium", color=SERIES[2])
    ax.plot(xs, muH, lw=2, label="μ High", color=SERIES[1])
    ax.plot(xs, wf, lw=2.2, ls="--", label=f"w_gate (r={rL},{rM},{rH})", color=INK)
    ax.set_xlabel("Normalised pupil feature z")
    ax.set_ylabel("Membership / weight")
    ax.set_ylim(-0.02, 1.08)
    ax.set_title("B11 — M1 triangular memberships and defuzzified gate weight", loc="left")
    ax.legend(fontsize=7, loc="center right")
    formula = (
        "w = (μL·rL + μM·rM + μH·rH) / (μL+μM+μH);  "
        f"cuts a1={FUZZY['a1']}, c={FUZZY['c']}, b2={FUZZY['b2']}"
    )
    fig.text(0.02, 0.02, formula, fontsize=7, color=MUTED)
    _save(fig, path)


def gate_weight_histogram(weights, path):
    """B12 — distribution of continuous fuzzy gate weights."""
    _style()
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.hist(weights, bins=24, color=SERIES[2], edgecolor="white", alpha=0.9)
    ax.axvline(np.mean(weights), color=SERIES[1], ls="--", lw=1.5, label=f"mean={np.mean(weights):.3f}")
    ax.axvline(np.median(weights), color=INK, ls=":", lw=1.5, label=f"median={np.median(weights):.3f}")
    ax.set_xlabel("Fuzzy gate weight w")
    ax.set_ylabel("Count")
    ax.set_title("B12 — Fuzzy gate weight distribution (synthetic windows)", loc="left")
    ax.legend(fontsize=7)
    _save(fig, path)


def gate_ablation_bands(band_names, series: dict, path):
    """B13 — per-band accuracy under no / hard / fuzzy gate selection."""
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(band_names))
    w = 0.25
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - 1) * w, vals, width=w, label=name, color=SERIES[i % 4])
    ax.set_xticks(x)
    ax.set_xticklabels(band_names)
    ax.set_ylabel("Band argmax accuracy (%)")
    ax.set_title("B13 — Per-band accuracy vs gate policy (illustrative)", loc="left")
    ax.legend(fontsize=7)
    _save(fig, path)


def gate_quality_bars(metrics: dict, path, title):
    """B14 / C19 — precision, recall, F1 for gate policies."""
    _style()
    labels = list(metrics.keys())
    keys = ["precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    x = np.arange(len(labels))
    w = 0.22
    for j, k in enumerate(keys):
        ax.bar(x + (j - 1) * w, [metrics[g][k] * 100 for g in labels], width=w, label=k.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in labels])
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title, loc="left")
    ax.legend(fontsize=7)
    _save(fig, path)


def pupil_raw_norm(t, raw, norm, path):
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.plot(t, raw, "o-", ms=3, lw=1.2, label="Raw pupil (mm)", color=SERIES[0])
    ax2 = ax.twinx()
    ax2.plot(t, norm, "s-", ms=2.5, lw=1.2, label="Trial-normalised z", color=SERIES[1])
    ax.set_xlabel("Window")
    ax.set_ylabel("Pupil diameter")
    ax2.set_ylabel("Normalised feature")
    ax.set_title("C15 — Raw vs participant-baseline-normalised pupil", loc="left")
    lines, labs = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labs + lab2, fontsize=7, loc="upper right")
    _save(fig, path)


def smoothing_effect(t, before, after, path):
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.plot(t, before, lw=1.2, label="Fuzzy weights (pre-smooth)", color=SERIES[1], alpha=0.85)
    ax.plot(t, after, lw=2, label="Temporally smoothed", color=SERIES[2])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Window")
    ax.set_ylabel("Gate score")
    ax.set_title("C16 — Temporal smoothing reduces isolated spikes", loc="left")
    ax.legend(fontsize=7)
    _save(fig, path)


def adaptive_timeline_panel(t, pupil, norm, smooth, adapt, path):
    _style()
    fig, axes = plt.subplots(4, 1, figsize=(7, 5.5), sharex=True)
    panels = [
        (pupil, "Pupil (mm)", SERIES[0]),
        (norm, "Normalised z", SERIES[1]),
        (smooth, "Smoothed fuzzy feature", SERIES[2]),
        (adapt, "Adaptive gate weight", SERIES[3]),
    ]
    for ax, (data, ylab, col) in zip(axes, panels):
        ax.plot(t, data, color=col, lw=1.5)
        ax.set_ylabel(ylab, fontsize=8)
    axes[-1].set_xlabel("Window")
    fig.suptitle("C17 — Adaptive temporal fuzzy gate components", x=0.02, ha="left", fontsize=10)
    _save(fig, path)


def gate_quality_extended(metrics: dict, path):
    """C19 — precision/recall/F1 + windows retained; episode AUC placeholder from F1."""
    _style()
    labels = list(metrics.keys())
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(labels))
    w = 0.18
    series = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("windows_retained_pct", "Windows retained"),
        ("episode_auc", "Episode-detection AUC"),
    ]
    for j, (k, lab) in enumerate(series):
        vals = [metrics[g][k] * (100 if k != "episode_auc" else 1) for g in labels]
        if k == "episode_auc":
            vals = [v * 100 for v in vals]
        ax.bar(x + (j - 2) * w, vals, width=w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in labels])
    ax.set_ylabel("Score (%)")
    ax.set_title("C19 — Gate quality comparison (synthetic episode labels)", loc="left")
    ax.legend(fontsize=6, ncol=2)
    _save(fig, path)


def noise_sensitivity_dual(noise_levels, gate_curves: dict, emo_curves: dict, path):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    for ax, curves, ylab, sub in [
        (axes[0], gate_curves, "Gate F1 (%)", "Gate quality"),
        (axes[1], emo_curves, "Emotion accuracy (%)", "Emotion classification"),
    ]:
        for name, ys in curves.items():
            ax.plot(noise_levels, ys, "o-", lw=2, label=name)
        ax.set_xlabel("Pupil noise σ (mm)")
        ax.set_ylabel(ylab)
        ax.set_title(sub, loc="left", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("C20 — Noise sensitivity (synthetic)", x=0.02, ha="left", fontsize=10)
    _save(fig, path)


def band_weights_bar(bands, weights, path):
    _style()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    x = np.arange(len(bands))
    ax.bar(x, weights, color=SERIES[0], width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel("Learned weight")
    ax.set_ylim(0, max(weights) * 1.25)
    ax.set_title(f"D22 — Learned band weights (Σ={sum(weights):.4f})", loc="left")
    for xi, v in zip(x, weights):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    _save(fig, path)


def fusion_strategy_compare(labels, accs, path, title):
    bar_compare(labels, accs, path, title)


def architecture_e28(path):
    _style()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    lines = [
        "E28 — Combined model architecture (illustrative diagram)",
        "",
        "Eye track + EEG  →  4 s windows",
        "  M1/M2: 8 eye features → pupil baseline norm → temporal smooth → adaptive fuzzy gate",
        "  M3: EEG preprocess → wavelet → 5 bands → 4 entropy features → mRMR → 5× Sugeno FIS",
        "  Fusion: validation-learned band weights → 4-class emotion probabilities",
        "  M4: M2 gate + M3 fusion → final sadness / happiness / fear / neutrality",
    ]
    y = 0.88
    for line in lines:
        weight = "bold" if line.startswith("E28") else "normal"
        ax.text(0.03, y, line, fontsize=11 if "E28" in line else 9, weight=weight, family="monospace")
        y -= 0.08
    boxes = [
        (0.05, 0.42, "M0 Hard gate", "#fde8e0"),
        (0.28, 0.42, "M1 Fuzzy gate", "#e8f4e8"),
        (0.51, 0.42, "M2 Adaptive gate", "#e8eef8"),
        (0.05, 0.18, "M3 Multi-band fusion", "#f0e8f8"),
        (0.42, 0.18, "M4 Combined", "#fff6da"),
    ]
    for x, yb, txt, col in boxes:
        rect = plt.Rectangle((x, yb), 0.2, 0.12, facecolor=col, edgecolor=MUTED)
        ax.add_patch(rect)
        ax.text(x + 0.01, yb + 0.04, txt, fontsize=8)
    _save(fig, path)


def m4_gate_timeline(t, w, path):
    _style()
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.fill_between(t, 0, w, alpha=0.25, color=SERIES[2])
    ax.plot(t, w, color=SERIES[2], lw=1.8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Window (M4 pooled illustrative trial)")
    ax.set_ylabel("Adaptive gate weight")
    ax.set_title("E29 — Adaptive gate weights in combined model", loc="left")
    _save(fig, path)


def m4_band_contribution(bands, weights, band_probs_mean, path):
    """E30 — mean class probability mass per band weighted by learned weights."""
    _style()
    contrib = weights[:, None] * band_probs_mean
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(bands))
    bottom = np.zeros(len(bands))
    for c, cls in enumerate(CLASSES):
        ax.bar(x, contrib[:, c], bottom=bottom, label=cls, color=SERIES[c])
        bottom += contrib[:, c]
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel("Weighted probability contribution")
    ax.set_title("E30 — Band-specific contributions to fused output (M4)", loc="left")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, path)


def classwise_grouped(y, y_pred, path):
    """E33 — per-class precision, recall, F1."""
    _style()
    p, r, f1, _ = precision_recall_fscore_support(y, y_pred, labels=range(4), zero_division=0)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(4)
    w = 0.22
    for j, (vals, lab) in enumerate([(p * 100, "Precision"), (r * 100, "Recall"), (f1 * 100, "F1")]):
        ax.bar(x + (j - 1) * w, vals, width=w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=20, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("E33 — Class-wise precision / recall / F1 (M4)", loc="left")
    ax.legend(fontsize=7)
    _save(fig, path)
