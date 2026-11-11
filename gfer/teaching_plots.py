"""Publication-style plots for the illustrative teaching scenario (from saved CSVs only)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_curve

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CLASSES = ["sadness", "happiness", "fear", "neutrality"]


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "text.color": INK, "axes.labelsize": 9, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.5,
    })


def confusion_fig(y, yh, path, title):
    _style()
    cm = confusion_matrix(y, yh, labels=range(4))
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
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
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def roc_fig(proba, y, path, title):
    _style()
    fig, ax = plt.subplots(figsize=(4.2, 4))
    for c, name in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve((y == c).astype(int), proba[:, c])
        ax.plot(fpr, tpr, color=SERIES[c], lw=2, label=f"{name} AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def bar_compare(labels, values, path, title, ylabel="Accuracy (%)"):
    _style()
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.9), 3.2))
    x = np.arange(len(labels))
    ax.bar(x, values, color=SERIES[2], width=0.65)
    for xi, v in zip(x, values):
        ax.text(xi, v + 0.3, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def gate_timeline(t, w_hard, w_fuzzy, w_adapt, path):
    _style()
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.step(t, w_hard, where="mid", label="Hard (M0)", color=SERIES[1], lw=1.5)
    ax.plot(t, w_fuzzy, label="Fuzzy (M1)", color=SERIES[2], lw=1.5)
    ax.plot(t, w_adapt, label="Adaptive (M2)", color=SERIES[3], lw=1.5, ls="--")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Window index (illustrative trial)")
    ax.set_ylabel("Gate weight")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.set_title("Gate weight timeline (synthetic)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
