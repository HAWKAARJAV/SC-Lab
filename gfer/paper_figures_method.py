"""Derived methodology diagrams A4–A5 for paper_reference (not numbered paper figures)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .gates import FUZZY, fuzzy_memberships
from .fuzzy import LEVEL_NAMES


def figure_A4_membership(out_path: str, dpi: int = 300) -> None:
    """Triangular MFs for three EEG inputs (Low / Medium / High)."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    xs = np.linspace(0, 1, 400)
    names = ["PhEn-FP2", "SampEn-T8", "PermEn-T7"]
    cuts = [(0.15, 0.45, 0.75), (0.2, 0.5, 0.8), (0.1, 0.55, 0.85)]
    colors = ["#2a78d6", "#1baf7a", "#eb6834"]
    for ax, feat, (a1, c, b2), col in zip(axes, names, cuts, colors):
        muL, muM, muH = fuzzy_memberships(xs, a1, c, b2)
        for mu, lab in zip((muL, muM, muH), LEVEL_NAMES[3]):
            ax.plot(xs, mu, lw=2, label=lab.capitalize())
        ax.set_title(feat, loc="left", fontsize=10)
        ax.set_xlabel("normalised feature")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("membership μ")
    fig.suptitle("A4 — Triangular membership functions (methodology diagram, derived)", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def figure_A5_rule_grid(out_path: str, dpi: int = 300) -> None:
    """27 rule combinations = 3³ antecedent patterns → 4 class outputs (schematic)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    levels = ["Low", "Medium", "High"]
    y = 0.92
    ax.text(0.02, y, "3 inputs × 3 levels = 27 IF–THEN rules (Sugeno, class consequents)", fontsize=11, weight="bold")
    y -= 0.06
    for i, feat in enumerate(["PhEn-FP2", "SampEn-T8", "PermEn-T7"]):
        ax.text(0.05, y - i * 0.08, f"{feat}:  " + "  |  ".join(levels), fontsize=9, family="monospace")
    y = 0.55
    rules = []
    idx = 1
    for a in levels:
        for b in levels:
            for c in levels:
                rules.append(f"R{idx:02d}: IF {a},{b},{c} THEN class ∈ {{sadness,happiness,fear,neutrality}}")
                idx += 1
    col1, col2, col3 = rules[:9], rules[9:18], rules[18:]
    for j, chunk in enumerate([col1, col2, col3]):
        x = 0.05 + j * 0.32
        for k, line in enumerate(chunk):
            ax.text(x, 0.48 - k * 0.045, line, fontsize=6.2, family="monospace")
    ax.text(0.02, 0.02, "A5 — Fuzzy rule combination space (explanatory; not a published figure)", fontsize=8, color="#666")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
