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
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    xs = np.linspace(0, 1, 400)
    names = ["PhEn-FP2", "SampEn-T8", "PermEn-T7"]
    cuts = [(0.15, 0.45, 0.75), (0.2, 0.5, 0.8), (0.1, 0.55, 0.85)]
    for ax, feat, (a1, c, b2) in zip(axes, names, cuts):
        muL, muM, muH = fuzzy_memberships(xs, a1, c, b2)
        for mu, lab in zip((muL, muM, muH), LEVEL_NAMES[3]):
            ax.plot(xs, mu, lw=2, label=lab.capitalize())
        ax.set_title(feat, loc="left", fontsize=10)
        ax.set_xlabel("normalised feature")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("membership μ")
    fig.suptitle("A4 — Triangular membership functions (derived methodology diagram)", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def figure_A5_rule_grid(out_path: str, dpi: int = 300) -> None:
    """27 antecedent combinations as 3×3×3 grid (consequents are learned GA outputs in paper)."""
    levels = ["Low", "Med", "High"]
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    idx = 0
    colors = []
    xs, ys, zs = [], [], []
    for i, a in enumerate(levels):
        for j, b in enumerate(levels):
            for k, c in enumerate(levels):
                xs.append(i)
                ys.append(j)
                zs.append(k)
                colors.append(idx / 27.0)
                idx += 1
    #sc = ax.scatter(xs, ys, zs, c=colors, cmap="viridis", s=120, depthshade=True, edgecolors="white", linewidths=0.5)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_zticks([0, 1, 2])
    ax.set_xticklabels(levels)
    ax.set_yticklabels(levels)
    ax.set_zticklabels(levels)
    ax.set_xlabel("PhEn-FP2")
    ax.set_ylabel("SampEn-T8")
    ax.set_zlabel("PermEn-T7")
    #fig.colorbar(sc, ax=ax, shrink=0.5, label="Rule index (1–27)")
    ax.set_title("A5 — 27 IF–THEN rule antecedents (3³); THEN → {sadness,happiness,fear,neutrality}\n"
                 "Explanatory diagram — consequents from GA (paper Tables 2–6)", fontsize=10, pad=12)
    ax.text2D(0.02, 0.02, "Not a published figure", transform=ax.transAxes, fontsize=8, color="#666")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
