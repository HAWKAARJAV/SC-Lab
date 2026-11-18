#!/usr/bin/env python3
"""Build the publication architecture figure used by the final paper."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

COLORS = {
    "eye": "#FCE8DF",
    "eeg": "#E4EDF9",
    "band": "#E7F3E8",
    "fusion": "#EEE6F8",
    "output": "#D7F0E2",
    "ink": "#25313C",
    "accent": "#D66A2C",
}


def box(ax, x, y, w, h, text, color, size=9.2, lw=1.25):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.06",
        facecolor=color, edgecolor=COLORS["ink"], linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, color=COLORS["ink"], linespacing=1.2)
    return patch


def arrow(ax, start, end, color=None, style="-|>", lw=1.35, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=12,
        linewidth=lw, color=color or COLORS["ink"],
        connectionstyle=f"arc3,rad={rad}",
    ))


def main():
    fig, ax = plt.subplots(figsize=(15.5, 7.4))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    ax.text(0.25, 7.05, "M4: Adaptive eye-gated, multi-band fuzzy EEG emotion recognition",
            fontsize=16, weight="bold", color=COLORS["ink"])
    ax.text(0.25, 6.72,
            "Eye tracking controls temporal relevance; five EEG experts provide class probabilities for learned late fusion.",
            fontsize=9.8, color="#536372")

    ax.text(0.20, 5.74, "EYE-TRACKING STREAM", fontsize=9, weight="bold",
            color=COLORS["accent"])
    eye = [
        (0.25, "Pupil +\nfixations"),
        (2.15, "8 eye\nfeatures"),
        (4.05, "mRMR + subject\nbaseline"),
        (6.20, "Fuzzy mapping +\ntemporal smoothing"),
    ]
    eye_boxes = [box(ax, x, 4.65, 1.55 if i < 2 else 1.80, 0.82, t,
                     COLORS["eye"], 8.8) for i, (x, t) in enumerate(eye)]
    for left, right in zip(eye_boxes, eye_boxes[1:]):
        arrow(ax, (left.get_x() + left.get_width(), 5.06), (right.get_x(), 5.06))
    gate = box(ax, 8.55, 4.55, 1.45, 1.02, "Adaptive gate\n$w(t)\\in[0.1,1]$",
               "#F8D2BE", 9.2, 1.6)

    ax.text(0.20, 3.63, "EEG STREAM", fontsize=9, weight="bold", color="#3B65A0")
    eeg = [
        (0.25, 1.50, "62/32-channel\nEEG"),
        (2.15, 1.65, "Filter, notch,\nCAR*, 128 Hz"),
        (4.15, 1.55, "4 s windows"),
        (6.10, 1.75, "db4 DWT\n5 bands"),
        (8.25, 1.85, "4 entropies/channel\n+ mRMR (3 inputs)"),
    ]
    eeg_boxes = [box(ax, x, 2.72, w, 0.82, t, COLORS["eeg"], 8.8)
                 for x, w, t in eeg]
    for left, right in zip(eeg_boxes, eeg_boxes[1:]):
        arrow(ax, (left.get_x() + left.get_width(), 3.13), (right.get_x(), 3.13))

    band_names = [r"$\delta$", r"$\theta$", r"$\alpha$", r"$\beta$", r"$\gamma$"]
    band_x = [0.45, 2.25, 4.05, 5.85, 7.65]
    bands = [box(ax, x, 1.08, 1.42, 0.82, f"{name} expert\nS-FIS + GA",
                 COLORS["band"], 8.6) for x, name in zip(band_x, band_names)]
    for b in bands:
        arrow(ax, (9.15, 2.72), (b.get_x() + b.get_width() / 2, 1.90),
              color="#456A91", rad=0.12)
        arrow(ax, (9.27, 4.55), (b.get_x() + b.get_width() / 2, 1.91),
              color=COLORS["accent"], lw=1.0, rad=0.08)

    fusion = box(ax, 10.15, 0.93, 2.55, 1.15,
                 "Validation-learned fusion\n$P(c)=\\sum_b\\lambda_b P_b(c)$",
                 COLORS["fusion"], 9.5, 1.55)
    for b in bands:
        arrow(ax, (b.get_x() + b.get_width(), 1.48), (10.15, 1.50), rad=-0.05)
    output = box(ax, 13.35, 0.93, 1.65, 1.15,
                 "Four-class output\nsad · happy · fear\n· neutral",
                 COLORS["output"], 8.8, 1.55)
    arrow(ax, (12.70, 1.50), (13.35, 1.50), lw=1.7)

    ax.text(10.45, 5.22, "M1", fontsize=8.5, weight="bold", color=COLORS["accent"])
    ax.text(10.86, 5.22, "continuous fuzzy gate", fontsize=8.5, color="#536372")
    ax.text(10.45, 4.82, "M2", fontsize=8.5, weight="bold", color=COLORS["accent"])
    ax.text(10.86, 4.82, "baseline + temporal adaptation", fontsize=8.5, color="#536372")
    ax.text(10.45, 4.42, "M3", fontsize=8.5, weight="bold", color="#6B4D8A")
    ax.text(10.86, 4.42, "five-band probability fusion", fontsize=8.5, color="#536372")
    ax.text(10.45, 4.02, "M4", fontsize=8.5, weight="bold", color="#2F7D55")
    ax.text(10.86, 4.02, "M2 + M3 (combined proposal)", fontsize=8.5, color="#536372")

    ax.text(0.25, 0.30,
            "* Common-average reference is applied where appropriate (MAHNOB-HCI path). "
            "Orange arrows show the temporal weight applied to each band expert.",
            fontsize=8.1, color="#536372")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "proposed_m4_architecture.pdf", bbox_inches="tight")
    fig.savefig(OUT / "proposed_m4_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
