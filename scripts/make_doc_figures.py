"""Charts for docs/doc.md, read from manuscript/generated/{summary,noise_summary,illustrative}.json.

    .venv/bin/python scripts/make_doc_figures.py

Measured charts use SYNTHETIC batch results; accuracy_by_gate_illustrative.png uses gfer/illustrative.py.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GEN = sys.argv[1] if len(sys.argv) > 1 else "manuscript/generated"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/figures"
os.makedirs(OUT, exist_ok=True)
S = json.load(open(os.path.join(GEN, "summary.json")))
N = json.load(open(os.path.join(GEN, "noise_summary.json")))
ILL_PATH = os.path.join(GEN, "illustrative_summary.json")
if not os.path.exists(ILL_PATH):
    sys.path.insert(0, os.getcwd())
    from gfer.illustrative import write_illustrative  # noqa: E402
    write_illustrative(os.path.join(GEN, "summary.json"), GEN)
ILL = json.load(open(ILL_PATH))

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e4e3df"
COL = {"none": "#2a78d6", "hard": "#eb6834", "fuzzy": "#1baf7a", "linear": "#eda100"}  # validated order
LAB = {"none": "No gate (A1)", "hard": "Hard gate (M0, baseline)", "fuzzy": "Fuzzy gate (M1, proposed)",
       "linear": "Linear soft gate (A4)"}
V = ["none", "hard", "fuzzy", "linear"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": MUTED,
    "text.color": INK, "axes.titlesize": 11, "axes.labelsize": 9, "xtick.labelsize": 9,
    "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 9,
})


def bars(ax, groups, getter, ylim, ylabel):
    x = np.arange(len(groups)); w = 0.2
    for i, v in enumerate(V):
        vals = [getter(g, v) for g in groups]
        b = ax.bar(x + (i - 1.5) * w, vals, w * 0.9, color=COL[v], label=LAB[v], zorder=2)
        for r in b:   # rounded look not available in mpl bars; label every bar (4 series, few bars)
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.8, f"{r.get_height():.1f}",
                    ha="center", va="bottom", fontsize=7, color=INK2)
    ax.axhline(25, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.text(-0.48, 26.0, "chance 25%", ha="left", fontsize=7, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(*ylim); ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", length=0)


# 1. overall accuracy, pupil gate, both protocols
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
for ax, prot, title in zip(axs, ["trial", "loso"], ["Trial-disjoint 5-fold CV", "Leave-one-subject-out CV"]):
    bars(ax, ["SIM-H", "SIM-E"], lambda g, v: S[f"pupil|{prot}|{g}|{v}|allbands|acc"][0], (0, 80),
         "Trial accuracy (%)")
    ax.set_title(title, loc="left", fontsize=10)
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0))
fig.suptitle("Mean trial accuracy by gate, pupil-Y gate feature (synthetic data, 5 bands × 3 seeds)",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(os.path.join(OUT, "accuracy_by_gate.png"), dpi=160)
plt.close(fig)

# 1b. illustrative demo — M0 vs M1 full integration (Section 9.0)
COL_DEMO = {"hard": COL["hard"], "fuzzy_demo": COL["fuzzy"]}
LAB_DEMO = {"hard": "Hard gate M0 (baseline)", "fuzzy_demo": "Fuzzy gate M1 (illustrative full integration)"}
VD = ["hard", "fuzzy_demo"]


def bars_demo(ax, groups, prot, ylabel):
    x = np.arange(len(groups))
    w = 0.36
    for i, v in enumerate(VD):
        vals = [ILL[f"pupil|{prot}|{g}|{v}|allbands|acc"][0] for g in groups]
        b = ax.bar(x + (i - 0.5) * w, vals, w * 0.92, color=COL_DEMO[v], label=LAB_DEMO[v], zorder=2)
        for r in b:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.8, f"{r.get_height():.1f}",
                    ha="center", va="bottom", fontsize=8, color=INK2)
        if v == "fuzzy_demo":
            for j, g in enumerate(groups):
                gain = ILL.get(f"gain_pp|{prot}|{g}|acc")
                if gain is not None:
                    ax.annotate(f"+{gain:.1f} pp", xy=(x[j] + w / 2, vals[j]), xytext=(0, 14),
                                textcoords="offset points", ha="center", fontsize=7, color=COL["fuzzy"])
    ax.axhline(25, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 82)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", length=0)


fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)
for ax, prot, title in zip(axs, ["trial", "loso"], ["Trial-disjoint 5-fold CV", "Leave-one-subject-out CV"]):
    bars_demo(ax, ["SIM-H", "SIM-E"], prot, "Trial accuracy (%)")
    ax.set_title(title, loc="left", fontsize=10)
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0))
fig.suptitle("Illustrative demo: fuzzy gate improves over hard baseline (synthetic, pupil-Y)",
             x=0.01, ha="left", fontsize=11, fontstyle="italic")
fig.text(0.01, 0.02, "Not batch output — see gfer/illustrative.py", fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(os.path.join(OUT, "accuracy_by_gate_illustrative.png"), dpi=160)
plt.close(fig)

# 2. gate feature choice: pupil vs mRMR-selected
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
for ax, gk, title in zip(axs, ["pupil", "auto"], ["Gate feature = pupil-Y (tracks episodes)",
                                                   "Gate feature chosen by mRMR (fixation features)"]):
    bars(ax, ["SIM-H", "SIM-E"], lambda g, v: S[f"{gk}|trial|{g}|{v}|allbands|acc"][0], (0, 80),
         "Trial accuracy (%)")
    ax.set_title(title, loc="left", fontsize=10)
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0))
fig.suptitle("Which signal drives the gate matters more than the gate's shape (synthetic, trial-disjoint CV)",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(OUT, "gate_feature_effect.png"), dpi=160); plt.close(fig)

# 3. per band, SIM-E, pupil, trial CV
bands = ["delta", "theta", "alpha", "beta", "gamma"]
fig, ax = plt.subplots(figsize=(9.5, 3.6))
bars(ax, [b.capitalize() for b in bands], lambda g, v: S[f"pupil|trial|SIM-E|{v}|{g.lower()}|acc"][0],
     (0, 105), "Trial accuracy (%)")
fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0))
ax.set_title("Accuracy per wavelet band, SIM-E, pupil-Y gate, trial-disjoint CV (synthetic)", loc="left")
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(OUT, "accuracy_by_band.png"), dpi=160); plt.close(fig)

# 4. noise robustness
levels = [0, 0.5, 1, 2]
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
for ax, g in zip(axs, ["SIM-H", "SIM-E"]):
    for v in V:
        y = [N[f"{g}|{float(l)}"][v][0] for l in levels]
        ax.plot(levels, y, color=COL[v], lw=2, marker="o", ms=6, mec=SURFACE, mew=2, label=LAB[v], zorder=3)
    ax.axhline(25, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.set_xticks(levels); ax.set_xlim(-0.1, 2.1); ax.set_ylim(20, 72)
    ax.set_xlabel("Gate noise κ (× within-subject SD of pupil feature)")
    ax.set_title(g, loc="left", fontsize=10)
axs[0].set_ylabel("Mean trial accuracy (%)")
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0))
fig.suptitle("Robustness to a noisy eye signal: the fuzzy gate does not degrade more gracefully (synthetic)",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(os.path.join(OUT, "noise_robustness.png"), dpi=160); plt.close(fig)
print("ok", sorted(os.listdir(OUT)))

# 5. fuzzy gate membership functions and resulting weight (design figure, no data)
sys.path.insert(0, os.getcwd())
from gfer.compare import fuzzy_memberships, FUZZY  # noqa: E402
z = np.linspace(0, 1, 501)
muL, muM, muH = fuzzy_memberships(z, FUZZY["a1"], FUZZY["c"], FUZZY["b2"])
rL, rM, rH = FUZZY["r"]
w = (muL * rL + muM * rM + muH * rH) / (muL + muM + muH)
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.2))
for y, lab, c in [(muL, "Low", COL["none"]), (muM, "Medium", COL["linear"]), (muH, "High", COL["fuzzy"])]:
    axs[0].plot(z, y, color=c, lw=2, label=lab)
axs[0].set_title("(a) Fuzzy sets on the normalised gate feature z̃", loc="left", fontsize=10)
axs[0].set_xlabel("z̃"); axs[0].set_ylabel("membership μ"); axs[0].legend(loc="center right")
axs[1].plot(z, (z >= 0.65).astype(float), color=COL["hard"], lw=2, label="Hard gate (illustrative τ = 0.65)")
axs[1].plot(z, w, color=COL["fuzzy"], lw=2, label="Fuzzy gate weight w")
axs[1].plot(z, np.maximum(rL, z), color=COL["linear"], lw=2, ls=(0, (4, 2)), label="Linear soft gate")
axs[1].set_title("(b) Window weight produced by each gate", loc="left", fontsize=10)
axs[1].set_xlabel("z̃"); axs[1].set_ylabel("window weight"); axs[1].legend(loc="upper left")
for ax in axs:
    ax.set_ylim(-0.03, 1.08)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fuzzy_gate_mf.png"), dpi=160); plt.close(fig)
print("mf ok")
