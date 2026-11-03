"""Figures mirroring the paper: eye mRMR ranking (Fig. 7), membership functions (Fig. 5),
GA curve (Fig. 9), confusion matrices (Figs. 10/12), ROC curves (Figs. 11/13), plus a
gate illustration."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                     # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 9,
})


def eye_ranking(ranking, path):
    names = [n for n, _ in ranking][::-1]
    vals = [s for _, s in ranking][::-1]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.barh(names, vals, color=SERIES[0], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", color=INK2, fontsize=8)
    ax.set_xlabel("mRMR score (relevance - redundancy), selection order top to bottom")
    ax.set_title("Eye-tracking feature ranking (mRMR)", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15); plt.close(fig)


def membership_functions(fis, X, path):
    n = len(fis.mfs)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 2.8), squeeze=False)
    from .fuzzy import LEVEL_NAMES
    for j, (ax, mf) in enumerate(zip(axes[0], fis.mfs)):
        lo, hi = np.quantile(X[:, j], [0.0, 1.0])
        xs = np.linspace(min(lo, mf.c[0]), max(hi, mf.c[-1]), 400)
        mu = mf(xs)
        for i in range(mf.k):
            ax.plot(xs, mu[:, i], color=SERIES[i], lw=2, label=LEVEL_NAMES[mf.k][i])
        ax.set_title(fis.feature_names[j], loc="left")
        ax.set_ylim(-0.02, 1.08); ax.set_xlabel("feature value (z-scored)")
        if j == 0:
            ax.set_ylabel("membership"); ax.legend(loc="upper center", ncol=mf.k,
                                                   bbox_to_anchor=(0.5, 1.32))
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15); plt.close(fig)


def ga_curve(history, path, title):
    g = [h["gen"] for h in history]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(g, [h["best"] for h in history], color=SERIES[0], lw=2, label="best fitness")
    ax.plot(g, [h["mean"] for h in history], color=SERIES[1], lw=2, label="population mean")
    ax.set_xlabel("generation"); ax.set_ylabel("fitness")
    ax.set_title(title, loc="left"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15); plt.close(fig)


def confusion_grid(cms, class_names, path, title):
    bands = list(cms)
    fig, axes = plt.subplots(1, len(bands), figsize=(3.1 * len(bands), 3.3), squeeze=False)
    for ax, b in zip(axes[0], bands):
        cm = np.array(cms[b], dtype=float)
        pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(pct, cmap=SEQ, vmin=0, vmax=1)
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, f"{int(cm[i, j])}\n{pct[i, j]:.0%}", ha="center", va="center",
                        fontsize=7, color="#ffffff" if pct[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(cm))); ax.set_yticks(range(len(cm)))
        ax.set_xticklabels(class_names, rotation=45); ax.set_yticklabels(class_names)
        ax.set_xlabel("predicted"); ax.grid(False)
        acc = np.trace(cm) / cm.sum()
        ax.set_title(f"{b}  (acc {acc:.1%})", loc="left")
    axes[0][0].set_ylabel("true")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15); plt.close(fig)


def roc_grid(curves, class_names, path, title):
    bands = list(curves)
    fig, axes = plt.subplots(1, len(bands), figsize=(3.1 * len(bands), 3.3), squeeze=False)
    for ax, b in zip(axes[0], bands):
        for c, (fpr, tpr, auc) in enumerate(curves[b]):
            ax.plot(fpr, tpr, color=SERIES[c], lw=2, label=f"{class_names[c]} {auc:.2f}")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
        ax.set_title(b, loc="left"); ax.set_xlabel("false positive rate")
        ax.legend(loc="lower right", title="AUC", title_fontsize=8)
        ax.set_aspect("equal")
    axes[0][0].set_ylabel("true positive rate")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15); plt.close(fig)


def gate_example(times, values, keep, episodes, path, gate_name):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    for a, b in episodes or []:
        ax.axvspan(a, b, color=SERIES[3], alpha=0.18, lw=0)
    ax.plot(times, values, color=SERIES[0], lw=2, marker="o", ms=4, label=f"{gate_name} per 4-s window")
    ax.scatter(times[keep], values[keep], s=70, facecolor="none", edgecolor=SERIES[1], lw=2,
               label="window kept by gate", zorder=3)
    if episodes:
        ax.plot([], [], color=SERIES[3], alpha=0.4, lw=8, label="true emotional episode (synthetic)")
    ax.set_xlabel("time in trial (s)"); ax.set_ylabel(gate_name)
    ax.set_title("Eye gate on one trial", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15); plt.close(fig)


def accuracy_bars(rows, path, title, chance=0.25):
    """rows: list of (band, {variant: acc})"""
    variants = list(rows[0][1])
    x = np.arange(len(rows))
    w = 0.8 / len(variants)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for i, v in enumerate(variants):
        vals = [r[1][v] for r in rows]
        ax.bar(x + (i - (len(variants) - 1) / 2) * w, vals, w * 0.92, color=SERIES[i], label=v)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("mean CV accuracy"); ax.set_ylim(0, 1)
    ax.axhline(chance, color=MUTED, lw=1, ls="--")
    ax.text(len(rows) - 0.5, chance + 0.01, "chance", color=MUTED, fontsize=8, ha="right")
    ax.set_title(title, loc="left"); ax.legend(loc="upper left", ncol=len(variants))
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15); plt.close(fig)
