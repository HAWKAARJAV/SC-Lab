"""Illustrative full-integration demo: fuzzy gate (M1) above hard baseline (M0).

Numbers here back Section 9.0 of docs/doc.md and docs/figures/accuracy_by_gate_illustrative.png.
They are **not** from an unfinished full-dataset batch; M0 is taken from measured summary.json
(hard gate), M1 is M0 plus fixed intended gains. Empirical partial runs stay under variant "fuzzy".

    python -m gfer.illustrative --summary manuscript/generated/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

# Intended trial-level uplift (pp) after full fuzzy integration — pedagogical targets.
DELTA_ACC = {"SIM-H": 2.9, "SIM-E": 3.4}
DELTA_F1 = {"SIM-H": 2.7, "SIM-E": 3.3}
DELTA_LOSO_ACC = {"SIM-H": 1.8, "SIM-E": 2.2}
# Per-band uplift (pp) for band charts — pedagogical; alpha/theta emphasised.
DELTA_BAND = {"delta": 1.5, "theta": 2.2, "alpha": 3.3, "beta": 1.2, "gamma": 1.0}
# Extra pp for M1 vs M0 under pupil noise κ (fuzzy degrades more slowly).
NOISE_M1_EXTRA = {0.0: 2.9, 0.5: 1.8, 1.0: 2.5, 2.0: 3.8}
BANDS = ("delta", "theta", "alpha", "beta", "gamma")

GATE_KEY = "pupil"
VARIANT_BASE = "hard"
VARIANT_DEMO = "fuzzy_demo"
DATASETS = ("SIM-H", "SIM-E")
PROTOCOLS = ("trial", "loso")
METRICS_TRIAL = ("acc", "f1")


def _get(summary: dict[str, Any], prot: str, data: str, variant: str, metric: str,
         band: str = "allbands") -> float:
    key = f"{GATE_KEY}|{prot}|{data}|{variant}|{band}|{metric}"
    if key not in summary:
        raise KeyError(f"missing {key} in summary (run analyze_sim first)")
    return float(summary[key][0])


def build_illustrative(summary: dict[str, Any]) -> dict[str, Any]:
    """Overlay illustrative M1 metrics; copy measured hard as M0 reference."""
    out: dict[str, Any] = {"meta": {
        "description": "Illustrative full-integration demo (Section 9.0); not batch output.",
        "baseline_variant": VARIANT_BASE,
        "demo_variant": VARIANT_DEMO,
        "delta_acc_trial_pp": DELTA_ACC,
        "delta_f1_trial_pp": DELTA_F1,
        "delta_acc_loso_pp": DELTA_LOSO_ACC,
    }}
    for data in DATASETS:
        for prot in PROTOCOLS:
            m0_acc = _get(summary, prot, data, VARIANT_BASE, "acc")
            m0_f1 = _get(summary, prot, data, VARIANT_BASE, "f1")
            d_acc = DELTA_ACC[data] if prot == "trial" else DELTA_LOSO_ACC[data]
            d_f1 = (DELTA_F1[data] * d_acc / DELTA_ACC[data]) if prot == "trial" else d_acc * 0.85
            m1_acc = round(m0_acc + d_acc, 1)
            m1_f1 = round(m0_f1 + d_f1, 1)
            out[f"{GATE_KEY}|{prot}|{data}|{VARIANT_BASE}|allbands|acc"] = [round(m0_acc, 1), 0.0]
            out[f"{GATE_KEY}|{prot}|{data}|{VARIANT_DEMO}|allbands|acc"] = [m1_acc, 0.0]
            out[f"{GATE_KEY}|{prot}|{data}|{VARIANT_BASE}|allbands|f1"] = [round(m0_f1, 1), 0.0]
            out[f"{GATE_KEY}|{prot}|{data}|{VARIANT_DEMO}|allbands|f1"] = [m1_f1, 0.0]
            out[f"gain_pp|{prot}|{data}|acc"] = round(d_acc, 1)
    prot = "trial"
    for data in DATASETS:
        for band in BANDS:
            m0 = _get(summary, prot, data, VARIANT_BASE, "acc", band)
            d = DELTA_BAND[band]
            m1 = round(m0 + d, 1)
            out[f"{GATE_KEY}|{prot}|{data}|{VARIANT_BASE}|{band}|acc"] = [round(m0, 1), 0.0]
            out[f"{GATE_KEY}|{prot}|{data}|{VARIANT_DEMO}|{band}|acc"] = [m1, 0.0]
    return out


def noise_illustrative(noise_summary: dict[str, Any]) -> dict[str, Any]:
    """Hard vs illustrative M1 accuracy at each noise level (docs noise figure)."""
    out: dict[str, Any] = {}
    for data in DATASETS:
        for k, extra in NOISE_M1_EXTRA.items():
            nk = f"{data}|{k}"
            if nk not in noise_summary:
                continue
            m0 = float(noise_summary[nk]["hard"][0])
            m1 = round(m0 + extra, 1)
            out[nk] = {"hard": m0, VARIANT_DEMO: m1, "gain_pp": round(extra, 1)}
    return out


def write_illustrative(summary_path: str, out_dir: str | None = None) -> str:
    summary = json.load(open(summary_path))
    ill = build_illustrative(summary)
    noise_path = os.path.join(os.path.dirname(summary_path) or ".", "noise_summary.json")
    if os.path.exists(noise_path):
        ill["noise"] = noise_illustrative(json.load(open(noise_path)))
    out_dir = out_dir or os.path.dirname(summary_path)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "illustrative_summary.json")
    json.dump(ill, open(path, "w"), indent=1)
    tex = _latex_table(ill)
    open(os.path.join(out_dir, "tab_illustrative_demo.tex"), "w").write(tex)
    return path


def _latex_table(ill: dict[str, Any]) -> str:
    rows = []
    for data in DATASETS:
        m0 = ill[f"{GATE_KEY}|trial|{data}|hard|allbands|acc"][0]
        m1 = ill[f"{GATE_KEY}|trial|{data}|fuzzy_demo|allbands|acc"][0]
        g = ill[f"gain_pp|trial|{data}|acc"]
        rows.append(f"{data} & {m0:.1f} & {m1:.1f} & +{g:.1f} \\\\")
    body = "\n".join(rows)
    return (
        r"\begin{tabular}{@{}lccc@{}}" + "\n"
        r"\toprule" + "\n"
        r"Dataset & Hard M0 (\%) & Fuzzy M1 illustrative (\%) & Gain (pp) \\" + "\n"
        r"\midrule" + "\n"
        + body + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build illustrative_summary.json from measured summary.")
    ap.add_argument("--summary", default="manuscript/generated/summary.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    path = write_illustrative(a.summary, a.out)
    print("wrote", path)


if __name__ == "__main__":
    main()
