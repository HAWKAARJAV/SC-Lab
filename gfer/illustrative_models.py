"""Illustrative expected-results for M0–M4 (pedagogy only; does not touch model training).

    python -m gfer.illustrative_models --out manuscript/generated/models_illustrative.json
"""
from __future__ import annotations

import argparse
import json
import os

# Pedagogical trial-level accuracies (%); not optimisation targets.
ILLUSTRATIVE = {
    "M0_hard": {"SIM-H": 55.2, "SIM-E": 66.0},
    "M1_fuzzy": {"SIM-H": 54.9, "SIM-E": 65.7},
    "M2_adaptive": {"SIM-H": 56.8, "SIM-E": 67.5},
    "M3_fusion_five_learned": {"SIM-H": 58.5, "SIM-E": 69.1},
    "M4_integrated": {"SIM-H": 60.2, "SIM-E": 70.4},
}
GATE_ABLATION = {
    "hard": 55.2,
    "fuzzy": 54.9,
    "adaptive_nosmooth": 56.1,
    "adaptive": 56.8,
}
FUSION_ABLATION = {
    "alpha_only": 52.0,
    "alpha_beta": 54.5,
    "alpha_beta_gamma": 55.8,
    "five_equal": 57.2,
    "five_learned": 58.5,
}


def build_package() -> dict:
    return {
        "meta": {
            "note": "Hypothetical teaching scenario; run gfer.integrated / gfer.compare for measured values.",
            "models": ["M0", "M1", "M2", "M3", "M4"],
        },
        "trial_accuracy_pct": ILLUSTRATIVE,
        "gate_ablation_pct": GATE_ABLATION,
        "fusion_ablation_pct": FUSION_ABLATION,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="manuscript/generated/models_illustrative.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pkg = build_package()
    json.dump(pkg, open(a.out, "w"), indent=1)
    print(f"Wrote illustrative M0–M4 package -> {a.out}")


if __name__ == "__main__":
    main()
