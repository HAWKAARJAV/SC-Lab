#!/usr/bin/env python3
"""Build docs/fuzzy_rulebook/ and docs/architecture/ (no new experiments)."""
from __future__ import annotations

import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gfer.gates import FUZZY, fuzzy_memberships
from gfer.paper_reference_data import CLASSES, PAPER, RULE_SETS, TABLE_7_RULES

RULEBOOK = os.path.join(ROOT, "docs", "fuzzy_rulebook")
ARCH = os.path.join(ROOT, "docs", "architecture")
BAND_TABLE_PAGE = {"delta": 11, "theta": 11, "alpha": 12, "beta": 12, "gamma": 12}
EMOTION_MAP = {"sad": "sadness", "happy": "happiness", "fear": "fear", "neutral": "neutrality"}

LEVELS = ["low", "medium", "high"]


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _parse_antecedents(text: str) -> tuple[str, str, str, str]:
    """Return phEn, sampEn, permEn, source_note for partial/spelling variants."""
    note = ""
    t = text.strip()
    if "If " in t[:4]:
        t = t[3:].strip()
    p1 = p2 = p3 = ""
    m = re.search(r"PhEn[-\s]?FP2\s+is\s+(\w+)", t, re.I)
    if m:
        p1 = m.group(1).lower()
    m = re.search(r"PHFP2\s+is\s+(\w+)", t, re.I)
    if m:
        p1 = m.group(1).lower()
        note += "Paper uses PHFP2 spelling. "
    m = re.search(r"PhEn[-\s]?T7\s+is\s+(\w+)", t, re.I)
    if m:
        note += "Paper uses PhEn-T7 (not FP2/T8). "
    m = re.search(r"SampEn[-\s]?T8\s+is\s+(\w+)", t, re.I)
    if m:
        p2 = m.group(1).lower()
    m = re.search(r"sampT8\s+(\w+)", t, re.I)
    if m:
        p2 = m.group(1).lower()
        note += "Paper uses sampT8 spelling. "
    m = re.search(r"sampEn[-\s]?T8\s+(\w+)", t, re.I)
    if m:
        p2 = m.group(1).lower()
    m = re.search(r"PermEn[-\s]?T7\s+is\s+(\w+)", t, re.I)
    if m:
        p3 = m.group(1).lower()
    m = re.search(r"Perm[-\s]?FP2\s+is\s+(\w+)", t, re.I)
    if m:
        p3 = m.group(1).lower()
        note += "Paper gamma rule uses Perm-FP2 (not PermEn-T7). "
    m = re.search(r"PeFP2\s+is\s+(\w+)", t, re.I)
    if m:
        note += "Table 7 uses PeFP2 (table7 only). "
    m = re.search(r"phT7\s+Is\s+(\w+)", t, re.I)
    if m:
        note += "Table 7 uses phT7 shorthand. "
    return p1, p2, p3, note.strip()


def export_published_rules():
    d = os.path.join(RULEBOOK, "01_published_eeg_emotion_rules")
    cols = [
        "rule_id", "dataset", "frequency_band", "rule_rank", "phEn_fp2", "sampEn_t8", "permEn_t7",
        "consequent_emotion", "source_table", "source_page", "provenance", "source_note", "if_text_original",
    ]
    all_rows = []
    for band, rules in RULE_SETS.items():
        rows = []
        tnum = {"delta": 2, "theta": 3, "alpha": 4, "beta": 5, "gamma": 6}[band]
        for rank, iff, then in rules:
            p1, p2, p3, note = _parse_antecedents(iff)
            row = {
                "rule_id": f"{band.upper()[:1]}{rank:02d}",
                "dataset": "SEED-IV (paper Tables 2–6)",
                "frequency_band": band,
                "rule_rank": rank,
                "phEn_fp2": p1,
                "sampEn_t8": p2,
                "permEn_t7": p3,
                "consequent_emotion": EMOTION_MAP.get(then, then),
                "source_table": f"Table {tnum}",
                "source_page": BAND_TABLE_PAGE[band],
                "provenance": "published_paper_transcription",
                "source_note": note,
                "if_text_original": iff,
            }
            rows.append(row)
            all_rows.append(row)
        write_csv(os.path.join(d, f"{band}_rules.csv"), rows, cols)
    write_csv(os.path.join(d, "all_published_rules.csv"), all_rows, cols)

    # 27 candidates (no consequents assigned)
    cand = []
    idx = 1
    for a in LEVELS:
        for b in LEVELS:
            for c in LEVELS:
                cand.append({
                    "rule_index": idx,
                    "phEn_fp2": a,
                    "sampEn_t8": b,
                    "permEn_t7": c,
                    "consequent_emotion": "",
                    "role": "initial_candidate_antecedent_only",
                    "provenance": "paper_methodology_3x3x3_space",
                    "note": "Not a published final rule unless selected by lookup-table/GA",
                })
                idx += 1
    write_csv(os.path.join(d, "candidate_27_combinations.csv"), cand,
              ["rule_index", "phEn_fp2", "sampEn_t8", "permEn_t7",
               "consequent_emotion", "role", "provenance", "note"])

    expl = """# Published EEG emotion rules (S-FIS)

Source: Farhadi Sedehi et al., *IEEE Access* 2024 — Tables 2–6 (SEED-IV rule mining).

- **50 rules** total: 10 per frequency band (δ, θ, α, β, γ).
- Inputs: **PhEn-FP2**, **SampEn-T8**, **PermEn-T7** (three levels each).
- **27 combinations** in `candidate_27_combinations.csv` are *candidate antecedents only* — not all are retained classifiers.
- Spelling/channel variants in the paper are preserved in `source_note` and `if_text_original`.
- This is the **EEG emotion classification FIS** — separate from the **eye-gating FIS** in `03_eye_gating_rules/`.
"""
    with open(os.path.join(d, "published_rule_explanation.md"), "w", encoding="utf-8") as f:
        f.write(expl)


def export_table7():
    d = os.path.join(RULEBOOK, "02_original_rule_performance")
    rows = []
    for i, r in enumerate(TABLE_7_RULES, 1):
        rows.append({
            "emotion": r["emotion"],
            "rule_rank": i,
            "rule_description": r["rule"],
            "coverage_pct": r["coverage_pct"],
            "rule_accuracy_pct": r["accuracy_pct"],
            "source_table": "Table 7",
            "source_page": 12,
        })
    write_csv(os.path.join(d, "table7_rule_coverage_accuracy.csv"), rows,
              ["emotion", "rule_rank", "rule_description", "coverage_pct",
               "rule_accuracy_pct", "source_table", "source_page"])
    with open(os.path.join(d, "rule_performance_explanation.md"), "w", encoding="utf-8") as f:
        f.write("""# Table 7 — rule coverage and accuracy

Table 7 lists **selected rules with reported coverage and accuracy** — not the same listing as Tables 2–6 band top-10 sets.

Do **not** join Table 7 metrics onto every Tables 2–6 rule without an explicit paper-stated match.
Some Table 7 descriptions use shorthand spellings (e.g. phT7, PeFP2); see `rule_description` verbatim.
""")


def export_eye_gates():
    d = os.path.join(RULEBOOK, "03_eye_gating_rules")
    rL, rM, rH = FUZZY["r"]
    rows_m1 = [
        {"rule_id": 1, "input_feature": "mean_pupil_diameter_Y (trial-normalised z)",
         "linguistic_level": "Low", "membership_note": f"triangular MF a1={FUZZY['a1']}, c={FUZZY['c']}, b2={FUZZY['b2']}",
         "then_gate_weight": rL, "provenance": "gfer/gates.py FUZZY", "system": "eye_gating_not_emotion"},
        {"rule_id": 2, "input_feature": "mean_pupil_diameter_Y (trial-normalised z)",
         "linguistic_level": "Medium", "membership_note": "see μM triangular overlap",
         "then_gate_weight": rM, "provenance": "gfer/gates.py FUZZY", "system": "eye_gating_not_emotion"},
        {"rule_id": 3, "input_feature": "mean_pupil_diameter_Y (trial-normalised z)",
         "linguistic_level": "High", "membership_note": "see μH triangular overlap",
         "then_gate_weight": rH, "provenance": "gfer/gates.py FUZZY", "system": "eye_gating_not_emotion"},
    ]
    write_csv(os.path.join(d, "m1_fuzzy_eye_gate_rules.csv"), rows_m1,
              ["rule_id", "input_feature", "linguistic_level", "membership_note",
               "then_gate_weight", "provenance", "system"])
    rows_m2 = rows_m1.copy()
    for r in rows_m2:
        r["input_feature"] = "baseline-normalised & temporally smoothed pupil feature S(t)"
        r["provenance"] = "gfer/adaptive_eye.py + gfer/gates.py"
    write_csv(os.path.join(d, "m2_adaptive_eye_gate_rules.csv"), rows_m2,
              ["rule_id", "input_feature", "linguistic_level", "membership_note",
               "then_gate_weight", "provenance", "system"])
    expl = f"""# Eye-gating fuzzy system (M1 / M2)

**Not an emotion classifier.** Outputs window importance weight `w(t) ∈ [{rL}, 1]`.

## M1 — `gfer/gates.py`

Defuzzification (Sugeno-style weighted average of level outputs):

`w(t) = (μL·rL + μM·rM + μH·rH) / (μL + μM + μH)`

With **rL={rL}, rM={rM}, rH={rH}** and triangular cuts **a1={FUZZY['a1']}, c={FUZZY['c']}, b2={FUZZY['b2']}** on trial-normalised pupil **z**.

## M2 — `gfer/adaptive_eye.py`

Pipeline before the same fuzzy mapping:

1. `μ_subject`, `σ_subject` from **training windows only** (`participant_baselines`).
2. Baseline-correct and robust scale: `normalize_pupil_feature` → **z̃ ∈ [0,1]**.
3. Per-subject adaptive MF percentiles (`adaptive_fuzzy_params`) on training data.
4. `adaptive_fuzzy_weights` → optional **temporal smoothing**:
   - **symmetric** (offline teaching default): neighbourhood mean over window index.
   - **causal** (`adaptive_causal`): uses past windows only (real-time compatible).

Smoothing does not add new IF-THEN rules; it filters the fuzzy gate score time series.
"""
    with open(os.path.join(d, "eye_gate_rule_explanation.md"), "w", encoding="utf-8") as f:
        f.write(expl)


def export_model_rules():
    d = os.path.join(RULEBOOK, "04_model_specific_rules")
    base = "Published Tables 2–6 per-band S-FIS+GA (same rule schema as paper)"
    models = {
        "M0": ("Hard quantile gate (q=0.6)", "Per-band S-FIS; evaluate bands separately", "None", base, "gfer/gates.hard_weights"),
        "M1": ("Fuzzy eye gate (continuous w)", "Per-band S-FIS", "None", base, "gfer/gates.fuzzy_weights"),
        "M2": ("Adaptive temporal fuzzy gate", "Per-band S-FIS (synthetic demo uses α/βγ mix)", "None", base, "gfer/adaptive_eye.compute_gate_weights"),
        "M3": ("Hard gate (M0 baseline)", "Five band classifiers → weighted fusion", "Validation-learned w_b", base, "gfer/fusion.py"),
        "M4": ("Adaptive gate (M2)", "Five band classifiers → weighted fusion", "Validation-learned w_b", base, "gfer/fusion.py + gfer/adaptive_eye.py"),
    }
    cols = ["model", "eye_gate", "eeg_classification", "multiband_fusion",
            "eeg_rule_source", "eye_rule_source", "ga_optimisation", "difference_vs_m0", "provenance"]
    for m, (gate, clf, fus, eeg, code) in models.items():
        rows = [{
            "model": m,
            "eye_gate": gate,
            "eeg_classification": clf,
            "multiband_fusion": fus,
            "eeg_rule_source": eeg,
            "eye_rule_source": "03_eye_gating_rules/m1" if m in ("M1", "M2", "M4") else ("hard threshold" if m != "M3" else "hard threshold (M3)"),
            "ga_optimisation": "Paper GA on 27-population rule chromosomes (gfer/ga.py tune_rules)" if "S-FIS" in clf or "band" in clf else "",
            "difference_vs_m0": "Same EEG rules as paper; gate/fusion per column",
            "provenance": "documented_architecture" if m == "M0" else f"SC-Lab extension; code: {code}",
        }]
        write_csv(os.path.join(d, f"{m}_rules.csv"), rows, cols)


def export_source_mapping():
    sm = os.path.join(RULEBOOK, "source_mapping")
    paper_rows = []
    for band, rules in RULE_SETS.items():
        tnum = {"delta": 2, "theta": 3, "alpha": 4, "beta": 5, "gamma": 6}[band]
        for rank, iff, then in rules:
            paper_rows.append({
                "rule_id": f"{band}-{rank}",
                "source_type": "published_table",
                "source_location": f"Table {tnum}, p.{BAND_TABLE_PAGE[band]}",
                "if_text": iff,
                "then_emotion": EMOTION_MAP.get(then, then),
            })
    write_csv(os.path.join(sm, "paper_rule_sources.csv"), paper_rows,
              ["rule_id", "source_type", "source_location", "if_text", "then_emotion"])
    code_rows = [
        {"module": "gfer/fuzzy.py", "artifact": "SugenoFIS.wang_mendel", "rule_type": "EEG emotion (mined)", "status": "implemented"},
        {"module": "gfer/ga.py", "artifact": "tune_rules", "rule_type": "GA rule optimisation", "status": "implemented"},
        {"module": "gfer/gates.py", "artifact": "fuzzy_weights", "rule_type": "Eye gate Sugeno outputs", "status": "implemented"},
        {"module": "gfer/adaptive_eye.py", "artifact": "compute_gate_weights", "rule_type": "M2 adaptive gate", "status": "implemented"},
        {"module": "gfer/fusion.py", "artifact": "fuse_weighted", "rule_type": "M3/M4 fusion (not FIS rules)", "status": "implemented"},
        {"module": "gfer/teaching_scenario.py", "artifact": "fuzzy_rules.csv", "rule_type": "Illustrative placeholder rules", "status": "illustrative_only"},
        {"module": "gfer/paper_reference_data.py", "artifact": "RULE_SETS", "rule_type": "Published transcription", "status": "reference_data"},
    ]
    write_csv(os.path.join(sm, "code_rule_sources.csv"), code_rows,
              ["module", "artifact", "rule_type", "status"])


def rule_visualisations():
    out = os.path.join(RULEBOOK, "05_rule_visualisations")
    os.makedirs(out, exist_ok=True)
    # 27 combinations
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    ax.set_title("27 candidate antecedent combinations (3³)\nno emotion consequent assigned", fontsize=11)
    idx = 1
    y = 0.95
    for a in ["Low", "Med", "High"]:
        for b in ["Low", "Med", "High"]:
            for c in ["Low", "Med", "High"]:
                ax.text(0.05 + (idx - 1) % 3 * 0.32, y - ((idx - 1) // 3) * 0.11,
                        f"R{idx:02d}: {a}, {b}, {c}", fontsize=8, family="monospace")
                idx += 1
    fig.savefig(os.path.join(out, "27_rule_combinations.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(out, "27_rule_combinations.svg"), bbox_inches="tight")
    plt.close(fig)

    # Eye gate MFs
    xs = np.linspace(0, 1, 200)
    muL, muM, muH = fuzzy_memberships(xs, FUZZY["a1"], FUZZY["c"], FUZZY["b2"])
    rL, rM, rH = FUZZY["r"]
    w = (muL * rL + muM * rM + muH * rH) / np.maximum(muL + muM + muH, 1e-12)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(xs, muL, label="μ Low")
    ax.plot(xs, muM, label="μ Medium")
    ax.plot(xs, muH, label="μ High")
    ax.plot(xs, w, "--", color="black", label="w gate")
    ax.set_title("Eye-gating fuzzy memberships & defuzzified weight (M1)")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(out, "eye_gate_fuzzy_rules.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(out, "eye_gate_fuzzy_rules.svg"), bbox_inches="tight")
    plt.close(fig)

    # EEG structure schematic
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.05, 0.85, "EEG S-FIS (per band): 3 inputs → 27 candidates → GA → ≤10 rules → 4 emotions", fontsize=11, weight="bold")
    ax.text(0.05, 0.65, "PhEn-FP2 (L/M/H)  ×  SampEn-T8 (L/M/H)  ×  PermEn-T7 (L/M/H)", fontsize=10)
    ax.text(0.05, 0.45, "Eye gate (separate): pupil_y → L/M/H → w(t) ∈ [0.1, 1.0]", fontsize=10, color="#c45c00")
    fig.savefig(os.path.join(out, "eeg_fuzzy_rule_structure.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(out, "eeg_fuzzy_rule_structure.svg"), bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, title, col in zip(axes, ["Published EEG FIS", "SC-Lab eye-gating FIS"], ["#2a78d6", "#eb6834"]):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0.1, 0.3), 0.8, 0.4, facecolor=col, alpha=0.2, edgecolor=col))
        ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=11)
    fig.suptitle("Two separate fuzzy systems", fontsize=12)
    fig.savefig(os.path.join(out, "original_vs_improved_fuzzy_systems.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(out, "original_vs_improved_fuzzy_systems.svg"), bbox_inches="tight")
    plt.close(fig)

    # emotion examples from alpha table
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    ax.set_title("Example published rules (alpha band, Table 4)", fontsize=11)
    for i, (rank, iff, then) in enumerate(RULE_SETS["alpha"][:5]):
        ax.text(0.05, 0.85 - i * 0.15, f"{rank}. {iff} → {EMOTION_MAP.get(then, then)}", fontsize=9, family="monospace")
    fig.savefig(os.path.join(out, "emotion_rule_examples.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(out, "emotion_rule_examples.svg"), bbox_inches="tight")
    plt.close(fig)


def write_rulebook_readme():
    with open(os.path.join(RULEBOOK, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"""# Fuzzy IF–THEN rulebook (SC-Lab)

Regenerate: `.venv/bin/python scripts/build_fuzzy_rulebook_and_architecture.py`

## Two fuzzy systems

| System | Purpose | Location |
|--------|---------|----------|
| **EEG S-FIS** | Emotion classification (sadness, happiness, fear, neutrality) | `01_published_eeg_emotion_rules/` |
| **Eye gate FIS** | Window importance weight `w(t)` | `03_eye_gating_rules/` |

Paper DOI: {PAPER['citation']['doi']}

## Contents

- **50 published EEG rules** (Tables 2–6) — not invented.
- **Table 7** performance metrics — separate from Tables 2–6.
- **27 candidate antecedents** — combinations only, no fabricated consequents.
- **M0–M4** model rule *roles* — see `04_model_specific_rules/`.
""")


def mermaid_sources():
    os.makedirs(ARCH, exist_ok=True)
    diagrams = {
        "01_original_paper_architecture": """flowchart LR
  subgraph EYE["Eye branch (paper M0)"]
    E1[8 eye features / 4s window] --> E2[mRMR]
    E2 --> E3[mean pupil Y]
    E3 --> E4[Hard gate 0/1]
  end
  subgraph EEG["EEG branch"]
    G1[Multichannel EEG] --> G2[4s segment]
    G2 --> G3[Filter + notch + CAR + 128 Hz]
    G3 --> G4[DWT db4]
    G4 --> G5[5 bands δ θ α β γ]
    G5 --> G6[4 entropies × channels]
    G6 --> G7[mRMR → 3 features]
    G7 --> G8[S-FIS 27 combos + GA]
    G8 --> G9[Per-band emotion class]
  end
  E4 -.->|gate masks window| G2
""",
        "02_M1_fuzzy_eye_gate": """flowchart LR
  E[Eye features] --> M[mRMR pupil Y]
  M --> Z[trial-normalised z]
  Z --> F[μL μM μH]
  F --> W["w∈[0.1,1]"]
  W -.-> EEG[EEG band S-FIS unchanged]
  H[Hard 0/1] -.->|replaced| F
""",
        "03_M2_adaptive_temporal_gate": """flowchart TB
  P[Pupil raw] --> B[Train-only baseline]
  B --> Z["z=scaled baseline-correct"]
  Z --> S["Temporal smooth S(t)"]
  S --> F[Fuzzy L/M/H]
  F --> W[Adaptive w(t)]
  W -.-> WIN[EEG windows]
""",
        "04_M3_multiband_fusion": """flowchart TB
  G[Hard eye gate] --> EEG[5-band preprocessing]
  EEG --> C1[δ FIS+GA]
  EEG --> C2[θ FIS+GA]
  EEG --> C3[α FIS+GA]
  EEG --> C4[β FIS+GA]
  EEG --> C5[γ FIS+GA]
  C1 & C2 & C3 & C4 & C5 --> FUSE["P(c)=Σ w_b P_b(c)"]
  FUSE --> OUT[4 emotions]
""",
        "05_M4_combined_proposed_architecture": """flowchart TB
  subgraph EYE2["M2 eye branch"]
    EYE2a[Eye track] --> EYE2b[Baseline + smooth]
    EYE2b --> EYE2c[Adaptive fuzzy w]
  end
  subgraph EEG2["M3 EEG branch"]
    EEG2a[EEG] --> EEG2b[5× S-FIS+GA]
    EEG2b --> EEG2c[Band probabilities]
  end
  EYE2c -.->|window weight| EEG2b
  EEG2c --> FUSE2[Learned fusion]
  FUSE2 --> OUT2[Emotion output]
""",
        "06_M0_to_M4_model_evolution": """flowchart LR
  M0[M0 hard + per-band] --> M1[M1 fuzzy gate]
  M1 --> M2[M2 adaptive gate]
  M0 --> M3[M3 hard + fusion]
  M2 --> M4[M4 adaptive + fusion]
""",
        "07_parallel_data_flow": """sequenceDiagram
  participant Eye
  participant EEG
  participant Sync
  Eye->>Sync: timestamps / 4s windows
  EEG->>Sync: aligned segments
  Sync->>Eye: eye features + gate w(t)
  Sync->>EEG: preprocess + band FIS
  Note over Eye,EEG: Gate weights conceptually modulate window evidence; code may still extract features then weight samples
  EEG->>Sync: band probabilities
  Sync->>Sync: weighted fusion (M3/M4)
""",
    }
    for name, body in diagrams.items():
        path = os.path.join(ARCH, f"{name}.mmd")
        with open(path, "w", encoding="utf-8") as f:
            f.write("---\nconfig:\n  theme: neutral\n---\n" + body)


def render_architecture_png_svg():
    """Matplotlib flowcharts (presentation PNG + SVG). Mermaid sources are authoritative for editing."""
    styles = {
        "eye": "#fde8e0",
        "eeg": "#e8eef8",
        "fuzzy": "#fff3cd",
        "ga": "#e8f4e8",
        "fusion": "#f0e8f8",
        "out": "#d4edda",
    }

    def save(name, fig):
        fig.savefig(os.path.join(ARCH, f"{name}.png"), dpi=220, bbox_inches="tight")
        fig.savefig(os.path.join(ARCH, f"{name}.svg"), bbox_inches="tight")
        plt.close(fig)

    def box(ax, x, y, w, h, text, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#555", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, wrap=True)

    def arrow(ax, x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#333"))

    # Diagram 1
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(0.2, 5.6, "01 — Original paper architecture (M0)", fontsize=12, weight="bold")
    box(ax, 0.2, 4.2, 2.2, 0.9, "Eye 8 feat\n4s", styles["eye"])
    box(ax, 2.6, 4.2, 1.6, 0.9, "mRMR", styles["eye"])
    box(ax, 4.4, 4.2, 2, 0.9, "Hard gate", styles["eye"])
    box(ax, 0.2, 2.2, 2.4, 0.9, "EEG ingest", styles["eeg"])
    box(ax, 2.8, 2.2, 1.8, 0.9, "DWT 5 bands", styles["eeg"])
    box(ax, 4.8, 2.2, 2, 0.9, "Entropies", styles["eeg"])
    box(ax, 7, 2.2, 2, 0.9, "mRMR 3 feat", styles["eeg"])
    box(ax, 9.2, 2.2, 2, 0.9, "S-FIS 27+GA", styles["ga"])
    box(ax, 11.6, 2.2, 2, 0.9, "Per-band class", styles["out"])
    arrow(ax, 2.4, 4.65, 2.6, 4.65)
    arrow(ax, 4.2, 4.65, 4.4, 4.65)
    arrow(ax, 2.6, 2.65, 2.8, 2.65)
    arrow(ax, 4.6, 2.65, 4.8, 2.65)
    arrow(ax, 6.8, 2.65, 7, 2.65)
    arrow(ax, 9, 2.65, 9.2, 2.65)
    arrow(ax, 11.2, 2.65, 11.6, 2.65)
    arrow(ax, 5.4, 4.2, 1.4, 3.1)
    save("01_original_paper_architecture", fig)

    # 02 M1
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
    ax.text(0.1, 3.5, "02 — M1 Fuzzy eye gate", fontsize=12, weight="bold")
    for x, t in [(0.2, "Eye feat"), (2.2, "mRMR pupil Y"), (4.2, "z norm"), (6.2, "μL/μM/μH"), (8.5, "w∈[0.1,1]")]:
        box(ax, x, 2, 1.6, 0.8, t, styles["fuzzy"])
    for x0, x1 in [(1.8, 2.2), (3.8, 4.2), (5.8, 6.2), (7.8, 8.5)]:
        arrow(ax, x0, 2.4, x1, 2.4)
    box(ax, 10.3, 1, 1.4, 2.6, "EEG S-FIS\n(per band)", styles["eeg"])
    arrow(ax, 9.3, 2.4, 10.3, 2.2)
    ax.text(0.2, 0.8, "Replaces hard {0,1} gate", fontsize=9)
    save("02_M1_fuzzy_eye_gate", fig)

    # 03 M2
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.5); ax.axis("off")
    ax.text(0.1, 4, "03 — M2 Adaptive temporal fuzzy gate", fontsize=12, weight="bold")
    chain = ["P(t)", "Baseline", "z(t)", "Smooth S(t)", "Fuzzy F(S)", "w(t)"]
    for i, t in enumerate(chain):
        box(ax, 0.2 + i * 1.9, 2.2, 1.5, 0.9, t, styles["fuzzy"] if i >= 3 else styles["eye"])
    for i in range(len(chain) - 1):
        arrow(ax, 0.2 + (i + 1) * 1.9, 2.65, 0.2 + (i + 1) * 1.9 + 0.15, 2.65)
    ax.text(0.2, 0.6, "z=(P-μ_sub)/(σ+ε); symmetric or causal smooth", fontsize=9)
    save("03_M2_adaptive_temporal_gate", fig)

    # 04 M3
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis("off")
    ax.text(0.1, 4.5, "04 — M3 Multiband fusion (hard gate)", fontsize=12, weight="bold")
    box(ax, 0.2, 3.5, 1.4, 0.7, "Hard gate", styles["eye"])
    box(ax, 2, 3.5, 2, 0.7, "EEG prep", styles["eeg"])
    for i, b in enumerate(["δ", "θ", "α", "β", "γ"]):
        box(ax, 0.4 + i * 2.4, 1.8, 1.8, 0.9, f"{b}\nFIS+GA", styles["ga"])
        arrow(ax, 3, 3.5, 1.3 + i * 2.4, 2.7)
    box(ax, 10, 1.8, 3.2, 1.2, "Σ w_b P_b(c)", styles["fusion"])
    for i in range(5):
        arrow(ax, 2.2 + i * 2.4, 2.3, 10, 2.4)
    box(ax, 10, 0.3, 3.2, 0.9, "4 emotions", styles["out"])
    arrow(ax, 11.6, 1.8, 11.6, 1.2)
    save("04_M3_multiband_fusion", fig)

    # 05 M4
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")
    ax.text(0.1, 5.5, "05 — M4 Combined (M2 gate + M3 fusion)", fontsize=12, weight="bold")
    box(ax, 0.2, 4.3, 6, 0.8, "Eye: extract → mRMR → baseline → smooth → adaptive fuzzy w(t)", styles["eye"])
    box(ax, 0.2, 2.5, 6, 1.4, "EEG: preprocess → DWT 5 bands → entropy → mRMR → 5× S-FIS+GA", styles["eeg"])
    arrow(ax, 6.2, 4.7, 6.2, 3.9)
    ax.text(6.4, 4.2, "w(t)", fontsize=8, color="#c45c00")
    box(ax, 7, 2.8, 1.1, 0.9, "δ", styles["ga"])
    box(ax, 8.2, 2.8, 1.1, 0.9, "θ", styles["ga"])
    box(ax, 9.4, 2.8, 1.1, 0.9, "α", styles["ga"])
    box(ax, 10.6, 2.8, 1.1, 0.9, "β", styles["ga"])
    box(ax, 11.8, 2.8, 1.1, 0.9, "γ", styles["ga"])
    box(ax, 7.5, 0.8, 4.5, 1.2, "Weighted fusion P(c)=Σ w_b P_b(c)", styles["fusion"])
    box(ax, 8.5, 0.1, 2.5, 0.55, "Output", styles["out"])
    arrow(ax, 10, 0.8, 9.7, 0.65)
    for i in range(5):
        arrow(ax, 7.55 + i * 1.2, 2.8, 9.5, 2)
    save("05_M4_combined_proposed_architecture", fig)

    # 06 evolution
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.5); ax.axis("off")
    ax.text(0.1, 3, "06 — M0→M4 evolution", fontsize=12, weight="bold")
    steps = [("M0", "Hard+per-band"), ("M1", "Fuzzy gate"), ("M2", "Adaptive gate"), ("M3", "Hard+fusion"), ("M4", "Adapt+fusion")]
    for i, (m, d) in enumerate(steps):
        box(ax, 0.3 + i * 2.3, 1.2, 1.9, 1.2, f"{m}\n{d}", styles["fuzzy"] if "M1" in m or "M2" in m or "M4" in m else styles["eeg"])
        if i < 4:
            arrow(ax, 2.2 + i * 2.3, 1.8, 2.3 + i * 2.3, 1.8)
    save("06_M0_to_M4_model_evolution", fig)

    # 07 data flow
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")
    ax.text(0.1, 4.5, "07 — Parallel eye / EEG data flow", fontsize=12, weight="bold")
    box(ax, 0.5, 3.2, 2, 0.9, "Eye recording", styles["eye"])
    box(ax, 0.5, 1.2, 2, 0.9, "EEG recording", styles["eeg"])
    box(ax, 3.5, 2.2, 2.2, 1.2, "Time sync\n4s windows", styles["out"])
    box(ax, 6.2, 3.2, 2.3, 0.9, "Eye features +\n gate w(t)", styles["fuzzy"])
    box(ax, 6.2, 1.2, 2.3, 0.9, "EEG features +\n5 FIS", styles["ga"])
    box(ax, 9, 2, 2.5, 1.4, "Fusion →\n prediction", styles["fusion"])
    arrow(ax, 2.5, 3.65, 3.5, 2.8)
    arrow(ax, 2.5, 1.65, 3.5, 2.4)
    arrow(ax, 5.7, 3.65, 6.2, 3.65)
    arrow(ax, 5.7, 1.65, 6.2, 1.65)
    arrow(ax, 8.5, 3.2, 9, 2.7)
    arrow(ax, 8.5, 1.65, 9, 2.2)
    ax.text(0.3, 0.4, "Repo note: features may be computed for all windows; gate applies as sample weight", fontsize=8, color="#666")
    save("07_parallel_data_flow", fig)


def architecture_readme():
    mapping = [
        {"architecture_stage": "Eye feature extraction", "model_version": "M0–M4", "python_module": "gfer/eye.py",
         "function_name": "eye_window_features", "responsibility": "8 eye features per 4s window", "status": "implemented"},
        {"architecture_stage": "Eye mRMR / gate feature", "model_version": "M0–M4", "python_module": "gfer/mrmr.py",
         "function_name": "mrmr_weighted", "responsibility": "Select gate-driving feature", "status": "implemented"},
        {"architecture_stage": "Hard eye gate", "model_version": "M0, M3", "python_module": "gfer/gates.py",
         "function_name": "hard_weights", "responsibility": "Binary window gate", "status": "implemented"},
        {"architecture_stage": "Fuzzy eye gate", "model_version": "M1", "python_module": "gfer/gates.py",
         "function_name": "fuzzy_weights", "responsibility": "Continuous gate weight", "status": "implemented"},
        {"architecture_stage": "Adaptive temporal gate", "model_version": "M2, M4", "python_module": "gfer/adaptive_eye.py",
         "function_name": "compute_gate_weights", "responsibility": "Baseline + smooth + fuzzy", "status": "implemented"},
        {"architecture_stage": "EEG preprocessing", "model_version": "M0–M4", "python_module": "gfer/signal.py",
         "function_name": "preprocess", "responsibility": "Filter, notch, DWT bands", "status": "implemented"},
        {"architecture_stage": "Entropy features", "model_version": "M0–M4", "python_module": "gfer/entropy.py",
         "function_name": "band_entropies", "responsibility": "SampEn, PermEn, etc.", "status": "implemented"},
        {"architecture_stage": "Sugeno FIS", "model_version": "M0–M4", "python_module": "gfer/fuzzy.py",
         "function_name": "SugenoFIS", "responsibility": "27-candidate + Wang-Mendel", "status": "implemented"},
        {"architecture_stage": "GA rule tuning", "model_version": "M0–M4", "python_module": "gfer/ga.py",
         "function_name": "tune_rules", "responsibility": "Optimise rule base", "status": "implemented"},
        {"architecture_stage": "Multiband fusion", "model_version": "M3, M4", "python_module": "gfer/fusion.py",
         "function_name": "fuse_weighted", "responsibility": "P(c)=Σ w_b P_b(c)", "status": "implemented"},
        {"architecture_stage": "End-to-end compare", "model_version": "all", "python_module": "gfer/compare.py",
         "function_name": "main", "responsibility": "Trial-level M0/M1/M2 runs", "status": "implemented"},
        {"architecture_stage": "Illustrative scenario", "model_version": "teaching", "python_module": "gfer/teaching_scenario.py",
         "function_name": "main", "responsibility": "Synthetic CSVs/figures", "status": "illustrative_only"},
        {"architecture_stage": "Illustrative predictions", "model_version": "teaching", "python_module": "gfer/teaching_data.py",
         "function_name": "TeachingWorld", "responsibility": "Not experimental", "status": "illustrative_only"},
    ]
    write_csv(os.path.join(ARCH, "code_module_mapping.csv"), mapping,
              ["architecture_stage", "model_version", "python_module", "function_name",
               "responsibility", "status"])
    readme = """# SC-Lab architecture documentation

Regenerate diagrams: `python scripts/build_fuzzy_rulebook_and_architecture.py`

## Objective

Extend the IEEE Access 2024 hard-gated, per-band S-FIS pipeline with:

- **M1** fuzzy eye gating,
- **M2** adaptive temporal fuzzy gating,
- **M3** multiband probability fusion (hard gate baseline),
- **M4** combined M2 + M3.

## Two fuzzy systems (do not merge)

1. **EEG emotion S-FIS** — three entropy inputs, 27 candidates, GA, four emotion classes (`docs/fuzzy_rulebook/01_published_eeg_emotion_rules/`).
2. **Eye-gating FIS** — pupil importance weights for EEG windows (`03_eye_gating_rules/`).

## Frequency bands & features

Five DWT bands (δ, θ, α, β, γ); four entropy measures; mRMR selects **PhEn-FP2, SampEn-T8, PermEn-T7** for the classifiers.

## GA & Sugeno FIS

Lookup table initialises up to 27 rules per band; GA (`gfer/ga.py`) selects/refines chromosomes (paper: population 27, early stop ~gen 450 on α).

## Simultaneous eye + EEG

Both modalities are segmented into **4-second windows** aligned by time. Gate weight **w(t)** scales the effective evidence of the matching EEG window in training/inference (see `gfer/compare.py` sample weights).

**Implementation note:** Feature extraction may compute EEG descriptors for all windows before applying gate weights as sample weights — see diagram 07.

## Diagram index

| File | Content |
|------|---------|
| `01_original_paper_architecture` | Paper baseline M0 |
| `02_M1_fuzzy_eye_gate` | Fuzzy gate |
| `03_M2_adaptive_temporal_gate` | Adaptive pipeline |
| `04_M3_multiband_fusion` | Fusion |
| `05_M4_combined_proposed_architecture` | Full proposal |
| `06_M0_to_M4_model_evolution` | Version ladder |
| `07_parallel_data_flow` | Synchronised streams |

Each base name has `.mmd` (edit), `.png`, and `.svg`.
"""
    with open(os.path.join(ARCH, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)


def main():
    os.makedirs(RULEBOOK, exist_ok=True)
    export_published_rules()
    export_table7()
    export_eye_gates()
    export_model_rules()
    export_source_mapping()
    rule_visualisations()
    write_rulebook_readme()
    mermaid_sources()
    render_architecture_png_svg()
    architecture_readme()
    print("wrote", RULEBOOK)
    print("wrote", ARCH)


if __name__ == "__main__":
    main()
