# SC-Lab architecture documentation

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
