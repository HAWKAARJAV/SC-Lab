# Published EEG emotion rules (S-FIS)

Source: Farhadi Sedehi et al., *IEEE Access* 2024 — Tables 2–6 (SEED-IV rule mining).

- **50 rules** total: 10 per frequency band (δ, θ, α, β, γ).
- Inputs: **PhEn-FP2**, **SampEn-T8**, **PermEn-T7** (three levels each).
- **27 combinations** in `candidate_27_combinations.csv` are *candidate antecedents only* — not all are retained classifiers.
- Spelling/channel variants in the paper are preserved in `source_note` and `if_text_original`.
- This is the **EEG emotion classification FIS** — separate from the **eye-gating FIS** in `03_eye_gating_rules/`.
