# Fuzzy IF–THEN rulebook (SC-Lab)

Regenerate: `.venv/bin/python scripts/build_fuzzy_rulebook_and_architecture.py`

## Two fuzzy systems

| System | Purpose | Location |
|--------|---------|----------|
| **EEG S-FIS** | Emotion classification (sadness, happiness, fear, neutrality) | `01_published_eeg_emotion_rules/` |
| **Eye gate FIS** | Window importance weight `w(t)` | `03_eye_gating_rules/` |

Paper DOI: 10.1109/ACCESS.2024.3430934

## Contents

- **50 published EEG rules** (Tables 2–6) — not invented.
- **Table 7** performance metrics — separate from Tables 2–6.
- **27 candidate antecedents** — combinations only, no fabricated consequents.
- **M0–M4** model rule *roles* — see `04_model_specific_rules/`.
