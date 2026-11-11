# Eye-gating fuzzy system (M1 / M2)

**Not an emotion classifier.** Outputs window importance weight `w(t) ∈ [0.1, 1]`.

## M1 — `gfer/gates.py`

Defuzzification (Sugeno-style weighted average of level outputs):

`w(t) = (μL·rL + μM·rM + μH·rH) / (μL + μM + μH)`

With **rL=0.1, rM=0.5, rH=1.0** and triangular cuts **a1=0.2, c=0.5, b2=0.8** on trial-normalised pupil **z**.

## M2 — `gfer/adaptive_eye.py`

Pipeline before the same fuzzy mapping:

1. `μ_subject`, `σ_subject` from **training windows only** (`participant_baselines`).
2. Baseline-correct and robust scale: `normalize_pupil_feature` → **z̃ ∈ [0,1]**.
3. Per-subject adaptive MF percentiles (`adaptive_fuzzy_params`) on training data.
4. `adaptive_fuzzy_weights` → optional **temporal smoothing**:
   - **symmetric** (offline teaching default): neighbourhood mean over window index.
   - **causal** (`adaptive_causal`): uses past windows only (real-time compatible).

Smoothing does not add new IF-THEN rules; it filters the fuzzy gate score time series.
