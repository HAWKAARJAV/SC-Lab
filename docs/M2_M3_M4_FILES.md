# M2–M4 source file guide

## New modules

| File | Purpose |
|---|---|
| `gfer/gates.py` | Shared M0/M1 normalisation and hard/fuzzy weights (refactored from `compare.py`). |
| `gfer/adaptive_eye.py` | M2 adaptive temporal fuzzy gate: baselines, adaptive MF parameters, smoothing, `compute_gate_weights`. |
| `gfer/fusion.py` | M3 per-band Sugeno classifiers, equal/learned probability fusion, fold runner. |
| `gfer/integrated.py` | M4 CLI combining M2 gate + M3 fusion. |
| `gfer/illustrative_models.py` | Hypothetical M0–M4 accuracy package (does not alter training). |
| `tests/test_m2_m3_m4.py` | Synthetic unit tests. |

## Modified modules

| File | Change |
|---|---|
| `gfer/compare.py` | M2 gate variants; uses `gates.py` and `compute_gate_weights` on training folds. |
| `gfer/evaluate.py` | `window_gate_weights`, `run_band_weighted` for weighted gates in `run.py`. |
| `gfer/run.py` | Optional `--gate-variant` for M0–M2 weighted pipeline. |
| `gfer/mrmr.py` | `mrmr_weighted` moved here to avoid import cycles. |
| `docs/doc.md`, `README.md` | Architecture and run instructions. |
| `requirements.txt` | Added `pytest`. |

## Execution

```bash
python3 -m pytest tests/test_m2_m3_m4.py
python3 -m gfer.integrated --data data/demo --experiment five_learned --gate adaptive
python3 -m gfer.illustrative_models
```

Measured accuracy on synthetic data may differ from illustrative targets in `illustrative_models.py`; do not claim superiority without running these commands.
