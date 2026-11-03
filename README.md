# Eye-gated EEG emotion recognition with mined fuzzy rules (re-implementation)

Python re-implementation of:

> J. Farhadi Sedehi, N. Jafarnia Dabanloo, K. Maghooli, A. Sheikhani,
> "Recognition of emotional categories using mined fuzzy rules from electroencephalogram
> signals with gated eye track data approach", *IEEE Access*, 2024. DOI 10.1109/ACCESS.2024.3430934

No official code was released; everything here is written from the paper.

## Pipeline

| Step | Paper section | Module |
|---|---|---|
| 8 eye features per 4-s window (fixation duration mean/std/max, fixation frequency, pupil X/Y mean/std) | III-E.1 | `gfer/eye.py` |
| mRMR ranking of eye features; top feature (paper: mean pupil Y) gates EEG windows | III-E.2, IV-A | `gfer/mrmr.py`, `gfer/eye.py` |
| Band-pass 1-60 Hz (Butterworth 4), notch, CAR (MAHNOB), resample to 128 Hz | III-F | `gfer/signal.py` |
| db4 4-level DWT → delta, theta, alpha, beta, gamma | III-F.1 | `gfer/signal.py` |
| SampEn, PermEn, SymbEn, PhEn (m=5, τ=2, r=0.15·std, c=3, k=5) per channel/band/window | III-F.2 | `gfer/entropy.py` |
| mRMR → 3 EEG features as fuzzy inputs | IV-B | `gfer/mrmr.py` |
| Entropy-based discretisation into 2/3/4 levels, triangular MFs | III-G.1 | `gfer/fuzzy.py` |
| Sugeno FIS, rules mined with the lookup-table (Wang-Mendel) method, conflict resolution | III-G.1 | `gfer/fuzzy.py` |
| GA rule tuning (fitness: accuracy + coverage; pop 27; stop after 10 stagnant generations) | III-G.2, IV-B | `gfer/ga.py` |
| 5-fold CV; accuracy, F1, precision, TPR, FNR, AUC; confusion matrices; ROC | III-H | `gfer/evaluate.py` |

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. synthetic demo data (SEED-IV-shaped: 62-ch EEG, pupil X/Y, fixations, 4 classes)
.venv/bin/python -m gfer.demo --out data/demo

# 2. run everything (features are cached in ./cache)
.venv/bin/python -m gfer.run --data data/demo --out results/demo_paper_protocol

# stricter evaluation: whole trials held out
.venv/bin/python -m gfer.run --data data/demo --out results/demo_trial_protocol --protocol trial

# the paper's fixed features instead of per-fold mRMR
.venv/bin/python -m gfer.run --data data/demo --features PhEn-FP2,SampEn-T8,PermEn-T7
```

Outputs in `results/<name>/`: `report.md` (tables, mined rules, figures), `summary.csv`,
`results.json` (every fold), `figures/`.

`scripts/download_zuco.py` downloads ZuCo 1.0 task 1 (free, EEG + eye-tracking).

## Using a real dataset

Write a loader that returns `List[Trial]` (see `gfer/data.py`) and register it in
`load_dataset`. A `Trial` holds the EEG (`channels × samples`), its sampling rate and
channel names, pupil samples (`time, diameter_x, diameter_y`), fixations
(`onset_s, duration_ms`) and an integer label.

## Assumptions where the paper is ambiguous

All are settings in `gfer/config.py`.

1. **Gate threshold.** The paper uses "the maximum" of mean pupil Y as threshold, which taken
   literally keeps one window per trial. We keep windows at or above the within-trial
   60th percentile (`gate_mode="quantile"`); `max_ratio` (≥ 90% of the trial maximum) is
   also available.
2. **Wavelet bands.** A db4 4-level DWT at 128 Hz gives 0-4 / 4-8 / 8-16 / 16-32 / 32-64 Hz,
   not the 8-13 / 13-25 / 25-60 Hz the paper quotes. We use the DWT bands, reconstructed to
   full length.
3. **Normalisation.** Not stated; entropy and eye features are z-scored per subject.
4. **EEG feature selection.** Paper reports PhEn-Fp2, SampEn-T8, PermEn-T7 (text and tables
   disagree on the pairing). Default is mRMR inside each training fold, which avoids leakage;
   the fixed set can be passed with `--features`.
5. **GA chromosome** (Fig. 6 only): one gene per rule = active bit + antecedent levels
   (with "don't care"); consequents and weights are recomputed from data.
6. **Defuzzification.** Rule outputs are class indicators, so Sugeno weighted-average
   ("centroid") output gives one score per class; arg-max is the prediction.
7. **Evaluation.** The paper's protocol pools 4-s windows from all subjects into 5 random
   folds. Neighbouring windows of the same trial then fall into train and test, which likely
   inflates accuracy. `--protocol trial` (held-out trials) and `--protocol loso`
   (leave-one-subject-out) give stricter estimates.

## Simulation study: hard vs fuzzy eye-tracking gate (manuscript)

`manuscript/main.pdf` is a controlled simulation study comparing the paper's hard eye gate (M0)
with a fuzzy gate (M1), a no-gate reference and a linear soft gate. **All results are on
synthetic data.**

```bash
# regenerate the two synthetic datasets used in the manuscript (not stored in git)
.venv/bin/python -m gfer.demo --out data/demo --seed 0                          # SIM-H
.venv/bin/python -m gfer.demo --out data/demo_easy --seed 1 --signal-strength 3 # SIM-E

# run the comparison grids (raw predictions -> results/sim_study_*/raw.json)
.venv/bin/python -m gfer.compare --out results/sim_study_pupilY --gate-feature pupil_y_mean
.venv/bin/python -m gfer.compare --out results/sim_study_autogate --gate-feature auto
.venv/bin/python -m gfer.compare --out results/sim_study_noise --gate-feature pupil_y_mean --protocols trial --noise 0.5,1,2

# regenerate every table and figure in manuscript/generated/, then build the PDF
.venv/bin/python -m gfer.analyze_sim --out manuscript/generated
.venv/bin/python -m gfer.analyze_noise --out manuscript/generated
cd manuscript && tectonic main.tex
```

Not in the repository: `data/` (synthetic data is regenerated with the commands above; ZuCo is
downloaded with `scripts/download_zuco.py`), `cache/`, `.venv/`.

Reference papers (not redistributed):
- Farhadi Sedehi et al., IEEE Access 12:118122–118140 (2024), https://doi.org/10.1109/ACCESS.2024.3430934
- Rani et al., J Supercomput 78:17284–17308 (2022), https://doi.org/10.1007/s11227-022-04584-3
- Rani & Shokeen, J Supercomput 80:20042–20079 (2024), https://doi.org/10.1007/s11227-024-06216-4
