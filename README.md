# Hard vs. fuzzy eye-tracking gating of EEG windows

This repository holds a Python re-implementation of the eye-gated EEG emotion recognizer of
Farhadi Sedehi et al. (*IEEE Access*, 2024), together with a proposed **fuzzy eye-tracking gate**.
It also contains a controlled simulation study comparing the two gates, and the LaTeX manuscript
that reports it.

> **All experimental results in this repository come from synthetic data.**
> The simulation is built so that the EEG windows that carry emotional information are known.
> Its numbers show how the gating mechanisms behave. They are **not** estimates of real-world EEG
> emotion-recognition accuracy and must not be compared with results on SEED-IV, MAHNOB-HCI or
> other real corpora.

- **Paper:** [`manuscript/main.pdf`](manuscript/main.pdf). It is titled *Hard versus Fuzzy
  Eye-Tracking Gating of EEG Windows in a Sugeno Fuzzy Rule-Mining Emotion Recognizer: A
  Controlled Simulation Study*.
- **Baseline method:** J. Farhadi Sedehi, N. Jafarnia Dabanloo, K. Maghooli, A. Sheikhani,
  "Recognition of emotional categories using mined fuzzy rules from electroencephalogram signals
  with gated eye track data approach," *IEEE Access* 12:118122–118140, 2024.
  [doi:10.1109/ACCESS.2024.3430934](https://doi.org/10.1109/ACCESS.2024.3430934).
  No official code was released, so everything here is written from the paper.

---

## The idea

The baseline keeps an EEG window only when an eye-tracking feature (mean vertical pupil diameter)
passes a threshold. This is a **hard gate**: each window gets a weight of either 0 or 1. The
proposed **fuzzy gate (M1)** handles the same feature differently:

1. It normalises the feature within each trial to a value z̃ in [0, 1].
2. It describes z̃ with three fuzzy sets, **Low / Medium / High**.
3. A zero-order Sugeno rule base turns those memberships into a continuous window weight
   w = Σ μᵢ rᵢ / Σ μᵢ, with r = (0.1, 0.5, 1.0).

The weights enter weighted mRMR feature selection, weighted entropy discretisation, weighted
Wang–Mendel rule mining, weighted GA fitness and weighted trial-level voting. Only the gate
differs between configurations; everything else is shared. The hard gate is a limiting case of
the fuzzy one.

| Configuration | Window weight |
|---|---|
| A1, no gate | w = 1 |
| **M0, hard gate (baseline)** | w = 1 if z̃ ≥ within-trial 60th percentile, else 0 |
| **M1, fuzzy gate (proposed)** | Sugeno output of the Low/Medium/High rules |
| A4, linear soft gate (ablation) | w = max(0.1, z̃) |

## Key findings (synthetic data)

These are mean trial-level accuracies over 5 bands × 3 seeds, using trial-disjoint 5-fold
cross-validation with the pupil gate. Chance level is 25 %.

| Dataset | No gate | Hard gate M0 | Fuzzy gate M1 | Linear gate A4 |
|---|---|---|---|---|
| SIM-H (harder) | 35.8 % | 55.2 % | 54.9 % | 53.2 % |
| SIM-E (easier) | 49.1 % | 66.0 % | 65.7 % | 64.8 % |

- **Gating helps when the gate signal tracks the informative windows.** Pupil-Y separates the
  windows inside and outside emotional episodes with an AUC of about 0.99. Both gates beat no
  gating by a median of 16.7–21.7 percentage points (p ≤ 0.005).
- **Fuzzy and hard gating are statistically indistinguishable** (Wilcoxon p > 0.4). The fuzzy
  gate does not beat a plain linear soft gate either.
- **The fuzzy gate is not more robust to a noisy eye signal.** Under added gate noise it trails
  the hard gate by 1–3 points, and none of these differences is significant.
- **Choosing the gate feature by mRMR backfires.** mRMR picks the feature most related to the
  class label, as the baseline does. Here it selects fixation features, which barely mark the
  episodes (AUC 0.57), so neither gate helps consistently. **The gating signal matters far more
  than the gate's shape.**

The paper gives the full tables, per-band results, confusion matrices, statistics, the noise
sweep and the cost analysis.

---

## Repository layout

```
gfer/                     pipeline package
  config.py               all parameters (paper defaults; assumptions marked)
  data.py                 dataset-independent Trial format + npz loader
  demo.py                 synthetic EEG + eye-tracking generator with known episodes
  signal.py               band-pass, notch, resample, db4 DWT sub-bands
  entropy.py              SampEn, PermEn, SymbEn, PhEn
  eye.py                  8 eye-tracking window features, hard gate
  features.py             per-window feature extraction with on-disk cache
  mrmr.py                 mRMR feature ranking
  fuzzy.py                entropy discretisation, triangular MFs, Sugeno FIS, Wang–Mendel mining
  ga.py                   GA rule-base tuning
  evaluate.py, run.py     baseline pipeline (window-level, original protocol)
  gates.py                shared hard/fuzzy gate (M0/M1)
  adaptive_eye.py         adaptive temporal fuzzy gate (M2)
  fusion.py               per-band classifiers + probability fusion (M3)
  integrated.py           M4: M2 gate + M3 fusion end-to-end CLI
  illustrative_models.py  illustrative M0–M4 teaching package (not training)
  compare.py              M0 / M1 / M2 / A1 / A4 comparison at trial level (+ gate-noise option)
  analyze_sim.py          builds every results table and figure of the paper
  analyze_noise.py        noise-robustness table and figure
  plots.py                plotting helpers
scripts/download_zuco.py  downloads ZuCo 1.0 (real EEG + eye tracking) from OSF
results/                  raw outputs of all runs (synthetic data)
  sim_study_pupilY/       comparison grid, pupil-Y gate
  sim_study_autogate/     comparison grid, mRMR-selected gate feature
  sim_study_noise/        gate-noise sweep
  demo_*/                 earlier baseline-only runs (window-level protocol)
logs/                     run logs
manuscript/               LaTeX source (main.tex, parts/), generated/ tables and figures, PDF
```

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Tested with Python 3.13.7, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, PyWavelets 1.8.0 and
Matplotlib 3.11.2, on an Apple M2. Building the paper needs LaTeX; the commands below use
[`tectonic`](https://tectonic-typesetting.github.io).

## Reproducing the paper

Every number in the manuscript is produced by these scripts; nothing is copied by hand.

```bash
# 1. regenerate the two synthetic datasets (not stored in git)
.venv/bin/python -m gfer.demo --out data/demo --seed 0                           # SIM-H
.venv/bin/python -m gfer.demo --out data/demo_easy --seed 1 --signal-strength 3  # SIM-E

# 2. run the comparison grids (first run extracts features, about 15 min per dataset; then cached)
.venv/bin/python -m gfer.compare --out results/sim_study_pupilY   --gate-feature pupil_y_mean
.venv/bin/python -m gfer.compare --out results/sim_study_autogate --gate-feature auto
.venv/bin/python -m gfer.compare --out results/sim_study_noise    --gate-feature pupil_y_mean \
                                 --protocols trial --noise 0.5,1,2

# 3. regenerate all tables/figures, then build the PDF
.venv/bin/python -m gfer.analyze_sim   --out manuscript/generated
.venv/bin/python -m gfer.analyze_noise --out manuscript/generated
cd manuscript && tectonic main.tex
```

`compare.py` uses trial-disjoint stratified 5-fold CV and leave-one-subject-out CV. Every
label-dependent step (gate-feature selection, EEG mRMR, discretisation, GA) is fitted on the
training trials only.

## Running the baseline pipeline alone

```bash
# original protocol: 4-s windows pooled into 5 random folds
.venv/bin/python -m gfer.run --data data/demo --out results/demo_paper_protocol
# stricter: whole trials held out / leave-one-subject-out
.venv/bin/python -m gfer.run --data data/demo --out results/demo_trial_protocol --protocol trial
.venv/bin/python -m gfer.run --data data/demo --protocol loso
# the paper's fixed EEG features instead of per-fold mRMR
.venv/bin/python -m gfer.run --data data/demo --features PhEn-FP2,SampEn-T8,PermEn-T7
```

Each run writes `report.md` (tables, mined rules, figures), `summary.csv`, `results.json` and
`figures/` to `results/<name>/`.

## Where the original paper is ambiguous

All of these are settings in `gfer/config.py`.

1. **Gate threshold.** The paper uses "the maximum" of mean pupil Y as the threshold, which taken
   literally keeps one window per trial. Here a window is kept if its value is at or above the
   within-trial 60th percentile.
2. **Wavelet bands.** A 4-level db4 DWT at 128 Hz gives 0–4 / 4–8 / 8–16 / 16–32 / 32–64 Hz, not
   the 8–13 / 13–25 / 25–60 Hz the paper quotes. Sub-bands are reconstructed to full length
   before computing entropies.
3. **Normalisation.** The paper does not state one. Entropy features are z-scored per subject,
   and the gate feature is scaled per trial with P5/P95.
4. **EEG features.** The paper reports PhEn-Fp2, SampEn-T8 and PermEn-T7. The default here is
   mRMR inside each training fold, which avoids leakage.
5. **GA chromosome.** The paper shows it only in a figure. Here each rule has an active bit and
   antecedent levels, with "don't care" allowed. Consequents and certainty factors are
   recomputed from the data.
6. **Evaluation.** The paper's random 5-fold split of pooled windows puts neighbouring windows
   of the same trial into both train and test. This inflates accuracy on real EEG, so the
   simulation study evaluates at trial level instead.

## Status and next steps

- **Real data has not been run yet.** SEED-IV and MAHNOB-HCI, the baseline's datasets, require
  signed academic licences.
- **ZuCo 1.0 is the free interim option.** It has EEG and eye tracking, but its labels are text
  sentiment (positive / neutral / negative), not emotions. `scripts/download_zuco.py` fetches
  it, and a ZuCo loader for `gfer/data.py` still needs to be written.
- **To add a dataset,** write a loader that returns `List[Trial]` (see `gfer/data.py`) and
  register it in `load_dataset`.

**Not stored in git:** `data/`, `cache/` and `.venv/`. The synthetic data is regenerated with the
commands above, and ZuCo is downloaded with the script.

## References

- Farhadi Sedehi J, Jafarnia Dabanloo N, Maghooli K, Sheikhani A (2024) *IEEE Access*
  12:118122–118140. https://doi.org/10.1109/ACCESS.2024.3430934
- Hollenstein N et al. (2018) ZuCo, a simultaneous EEG and eye-tracking resource for natural
  sentence reading. *Sci Data* 5:180291. https://doi.org/10.1038/sdata.2018.291
- Writing-style references (not redistributed): Rani et al., *J Supercomput* 78:17284–17308 (2022),
  https://doi.org/10.1007/s11227-022-04584-3; Rani & Shokeen, *J Supercomput* 80:20042–20079
  (2024), https://doi.org/10.1007/s11227-024-06216-4
