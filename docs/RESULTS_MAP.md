# Results map: which output comes from which input and which code

This file traces every result in the repository (each folder, JSON, CSV, table and figure)
back to the input data, the command and the source code that produced it.

> Every result here comes from **synthetic data** made by `gfer/demo.py`.
> No real EEG dataset has been run yet.

---

## 1. The big picture: four layers

The results come in four layers. Each layer reads only from the layer above it.

```
 LAYER 0  INPUT DATA (not in git, regenerated)
 ─────────────────────────────────────────────
  gfer/demo.py --seed 0                       ──► data/demo/        = "SIM-H" (harder)
  gfer/demo.py --seed 1 --signal-strength 3   ──► data/demo_easy/   = "SIM-E" (easier)
        each holds trials .npz + ground_truth_episodes.json (where the emotional episodes are)
        features are cached in cache/<hash>/ by gfer/features.py
                         │
        ┌────────────────┴───────────────────────┐
        ▼                                        ▼
 LAYER 1a  BASELINE RUNS (gfer/run.py)    LAYER 1b  GATE COMPARISON RUNS (gfer/compare.py)
 window-level, hard gate only             trial-level, 4 gates side by side
  results/demo_paper_protocol/             results/sim_study_pupilY/raw.json
  results/demo_trial_protocol/             results/sim_study_autogate/raw.json
  results/demo_easy_paper_protocol/        results/sim_study_noise/raw.json
  results/demo_easy_pupilY_gate/           results/sim_round2/raw.json   (pilot, unused)
   each: report.md, results.json,
         summary.csv, figures/*.png
        │                                        │
        │   (summary.csv only)                   ▼
        └──────────────────────────► LAYER 2  ANALYSIS (measured numbers)
                                      gfer/analyze_sim.py   ──► manuscript/generated/
                                      gfer/analyze_noise.py      tab_*.tex, fig_*.pdf,
                                                                 summary.json, noise_summary.json
                                                │
                                                ▼
                                      gfer/illustrative.py  ──► illustrative_summary.json,
                                      (called at end of            tab_illustrative_demo.tex
                                       analyze_sim)               ⚠ NOT measured, see section 6
                                                │
                         ┌──────────────────────┴──────────────────────┐
                         ▼                                             ▼
 LAYER 3a  PAPER                                  LAYER 3b  PROJECT GUIDE
  manuscript/main.tex  \input's the tab_*.tex      scripts/make_doc_figures.py
  and fig_*.pdf ──► manuscript/main.pdf             ──► docs/figures/*.png ──► docs/doc.md
```

---

## 2. Layer 0: input data

| Name used in results | Folder | Command | What makes it different |
|---|---|---|---|
| **SIM-H** (harder) | `data/demo` | `python -m gfer.demo --out data/demo --seed 0` | Default signal strength 1.0 |
| **SIM-E** (easier) | `data/demo_easy` | `python -m gfer.demo --out data/demo_easy --seed 1 --signal-strength 3` | Emotional EEG component is 3× stronger |

Both have 5 subjects × 4 classes (sad, happy, fear, neutral) × 3 trials = **60 trials** of 60 s.
With 4-s windows that gives **900 windows**, with 62 EEG channels and 8 eye features per window.

`gfer/demo.py` builds the data so the answer is known:
- Each trial has a few **emotional episodes**. Only inside an episode does the EEG (around T7, T8 and Fp2)
  carry class information.
- During an episode the **pupil dilates** (Y more than X). This is why `pupil_y_mean` is a good gate signal.
- **Fixation** features also change by class, but they mark the episodes only weakly.
  This is why mRMR picks them, and why they make a poor gate.
- The episode times are saved to `ground_truth_episodes.json`. The analysis uses them to measure gate quality.

Feature extraction (`gfer/features.py` → `signal.py`, `entropy.py`, `eye.py`) gives each window:
- `F["eeg"]`: 5 bands × 4 entropies (SampEn, PermEn, SymbEn, PhEn) × 62 channels
- `F["eye"]`: 8 eye features (`config.EYE_FEATURES`)

These features are cached under `cache/`, so later runs skip extraction.

---

## 3. Layer 1a: baseline runs (`gfer/run.py`)

These runs re-implement the **original paper**: one hard gate, accuracy measured per window.
They do not compare gates. Their only ablation is "no gate" plus a shuffled-label sanity check.

**Pipeline in `run.py`:** load → extract features → choose the gate feature with mRMR over eye features
(`evaluate.select_gate_feature`) → hard gate (`evaluate.apply_gate`: keep windows ≥ the within-trial 60th
percentile, so 360 of 900 are kept) → for each band and each number of fuzzy levels: EEG mRMR → S-FIS look-up
table → GA → CV metrics (`evaluate.run_band`) → ablations → outputs.

### Which folder came from which settings

| Folder | Data | `--protocol` | Gate feature | Fuzzy levels | Log | Best band |
|---|---|---|---|---|---|---|
| `results/demo_paper_protocol/` | SIM-H | `paper` (random 5-fold over windows) | auto → `fix_dur_mean` | 2, 3, 4 | `logs/r1.log` (earlier: `demo_run.log`) | beta |
| `results/demo_trial_protocol/` | SIM-H | `trial` (whole trials held out) | auto → `fix_dur_mean` | 3 | `logs/r3.log` | gamma |
| `results/demo_easy_paper_protocol/` | SIM-E | `paper` | auto → `fix_dur_mean` | 3 | `logs/demo_easy_run.log` | beta |
| `results/demo_easy_pupilY_gate/` | SIM-E | `paper` | **fixed** `pupil_y_mean` | 3 | `logs/r2.log` | gamma |

The settings are recorded in each folder's `results.json` under `config`.
Compare `demo_easy_paper_protocol` with `demo_easy_pupilY_gate`: only the gate feature differs.

### What each file in a baseline folder holds

| File | Content | Written by |
|---|---|---|
| `results.json` | Everything: config, gate feature, eye mRMR ranking, gate check against true episodes, per-fold metrics for every (levels, band), ablations, mined rules | `run._write_outputs` |
| `summary.csv` | One row per (levels, band): accuracy, F1, precision, recall, AUC, look-up-only accuracy, per-class TPR/FNR/AUC | `run._write_outputs` |
| `report.md` | Readable version: ranking table, accuracy tables, per-class metrics, ablation table, top-10 rules per band, best rule per class | `run._write_report` |
| `figures/eye_mrmr_ranking.png` | mRMR score of the 8 eye features (the top one becomes the gate) | `plots.eye_ranking` |
| `figures/gate_example.png` | Trial 0: gate-feature value over time, kept windows, true episodes | `plots.gate_example` via `run._gate_check` |
| `figures/confusion_matrices.png` | One confusion matrix per band, folds pooled, best number of levels | `plots.confusion_grid` |
| `figures/roc_curves.png` | One-vs-rest ROC per class, per band | `plots.roc_grid` |
| `figures/ga_curve_<band>.png` | GA fitness over generations, best band, best fold | `plots.ga_curve` |
| `figures/membership_<band>.png` | Triangular membership functions of the 3 selected EEG features | `plots.membership_functions` |
| `figures/gate_ablation.png` | Accuracy per band: **eye-gated vs no gate** (only 2 bars, no fuzzy gate) | `plots.accuracy_bars` |

The `<band>` in `ga_curve_*` and `membership_*` is the best band: beta or gamma, depending on the run.

**Layer 2 uses only one thing from these runs:** `analyze_sim.py` reads the 3-level rows of each
`summary.csv` to build `tab_previous.tex`, the "earlier baseline runs" table in the paper.

---

## 4. Layer 1b: gate comparison runs (`gfer/compare.py`)

This is the **main experiment**. Only the gate changes; every other step is shared, and accuracy is
measured **per trial** (one prediction per 60-s trial).

### The four gate variants (`compare.gate_weights`)

| Code `variant` | Paper name | Window weight w (z̃ = gate feature scaled to [0, 1] within the trial) |
|---|---|---|
| `none` | A1, no gate | 1 |
| `hard` | **M0, hard gate (baseline)** | 1 if z̃ ≥ within-trial 60th percentile, else 0 |
| `fuzzy` | **M1, fuzzy gate (proposed)** | Sugeno: Low/Medium/High sets (a1=0.2, c=0.5, b2=0.8), r=(0.1, 0.5, 1.0) |
| `linear` | A4, linear soft gate | max(0.1, z̃) |

`compare.py` also defines `fuzzy_sharp`, `fuzzy_tuned` and `hard_tuned`, but none of the stored runs used them.

The weight w enters 5 steps: weighted mRMR (`mrmr_weighted`), weighted discretisation
(`SugenoFIS.fit_mfs(sw=)`), weighted Wang–Mendel (`wang_mendel(sw=)`), weighted GA fitness
(`tune_rules(sw=)`), and the weighted trial vote in `fit_predict`.

### One task = one record in raw.json

`compare.main` builds a grid of tasks `(dataset, protocol, seed, variant, band, gate_feature, noise)`.
`run_task` runs every CV fold for one task and appends one record:

```
{root, protocol, seed, variant, band, noise, gate_feature_mode, elapsed_s,
 folds: [{fold, gate_feature, features, n_rules, ga_gen, train_eff_windows, train_windows,
          window_acc_unweighted, trial_preds: [[trial_id, true, predicted], ...]}, ...]}
```

All later metrics are computed from `trial_preds`.

### Which raw.json came from which grid

| File | Command | Grid | Records | Log |
|---|---|---|---|---|
| `results/sim_study_pupilY/raw.json` | `--gate-feature pupil_y_mean` | 2 datasets × {trial, loso} × 3 seeds × 4 variants × 5 bands | 240 | `logs/sim_pupil.log` |
| `results/sim_study_autogate/raw.json` | `--gate-feature auto` (mRMR picks the gate per training fold) | same | 240 | `logs/sim_auto.log` |
| `results/sim_study_noise/raw.json` | `--gate-feature pupil_y_mean --protocols trial --noise 0.5,1,2` | 3 noise levels × 2 datasets × trial × 3 seeds × 4 variants × 5 bands | 360 | `logs/sim_noise.log` |
| `results/sim_round2/raw.json` | early pilot | SIM-H, trial, seed 0, pupil-Y, 4 variants × 5 bands | 20 | none |

`sim_round2` is not read by any analysis script. It is a leftover pilot run.

The noise run adds Gaussian noise to the gate feature, with SD = κ × the within-subject SD.
The κ = 0 point comes from `sim_study_pupilY` (trial protocol).

---

## 5. Layer 2: analysis (`manuscript/generated/`)

### `gfer/analyze_sim.py` (reads `sim_study_pupilY` + `sim_study_autogate` + the baseline `summary.csv` files)

| Output | What it shows | Inputs | Used in the paper as |
|---|---|---|---|
| `tab_macro_pupil_trial.tex` | Accuracy, balanced accuracy, macro P/R/F1, averaged over bands × seeds, 4 gates × 2 datasets | pupilY, trial | Main results table |
| `tab_macro_pupil_loso.tex` | Same, LOSO | pupilY, loso | Main results table |
| `tab_macro_auto_trial.tex`, `tab_macro_auto_loso.tex` | Same, with the mRMR-chosen gate feature | autogate | Gate-feature table |
| `tab_bands_pupil_trial.tex`, `tab_bands_pupil_loso.tex` | Accuracy per band (mean ± SD over 3 seeds); best value in bold | pupilY | Per-band table |
| `tab_bands_auto_*.tex` | Same for the mRMR gate | autogate | Generated but not `\input` in main.tex |
| `tab_stats.tex` | Wilcoxon signed-rank test, M1 vs M0 / A4 / A1, paired over seed × band (n = 15): median Δ, wins–ties–losses, p | both | Statistics table |
| `tab_gate_quality.tex` | How well each gate signal finds the true episodes: episode rate, AUC(z̃), hard gate precision/recall, fuzzy/linear weighted precision | data + `ground_truth_episodes.json` (no raw.json) | Gate-quality table |
| `tab_cost.tex` | Effective training windows (%) and run time per gate | both | Cost table |
| `tab_previous.tex` | 3-level accuracy per band from the 4 baseline folders | `results/demo_*/summary.csv` | "Earlier runs" table |
| `fig_bands.pdf` | Grouped bars: 4 gates × 5 bands, SIM-H and SIM-E, pupil-Y, trial CV | pupilY | Per-band figure |
| `fig_confusion.pdf` | Confusion matrices, M0 vs M1, SIM-E, best band, 3 seeds pooled | pupilY | Confusion figure |
| `fig_gate_trace.pdf` | One SIM-E trial over time: z̃, hard weight, fuzzy weight, shaded true episodes | data only | Gate-trace figure |
| `summary.json` | Every number above in machine-readable form, plus McNemar test, gate choices made by mRMR, confusion matrices | all | Source for `docs/figures` |

**Key format in `summary.json`:** `"<gate>|<protocol>|<dataset>|<variant>|<band or allbands>|<metric>": [mean, sd]`,
for example `"pupil|trial|SIM-E|fuzzy|allbands|acc": [65.7, …]`.
`<gate>` is `pupil` or `auto`. `<metric>` is one of `acc`, `bacc`, `prec`, `rec`, `f1`.
There are also the blocks `stats`, `mcnemar`, `gate_quality`, `auto_gate_choices`, `cost` and `previous_runs`.

### `gfer/analyze_noise.py` (reads `sim_study_pupilY` [trial, κ = 0] + `sim_study_noise`)

| Output | What it shows |
|---|---|
| `tab_noise.tex` | For each κ ∈ {0, 0.5, 1, 2}: gate AUC, accuracy of the 4 gates, mean Δ(M1−M0), Wilcoxon p |
| `fig_noise.pdf` | Accuracy vs κ, one line per gate, SIM-H and SIM-E |
| `noise_summary.json` | Keys `"<dataset>|<κ>"` → auc, hard_prec, per-variant [mean, sd], d_mean, d_median, p |

### Not produced by any script

- `manuscript/figures/fuzzy_gate_mf.pdf` (the paper's membership-function figure) has no generator in the repo.
  It was made by hand. The similar `docs/figures/fuzzy_gate_mf.png` is generated (see below).

---

## 6. Layer 2b: the "illustrative" numbers (`gfer/illustrative.py`)

> ⚠ **Read this before using any chart in `docs/`.**

`analyze_sim.py` ends by calling `write_illustrative`. This builds `illustrative_summary.json` and
`tab_illustrative_demo.tex` as follows:

- **M0 (`hard`)** is copied from the measured `summary.json`.
- **M1 (`fuzzy_demo`)** = M0 **plus fixed, hard-coded gains** set in the source code:
  `DELTA_ACC = {SIM-H: +2.9, SIM-E: +3.4}`, `DELTA_LOSO_ACC = {+1.8, +2.2}`,
  `DELTA_BAND = {delta +1.5, theta +2.2, alpha +3.3, beta +1.2, gamma +1.0}`,
  `NOISE_M1_EXTRA = {κ0 +2.9, κ0.5 +1.8, κ1 +2.5, κ2 +3.8}`.

So **the `fuzzy_demo` values are not results of running the fuzzy gate**. They are targets typed into the code.
The measured fuzzy gate (`fuzzy` in `summary.json`) is about equal to M0. For example, on SIM-E with
trial CV, the measured values are M0 66.0 % and M1 65.7 % (p > 0.4). The illustrative values are 66.0 % and 69.4 %.

---

## 7. Layer 3b: doc figures (`scripts/make_doc_figures.py` → `docs/figures/`)

This script reads `summary.json`, `noise_summary.json` and `illustrative_summary.json` from `manuscript/generated/`.

| PNG | Content | Data source | Measured? | Used in `docs/doc.md` |
|---|---|---|---|---|
| `m0_vs_m1_proposed.png` | M0 vs M1, SIM-H/SIM-E, trial and LOSO, with "+x.x" gain labels | illustrative | ❌ M1 is illustrative | §9.1 (main result) |
| `accuracy_by_gate.png` | **Same image as above** (saved under an old file name) | illustrative | ❌ | not referenced |
| `accuracy_by_gate_illustrative.png` | **Same image as above** (saved under an old file name) | illustrative | ❌ | not referenced |
| `accuracy_by_gate_measured.png` | **All 4 gates** (no gate, hard, fuzzy, linear), SIM-H/SIM-E, trial and LOSO | `summary.json` | ✅ | Appendix A |
| `gate_feature_effect.png` | Left: pupil-Y M0 vs M1. Right: mRMR gate, all 4 gates | left: illustrative; right: `summary.json` | ⚠ left ❌, right ✅ | §9.2 |
| `accuracy_by_band.png` | SIM-E, M0 vs M1 per band | illustrative | ❌ | §9.3 |
| `noise_robustness.png` | M0 vs M1 against κ | illustrative noise block | ❌ (falls back to measured only if that block is missing) | §9.4 |
| `fuzzy_gate_mf.png` | (a) Low/Medium/High sets; (b) weight vs z̃ for the hard, fuzzy and linear gates | `compare.FUZZY` parameters | ✅ design plot (but the hard gate is drawn at τ = 0.65 as a sketch; the real hard gate uses the per-trial 60th percentile) | §4, §9.5 |

---

## 8. Quick lookup: "I see X, where did it come from?"

| You are looking at… | It came from |
|---|---|
| A number in `docs/doc.md` §9 or any chart labelled "proposed / illustrative" | `gfer/illustrative.py` (M0 measured + hard-coded gain) |
| The four-gate bar chart (no / hard / fuzzy / linear) | `docs/figures/accuracy_by_gate_measured.png` ← `summary.json` ← `sim_study_pupilY/raw.json` |
| A table in `manuscript/main.pdf` | `manuscript/generated/tab_*.tex` ← `analyze_sim.py` / `analyze_noise.py` ← `results/sim_study_*/raw.json` |
| The README "Key findings" table | `tab_macro_pupil_trial.tex` (measured) |
| "eye-gated vs no gate" per band, blue/orange | `results/demo_*/figures/gate_ablation.png` ← `run.py` (baseline only) |
| Mined fuzzy rules in text form | `results/demo_*/report.md` and `results.json → rules` (only the baseline runs save rules) |
| GA convergence or membership-function plot of EEG features | `results/demo_*/figures/ga_curve_*.png`, `membership_*.png` |
| ROC curves | Only the baseline runs (`results/demo_*/figures/roc_curves.png`); the comparison study reports no ROC |
| Which eye feature mRMR picked as the gate | baseline: `results.json → gate_feature`; comparison: `summary.json → auto_gate_choices` |

## 9. Reproducing everything in order

```bash
python -m gfer.demo --out data/demo --seed 0
python -m gfer.demo --out data/demo_easy --seed 1 --signal-strength 3
# layer 1a (optional, only feeds tab_previous.tex)
python -m gfer.run --data data/demo      --out results/demo_paper_protocol --levels 2,3,4
python -m gfer.run --data data/demo      --out results/demo_trial_protocol --protocol trial --levels 3
python -m gfer.run --data data/demo_easy --out results/demo_easy_paper_protocol --levels 3
python -m gfer.run --data data/demo_easy --out results/demo_easy_pupilY_gate --gate-feature pupil_y_mean --levels 3
# layer 1b
python -m gfer.compare --out results/sim_study_pupilY   --gate-feature pupil_y_mean
python -m gfer.compare --out results/sim_study_autogate --gate-feature auto
python -m gfer.compare --out results/sim_study_noise    --gate-feature pupil_y_mean --protocols trial --noise 0.5,1,2
# layer 2 (analyze_sim also writes illustrative_summary.json)
python -m gfer.analyze_sim   --out manuscript/generated
python -m gfer.analyze_noise --out manuscript/generated
# layer 3
python scripts/make_doc_figures.py
cd manuscript && tectonic main.tex
```

Order matters on a clean checkout. `analyze_sim` writes `illustrative_summary.json` before
`analyze_noise` has written `noise_summary.json`, so the illustrative file has no `noise` block. When that
block is missing, `make_doc_figures.py` draws the **measured** noise curves instead.

The `--levels` flags for the baseline runs are inferred from the `config` saved in each `results.json`.
