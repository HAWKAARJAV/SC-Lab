# SEED-IV on the GPU: M0 vs M1 vs M2

This folder runs the project's three eye-gate models on the SEED-IV dataset using an NVIDIA GPU (PyTorch + CUDA):

| Model | Eye gate |
|---|---|
| **M0** | hard pupil threshold (the original paper) |
| **M1** | fuzzy Low/Medium/High pupil gate |
| **M2** | adaptive temporal fuzzy gate |
| No gate | reference |

You only need to run **one command**. It writes everything into `gpu_seed/results/`, which you then commit and push.

---

## 1. One-time setup (about 10 minutes)

You need **Python 3.10–3.12** and an NVIDIA driver (any recent driver for an RTX 40-series card works).

```bash
git clone https://github.com/HAWKAARJAV/SC-Lab.git
cd SC-Lab
git checkout seed-iv-gpu

python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

# 1) PyTorch WITH CUDA (do this first; the plain "pip install torch" may be CPU-only on Windows)
pip install torch --index-url https://download.pytorch.org/whl/cu124
# 2) the rest
pip install -r gpu_seed/requirements.txt

# check the GPU is visible. This must print: True NVIDIA GeForce RTX 4070 ...
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If the check prints `False`, the CPU-only PyTorch got installed. Fix it:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## 2. Get the dataset (about 7.4 GB)

Download https://www.kaggle.com/datasets/phhasian0710/seed-iv, either in the browser (Download button) or with the Kaggle CLI:

```bash
pip install kaggle   # needs your Kaggle API token in ~/.kaggle/kaggle.json
kaggle datasets download -d phhasian0710/seed-iv -p D:/seed_iv --unzip
```

Unzip it **outside the repo folder** (for example `D:/seed_iv`). Do not commit the dataset.

You don't need to reorganise anything. The code searches the folder for the raw-EEG files and the eye-feature files on its own.

## 3. Run

**Step A: quick test (about 5–15 minutes).** This checks that everything works before the long run.

```bash
python -m gpu_seed --root D:/seed_iv --quick
```

It should end with `DONE. Results in gpu_seed/results/quick`. Then open `gpu_seed/results/dataset_inventory.txt` and check that it says **15 subjects**, each with sessions **[1, 2, 3]**, about **45 paired sessions**, and `fs used 200.0` (or 1000.0).

**Step B: full run (several hours; leave the laptop plugged in).**

```bash
python -m gpu_seed --root D:/seed_iv
```

- **Plug in the charger and set Windows power mode to "Best performance".** On battery the GPU is throttled.
- **Disable sleep for the duration:** Settings → System → Power → Screen and sleep → "Never" while plugged in.
- **Safe to stop and restart.** Re-run the same command and it continues where it stopped. Features are cached in `gpu_seed/cache/`, and every finished experiment is saved in `gpu_seed/results/full/tasks/`.
- **If you see "CUDA out of memory",** add `--sampen-chunk 64` and run again.

**What it does:**
1. Extracts entropy features on the GPU. This step runs once and is cached.
2. Runs all experiments: 4 models × 5 EEG bands × 3 seeds × 3 evaluation protocols = **180 tasks**. Progress is printed and saved to `gpu_seed/results/full/log.txt`.
3. Writes the summary.

## 4. Send the results back

```bash
git add gpu_seed/results
git commit -m "SEED-IV GPU results"
git push
```

(Your GitHub account must be added as a collaborator on the repo. Otherwise zip `gpu_seed/results` and send it.)

The results folder contains:

| File | Contents |
|---|---|
| `results/dataset_inventory.txt` | what was found in the dataset (check this first if anything looks wrong) |
| `results/full/report.md` | the tables: accuracy per model and band, M1−M0 and M2−M0 with significance tests |
| `results/full/summary.csv`, `accuracy.png` | the same numbers as CSV and as a chart |
| `results/full/tasks/*.json` | every prediction (raw results) |
| `results/full/trials.json`, `log.txt` | trial list, settings, device, and the full log |

**Do not commit** `gpu_seed/cache/` (it is git-ignored) or the dataset.

---

## Details (for the project report)

**Data.** Raw EEG has 62 channels. The sampling rate is inferred from the data, or set with `--fs`. The 31 eye features per 4-s window come from `eye_feature_smooth`. Labels are 0 neutral, 1 sad, 2 fear, 3 happy, in the fixed per-session order from the SEED-IV ReadMe; if your copy's ReadMe differs, pass `--labels labels.json`. That gives 15 subjects × 3 sessions × 24 trials = 1,080 trials.

**Pipeline** (identical to the ZuCo study, with the paper's settings):
- 1–60 Hz band-pass and 50 Hz notch, then resampling to 128 Hz.
- A db4 wavelet splits the signal into delta, theta, alpha, beta and gamma.
- Four entropies are computed for each 4-s window, band and channel (SampEn, PermEn, SymbEn, PhEn; m = 5, τ = 2): **GPU**.
- Features are z-scored within each session.
- Weighted mRMR picks 3 EEG features: **GPU**.
- A Sugeno fuzzy system with 3 levels per feature gets its initial rules from Wang–Mendel, then a genetic algorithm tunes them. The whole population is scored in one batched step: **GPU**.
- One prediction per trial, as a gate-weighted vote over its windows.

**Gates** (code shared with the ZuCo study in `gfer/gates.py` and `gfer/adaptive_gate.py`). The gate signal is the top eye feature by mRMR on the training data.
- **M0** keeps the top 40% of each trial's windows.
- **M1** gives a fuzzy Low/Medium/High weight after scaling the signal per session.
- **M2** uses per-session percentile breakpoints and smoothing over neighbouring windows.

**Protocols:**
- `trial`: 5-fold over trials; no trial is split between training and test.
- `loso`: leave-one-subject-out, 15 folds.
- `paper`: the original paper's 5-fold over pooled windows. This is optimistic, because windows from one trial appear in both training and test; it's reported only to compare with the paper's numbers.

**Useful options:**

| Option | Effect |
|---|---|
| `--variants none,hard,fuzzy,adaptive,adaptive_causal,fuzzy_smooth` | more models |
| `--protocols trial,loso` | skip the paper protocol |
| `--seeds 0` | 1 seed instead of 3 (3× faster) |
| `--device cpu` | run without a GPU (very slow) |

**Checks:** `tests/test_gpu_seed.py` verifies that the GPU entropies, mRMR and GA fitness match the original NumPy code. Run it with `python -m pytest tests/test_gpu_seed.py`.
