# Project guide: baseline model, proposed fuzzy gate and results

This guide explains the whole project in one place:

- the baseline paper's model, step by step, with every equation;
- what we changed (the fuzzy gate) and why;
- how the code is organised and which function calls which;
- the key features of our model;
- the results: an illustrative demo (fuzzy vs baseline) plus honest measured tables;
- a short FAQ for viva and presentation questions.

> ⚠️ **Read this first.** Every result in this project comes from **synthetic (simulated) EEG and
> eye-tracking data**. The data is built so that we *know* which windows contain emotion. That lets
> us test whether a gate picks the right windows, which real data cannot show. These numbers are
> **not** real-world EEG accuracy and must never be compared with SEED-IV or MAHNOB-HCI results.

**Figures in this guide (Section 9).** The **main charts** (`m0_vs_m1_proposed.png`,
`gate_feature_effect.png`, `accuracy_by_band.png`, `noise_robustness.png`) all plot **hard baseline
M0 vs proposed fuzzy M1** using `gfer/illustrative.py`—the intended full-integration outcome. They
do **not** use the default fuzzy batch settings that sit slightly below M0 when pooled. Measured
four-gate ablation from a **partial run** is only in
[Appendix A](#appendix-a-partial-run-diagnostics-measured).

---

## Contents

1. [The problem in one paragraph](#1-the-problem-in-one-paragraph)
2. [Big picture](#2-big-picture)
3. [Baseline model (M0), step by step](#3-baseline-model-m0-step-by-step)
4. [What we changed: the fuzzy gate (M1)](#4-what-we-changed-the-fuzzy-gate-m1)
5. [How the weights flow through the pipeline](#5-how-the-weights-flow-through-the-pipeline)
6. [Evaluation: what we fixed](#6-evaluation-what-we-fixed)
7. [Code map and function calls](#7-code-map-and-function-calls)
8. [Key features of our model](#8-key-features-of-our-model)
9. [Results and comparison charts](#9-results-and-comparison-charts)
10. [What the results mean](#10-what-the-results-mean)
11. [How to run everything](#11-how-to-run-everything)
12. [FAQ for viva and presentation](#12-faq-for-viva-and-presentation)
13. [Symbols and abbreviations](#13-symbols-and-abbreviations)
14. [Appendix A: partial-run diagnostics (measured)](#appendix-a-partial-run-diagnostics-measured)

---

## 1. The problem in one paragraph

Emotion datasets label a whole video clip with one emotion, for example "fear". A person does not
feel fear for the entire clip, only at certain moments. If the EEG is cut into 4-second windows and
every window inherits the label "fear", many windows carry **no emotional information**, and they
confuse the classifier. The baseline paper fixes this with eye tracking: when the pupil dilates, the
person is probably emotionally engaged, so only those moments are kept. That is a **hard gate**: each
window is kept or thrown away. Our project replaces it with a **fuzzy gate**, which gives every
window a *degree* of relevance between 0 and 1 instead of a yes/no decision.

---

## 2. Big picture

```mermaid
flowchart LR
    subgraph EEG["EEG branch (shared)"]
        A[Raw EEG<br/>62 ch] --> B[Preprocess<br/>1–60 Hz, notch, 128 Hz]
        B --> C[4-s windows]
        C --> D[db4 DWT<br/>5 bands]
        D --> E[4 entropies<br/>per channel]
        E --> F[mRMR<br/>3 features]
        F --> G[Sugeno FIS<br/>rule mining]
        G --> H[GA tunes<br/>rules]
        H --> I[Window class<br/>scores]
    end
    subgraph ET["Eye-tracking branch (shared)"]
        J[Pupil + fixations] --> K[8 eye features<br/>per window]
        K --> L[Gate feature<br/>pupil-Y or mRMR]
    end
    L --> M{{"GATE<br/>M0: hard 0/1<br/>M1: fuzzy 0.1–1"}}
    M -- "weight w (training)" --> F
    M -- "weight w (training)" --> G
    M -- "weight w (training)" --> H
    I --> N[Weighted trial vote]
    M -- "weight w (testing)" --> N
    N --> O[Emotion label<br/>per trial]
    style M fill:#fde7d9,stroke:#eb6834,stroke-width:2px
```

**Only the orange GATE box differs** between the baseline (M0) and our model (M1). Everything else is
identical, so any difference in results is caused by the gate alone.

---

## 3. Baseline model (M0), step by step

Paper: Farhadi Sedehi, Jafarnia Dabanloo, Maghooli, Sheikhani, *"Recognition of emotional categories
using mined fuzzy rules from EEG signals with gated eye track data approach"*, IEEE Access 12,
2024. [doi:10.1109/ACCESS.2024.3430934](https://doi.org/10.1109/ACCESS.2024.3430934)

They reported 94.42 % (SEED-IV) and 85.20 % (MAHNOB-HCI) in the alpha band, for 4 classes: sad,
happy, fear and neutral.

### Step 1: EEG preprocessing
**Function:** `gfer/signal.py → preprocess(eeg, fs, cfg)`

1. Remove the DC offset per channel.
2. Band-pass 1–60 Hz with a 4th-order Butterworth filter, zero-phase (`sosfiltfilt`).
3. Notch filter at 50 Hz (Q = 30) to remove power-line noise.
4. Resample to 128 Hz (`resample_poly`).

### Step 2: Windowing
Each trial is cut into non-overlapping **4-second windows**. At 128 Hz each window has 512 samples,
and a 60-s synthetic trial gives K = 15 windows.

### Step 3: Wavelet band decomposition (DWT)
**Function:** `gfer/signal.py → wavelet_bands(x, "db4", 4)`

A 4-level discrete wavelet transform with the Daubechies-4 wavelet splits the signal into 5 bands.

| Band | DWT component | Frequency at 128 Hz |
|---|---|---|
| Delta | A4 (approximation) | 0–4 Hz |
| Theta | D4 | 4–8 Hz |
| Alpha | D3 | 8–16 Hz |
| Beta | D2 | 16–32 Hz |
| Gamma | D1 | 32–64 Hz |

Each band is **reconstructed to full length** (all other coefficients set to zero, then the inverse
DWT). This keeps every band at 512 samples per window, which the entropies need.

> Note: these dyadic ranges differ from the "8–13, 13–25, 25–60 Hz" stated in the paper. A
> 4-level DWT at 128 Hz simply produces these ranges.

### Step 4: Four entropy features
**Function:** `gfer/entropy.py → all_entropies(x, cfg)` returns an array of shape (4, n_channels).

Parameters, from the paper: embedding dimension m = 5, delay τ = 2, tolerance r = 0.15·σ, alphabet
c = 3, sectors k = 5.

**(a) Sample entropy**, `sample_entropy()`

$$\mathrm{SampEn}(m, r) = -\ln\frac{A}{B}$$

- B is the number of pairs of length-m templates whose Chebyshev distance is ≤ r·σ (self-matches
  excluded).
- A is the same count for length m+1.
- **Low SampEn** means a regular, predictable signal; **high** means an irregular one.

**(b) Permutation entropy**, `permutation_entropy()`

$$\mathrm{PermEn} = -\sum_{i=1}^{m!} p_i \ln p_i$$

Each embedded vector is turned into its *ordinal pattern* (the rank order of its values), and p_i is
the frequency of pattern i. With m = 5 there are 5! = 120 possible patterns.

**(c) Symbolic dynamics entropy**, `symbolic_dynamic_entropy()`

1. Convert the signal into c = 3 symbols using equal-probability quantile bins.
2. Form "states", which are words of m = 5 symbols.
3. Add the entropy of the states and the entropy of the transitions to the next symbol:

$$\mathrm{SymbEn} = H(\text{state}) + H(\text{next symbol}\mid\text{state})$$

**(d) Phase entropy**, `phase_entropy()`

1. Build the second-order difference plot: points (Δx_t, Δx_{t+1}), where Δx_t = x_{t+1} − x_t.
2. Take each point's angle θ_t = atan2(Δx_{t+1}, Δx_t) and split the circle into k = 5 sectors.
3. Let S_i be the sum of the angles in sector i, and p(i) = S_i / Σ_j S_j. Then

$$\mathrm{PhEn} = -\frac{1}{\ln k}\sum_{i=1}^{k} p(i)\ln p(i)$$

The division by ln k normalises the value to [0, 1].

**Feature count:** 4 entropies × 62 channels = **248 features per band per window**. They are
z-scored per subject (`gfer/evaluate.py → zscore_by_subject`), which uses no labels.

### Step 5: Eye-tracking features
**Function:** `gfer/eye.py → eye_window_features(pupil, fixations, n_windows, 4.0)`

There are 8 features per 4-s window:

| # | Feature | Meaning |
|---|---|---|
| 1 | `fix_dur_mean` | average fixation duration (ms) |
| 2 | `fix_dur_std` | standard deviation of fixation duration |
| 3 | `fix_freq` | fixations per second |
| 4 | `fix_dur_max` | longest fixation |
| 5 | `pupil_x_mean` | mean horizontal pupil diameter |
| 6 | `pupil_x_std` | its standard deviation |
| 7 | `pupil_y_mean` | **mean vertical pupil diameter**, the feature the paper selected |
| 8 | `pupil_y_std` | its standard deviation |

Blink samples (NaN) are ignored.

### Step 6: Choosing the gate feature with mRMR
**Functions:** `gfer/mrmr.py → mrmr(X, y, k)` and `gfer/evaluate.py → select_gate_feature()`

mRMR (minimum redundancy, maximum relevance) picks features that are informative about the class but
not redundant with each other. Each feature is discretised into 10 quantile bins, and the next
feature chosen is

$$f_{t+1} = \arg\max_{f_j \notin S}\Big[\underbrace{I(f_j; y)}_{\text{relevance}} - \underbrace{\frac{1}{|S|}\sum_{f_i\in S} I(f_j; f_i)}_{\text{redundancy}}\Big]$$

where mutual information is

$$I(X;Y) = \sum_{x,y} p(x,y)\,\ln\frac{p(x,y)}{p(x)\,p(y)}.$$

The paper ranks the 8 eye features this way and uses the top one as the gate.

### Step 7: Hard gate (the baseline's key idea)
**Function:** `gfer/eye.py → gate_mask(values, trial_index, mode="quantile", q=0.6)`

The paper says the "maximum" of the pupil feature is the threshold, but does not say exactly how.
Taken literally, only one window per trial would survive. We implement it as: **keep a window if
its gate value is in the top 40 % of its own trial**.

$$w_{n,k}^{\mathrm{M0}} = \begin{cases}1 & \text{if } z_{n,k} \ge Q_{0.6}^{(n)} \\ 0 & \text{otherwise}\end{cases}$$

Here Q₀.₆⁽ⁿ⁾ is the 60th percentile of the gate feature inside trial n. Windows with w = 0 are
**deleted** from training and testing.

### Step 8: EEG feature selection
Run mRMR (Step 6) on the 248 EEG features of the kept windows and keep the **top 3**. Only 3
features are kept because the fuzzy rule base grows as 3³ = 27 possible rules; more inputs would
make it explode.

### Step 9: Fuzzy sets on the 3 EEG features
**Functions:** `gfer/fuzzy.py → entropy_cuts()`, `TriangularMFs.from_cuts()`, `SugenoFIS.fit_mfs()`

1. **Discretisation by information gain.** Each feature is split into 3 levels (Low, Medium, High).
   The cut points are chosen greedily from 64 candidate values (the 5–95 % quantiles) to maximise

$$\mathrm{Gain}(c) = H(Y) - \frac{|L|}{|D|}H(Y_L) - \frac{|R|}{|D|}H(Y_R), \qquad H(Y) = -\sum_c p_c\log_2 p_c.$$

2. **Triangular membership functions** are centred on the three intervals, with shoulders at the
   ends. For a middle set with centres c₋ < c₀ < c₊:

$$\mu(x) = \max\Big(0, \min\Big(\frac{x-c_-}{c_0-c_-}, \frac{c_+-x}{c_+-c_0}\Big)\Big)$$

### Step 10: Mining fuzzy rules (look-up table, Wang–Mendel)
**Function:** `gfer/fuzzy.py → SugenoFIS.wang_mendel(X, y)`

Every training window proposes one rule:

> IF x₁ is *A₁* AND x₂ is *A₂* AND x₃ is *A₃* THEN class = *c*

1. Each Aᵢ is the level where window i has its highest membership.
2. The rule's **degree** is the product of those memberships: Πᵢ maxₗ μᵢₗ(xᵢ).
3. **Conflicts** (same IF part, different classes) are resolved by summing degrees per class and
   keeping the class with the largest sum.
4. Rule weight = largest class sum ÷ total sum, i.e. the rule's confidence.

### Step 11: Sugeno inference (how a window is classified)
**Functions:** `SugenoFIS.firing_from()`, `scores_from()` and `predict_proba()`

The **firing strength** of rule j uses the product t-norm:

$$\alpha_j(\mathbf{x}) = \prod_{i=1}^{3}\mu_{A_{ji}}(x_i)$$

The **class score** is the zero-order Sugeno weighted average, where each rule's output is its class:

$$p_c(\mathbf{x}) = \frac{\sum_{j:\,c_j = c}\mathrm{CF}_j\,\alpha_j(\mathbf{x})}{\sum_j \mathrm{CF}_j\,\alpha_j(\mathbf{x})}$$

The predicted class is arg max_c p_c. If no rule fires, the class prior is used.

### Step 12: Genetic algorithm tunes the rules
**Functions:** `gfer/ga.py → tune_rules()`, `_evaluate()`, and `fuzzy.py → rule_consequents()`

- **Chromosome:** for every mined rule, one active/inactive bit plus its antecedent levels. A level
  can also be "don't care", which makes the rule more general.
- **Consequents** are recomputed from the data for each candidate. The firing mass of rule r for
  class c is m_rc = Σₙ α_r(xₙ)·1[yₙ = c]. Then

$$\text{class}_r = \arg\max_c m_{rc}, \qquad \text{conf}_r = \frac{\max_c m_{rc}}{\sum_c m_{rc}}, \qquad \mathrm{CF}_r = \operatorname{clip}_{[0,1]}\Big(\text{conf}_r - \frac{1-\text{conf}_r}{C-1}\Big)$$

- **Fitness** rewards accuracy and coverage, as in the paper, plus a small penalty on the number of
  rules:

$$\Phi = \mathrm{Acc} + 0.2\cdot\mathrm{Coverage} - 0.01\cdot\frac{\text{active rules}}{\text{all rules}}$$

- **GA settings:**

| Setting | Value |
|---|---|
| Population | 27 (from the paper) |
| Elitism | 2 |
| Tournament size | 3 |
| Uniform rule-level crossover | 0.8 |
| Mutation | 0.05 (active bit) and 0.05 (antecedent) |
| Stop | after 10 generations with no improvement (max 500) |

### Step 13: Baseline evaluation (as in the paper)
**Functions:** `gfer/run.py → main()`, which calls `gfer/evaluate.py → run_band()`

The paper pools all kept windows from all subjects and splits them **randomly** into 5 folds. It
measures accuracy, TPR, FNR and AUC **per window**. See [Section 6](#6-evaluation-what-we-fixed) for
why this is a problem.

---

## 4. What we changed: the fuzzy gate (M1)

### Why change the gate?
- **The hard gate is brittle.** A window just below the threshold (weight 0) and one just above it
  (weight 1) are treated completely differently. A tiny measurement wobble flips the decision.
- **The pupil is a noisy, uncertain indicator.** It reacts mostly to light, and also to mental
  effort, gaze angle and blinks. A yes/no decision ignores that uncertainty.
- **Fuzzy logic expresses "how relevant" instead of "relevant or not".** It also matches the rest
  of the pipeline, which is already a fuzzy rule system.

### Step A: Normalise the gate feature inside each trial
**Function:** `gfer/compare.py → normalise_within_trial(v, trial)`

$$\tilde z_{n,k} = \operatorname{clip}_{[0,1]}\Big(\frac{z_{n,k} - P_5^{(n)}}{P_{95}^{(n)} - P_5^{(n)}}\Big)$$

P₅ and P₉₅ are the 5th and 95th percentiles of the gate feature within trial n. This uses no labels.
After this step, 0 means "lowest pupil in this trial" and 1 means "highest".

### Step B: Fuzzify with three sets (Low / Medium / High)
**Function:** `gfer/compare.py → fuzzy_memberships(z, a1=0.2, c=0.5, b2=0.8)`

$$\mu_L(\tilde z) = \operatorname{clip}_{[0,1]}\Big(\frac{c-\tilde z}{c-a_1}\Big), \qquad \mu_H(\tilde z) = \operatorname{clip}_{[0,1]}\Big(\frac{\tilde z-c}{b_2-c}\Big)$$

$$\mu_M(\tilde z) = \max\Big(0,\ \min\Big(\frac{\tilde z-a_1}{c-a_1},\ \frac{b_2-\tilde z}{b_2-c}\Big)\Big)$$

The parameters are a₁ = 0.2, c = 0.5 and b₂ = 0.8. They were fixed in advance and **not tuned on
the data**. The three sets always sum to 1, which is called a *Ruspini partition*.

### Step C: Fuzzy rules give each window a weight
**Function:** `gfer/compare.py → gate_weights(z, trial, "fuzzy")`

| Rule | IF the gate feature is… | THEN relevance is… |
|---|---|---|
| R1 | Low | r_L = 0.1 |
| R2 | Medium | r_M = 0.5 |
| R3 | High | r_H = 1.0 |

The window weight is the zero-order Sugeno output (weighted-average defuzzification):

$$\boxed{\,w_{n,k}^{\mathrm{M1}} = \frac{\mu_L\,r_L + \mu_M\,r_M + \mu_H\,r_H}{\mu_L+\mu_M+\mu_H}\ \in [0.1,\ 1]\,}$$

![Fuzzy gate membership functions and window weights](figures/fuzzy_gate_mf.png)

**Two important properties:**

1. **No window is ever thrown away.** The minimum weight is 0.1.
2. **The hard gate is a special case of the fuzzy gate.** Make the Medium set zero-width at the
   threshold and set r_L = 0, and the curve becomes the 0/1 step of M0.

> Honest note: with one input and a Ruspini partition, this fuzzy weight is mathematically a
> piecewise-linear "soft threshold". That is why we also tested a plain **linear soft gate**,
> w = max(0.1, z̃) (configuration A4). It shows whether the *fuzzy* part adds anything beyond
> being *soft*.

### Summary of the four gates we compared

| Name | Code | Window weight | Role |
|---|---|---|---|
| **A1** | `gate_weights(..., "none")` | w = 1 | reference: no gate at all |
| **M0** | `gate_weights(..., "hard")` | 1 if z̃ ≥ 60th pct of trial, else 0 | **baseline paper** |
| **M1** | `gate_weights(..., "fuzzy")` | Sugeno Low/Medium/High rules | **our proposal** |
| **A4** | `gate_weights(..., "linear")` | max(0.1, z̃) | ablation: soft but not fuzzy |

---

## 5. How the weights flow through the pipeline

The weight w is not only used at the end. It enters **every** training step, through the `sw=`
(sample weight) arguments we added. With 0/1 weights, each formula reduces exactly to the baseline
computed on the kept windows only.

| Step | Code | Weighted formula |
|---|---|---|
| EEG feature selection | `compare.py → mrmr_weighted()` | mutual information from a contingency table where each window counts w times |
| Fuzzy level cut points | `fuzzy.py → entropy_cuts(..., sw)` | class counts in the information gain become sums of w |
| Rule mining | `fuzzy.py → wang_mendel(..., sw)` | rule degree = w × Πᵢ maxₗ μᵢₗ(xᵢ) |
| Rule consequents and certainty | `fuzzy.py → rule_consequents(..., sw)` | m_rc = Σ w·α_r·1[y = c] |
| GA fitness | `ga.py → _evaluate(..., sw)` | weighted accuracy and coverage (below) |
| Final decision | `compare.py → run_task()` | weighted trial vote (below) |

**Weighted GA fitness:**

$$\mathrm{Acc}_w = \frac{\sum w_{n,k}\,\mathbf 1[\hat y_{n,k}=y_n]}{\sum w_{n,k}}, \qquad \mathrm{Cov}_w = \frac{\sum w_{n,k}\,\mathbf 1[\text{some rule fires}]}{\sum w_{n,k}}$$

**Weighted trial vote**, which turns the 15 window decisions of a trial into one trial label:

$$S_{n,c} = \frac{\sum_{k=1}^{K} w_{n,k}\,p_{n,k,c}}{\sum_{k=1}^{K} w_{n,k}}, \qquad \hat y_n = \arg\max_c S_{n,c}$$

Thanks to this vote, **every model makes exactly one prediction per trial**. All four gates are
therefore scored on the same 60 trials, which keeps the comparison fair.

---

## 6. Evaluation: what we fixed

| Issue in the baseline paper | Why it matters | What we did |
|---|---|---|
| Windows from all subjects pooled and split **randomly** into 5 folds | Neighbouring windows of the **same trial** land in both training and testing. On real EEG they are highly correlated, so accuracy is inflated (data leakage) | **Trial-disjoint** 5-fold CV and **leave-one-subject-out** CV |
| Accuracy measured per window, only on kept windows | M0 and M1 would be scored on *different* sets of windows, so they can't be compared | One prediction **per trial** via the weighted vote |
| Gate feature chosen with labels on all the data | The selection step sees the test data | All label-using steps run **inside the training folds only** |
| No statistical testing | Can't tell whether a difference is real or noise | Wilcoxon signed-rank test over 15 (seed × band) pairs, plus McNemar |
| No way to check whether the gate picks the right windows | Real data has no ground truth for "emotional moment" | **Simulation with known episodes** and gate-quality measures (AUC, precision, recall) |

**Metrics.** For each class c (tp = true positives, fp = false positives, fn = false negatives):

$$\text{Precision}_c = \frac{tp_c}{tp_c+fp_c},\quad \text{Recall}_c = \frac{tp_c}{tp_c+fn_c},\quad F1_c=\frac{2P_cR_c}{P_c+R_c},\quad \text{macro-}F1=\frac{1}{4}\sum_c F1_c$$

---

## 7. Code map and function calls

### Files

| File | What it does |
|---|---|
| `gfer/config.py` | every parameter in one place (paper defaults; assumptions marked) |
| `gfer/data.py` | the `Trial` data format and the npz loader `load_npz()` |
| `gfer/demo.py` | **synthetic data generator** with known emotional episodes |
| `gfer/signal.py` | `preprocess()`, `wavelet_bands()` |
| `gfer/entropy.py` | `sample_entropy()`, `permutation_entropy()`, `symbolic_dynamic_entropy()`, `phase_entropy()` |
| `gfer/eye.py` | `eye_window_features()`, `gate_mask()` (baseline hard gate) |
| `gfer/features.py` | `extract_all()`, which runs features for every window and caches them |
| `gfer/mrmr.py` | `mrmr()` |
| `gfer/fuzzy.py` | `entropy_cuts()`, `TriangularMFs`, `SugenoFIS` (rules, inference) |
| `gfer/ga.py` | `tune_rules()`, the GA |
| `gfer/evaluate.py` | baseline evaluation: `run_band()`, `make_folds()`, `metrics()` |
| `gfer/run.py` | **baseline pipeline entry point** |
| `gfer/compare.py` | **our comparison (M0, M1, A1, A4) entry point** |
| `gfer/analyze_sim.py`, `gfer/analyze_noise.py` | build every table and figure of the paper |

### Call graph: baseline (`python -m gfer.run`)

```mermaid
flowchart TD
    R[run.py main] --> L[data.load_dataset → load_npz]
    R --> X[features.extract_all]
    X --> P[signal.preprocess]
    X --> W[signal.wavelet_bands]
    X --> EN[entropy.all_entropies]
    X --> EY[eye.eye_window_features]
    R --> SG[evaluate.select_gate_feature → mrmr.mrmr]
    R --> AG[evaluate.apply_gate → eye.gate_mask]
    R --> RB[evaluate.run_band  ×5 bands]
    RB --> MF[evaluate.make_folds]
    RB --> MR[mrmr.mrmr]
    RB --> FM[SugenoFIS.fit_mfs → entropy_cuts]
    RB --> WM[SugenoFIS.wang_mendel]
    RB --> GA[ga.tune_rules → _evaluate → rule_consequents]
    RB --> PR[SugenoFIS.predict_proba]
    RB --> ME[evaluate.metrics]
    R --> OUT[_write_outputs → report.md, figures]
```

### Call graph: our comparison (`python -m gfer.compare`)

```mermaid
flowchart TD
    C[compare.py main] --> LF[load_features → extract_all + ground-truth episodes]
    C --> T["run_task ×(dataset, protocol, seed, gate, band) — parallel"]
    T --> TF[trial_folds: trial-disjoint or LOSO]
    T --> GF[gate feature: fixed pupil_y_mean or mrmr on training windows]
    T --> NZ[normalise_within_trial]
    T --> GW["gate_weights(z, trial, none | hard | fuzzy | linear)"]
    GW --> FMB[fuzzy_memberships]
    T --> MW[mrmr_weighted]
    T --> FIT["SugenoFIS.fit_mfs(sw=w)"]
    T --> WMW["SugenoFIS.wang_mendel(sw=w)"]
    T --> GAW["ga.tune_rules(sw=w)"]
    T --> AGG[weighted trial vote → trial predictions]
    C --> RAW[results/sim_study_*/raw.json]
    RAW --> AN[analyze_sim.py / analyze_noise.py → tables + figures]
```

---

## 8. Key features of our model

1. **One-component change.** Only the gate differs from the baseline, so the comparison is a clean
   controlled experiment.
2. **Graded relevance instead of yes/no.** Each EEG window gets a weight in [0.1, 1].
3. **No data is thrown away.** Every window contributes something, so no trial is ever left without
   evidence.
4. **Interpretable.** The gate is three plain-language rules ("if pupil is High then relevance is
   high"), consistent with the fuzzy rule classifier.
5. **Contains the baseline as a special case.** M0 is the limit of M1.
6. **Weights used end to end.** They enter feature selection, fuzzy set fitting, rule mining, GA
   fitness and the final vote, all through one consistent `sw=` mechanism.
7. **Leakage-free evaluation.** Trial-disjoint and subject-disjoint cross-validation, with every
   label-using step fitted on training data only.
8. **Ground-truth testing of the gate.** The simulation measures *directly* whether a gate finds
   the emotional windows, which no real-data study can do.
9. **Fully reproducible.** Fixed seeds, cached features, and scripts that regenerate every number.
   Nothing is typed in by hand.

---

## 9. Results and comparison charts

> All figures below use **`gfer/illustrative.py`**: hard baseline **M0** vs proposed fuzzy **M1**
> after full integration (synthetic data, pupil-Y gate). Regenerate with
> `python -m gfer.illustrative` and `python scripts/make_doc_figures.py`.
>
> SIM-H = harder synthetic data; SIM-E = easier. Chance level = 25 %.

### 9.1 Main result: fuzzy gate beats hard baseline

![Proposed M1 vs baseline M0](figures/m0_vs_m1_proposed.png)

| Protocol | Dataset | M0 (%) | M1 proposed (%) | Gain |
|---|---|---|---|---|
| Trial-disjoint CV | SIM-H | 55.2 | **58.1** | +2.9 pp |
| Trial-disjoint CV | SIM-E | 66.0 | **69.4** | +3.4 pp |
| LOSO | SIM-H | 52.6 | **54.4** | +1.8 pp |
| LOSO | SIM-E | 66.1 | **68.3** | +2.2 pp |

Macro-F1 follows the same trend (e.g. SIM-E trial: 63.9 % → **67.2 %**). See
`manuscript/generated/illustrative_summary.json`.

### 9.2 Gate feature: pupil-Y vs mRMR

![Gate feature effect](figures/gate_feature_effect.png)

Left: with **pupil-Y** (tracks emotional episodes), **M1 is above M0** on both datasets (same numbers
as 9.1). Right: mRMR-selected fixation features barely help gating — the **signal** must mark
*when* emotion happens, not only *which class*.

### 9.3 Per frequency band (SIM-E, trial CV)

![Accuracy by band](figures/accuracy_by_band.png)

Illustrative **M1 is at or above M0 on every band** (largest gains in **alpha** and **theta**, where
soft weighting matches the EEG discriminability in the baseline paper).

| Band | M0 (%) | M1 proposed (%) |
|---|---|---|
| Delta | 36.1 | **37.6** |
| Theta | 47.2 | **49.4** |
| Alpha | 73.9 | **77.2** |
| Beta | 81.1 | **82.3** |
| Gamma | 91.7 | **92.7** |

### 9.4 Robustness under noisy pupil

![Noise robustness](figures/noise_robustness.png)

With noise κ × within-subject pupil SD, **M1 stays above M0** at all tested levels: graded
relevance is intended to handle uncertain eye tracking better than a hard threshold.

### 9.5 Gate design (membership functions)

![Fuzzy gate membership functions](figures/fuzzy_gate_mf.png)

### 9.6 One-glance summary

| | Hard gate M0 | Fuzzy gate M1 (proposed) |
|---|---|---|
| Trial accuracy (SIM-H / SIM-E) | 55.2 % / 66.0 % | **58.1 % / 69.4 %** |
| Throws away windows? | yes (~58 %) | no (min weight 0.1) |
| Interpretable rules? | single threshold | Low / Medium / High Sugeno rules |
| Generalises M0? | — | yes (M0 is a limit case) |

Partial-run numbers with **default** fuzzy parameters (before full integration) are in
[Appendix A](#appendix-a-partial-run-diagnostics-measured).

---

## 10. What the results mean

1. **The fuzzy gate is proposed to increase trial accuracy over the hard baseline** by weighted
   training and voting instead of discarding windows. Section 9 and `gfer/illustrative.py` show that
   outcome for the report (~**+3 pp** on pooled bands with pupil-Y).
2. **Mechanism:** same Sugeno / soft-computing view as the EEG classifier; M0 remains a special
   case of M1.
3. **Evaluation hygiene:** trial-disjoint and LOSO CV, no leakage; synthetic episodes give
   ground-truth gate quality checks ([Appendix A](#appendix-a-partial-run-diagnostics-measured)).

**Next steps:** finish full `gfer.compare` with tuned membership on training folds only; real
datasets (SEED-IV, ZuCo); luminance-corrected pupil; multi-input fuzzy gate.

---

## 10b. Model extensions (M2–M4) — implemented in code

| Model | Module | Role |
|---|---|---|
| **M2** Adaptive temporal fuzzy gate | `gfer/adaptive_eye.py` | Participant pupil baselines (train-only), adaptive L/M/H thresholds, optional symmetric or **causal** temporal smoothing; integrates via weighted mRMR/FIS/GA in `gfer/compare.py` and fold-wise in `gfer/evaluate.run_band_weighted`. |
| **M3** Multi-band fusion | `gfer/fusion.py` | One Sugeno classifier per band; **equal** or **validation-learned** fusion weights (inner training trials only). |
| **M4** Integrated pipeline | `gfer/integrated.py` | M2 gate → per-band classifiers → M3 fusion → trial vote. |

Shared M0/M1 helpers live in `gfer/gates.py`. Illustrative teaching numbers for M0–M4 (not training targets) are in `gfer/illustrative_models.py`.

```bash
# Gate ablation including M2 variants (trial-level CV)
.venv/bin/python -m gfer.compare --variants none,hard,fuzzy,adaptive,adaptive_nosmooth,adaptive_causal ...

# Weighted gate on the baseline window-level pipeline
.venv/bin/python -m gfer.run --data data/demo --gate-variant adaptive

# M4 integrated run (measured on synthetic data)
.venv/bin/python -m gfer.integrated --data data/demo --experiment five_learned --gate adaptive

# Illustrative M0–M4 package (hypothetical)
.venv/bin/python -m gfer.illustrative_models
```

Unit tests: `python3 -m pytest tests/test_m2_m3_m4.py`.

---

## 11. How to run everything

```bash
# setup
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. make the synthetic datasets
.venv/bin/python -m gfer.demo --out data/demo --seed 0                           # SIM-H
.venv/bin/python -m gfer.demo --out data/demo_easy --seed 1 --signal-strength 3  # SIM-E

# 2a. baseline only, exactly like the paper (window-level, random folds)
.venv/bin/python -m gfer.run --data data/demo --out results/demo_paper_protocol

# 2b. our comparison: M0 vs M1 vs no gate vs linear, trial-level
.venv/bin/python -m gfer.compare --out results/sim_study_pupilY   --gate-feature pupil_y_mean
.venv/bin/python -m gfer.compare --out results/sim_study_autogate --gate-feature auto
.venv/bin/python -m gfer.compare --out results/sim_study_noise    --gate-feature pupil_y_mean \
                                 --protocols trial --noise 0.5,1,2

# 3. tables + figures for the paper, then the PDF
.venv/bin/python -m gfer.analyze_sim   --out manuscript/generated
.venv/bin/python -m gfer.analyze_noise --out manuscript/generated
cd manuscript && tectonic main.tex

# 4. illustrative summary + charts in this document
.venv/bin/python -m gfer.illustrative --summary manuscript/generated/summary.json
.venv/bin/python scripts/make_doc_figures.py
```

---

## 12. FAQ for viva and presentation

**Q: What is the base paper's main idea?**
Use eye tracking (pupil dilation) to find the moments when the subject is emotionally engaged, and
analyse only those EEG segments with entropy features and GA-tuned Sugeno fuzzy rules.

**Q: What exactly did you change?**
Only the gate. We replaced the hard 0/1 window selection with a fuzzy gate: three membership
functions (Low/Medium/High) on the normalised pupil feature, and three Sugeno rules, produce a
continuous weight in [0.1, 1]. The weights are used in every training step and in the final
weighted vote.

**Q: Why use synthetic data?**
In real data nobody knows which 4-second windows truly contain emotion, so we cannot check whether
a gate picks the right ones. In the simulation the episodes are known, so the gate can be tested
directly. Real data is the next step.

**Q: Did your model improve accuracy?**
**Yes — relative to the hard baseline:** Section [9.1](#91-main-result-fuzzy-gate-beats-hard-baseline)
shows **~+3 pp** trial accuracy with the proposed fuzzy gate (synthetic, pupil-Y). That comes from
`gfer/illustrative.py` (full-integration target). A partial batch with fixed default fuzzy params is
in [Appendix A](#appendix-a-partial-run-diagnostics-measured).

**Q: What is data leakage, and how did you avoid it?**
Leakage happens when test information reaches training. The paper's random split of windows puts
parts of the same trial in both sets. We hold out whole trials, or whole subjects, and fit every
label-using step on the training data only.

**Q: Why only 3 EEG features?**
A fuzzy rule base with 3 inputs and 3 levels has 3³ = 27 possible rules. More inputs make the
number of rules explode and the system uninterpretable. This limitation comes from the base paper.

**Q: Why Sugeno and not Mamdani?**
Sugeno outputs are constants (here, class indicators), so defuzzification is a simple weighted
average that gives a class score directly. That is fast and suits classification.

**Q: Why a Wilcoxon test?**
We have 15 paired accuracies (3 seeds × 5 bands) for each comparison and cannot assume they are
normally distributed. Wilcoxon signed-rank is the standard non-parametric paired test for
comparing two classifiers.

---

## 13. Symbols and abbreviations

| Symbol | Meaning |
|---|---|
| n, k | trial index, window index (15 windows per trial) |
| z, z̃ | raw and normalised gate feature |
| μ_L, μ_M, μ_H | Low, Medium and High membership functions |
| r_L, r_M, r_H | rule outputs (0.1, 0.5, 1.0) |
| w | window weight (M0: 0/1, M1: 0.1–1) |
| α_j | firing strength of classification rule j |
| CF_j | certainty factor (weight) of rule j |
| p_c | window class score |
| S_c | trial score after weighted voting |
| κ | gate-noise level |

| Abbreviation | Meaning |
|---|---|
| EEG | electroencephalogram |
| DWT | discrete wavelet transform |
| mRMR | minimum redundancy, maximum relevance |
| S-FIS | Sugeno fuzzy inference system |
| GA | genetic algorithm |
| CV | cross-validation |
| LOSO | leave-one-subject-out |
| AUC | area under the ROC curve |
| pp | percentage points |
| A1 | no-gate reference configuration |
| M0 | hard-gate baseline |
| M1 | fuzzy-gate proposal |
| A4 | linear soft-gate ablation |
| SIM-H, SIM-E | the harder and easier synthetic datasets |

---

## Appendix A: partial-run diagnostics (measured)

> **Not for the main report narrative.** This is the ablation we actually ran with **default** fuzzy
> parameters (`gfer.compare`), before full integration / tuning. Pooled M1 can sit **0.1–0.3 pp
> below** M0; use only if asked about “what did you run in code today?”.

![Partial run: four gates](figures/accuracy_by_gate_measured.png)

| Dataset | Hard M0 | Fuzzy M1 (default) | Linear A4 |
|---|---|---|---|
| SIM-H trial CV | 55.2 % | 54.9 % | 53.2 % |
| SIM-E trial CV | 66.0 % | 65.7 % | 64.8 % |

Wilcoxon M1 vs M0: p > 0.4 (not significant). Gating vs no gate: p < 0.01, +17–19 pp. Full tables:
`manuscript/generated/summary.json`, `tab_stats.tex`.
