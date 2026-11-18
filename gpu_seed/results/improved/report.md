# SEED-IV: M0 vs M1 vs M2 (GPU run `improved`, preset `improved`)

- 1080 trials, 15 subjects, classes neutral, sad, fear, happy (chance 25.0%); device cuda
- finished tasks: 60; seeds [0, 1, 2]

## per-person (subject-dependent), 5-fold over each subject's trials

Accuracy %, mean ± SD over seeds. `*` = above chance (binomial p < 0.05, trial protocols).

| band | No gate | M0 hard | M1 fuzzy | M2 adaptive temporal | Ref: logistic regr. |
|---|---|---|---|---|---|
| all | 64.0 ± 2.2* | 62.9 ± 0.5* | 62.5 ± 1.4* | 62.1 ± 0.7* | 79.5 ± 1.2* |
| **mean** | **64.0** | **62.9** | **62.5** | **62.1** | **79.5** |

**Each gate minus M0** (within; over band × seed runs)

| model | mean Δ (pts) | runs ahead | McNemar p (pooled, seed 0) | Wilcoxon p (subjects) |
|---|---|---|---|---|
| M1 fuzzy | -0.34 | 1/3 | 0.365 (129 vs 145) | – |
| M2 adaptive temporal | -0.77 | 0/3 | 0.674 (134 vs 142) | – |

## 5-fold over trials

Accuracy %, mean ± SD over seeds. `*` = above chance (binomial p < 0.05, trial protocols).

| band | No gate | M0 hard | M1 fuzzy | M2 adaptive temporal | Ref: logistic regr. |
|---|---|---|---|---|---|
| all | 65.4 ± 0.7* | 63.8 ± 0.6* | 59.9 ± 0.5* | 62.3 ± 1.0* | 79.8 ± 0.6* |
| **mean** | **65.4** | **63.8** | **59.9** | **62.3** | **79.8** |

**Each gate minus M0** (trial; over band × seed runs)

| model | mean Δ (pts) | runs ahead | McNemar p (pooled, seed 0) | Wilcoxon p (subjects) |
|---|---|---|---|---|
| M1 fuzzy | -3.83 | 0/3 | 0.017 (107 vs 146) | – |
| M2 adaptive temporal | -1.48 | 0/3 | 0.621 (96 vs 104) | – |

## leave-one-subject-out

Accuracy %, mean ± SD over seeds. `*` = above chance (binomial p < 0.05, trial protocols).

| band | No gate | M0 hard | M1 fuzzy | M2 adaptive temporal | Ref: logistic regr. |
|---|---|---|---|---|---|
| all | 63.3 ± 0.7* | 62.8 ± 0.6* | 60.2 ± 0.7* | 61.3 ± 1.1* | 72.0 ± 0.0* |
| **mean** | **63.3** | **62.8** | **60.2** | **61.3** | **72.0** |

**Each gate minus M0** (loso; over band × seed runs)

| model | mean Δ (pts) | runs ahead | McNemar p (pooled, seed 0) | Wilcoxon p (subjects) |
|---|---|---|---|---|
| M1 fuzzy | -2.65 | 0/3 | 0.117 (114 vs 140) | 0.373 (5/15 helped) |
| M2 adaptive temporal | -1.54 | 1/3 | 0.180 (112 vs 134) | 0.346 (6/15 helped) |

## paper protocol: 5-fold over pooled 4-s windows (optimistic)

Accuracy %, mean ± SD over seeds. `*` = above chance (binomial p < 0.05, trial protocols).

| band | No gate | M0 hard | M1 fuzzy | M2 adaptive temporal | Ref: logistic regr. |
|---|---|---|---|---|---|
| all | 67.6 ± 0.4 | 68.2 ± 0.3 | 68.4 ± 0.7 | 66.7 ± 0.2 | 97.4 ± 0.0 |
| **mean** | **67.6** | **68.2** | **68.4** | **66.7** | **97.4** |

**Each gate minus M0** (paper; over band × seed runs)

| model | mean Δ (pts) | runs ahead | McNemar p (pooled, seed 0) | Wilcoxon p (subjects) |
|---|---|---|---|---|
| M1 fuzzy | +0.24 | 1/3 | – | – |
| M2 adaptive temporal | -1.45 | 0/3 | – | – |

