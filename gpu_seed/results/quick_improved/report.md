# SEED-IV: M0 vs M1 vs M2 (GPU run `quick_improved`, preset `improved`)

- 144 trials, 2 subjects, classes neutral, sad, fear, happy (chance 25.0%); device cuda
- finished tasks: 10; seeds [0]

## per-person (subject-dependent), 5-fold over each subject's trials

Accuracy %, mean ± SD over seeds. `*` = above chance (binomial p < 0.05, trial protocols).

| band | No gate | M0 hard | M1 fuzzy | M2 adaptive temporal | Ref: logistic regr. |
|---|---|---|---|---|---|
| all | 63.2 ± 0.0* | 65.3 ± 0.0* | 63.2 ± 0.0* | 67.4 ± 0.0* | 78.5 ± 0.0* |
| **mean** | **63.2** | **65.3** | **63.2** | **67.4** | **78.5** |

**Each gate minus M0** (within; over band × seed runs)

| model | mean Δ (pts) | runs ahead | McNemar p (pooled, seed 0) | Wilcoxon p (subjects) |
|---|---|---|---|---|
| M1 fuzzy | -2.08 | 0/1 | 0.736 (16 vs 19) | – |
| M2 adaptive temporal | +2.08 | 1/1 | 0.728 (18 vs 15) | – |

## 5-fold over trials

Accuracy %, mean ± SD over seeds. `*` = above chance (binomial p < 0.05, trial protocols).

| band | No gate | M0 hard | M1 fuzzy | M2 adaptive temporal | Ref: logistic regr. |
|---|---|---|---|---|---|
| all | 68.1 ± 0.0* | 66.7 ± 0.0* | 66.0 ± 0.0* | 68.8 ± 0.0* | 79.2 ± 0.0* |
| **mean** | **68.1** | **66.7** | **66.0** | **68.8** | **79.2** |

**Each gate minus M0** (trial; over band × seed runs)

| model | mean Δ (pts) | runs ahead | McNemar p (pooled, seed 0) | Wilcoxon p (subjects) |
|---|---|---|---|---|
| M1 fuzzy | -0.69 | 0/1 | 1.000 (17 vs 18) | – |
| M2 adaptive temporal | +2.08 | 1/1 | 0.701 (15 vs 12) | – |

