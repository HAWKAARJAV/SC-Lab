# SEED-IV: M0 vs M1 vs M2 (GPU run `quick`)

- 144 trials, 2 subjects, classes neutral, sad, fear, happy (chance 25.0%); device cuda
- finished tasks: 4; seeds [0]

## 5-fold over trials

Accuracy %, mean ± SD over seeds. `*` = above chance (binomial p < 0.05, trial protocols).

| band | No gate | M0 hard | M1 fuzzy | M2 adaptive temporal |
|---|---|---|---|---|
| alpha | 26.4 ± 0.0 | 33.3 ± 0.0* | 27.8 ± 0.0 | 29.9 ± 0.0 |
| **mean** | **26.4** | **33.3** | **27.8** | **29.9** |

**Each gate minus M0** (trial; over band × seed runs)

| model | mean Δ (pts) | runs ahead | McNemar p (pooled, seed 0) | Wilcoxon p (subjects) |
|---|---|---|---|---|
| M1 fuzzy | -5.56 | 0/1 | 0.366 (26 vs 34) | – |
| M2 adaptive temporal | -3.47 | 0/1 | 0.609 (28 vs 33) | – |

