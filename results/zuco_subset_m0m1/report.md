# M0 vs M1 on zuco1_task1_sentiment

> REAL EEG (ZuCo 1.0 task 1). Labels = text sentiment, not induced emotion. pupil columns = sqrt(pupil area), x and y identical.

- Trials (sentences): 1807; subjects: 5; classes: negative, neutral, positive (chance = 33.3%)
- Window 1.0 s; seeds [0]; protocols ['loso', 'trial']

## Protocol `loso` — trial-level accuracy %, mean ± SD over seeds

| band | No gate | M0 hard gate | M1 fuzzy gate | M1 − M0 | McNemar p (seed 0) | Wilcoxon p (subjects) |
|---|---|---|---|---|---|---|
| delta | 35.0 ± 0.0 | 35.0 ± 0.0 | 32.8 ± 0.0 | -2.2 | 0.012 (101 vs 141) | 0.250 |
| theta | 35.3 ± 0.0* | 34.9 ± 0.0 | 35.2 ± 0.0* | +0.3 | 0.659 (67 vs 61) | 0.375 |
| alpha | 35.0 ± 0.0 | 35.3 ± 0.0* | 35.9 ± 0.0* | +0.6 | 0.503 (117 vs 106) | 0.625 |
| beta | 34.9 ± 0.0 | 35.6 ± 0.0* | 35.0 ± 0.0 | -0.6 | 0.648 (189 vs 199) | 0.312 |
| gamma | 34.6 ± 0.0 | 34.9 ± 0.0 | 35.4 ± 0.0* | +0.6 | 0.434 (71 vs 61) | 0.250 |

`*` = above chance, one-sided binomial p < 0.05 (median over seeds). McNemar column: trials M1 got right and M0 wrong vs the reverse.

## Protocol `trial` — trial-level accuracy %, mean ± SD over seeds

| band | No gate | M0 hard gate | M1 fuzzy gate | M1 − M0 | McNemar p (seed 0) |
|---|---|---|---|---|---|
| delta | 34.8 ± 0.0 | 35.3 ± 0.0* | 34.6 ± 0.0 | -0.6 | 0.355 (53 vs 64) |
| theta | 35.0 ± 0.0 | 35.1 ± 0.0 | 35.5 ± 0.0* | +0.4 | 0.456 (48 vs 40) |
| alpha | 35.0 ± 0.0 | 34.4 ± 0.0 | 35.3 ± 0.0* | +0.9 | 0.227 (85 vs 69) |
| beta | 35.1 ± 0.0 | 35.0 ± 0.0 | 34.5 ± 0.0 | -0.4 | 0.554 (66 vs 74) |
| gamma | 34.2 ± 0.0 | 35.4 ± 0.0* | 34.5 ± 0.0 | -0.9 | 0.385 (141 vs 157) |

`*` = above chance, one-sided binomial p < 0.05 (median over seeds). McNemar column: trials M1 got right and M0 wrong vs the reverse.

