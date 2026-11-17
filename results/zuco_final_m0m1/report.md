# M0 vs M1 on zuco1_task1_sentiment

> REAL EEG (ZuCo 1.0 task 1). Labels = text sentiment, not induced emotion. pupil columns = sqrt(pupil area), x and y identical.

- Trials (sentences): 4430; subjects: 12; classes: negative, neutral, positive (chance = 33.3%)
- Window 1.0 s; seeds [0, 1, 2]; protocols ['loso', 'trial']

## Protocol `loso` — trial-level accuracy %, mean ± SD over seeds

| band | No gate | M0 hard gate | M1 fuzzy gate | M1 (per-sentence scaling, v1) | M1 − M0 | McNemar p (seed 0) | Wilcoxon p (subjects) |
|---|---|---|---|---|---|---|---|
| delta | 32.6 ± 0.4 | 32.9 ± 0.3 | 33.8 ± 0.2 | 33.8 ± 0.2 | +0.9 | 0.534 (955 vs 927) | 0.278 |
| theta | 32.9 ± 0.3 | 32.9 ± 0.4 | 31.7 ± 0.4 | 33.1 ± 0.2 | -1.2 | 0.045 (899 vs 987) | 0.010 |
| alpha | 33.1 ± 0.2 | 33.8 ± 0.6 | 33.5 ± 0.7 | 31.9 ± 0.3 | -0.3 | 1.000 (1005 vs 1006) | 0.638 |
| beta | 35.2 ± 0.5* | 34.2 ± 0.2 | 33.4 ± 0.4 | 33.5 ± 0.3 | -0.8 | 0.254 (861 vs 910) | 0.470 |
| gamma | 34.4 ± 0.4* | 34.2 ± 0.3 | 33.1 ± 0.4 | 34.3 ± 0.3* | -1.1 | 0.431 (969 vs 1005) | 0.266 |

`*` = above chance, one-sided binomial p < 0.05 (median over seeds). McNemar column: trials M1 got right and M0 wrong vs the reverse.

## Protocol `trial` — trial-level accuracy %, mean ± SD over seeds

| band | No gate | M0 hard gate | M1 fuzzy gate | M1 (per-sentence scaling, v1) | M1 − M0 | McNemar p (seed 0) |
|---|---|---|---|---|---|---|
| delta | 32.3 ± 1.1 | 32.7 ± 0.3 | 33.1 ± 0.4 | 32.8 ± 1.0 | +0.5 | 0.909 (953 vs 947) |
| theta | 33.2 ± 0.1 | 33.0 ± 0.7 | 33.4 ± 0.4 | 33.2 ± 0.2 | +0.4 | 0.414 (899 vs 935) |
| alpha | 33.3 ± 1.2 | 33.1 ± 0.5 | 33.6 ± 0.4 | 33.2 ± 0.8 | +0.5 | 0.609 (937 vs 914) |
| beta | 34.6 ± 0.6 | 33.6 ± 0.6 | 33.5 ± 0.4 | 32.9 ± 0.5 | -0.0 | 0.442 (938 vs 904) |
| gamma | 33.1 ± 0.8 | 32.8 ± 0.4 | 33.1 ± 0.6 | 34.1 ± 0.6 | +0.3 | 0.700 (964 vs 982) |

`*` = above chance, one-sided binomial p < 0.05 (median over seeds). McNemar column: trials M1 got right and M0 wrong vs the reverse.

