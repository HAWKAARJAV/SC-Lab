# M0 vs M2 on zuco1_task1_sentiment

> REAL EEG (ZuCo 1.0 task 1). Labels = text sentiment, not induced emotion. pupil columns = sqrt(pupil area), x and y identical.

- Trials (sentences): 4430; subjects: 12; classes: negative, neutral, positive (chance = 33.3%)
- Window 1.0 s; seeds [0, 1, 2]; protocols ['loso', 'trial']

## Protocol `loso` — trial-level accuracy %, mean ± SD over seeds

| band | No gate | M0 hard gate | M2 adaptive temporal | M2 adaptive only | M2 causal | M2 − M0 | McNemar p (seed 0) | Wilcoxon p (subjects) |
|---|---|---|---|---|---|---|---|---|
| delta | 32.6 ± 0.4 | 32.9 ± 0.3 | 32.4 ± 0.2 | 33.8 ± 0.4 | 32.8 ± 0.4 | -0.5 | 0.650 (960 vs 981) | 0.266 |
| theta | 32.9 ± 0.3 | 32.9 ± 0.4 | 31.4 ± 0.5 | 32.5 ± 0.3 | 32.1 ± 0.3 | -1.5 | 0.127 (959 vs 1028) | 0.339 |
| alpha | 33.1 ± 0.2 | 33.8 ± 0.6 | 34.6 ± 0.9 | 33.7 ± 0.3 | 34.9 ± 0.4* | +0.8 | 0.031 (1064 vs 966) | 0.064 |
| beta | 35.2 ± 0.5* | 34.2 ± 0.2 | 34.0 ± 0.1 | 32.9 ± 0.7 | 33.6 ± 0.1 | -0.2 | 0.982 (971 vs 969) | 0.910 |
| gamma | 34.4 ± 0.4* | 34.2 ± 0.3 | 32.5 ± 0.3 | 33.5 ± 0.3 | 33.1 ± 0.6 | -1.7 | 0.087 (849 vs 922) | 0.016 |

`*` = above chance, one-sided binomial p < 0.05 (median over seeds). McNemar column: trials M2 got right and M0 wrong vs the reverse.

## Protocol `trial` — trial-level accuracy %, mean ± SD over seeds

| band | No gate | M0 hard gate | M2 adaptive temporal | M2 adaptive only | M2 causal | M2 − M0 | McNemar p (seed 0) |
|---|---|---|---|---|---|---|---|
| delta | 32.3 ± 1.1 | 32.7 ± 0.3 | 33.7 ± 0.6 | 32.8 ± 0.8 | 33.2 ± 0.7 | +1.0 | 0.056 (1151 vs 1060) |
| theta | 33.2 ± 0.1 | 33.0 ± 0.7 | 32.8 ± 0.3 | 32.8 ± 0.4 | 32.3 ± 1.0 | -0.2 | 0.523 (944 vs 973) |
| alpha | 33.3 ± 1.2 | 33.1 ± 0.5 | 33.6 ± 1.0 | 33.5 ± 0.5 | 34.2 ± 0.7* | +0.5 | 0.127 (970 vs 903) |
| beta | 34.6 ± 0.6 | 33.6 ± 0.6 | 34.0 ± 1.2* | 34.0 ± 0.3 | 34.0 ± 0.5 | +0.4 | 0.175 (944 vs 885) |
| gamma | 33.1 ± 0.8 | 32.8 ± 0.4 | 33.3 ± 0.4 | 32.9 ± 0.5 | 32.8 ± 0.9 | +0.5 | 0.381 (1010 vs 970) |

`*` = above chance, one-sided binomial p < 0.05 (median over seeds). McNemar column: trials M2 got right and M0 wrong vs the reverse.

