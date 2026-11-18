# M0 vs M1+temporal on zuco1_task1_sentiment

> REAL EEG (ZuCo 1.0 task 1). Labels = text sentiment, not induced emotion. pupil columns = sqrt(pupil area), x and y identical.

- Trials (sentences): 4430; subjects: 12; classes: negative, neutral, positive (chance = 33.3%)
- Window 1.0 s; seeds [0, 1, 2]; protocols ['loso', 'trial']

## Protocol `loso` — trial-level accuracy %, mean ± SD over seeds

| band | No gate | M0 hard gate | M1 fuzzy gate | M2 adaptive temporal | M1+temporal | M1+temporal − M0 | McNemar p (seed 0) | Wilcoxon p (subjects) |
|---|---|---|---|---|---|---|---|---|
| delta | 32.6 ± 0.4 | 32.9 ± 0.3 | 33.8 ± 0.2 | 32.4 ± 0.2 | 32.9 ± 0.3 | +0.1 | 0.769 (989 vs 975) | 0.569 |
| theta | 32.9 ± 0.3 | 32.9 ± 0.4 | 31.7 ± 0.4 | 31.4 ± 0.5 | 32.0 ± 0.3 | -0.9 | 0.275 (1022 vs 1073) | 0.129 |
| alpha | 33.1 ± 0.2 | 33.8 ± 0.6 | 33.5 ± 0.7 | 34.6 ± 0.9 | 34.3 ± 0.4 | +0.5 | 0.583 (1052 vs 1026) | 0.622 |
| beta | 35.2 ± 0.5* | 34.2 ± 0.2 | 33.4 ± 0.4 | 34.0 ± 0.1 | 34.8 ± 0.5* | +0.6 | 0.298 (957 vs 911) | 0.240 |
| gamma | 34.4 ± 0.4* | 34.2 ± 0.3 | 33.1 ± 0.4 | 32.5 ± 0.3 | 32.9 ± 0.1 | -1.3 | 0.330 (908 vs 951) | 0.365 |

`*` = above chance, one-sided binomial p < 0.05 (median over seeds). McNemar column: trials M1+temporal got right and M0 wrong vs the reverse.

## Protocol `trial` — trial-level accuracy %, mean ± SD over seeds

| band | No gate | M0 hard gate | M1 fuzzy gate | M2 adaptive temporal | M1+temporal | M1+temporal − M0 | McNemar p (seed 0) |
|---|---|---|---|---|---|---|---|
| delta | 32.3 ± 1.1 | 32.7 ± 0.3 | 33.1 ± 0.4 | 33.7 ± 0.6 | 33.3 ± 0.5 | +0.6 | 0.206 (1046 vs 988) |
| theta | 33.2 ± 0.1 | 33.0 ± 0.7 | 33.4 ± 0.4 | 32.8 ± 0.3 | 32.3 ± 0.2 | -0.6 | 0.153 (909 vs 972) |
| alpha | 33.3 ± 1.2 | 33.1 ± 0.5 | 33.6 ± 0.4 | 33.6 ± 1.0 | 33.9 ± 0.3 | +0.8 | 0.833 (918 vs 908) |
| beta | 34.6 ± 0.6 | 33.6 ± 0.6 | 33.5 ± 0.4 | 34.0 ± 1.2* | 34.2 ± 0.8* | +0.6 | 0.152 (966 vs 903) |
| gamma | 33.1 ± 0.8 | 32.8 ± 0.4 | 33.1 ± 0.6 | 33.3 ± 0.4 | 33.7 ± 0.3 | +0.9 | 0.430 (1061 vs 1024) |

`*` = above chance, one-sided binomial p < 0.05 (median over seeds). McNemar column: trials M1+temporal got right and M0 wrong vs the reverse.

