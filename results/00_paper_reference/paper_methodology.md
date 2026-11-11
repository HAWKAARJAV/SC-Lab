# Original paper methodology (reference summary)

**Source:** Farhadi Sedehi et al., IEEE Access, 2024. Transcribed for teaching/lab reference.

## Eye tracking (8 features / 4 s)

Mean and SD of fixation duration; fixation frequency; max fixation duration; mean and SD of
pupil X and Y. mRMR ranks features; **mean pupil diameter Y** is the gate feature (Fig. 7).

## EEG

Butterworth band-pass, notch, downsampling to 128 Hz; MAHNOB-HCI uses common average referencing.

## Frequency bands (as stated in paper)

| Band | Hz |
|------|-----|
| Delta | 1–4 |
| Theta | 4–8 |
| Alpha | 8–13 |
| Beta | 13–25 |
| Gamma | 25–60 |

**Repository note:** `gfer/signal.py` uses 4-level db4 DWT at 128 Hz → dyadic bands differ slightly
(see `docs/doc.md`).

## Entropies

SampEn, PermEn, SymbEn, PhEn (m=5, τ=2). 62×4 features (SEED-IV) or 32×4 (MAHNOB-HCI).
mRMR selects **PhEn-FP2, SampEn-T8, PermEn-T7** for the S-FIS.

## S-FIS + GA

3 inputs × 3 levels → 27 initial rules; GA population 27; alpha-band training stops at generation
450 with patience 10 (Fig. 9).

## Evaluation

5-fold CV; accuracy, TPR, FNR, AUC (segment-level in the original paper).
