# Illustrative scenario assumptions (synthetic)

1. **Shared windows**: All models M0–M4 use identical subject/trial/window IDs and ground-truth labels.
2. **Eye gate**: Pupil is generated with subject baselines (train-masked medians); episodes modulate diameter.
3. **Band classifiers**: Each band has distinct logit scale, noise, and confusion partners (not 100% separable).
4. **Gate coupling**: Gate weight scales class logit separation (weak gate → probabilities closer to uniform).
5. **M0/M1/M2**: Alpha-band softmax after hard / fuzzy / adaptive gating (other bands logged in `band_probs`).
6. **M3/M4**: Hard (M3) or adaptive (M4) gating on band logits; fusion weights learned on validation indices only.
7. **Fusion**: `P_final(c) = Σ_b w_b P_b(c)` — saved probabilities are never post-edited for target accuracy.
8. **Reference targets** in `summary.csv` are paper-scale anchors; `achieved_*` may differ.
