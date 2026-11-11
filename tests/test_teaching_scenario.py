"""Teaching scenario: fusion consistency, band realism, reproducibility."""
import numpy as np
import pandas as pd
import pytest

from gfer.teaching_data import (
    N_CLASSES,
    TARGET_REFERENCE,
    build_teaching_world,
    fuse_band_probs,
    stable_seed,
    verify_fusion,
)


def test_world_band_not_perfect():
    w = build_teaching_world("SEED-IV_reference_scale", 400, stable_seed(1, "test"))
    bp = w.band_probabilities("M3")
    accs = [(bp[:, b].argmax(1) == w.y).mean() for b in range(5)]
    assert max(accs) < 0.98
    assert min(accs) > 0.15


def test_fusion_formula_m3_m4():
    w = build_teaching_world("SEED-IV_reference_scale", 300, stable_seed(2, "fuse"))
    for model in ("M3", "M4"):
        out = w.fusion_outputs(model)
        verify_fusion(out.proba, out.band_probs, out.weights_learned)
        assert np.isclose(out.weights_learned.sum(), 1.0)


def test_m0_m2_fold_metrics_differ():
    w = build_teaching_world("SEED-IV_reference_scale", 400, stable_seed(3, "folds"))
    outs = {m: w.classifier_outputs(m) for m in ("M0", "M1", "M2")}
    assert len({tuple(o.y_pred) for o in outs.values()}) >= 2
    accs = {m: outs[m].metrics_pooled["accuracy_pct"] for m in outs}
    assert len(set(accs.values())) >= 2


def test_reproducibility():
    a = build_teaching_world("SEED-IV_reference_scale", 200, stable_seed(99, "rep"))
    b = build_teaching_world("SEED-IV_reference_scale", 200, stable_seed(99, "rep"))
    assert np.array_equal(a.y, b.y)
    assert np.array_equal(a.w_adaptive, b.w_adaptive)
    oa, ob = a.fusion_outputs("M4"), b.fusion_outputs("M4")
    assert np.allclose(oa.proba, ob.proba)


def test_saved_fusion_matches_csv(tmp_path, monkeypatch):
    from gfer import teaching_scenario as ts

    monkeypatch.setattr(ts, "OUT", str(tmp_path / "scenario"))
    ts.MASTER_SEED = 42
    ts.N_PER_DATASET = 120
    ts.main()
    m3 = pd.read_csv(tmp_path / "scenario/M3_multiband_fusion/fusion_predictions.csv")
    bp_path = tmp_path / "scenario/M3_multiband_fusion/band_probabilities.csv"
    assert bp_path.exists()
    wts = pd.read_csv(tmp_path / "scenario/M3_multiband_fusion/learned_band_weights.csv")
    # spot-check one dataset row
    sub = m3[m3.dataset == "SEED-IV_reference_scale"].iloc[0]
    assert abs(sum(sub[[f"p_{c}" for c in ("sadness", "happiness", "fear", "neutrality")]]) - 1) < 1e-5
