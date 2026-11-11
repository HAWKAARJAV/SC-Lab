"""Unit tests for M2 adaptive gate, M3 fusion, and M4 integration (synthetic inputs)."""
import numpy as np
import pytest

from gfer.adaptive_eye import (
    compare_gate_modes,
    compute_gate_weights,
    impute_gate_values,
    normalize_pupil_feature,
    participant_baselines,
    temporal_smooth_weights,
)
from gfer.fusion import (
    fuse_equal,
    fuse_weighted,
    learn_fusion_weights,
    normalize_fusion_weights,
    trial_level_vote,
)
from gfer.fuzzy import SugenoFIS


def _synth_windows(n=12, n_trials=3):
    per = n // n_trials
    trial = np.repeat(np.arange(n_trials), per)
    window = np.tile(np.arange(per), n_trials)
    subjects = np.array(["S1"] * (n // 2) + ["S2"] * (n - n // 2))
    values = np.linspace(1.0, 2.0, n) + 0.1 * window[: len(trial)]
    return values, subjects, trial, window


def test_participant_baseline_uses_train_only():
    values, subjects, trial, window = _synth_windows()
    train = np.zeros(len(values), bool)
    train[:6] = True
    bases = participant_baselines(values, subjects, train)
    assert "S1" in bases and "__global__" in bases
    assert bases["S1"] == pytest.approx(np.median(values[:6]))


def test_impute_and_normalize_range():
    values, subjects, trial, window = _synth_windows()
    values[3] = np.nan
    imputed = impute_gate_values(values, trial, window)
    assert np.isfinite(imputed).all()
    bases = participant_baselines(imputed, subjects, np.ones(len(values), bool))
    z = normalize_pupil_feature(imputed, subjects, trial, bases)
    assert z.min() >= 0.0 and z.max() <= 1.0


def test_temporal_smoothing_causal_vs_symmetric():
    w = np.array([0.2, 0.8, 0.2, 0.9, 0.1, 0.5], dtype=float)
    trial = np.array([0, 0, 0, 1, 1, 1])
    window = np.array([0, 1, 2, 0, 1, 2])
    sym = temporal_smooth_weights(w, trial, window, mode="symmetric", span=1)
    cau = temporal_smooth_weights(w, trial, window, mode="causal", span=1)
    assert sym.shape == w.shape
    assert not np.allclose(sym, cau)


def test_gate_weights_in_valid_range():
    values, subjects, trial, window = _synth_windows(15, 3)
    train = np.ones(len(values), bool)
    for variant in ("hard", "fuzzy", "adaptive", "adaptive_nosmooth", "adaptive_causal"):
        w = compute_gate_weights(values, subjects, trial, window, train, variant)
        assert w.shape == (len(values),)
        assert np.all(w >= 0.0) and np.all(w <= 1.0 + 1e-9)


def test_adaptive_params_not_using_test_fold():
    from gfer.adaptive_eye import adaptive_fuzzy_params, normalize_pupil_feature, participant_baselines

    values, subjects, trial, window = _synth_windows(20, 4)
    values[10:] += 8.0
    train = np.zeros(20, bool)
    train[:10] = True
    v = impute_gate_values(values, trial, window)
    b_tr = participant_baselines(v, subjects, train)
    b_all = participant_baselines(v, subjects, np.ones(20, bool))
    assert b_tr["S2"] != pytest.approx(b_all["S2"])
    z = normalize_pupil_feature(v, subjects, trial, b_tr)
    p_tr = adaptive_fuzzy_params(z, subjects, train)
    p_all = adaptive_fuzzy_params(z, subjects, np.ones(20, bool))
    assert p_tr["S2"] != p_all["S2"]


def test_missing_eye_measurements():
    values, subjects, trial, window = _synth_windows()
    values[:] = np.nan
    values[0] = 1.5
    w = compute_gate_weights(values, subjects, trial, window, np.ones(len(values), bool), "adaptive")
    assert np.isfinite(w).all()


def test_fusion_weight_normalization():
    w = np.array([1.0, 2.0, 3.0])
    n = normalize_fusion_weights(w)
    assert n.sum() == pytest.approx(1.0)
    assert np.all(n >= 0)


def test_probability_fusion_four_classes():
    rng = np.random.default_rng(0)
    P1 = rng.dirichlet([1, 1, 1, 1], size=5)
    P2 = rng.dirichlet([1, 1, 1, 1], size=5)
    fused = fuse_equal([P1, P2])
    assert fused.shape == (5, 4)
    assert np.allclose(fused.sum(1), 1.0)
    w = learn_fusion_weights({"a": P1[:3], "b": P2[:3]}, np.array([0, 1, 2]))
    fused_w = fuse_weighted([P1, P2], w)
    pred = fused_w.argmax(1)
    assert pred.shape == (5,)
    assert set(pred).issubset({0, 1, 2, 3})


def test_band_classifier_output_shape():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 6))
    y = rng.integers(0, 4, size=40)
    w = np.linspace(0.2, 1.0, 40)
    from gfer.config import Config
    from gfer.fusion import band_predict_proba, train_band_classifier

    cfg = Config(seed=0)
    fis = train_band_classifier(X, y, w, cfg, 4, [f"f{i}" for i in range(6)], rng)
    P = band_predict_proba(fis, X[:5])
    assert P.shape == (5, 4)
    assert np.allclose(P.sum(1), 1.0, rtol=1e-5)


def test_m4_gate_modes_comparison():
    values, subjects, trial, window = _synth_windows(16, 4)
    train = np.ones(len(values), bool)
    modes = compare_gate_modes(values, subjects, trial, window, train)
    assert set(modes) == {"hard", "fuzzy", "adaptive_nosmooth", "adaptive"}


def test_trial_level_vote():
    proba = np.eye(4)[[0, 1, 2, 0]]
    w = np.array([1.0, 0.5, 0.2, 1.0])
    trial = np.array([0, 0, 1, 1])
    y = np.array([0, 0, 2, 3])
    preds = trial_level_vote(proba, w, trial, y)
    assert len(preds) == 2
