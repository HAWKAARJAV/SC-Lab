"""M0 (hard) / M1 (fuzzy) gate behaviour and the ZuCo label parser."""
import os

import numpy as np
import pytest

from gfer.gates import FUZZY, gate_weights, normalise_within_trial


def _windows():
    rng = np.random.default_rng(0)
    trial = np.repeat(np.arange(4), 6)
    return rng.normal(size=len(trial)), trial


def test_hard_gate_is_binary_and_keeps_top_windows_per_trial():
    v, trial = _windows()
    w = gate_weights(normalise_within_trial(v, trial), trial, "hard")
    assert set(np.unique(w)) <= {0.0, 1.0}
    for t in np.unique(trial):
        m = trial == t
        assert w[m].sum() >= 1
        assert v[m][w[m] == 1].min() >= v[m][w[m] == 0].max()


def test_fuzzy_gate_is_bounded_and_monotone():
    z = np.linspace(0, 1, 101)
    w = gate_weights(z, np.zeros_like(z, dtype=int), "fuzzy")
    assert w.min() >= FUZZY["r"][0] - 1e-12 and w.max() <= 1.0
    assert np.all(np.diff(w) >= -1e-12)


def test_normalisation_handles_missing_eye_data():
    v, trial = _windows()
    v[:3] = np.nan
    z = normalise_within_trial(v, trial)
    assert np.all((z >= 0) & (z <= 1)) and np.all(z[:3] == 0.5)


@pytest.mark.skipif(not os.path.exists("data/zuco/meta/sentiment_labels_task1.csv"),
                    reason="ZuCo metadata not downloaded")
def test_zuco_labels_cover_all_400_sentences():
    from gfer.zuco import sentence_labels
    ids, lab = sentence_labels("data/zuco/meta")
    assert len(lab) == 400 and np.isfinite(lab).sum() >= 399
    assert set(np.unique(lab[np.isfinite(lab)])) == {-1, 0, 1}
