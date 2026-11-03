"""Eye-tracking features per 4-s window (Sec. III-E.1) and the eye gate (Sec. IV-A)."""
import numpy as np

from .config import EYE_FEATURES


def eye_window_features(pupil, fixations, n_windows, win_s):
    """Return (n_windows, 8) features in the order of config.EYE_FEATURES."""
    out = np.full((n_windows, len(EYE_FEATURES)), np.nan)
    t, px, py = pupil[:, 0], pupil[:, 1], pupil[:, 2]
    fon, fdur = (fixations[:, 0], fixations[:, 1]) if len(fixations) else (np.array([]), np.array([]))
    for w in range(n_windows):
        a, b = w * win_s, (w + 1) * win_s
        fd = fdur[(fon >= a) & (fon < b)]
        if fd.size:
            out[w, 0:4] = [fd.mean(), fd.std(), fd.size / win_s, fd.max()]
        else:
            out[w, 0:4] = [0.0, 0.0, 0.0, 0.0]
        sel = (t >= a) & (t < b)
        xs, ys = px[sel], py[sel]
        xs, ys = xs[np.isfinite(xs)], ys[np.isfinite(ys)]
        if xs.size > 1:
            out[w, 4:6] = [xs.mean(), xs.std()]
        if ys.size > 1:
            out[w, 6:8] = [ys.mean(), ys.std()]
    return out


def gate_mask(values, trial_index, mode="quantile", q=0.6, ratio=0.9):
    """Boolean mask of windows to keep, decided separately inside each trial.

    quantile : keep windows whose gate value >= the q-quantile of that trial
    max_ratio: keep windows whose gate value >= ratio * max of that trial
    none     : keep everything (ablation)
    """
    keep = np.zeros(len(values), dtype=bool)
    if mode == "none":
        keep[:] = True
        return keep
    for tr in np.unique(trial_index):
        idx = np.where(trial_index == tr)[0]
        v = values[idx]
        ok = np.isfinite(v)
        if not ok.any():
            continue
        if mode == "quantile":
            thr = np.quantile(v[ok], q)
        elif mode == "max_ratio":
            thr = ratio * np.nanmax(v)
        else:
            raise ValueError(mode)
        keep[idx[ok & (v >= thr)]] = True
    return keep
