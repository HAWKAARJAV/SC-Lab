"""M2: adaptive temporal fuzzy eye gate.

Two additions on top of a fuzzy Low/Medium/High pupil gate:

* Adaptive: each participant's pupil signal is imputed (within trial), robust-scaled against
  that participant's own range (P5-P95), and the Low/Medium/High breakpoints are placed at the
  participant's own 20th/50th/80th percentiles of the scaled signal, so "high pupil" means
  "high for this person". Everything is label-free and uses only that participant's eye data,
  so it also works for an unseen participant in leave-one-subject-out testing.
* Temporal: the window weights are smoothed along time within each trial, either symmetric
  (t-1, t, t+1; offline) or causal (t-1, t; usable in real time).

Rewritten from the earlier version, whose per-subject baseline subtraction was cancelled by a
subsequent within-trial rescaling, and whose within-trial scaling collapses on short trials.
"""
import numpy as np

from .gates import FUZZY, fuzzy_memberships, normalise_within_subject

PERCENTILES = (20.0, 50.0, 80.0)
SPAN = 1                    # windows on each side used for smoothing

M2_VARIANTS = {             # name -> temporal smoothing mode
    "adaptive": "symmetric",            # M2 (full)
    "adaptive_nosmooth": None,          # M2 without the temporal part
    "adaptive_causal": "causal",        # M2 with real-time-compatible smoothing
}


def impute_within_trial(v, trial, window):
    """Forward/backward fill missing values along time inside each trial; trial median last."""
    out = np.asarray(v, dtype=float).copy()
    for t in np.unique(trial):
        idx = np.where(trial == t)[0]
        idx = idx[np.argsort(window[idx])]
        x = out[idx]
        ok = np.isfinite(x)
        if not ok.any():
            continue
        pos = np.where(ok)[0]
        fill = np.interp(np.arange(len(x)), pos, x[pos])   # flat extrapolation at the ends
        out[idx] = np.where(ok, x, fill)
    return out


def subject_breakpoints(z, subject, pct=PERCENTILES, min_gap=0.05):
    """Per-subject (a1, c, b2) for the Low/Medium/High memberships."""
    params = {}
    for s in np.unique(subject):
        zs = z[subject == s]
        zs = zs[np.isfinite(zs)]
        if zs.size < 5:
            params[s] = (FUZZY["a1"], FUZZY["c"], FUZZY["b2"])
            continue
        a1, c, b2 = np.percentile(zs, pct)
        c = float(np.clip(c, min_gap, 1 - min_gap))
        a1 = float(min(a1, c - min_gap))
        b2 = float(max(b2, c + min_gap))
        params[s] = (a1, c, b2)
    return params


def adaptive_weights(z, subject, params, r=FUZZY["r"]):
    w = np.empty(len(z))
    for s, (a1, c, b2) in params.items():
        m = subject == s
        muL, muM, muH = fuzzy_memberships(z[m], a1, c, b2)
        den = muL + muM + muH
        w[m] = np.where(den > 1e-12, (muL * r[0] + muM * r[1] + muH * r[2]) / np.maximum(den, 1e-12), r[0])
    return np.clip(w, r[0], 1.0)


def smooth_within_trial(w, trial, window, mode="symmetric", span=SPAN):
    if mode is None or span <= 0:
        return w.copy()
    out = w.copy()
    for t in np.unique(trial):
        idx = np.where(trial == t)[0]
        idx = idx[np.argsort(window[idx])]
        x = w[idx]
        n = len(x)
        c = np.concatenate([[0.0], np.cumsum(x)])
        lo = np.maximum(np.arange(n) - span, 0)
        hi = np.arange(n) + 1 if mode == "causal" else np.minimum(np.arange(n) + span + 1, n)
        out[idx] = (c[hi] - c[lo]) / (hi - lo)
    return out


def m2_weights(v, subject, trial, window, variant="adaptive"):
    mode = M2_VARIANTS[variant]
    v = impute_within_trial(v, trial, window)
    z = normalise_within_subject(v, subject)
    w = adaptive_weights(z, subject, subject_breakpoints(z, subject))
    return smooth_within_trial(w, trial, window, mode)
