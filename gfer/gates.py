"""Shared eye-gate feature normalisation and hard/fuzzy weighting (M0/M1)."""
import numpy as np

FUZZY = dict(a1=0.2, c=0.5, b2=0.8, r=(0.1, 0.5, 1.0))
HARD_Q = 0.6


def normalise_within_trial(v, trial):
    z = np.full(len(v), 0.5)
    for t in np.unique(trial):
        m = trial == t
        x = v[m]
        ok = np.isfinite(x)
        if ok.sum() < 2:
            continue
        lo, hi = np.percentile(x[ok], [5, 95])
        zz = np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1)
        zz[~ok] = 0.5
        z[m] = zz
    return z


def fuzzy_memberships(z, a1=0.2, c=0.5, b2=0.8):
    z = np.asarray(z, dtype=float)
    muL = np.clip((c - z) / max(c - a1, 1e-12), 0, 1)
    muH = np.clip((z - c) / max(b2 - c, 1e-12), 0, 1)
    muM = np.clip(np.minimum((z - a1) / max(c - a1, 1e-12), (b2 - z) / max(b2 - c, 1e-12)), 0, 1)
    return muL, muM, muH


def hard_weights(z, trial, q=HARD_Q):
    w = np.zeros(len(z))
    for t in np.unique(trial):
        m = np.where(trial == t)[0]
        w[m[z[m] >= np.quantile(z[m], q)]] = 1.0
    return w


def fuzzy_weights(z, P=None):
    P = FUZZY if P is None else P
    muL, muM, muH = fuzzy_memberships(z, P["a1"], P["c"], P["b2"])
    rL, rM, rH = P["r"]
    denom = muL + muM + muH
    out = np.where(denom > 1e-12, (muL * rL + muM * rM + muH * rH) / np.maximum(denom, 1e-12), rL)
    return np.clip(out, rL, 1.0)


def gate_weights(z, trial, variant):
    if variant == "none":
        return np.ones(len(z))
    if variant == "hard":
        return hard_weights(z, trial, HARD_Q)
    if variant == "fuzzy":
        return fuzzy_weights(z, FUZZY)
    if variant == "linear":
        return np.maximum(FUZZY["r"][0], z)
    raise ValueError(variant)
