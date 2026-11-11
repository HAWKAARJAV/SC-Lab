"""Adaptive Temporal Fuzzy Eye Gate (M2).

Participant baselines and adaptive Low/Medium/High thresholds are fit on training windows
only. Temporal smoothing is applied within each trial over consecutive 4-s window indices.
"""
from __future__ import annotations

import numpy as np

from .gates import FUZZY, fuzzy_memberships, fuzzy_weights, hard_weights, normalise_within_trial

DEFAULT_SUGENO = FUZZY["r"]
SMOOTH_DEFAULT_SPAN = 1

GATE_VARIANTS = (
    "none",
    "hard",
    "fuzzy",
    "adaptive",
    "adaptive_nosmooth",
    "adaptive_causal",
    "linear",
)


def impute_gate_values(values: np.ndarray, trial: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Fill missing pupil/gate measurements within each trial (forward then backward)."""
    out = values.astype(np.float64, copy=True)
    for t in np.unique(trial):
        idx = np.where(trial == t)[0]
        order = idx[np.argsort(window[idx])]
        last = np.nan
        for i in order:
            if np.isfinite(out[i]):
                last = out[i]
            elif np.isfinite(last):
                out[i] = last
        last = np.nan
        for i in order[::-1]:
            if np.isfinite(out[i]):
                last = out[i]
            elif np.isfinite(last):
                out[i] = last
        bad = ~np.isfinite(out[order])
        if bad.any():
            ok = np.isfinite(out[order])
            med = float(np.nanmedian(out[order][ok])) if ok.any() else 0.0
            out[order[bad]] = med
    return out


def participant_baselines(
    values: np.ndarray,
    subjects: np.ndarray,
    train_mask: np.ndarray,
    *,
    min_count: int = 3,
    global_fallback: float | None = None,
) -> dict[str, float]:
    """Training-only per-subject medians; pooled median for sparse subjects."""
    tr = train_mask & np.isfinite(values)
    if global_fallback is None:
        global_fallback = float(np.median(values[tr])) if tr.any() else 0.0
    bases: dict[str, float] = {"__global__": global_fallback}
    for s in np.unique(subjects):
        m = tr & (subjects == s)
        bases[s] = float(np.median(values[m])) if m.sum() >= min_count else global_fallback
    return bases


def baseline_correct(values: np.ndarray, subjects: np.ndarray, baselines: dict[str, float]) -> np.ndarray:
    out = values.astype(np.float64, copy=True)
    g = baselines.get("__global__", 0.0)
    for i, s in enumerate(subjects):
        if np.isfinite(out[i]):
            out[i] = out[i] - baselines.get(s, g)
    return out


def normalize_pupil_feature(
    values: np.ndarray,
    subjects: np.ndarray,
    trial: np.ndarray,
    baselines: dict[str, float],
) -> np.ndarray:
    """Baseline-correct then robust within-trial scaling to z̃ ∈ [0, 1] (label-free)."""
    v = baseline_correct(impute_gate_values(values, trial, np.arange(len(values))), subjects, baselines)
    return normalise_within_trial(v, trial)


def adaptive_fuzzy_params(
    z: np.ndarray,
    subjects: np.ndarray,
    train_mask: np.ndarray,
    *,
    p_low: float = 0.2,
    p_mid: float = 0.5,
    p_high: float = 0.8,
) -> dict[str, tuple[float, float, float]]:
    """Per-subject (a1, c, b2) from training percentiles of z̃."""
    tr = train_mask & np.isfinite(z)
    params: dict[str, tuple[float, float, float]] = {}
    pooled = z[tr] if tr.any() else np.array([0.2, 0.5, 0.8])
    g = _triplet(pooled, p_low, p_mid, p_high)
    params["__global__"] = g
    for s in np.unique(subjects):
        m = tr & (subjects == s)
        params[s] = _triplet(z[m], p_low, p_mid, p_high) if m.sum() >= 5 else g
    return params


def _triplet(z: np.ndarray, p_low: float, p_mid: float, p_high: float) -> tuple[float, float, float]:
    if z.size < 2:
        return (0.2, 0.5, 0.8)
    a1, c, b2 = np.percentile(z, [100 * p_low, 100 * p_mid, 100 * p_high])
    a1, c, b2 = float(a1), float(c), float(b2)
    a1 = min(a1, c - 0.05)
    b2 = max(b2, c + 0.05)
    a1 = np.clip(a1, 0.0, 0.45)
    b2 = np.clip(b2, 0.55, 1.0)
    c = np.clip(c, a1 + 0.05, b2 - 0.05)
    return a1, c, b2


def adaptive_fuzzy_weights(
    z: np.ndarray,
    subjects: np.ndarray,
    params: dict[str, tuple[float, float, float]],
    sugeno_r: tuple[float, float, float] = DEFAULT_SUGENO,
) -> np.ndarray:
    w = np.zeros(len(z), dtype=np.float64)
    g = params.get("__global__", (0.2, 0.5, 0.8))
    for i in range(len(z)):
        a1, c, b2 = params.get(subjects[i], g)
        muL, muM, muH = fuzzy_memberships(np.array([z[i]]), a1, c, b2)
        denom = muL[0] + muM[0] + muH[0]
        if denom <= 1e-12:
            w[i] = sugeno_r[0]
        else:
            w[i] = (muL[0] * sugeno_r[0] + muM[0] * sugeno_r[1] + muH[0] * sugeno_r[2]) / denom
    return np.clip(w, sugeno_r[0], 1.0)


def temporal_smooth_weights(
    w: np.ndarray,
    trial: np.ndarray,
    window: np.ndarray,
    *,
    mode: str = "symmetric",
    span: int = SMOOTH_DEFAULT_SPAN,
) -> np.ndarray:
    """Smooth gate weights along window index within each trial."""
    if mode == "none" or span <= 0:
        return w.astype(np.float64, copy=True)
    out = w.astype(np.float64, copy=True)
    for t in np.unique(trial):
        idx = np.where(trial == t)[0]
        order = idx[np.argsort(window[idx])]
        seq = out[order]
        sm = np.empty_like(seq)
        n = len(seq)
        for j in range(n):
            if mode == "causal":
                lo, hi = max(0, j - span), j + 1
            elif mode == "symmetric":
                lo, hi = max(0, j - span), min(n, j + span + 1)
            else:
                raise ValueError(mode)
            sm[j] = seq[lo:hi].mean()
        out[order] = sm
    return out


def compute_gate_weights(
    values: np.ndarray,
    subjects: np.ndarray,
    trial: np.ndarray,
    window: np.ndarray,
    train_mask: np.ndarray,
    variant: str,
    *,
    hard_q: float = 0.6,
    smooth_span: int = SMOOTH_DEFAULT_SPAN,
) -> np.ndarray:
    """Return continuous (or hard 0/1) weights for every window."""
    if variant == "none":
        return np.ones(len(values), dtype=np.float64)
    v = impute_gate_values(values, trial, window)
    if variant == "hard":
        z = normalise_within_trial(v, trial)
        return hard_weights(z, trial, hard_q)
    if variant == "fuzzy":
        z = normalise_within_trial(v, trial)
        return fuzzy_weights(z, FUZZY)
    if variant == "linear":
        z = normalise_within_trial(v, trial)
        return np.maximum(FUZZY["r"][0], z)
    if variant not in ("adaptive", "adaptive_nosmooth", "adaptive_causal"):
        raise ValueError(f"unknown gate variant: {variant}")

    bases = participant_baselines(v, subjects, train_mask)
    z = normalize_pupil_feature(v, subjects, trial, bases)
    params = adaptive_fuzzy_params(z, subjects, train_mask)
    w = adaptive_fuzzy_weights(z, subjects, params)
    if variant == "adaptive_nosmooth":
        return w
    smooth_mode = "causal" if variant == "adaptive_causal" else "symmetric"
    return temporal_smooth_weights(w, trial, window, mode=smooth_mode, span=smooth_span)


def compare_gate_modes(
    values: np.ndarray,
    subjects: np.ndarray,
    trial: np.ndarray,
    window: np.ndarray,
    train_mask: np.ndarray,
    *,
    modes: tuple[str, ...] = ("hard", "fuzzy", "adaptive_nosmooth", "adaptive"),
) -> dict[str, np.ndarray]:
    return {m: compute_gate_weights(values, subjects, trial, window, train_mask, m) for m in modes}
