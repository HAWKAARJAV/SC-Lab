"""Multi-band weighted EEG probability fusion (M3).

Trains one Sugeno fuzzy classifier per frequency band and fuses class probabilities
(equal or validation-learned weights). Weight learning uses training data only
(inner hold-out trials within each CV fold; never the test fold).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .compare import fit_predict
from .config import Config
from .evaluate import band_matrix, zscore_by_subject
from .fuzzy import SugenoFIS
from .ga import tune_rules
from .mrmr import mrmr_weighted

FUSION_EXPERIMENTS: dict[str, tuple[str, ...]] = {
    "alpha_only": ("alpha",),
    "alpha_beta": ("alpha", "beta"),
    "alpha_beta_gamma": ("alpha", "beta", "gamma"),
    "five_equal": ("delta", "theta", "alpha", "beta", "gamma"),
    "five_learned": ("delta", "theta", "alpha", "beta", "gamma"),
}

DEFAULT_N_LEVELS = 3


def train_band_classifier(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    cfg: Config,
    n_classes: int,
    feature_names: list[str],
    rng: np.random.Generator,
    n_levels: int = DEFAULT_N_LEVELS,
) -> SugenoFIS:
    sel = mrmr_weighted(X, y, w, k=cfg.n_selected, n_bins=cfg.mrmr_bins)
    Xs = X[:, sel]
    fnames = [feature_names[i] for i in sel]
    fis = SugenoFIS.fit_mfs(Xs, y, n_levels, n_classes, fnames, sw=w)
    fis.wang_mendel(Xs, y, sw=w)
    fis, _ = tune_rules(fis, Xs, y, cfg, rng, sw=w)
    fis._feature_idx = np.asarray(sel)  # type: ignore[attr-defined]
    return fis


def band_predict_proba(fis: SugenoFIS, X: np.ndarray) -> np.ndarray:
    idx = getattr(fis, "_feature_idx", np.arange(X.shape[1]))
    return fis.predict_proba(X[:, idx])


def normalize_fusion_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s <= 0:
        return np.ones(len(w)) / len(w)
    return w / s


def fuse_equal(probas: list[np.ndarray]) -> np.ndarray:
    if not probas:
        raise ValueError("no band probabilities")
    P = np.stack(probas, axis=0).mean(axis=0)
    s = P.sum(axis=1, keepdims=True)
    return np.where(s > 0, P / np.maximum(s, 1e-12), P)


def fuse_weighted(probas: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    w = normalize_fusion_weights(weights)
    P = sum(wi * Pi for wi, Pi in zip(w, probas))
    s = P.sum(axis=1, keepdims=True)
    return np.where(s > 0, P / np.maximum(s, 1e-12), P)


def learn_fusion_weights(
    band_probas_val: dict[str, np.ndarray],
    y_val: np.ndarray,
    *,
    n_grid: int = 11,
) -> np.ndarray:
    names = list(band_probas_val.keys())
    n_b = len(names)
    if n_b == 1:
        return np.ones(1)
    probas = [band_probas_val[n] for n in names]
    if n_b == 2:
        grid = np.linspace(0.0, 1.0, n_grid)
        best_w, best_acc = None, -1.0
        for a in grid:
            w = np.array([a, 1.0 - a])
            pred = fuse_weighted(probas, w).argmax(axis=1)
            acc = (pred == y_val).mean()
            if acc > best_acc:
                best_acc, best_w = acc, w
        return normalize_fusion_weights(best_w)

    w = np.ones(n_b) / n_b
    for _ in range(8):
        improved = False
        for bi in range(n_b):
            grid = np.linspace(0.0, 1.0, n_grid)
            local_best, local_acc = w.copy(), -1.0
            for t in grid:
                trial = w.copy()
                rem = max(1.0 - t, 1e-12)
                others = np.delete(w, bi)
                if others.sum() > 0:
                    trial[np.arange(n_b) != bi] = others / others.sum() * rem
                trial[bi] = t
                trial = normalize_fusion_weights(trial)
                pred = fuse_weighted(probas, trial).argmax(axis=1)
                acc = (pred == y_val).mean()
                if acc > local_acc:
                    local_acc, local_best = acc, trial
            if local_acc > (fuse_weighted(probas, w).argmax(1) == y_val).mean() + 1e-9:
                w = local_best
                improved = True
        if not improved:
            break
    return normalize_fusion_weights(w)


def trial_level_vote(
    proba: np.ndarray,
    gate_w: np.ndarray,
    trial_ix: np.ndarray,
    y_true: np.ndarray,
) -> list[tuple[int, int, int]]:
    preds = []
    for t in np.unique(trial_ix):
        m = trial_ix == t
        ww = gate_w[m]
        S = (ww[:, None] * proba[m]).sum(0) / max(ww.sum(), 1e-12)
        preds.append((int(t), int(y_true[m][0]), int(S.argmax())))
    return preds


def _inner_val_trials(trial_ids: np.ndarray, labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(trial_ids) < 6:
        mid = len(trial_ids) // 2
        return trial_ids[:mid], trial_ids[mid:]
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    tr_i, va_i = next(skf.split(trial_ids, labels))
    return trial_ids[tr_i], trial_ids[va_i]


def run_fusion_fold(
    F: dict,
    bands: tuple[str, ...],
    fusion_mode: str,
    tr_win: np.ndarray,
    te_win: np.ndarray,
    w_all: np.ndarray,
    cfg: Config,
    n_classes: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    y = F["label"]
    trial = F["trial"]
    tr_tids = np.unique(trial[tr_win])
    tlab = np.array([y[trial == t][0] for t in tr_tids])
    fit_tids, val_tids = _inner_val_trials(tr_tids, tlab, int(rng.integers(1_000_000)))
    fit_win = tr_win & np.isin(trial, fit_tids)
    val_win = tr_win & np.isin(trial, val_tids)

    models: dict[str, SugenoFIS] = {}
    band_probas_te: dict[str, np.ndarray] = {}
    band_probas_val: dict[str, np.ndarray] = {}

    for bi, band in enumerate(bands):
        X, names = band_matrix(F, band)
        X = np.nan_to_num(zscore_by_subject(X, F["subject"]))
        band_rng = np.random.default_rng(int(rng.integers(1_000_000)) + bi)
        fis = train_band_classifier(X[fit_win], y[fit_win], w_all[fit_win], cfg, n_classes,
                                    names, band_rng)
        models[band] = fis
        band_probas_val[band] = band_predict_proba(fis, X[val_win])
        band_probas_te[band] = band_predict_proba(fis, X[te_win])

    if fusion_mode == "equal" or len(bands) == 1:
        wvec = np.ones(len(bands)) / len(bands)
    else:
        wvec = learn_fusion_weights(band_probas_val, y[val_win])

    fused_te = fuse_weighted([band_probas_te[b] for b in bands], wvec)
    preds = trial_level_vote(fused_te, w_all[te_win], trial[te_win], y[te_win])
    per_band_preds = {
        b: trial_level_vote(band_probas_te[b], w_all[te_win], trial[te_win], y[te_win])
        for b in bands
    }
    return {
        "fusion_weights": {b: float(wvec[i]) for i, b in enumerate(bands)},
        "fusion_mode": fusion_mode,
        "bands": list(bands),
        "trial_preds": preds,
        "per_band_preds": per_band_preds,
        "models": models,
    }


def run_single_band_fold(
    F: dict,
    band: str,
    tr_win: np.ndarray,
    te_win: np.ndarray,
    w_all: np.ndarray,
    cfg: Config,
    n_classes: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    X, names = band_matrix(F, band)
    X = np.nan_to_num(zscore_by_subject(X, F["subject"]))
    y = F["label"]
    preds, info = fit_predict(X, y, F["trial"], tr_win, te_win, w_all, cfg, n_classes, names, rng)
    return {"band": band, "trial_preds": preds, "info": info}
