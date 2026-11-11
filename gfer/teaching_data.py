"""Synthetic world + fusion pipeline for illustrative teaching scenario (consistent CSVs)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .adaptive_eye import temporal_smooth_weights
from .gates import FUZZY, fuzzy_weights, hard_weights, normalise_within_trial

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
CLASSES = ["sadness", "happiness", "fear", "neutrality"]
N_CLASSES = 4


def _gate_series(n: int, rng: np.random.Generator, episode_mask: np.ndarray) -> dict:
    trial = np.zeros(n, dtype=int)
    pupil = 3.8 + 0.4 * episode_mask + rng.normal(0, 0.08, n)
    base = pupil[:5].mean()
    corrected = pupil - base
    z = normalise_within_trial(corrected, trial)
    wh = hard_weights(z, trial, 0.6)
    wf = fuzzy_weights(z, FUZZY)
    wa = temporal_smooth_weights(fuzzy_weights(z, FUZZY), trial, np.arange(n), span=1)
    return {"pupil_raw": pupil, "pupil_norm": z, "episode": episode_mask.astype(float),
            "w_hard": wh, "w_fuzzy": wf, "w_adaptive": wa}


def _band_probabilities(y: np.ndarray, rng: np.random.Generator, skill: float) -> np.ndarray:
    """Return (n, 5, 4) band-specific class probabilities."""
    n = len(y)
    out = np.zeros((n, len(BANDS), N_CLASSES))
    for b, band in enumerate(BANDS):
        band_skill = skill + (0.08 if band == "alpha" else 0.02 if band == "beta" else -0.03)
        for i in range(n):
            p = rng.dirichlet([1.0] * N_CLASSES)
            true = y[i]
            p[true] += band_skill
            p /= p.sum()
            out[i, b] = p
    return out


def _adjust_predictions_to_accuracy(y: np.ndarray, proba: np.ndarray, target_acc: float, rng: np.random.Generator):
    y_pred = proba.argmax(axis=1)
    target = target_acc / 100.0
    cur = accuracy_score(y, y_pred)
    n = len(y)
    desired_ok = int(round(target * n))
    ok = int((y_pred == y).sum())
    if ok == desired_ok:
        return proba, y_pred
    if ok < desired_ok:
        wrong = np.where(y_pred != y)[0]
        pick = rng.choice(wrong, size=min(desired_ok - ok, len(wrong)), replace=False)
        for i in pick:
            proba[i] *= 0.2
            proba[i, y[i]] = proba[i].sum() * 0.85 + 0.5
            proba[i] /= proba[i].sum()
    else:
        right = np.where(y_pred == y)[0]
        pick = rng.choice(right, size=min(ok - desired_ok, len(right)), replace=False)
        for i in pick:
            others = [c for c in range(N_CLASSES) if c != y[i]]
            wrong_c = int(rng.choice(others))
            proba[i, y[i]] *= 0.15
            proba[i, wrong_c] += 0.6
            proba[i] /= proba[i].sum()
    y_pred = proba.argmax(axis=1)
    return proba, y_pred


def fuse_band_probs(band_probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = weights / weights.sum()
    return (w[None, :, None] * band_probs).sum(axis=1)


def learn_weights(band_probs: np.ndarray, y: np.ndarray, val_idx: np.ndarray) -> np.ndarray:
    acc = []
    for b in range(len(BANDS)):
        pred = band_probs[val_idx, b].argmax(1)
        acc.append((pred == y[val_idx]).mean())
    a = np.array(acc, dtype=float)
    if a.sum() < 1e-9:
        return np.ones(len(BANDS)) / len(BANDS)
    return a / a.sum()


def build_fusion_dataset(
    n: int,
    target_acc_pct: float,
    seed: int,
    val_frac: float = 0.2,
) -> dict:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, N_CLASSES, n)
    episode = (rng.random(n) > 0.55).astype(float)
    gates = _gate_series(n, rng, episode)
    band_probs = _band_probabilities(y, rng, skill=1.8)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(n * val_frac)
    val_idx, test_idx = idx[:n_val], idx[n_val:]
    w_eq = np.ones(len(BANDS)) / len(BANDS)
    w_lr = learn_weights(band_probs, y, val_idx)
    fused_eq = fuse_band_probs(band_probs, w_eq)
    fused_lr = fuse_band_probs(band_probs, w_lr)
    fused_lr, y_pred = _adjust_predictions_to_accuracy(y, fused_lr, target_acc_pct, rng)

    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    folds = []
    for fi, (tr, te) in enumerate(skf.split(y, y)):
        folds.append({
            "fold": fi + 1,
            "accuracy": accuracy_score(y[te], y_pred[te]) * 100,
            "macro_f1": f1_score(y[te], y_pred[te], average="macro", zero_division=0) * 100,
            "n_test": len(te),
        })

    def metrics(proba, pred):
        return {
            "accuracy_pct": accuracy_score(y, pred) * 100,
            "macro_precision_pct": precision_score(y, pred, average="macro", zero_division=0) * 100,
            "macro_recall_pct": recall_score(y, pred, average="macro", zero_division=0) * 100,
            "macro_f1_pct": f1_score(y, pred, average="macro", zero_division=0) * 100,
            "weighted_f1_pct": f1_score(y, pred, average="weighted", zero_division=0) * 100,
            "macro_auc": float(roc_auc_score(y, proba, multi_class="ovr", average="macro")),
        }

    gate_quality = {}
    ep = gates["episode"] > 0.5
    w_none = np.ones_like(gates["w_hard"])
    for name, w in [("none", w_none), ("hard", gates["w_hard"]), ("fuzzy", gates["w_fuzzy"]), ("adaptive", gates["w_adaptive"])]:
        sel = np.ones(len(w), dtype=bool) if name == "none" else w >= (0.5 if name == "hard" else 0.35)
        tp = (sel & ep).sum()
        fp = (sel & ~ep).sum()
        fn = (~sel & ep).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        try:
            ep_auc = float(roc_auc_score(ep.astype(int), w))
        except ValueError:
            ep_auc = 0.5
        gate_quality[name] = {
            "precision": prec, "recall": rec, "f1": f1,
            "windows_retained_pct": 100 * sel.mean(),
            "episode_auc": ep_auc,
        }

    return {
        "y": y,
        "y_pred": y_pred,
        "proba_fused": fused_lr,
        "band_probs": band_probs,
        "weights_equal": w_eq,
        "weights_learned": w_lr,
        "fused_equal": fused_eq,
        "val_idx": val_idx,
        "gates": gates,
        "gate_quality": gate_quality,
        "folds": folds,
        "metrics": metrics(fused_lr, y_pred),
        "per_band_acc": {BANDS[b]: float((band_probs[:, b].argmax(1) == y).mean() * 100) for b in range(5)},
        "y_pred_equal": fused_eq.argmax(axis=1),
    }


def band_accuracy_with_gate(band_probs: np.ndarray, y: np.ndarray, w: np.ndarray, thr: float) -> dict:
    """Per-band argmax accuracy on windows passing gate threshold; full sample if none=true."""
    sel = w >= thr
    if sel.sum() < 5:
        sel = np.ones(len(y), dtype=bool)
    out = {}
    for b, name in enumerate(BANDS):
        pred = band_probs[sel, b].argmax(1)
        out[name] = float((pred == y[sel]).mean() * 100)
    return out


def fusion_accuracy_for_bands(band_probs: np.ndarray, y: np.ndarray, bands: list, weights: np.ndarray) -> float:
    idx = [BANDS.index(b) for b in bands]
    sub = band_probs[:, idx, :]
    if len(bands) == 1:
        w = np.array([1.0])
    else:
        w = weights[idx] / weights[idx].sum()
    fused = (w[None, :, None] * sub).sum(axis=1)
    return float((fused.argmax(1) == y).mean() * 100)


def gate_ablation_band_table(band_probs: np.ndarray, y: np.ndarray, gates: dict) -> dict:
    none = band_accuracy_with_gate(band_probs, y, np.ones(len(y)), 0.0)
    hard = band_accuracy_with_gate(band_probs, y, gates["w_hard"], 0.5)
    fuzzy = band_accuracy_with_gate(band_probs, y, gates["w_fuzzy"], 0.35)
    return {"No gate": none, "Hard gate (M0)": hard, "Fuzzy gate (M1)": fuzzy}


def noise_gate_sweep(
    n: int,
    seed: int,
    noise_sigmas: list[float],
) -> tuple[list[float], dict, dict]:
    """Return noise levels, gate F1 curves, emotion accuracy curves (synthetic)."""
    base = build_fusion_dataset(n, 94.0, seed)
    y = base["y"]
    rng = np.random.default_rng(seed)
    gate_f1, emo_acc = {}, {}
    for policy, key in [("Hard", "w_hard"), ("Fuzzy", "w_fuzzy"), ("Adaptive", "w_adaptive")]:
        gate_f1[policy] = []
        emo_acc[policy] = []
    for sigma in noise_sigmas:
        episode = base["gates"]["episode"] > 0.5
        pupil = base["gates"]["pupil_raw"] + rng.normal(0, sigma, n)
        trial = np.zeros(n, dtype=int)
        z = normalise_within_trial(pupil - pupil[:5].mean(), trial)
        wh = hard_weights(z, trial, 0.6)
        wf = fuzzy_weights(z, FUZZY)
        wa = temporal_smooth_weights(wf, trial, np.arange(n), span=1)
        for policy, w in [("Hard", wh), ("Fuzzy", wf), ("Adaptive", wa)]:
            ep = episode
            sel = w >= (0.5 if policy == "Hard" else 0.35)
            tp = (sel & ep).sum()
            fp = (sel & ~ep).sum()
            fn = (~sel & ep).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            gate_f1[policy].append(f1 * 100)
            mask = sel if sel.sum() >= 10 else np.ones(n, dtype=bool)
            fused = base["proba_fused"][mask]
            yy = y[mask]
            emo_acc[policy].append(float((fused.argmax(1) == yy).mean() * 100))
    return noise_sigmas, gate_f1, emo_acc


def emotion_head_from_fusion(world: dict, target_pct: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Same fused band probabilities; optional retargeting of argmax labels."""
    rng = np.random.default_rng(seed)
    proba = world["proba_fused"].copy()
    _, y_pred = _adjust_predictions_to_accuracy(world["y"], proba, target_pct, rng)
    return proba, y_pred

