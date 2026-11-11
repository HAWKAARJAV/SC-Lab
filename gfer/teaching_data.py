"""Deterministic synthetic teaching world: shared windows, gate-conditioned band classifiers, fusion."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .adaptive_eye import (
    adaptive_fuzzy_params,
    compute_gate_weights,
    impute_gate_values,
    normalize_pupil_feature,
    participant_baselines,
    temporal_smooth_weights,
)
from .gates import FUZZY, fuzzy_weights, hard_weights

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
CLASSES = ["sadness", "happiness", "fear", "neutrality"]
N_CLASSES = 4

TARGET_REFERENCE = {
    "SEED-IV_reference_scale": {"M0": 94.42, "M1": 94.75, "M2": 95.25, "M3": 95.40, "M4": 95.95},
    "MAHNOB_reference_scale": {"M0": 85.20, "M1": 85.55, "M2": 86.05, "M3": 86.20, "M4": 86.75},
}

# Band skill / confusion priors (illustrative, not calibrated to paper accuracies).
BAND_LOGIT_SCALE = {
    "alpha": 1.35,
    "beta": 1.15,
    "gamma": 0.95,
    "theta": 0.75,
    "delta": 0.65,
}
BAND_NOISE = {
    "alpha": 0.55,
    "beta": 0.65,
    "gamma": 0.75,
    "theta": 0.90,
    "delta": 1.05,
}
# Structured confusions: fraction of mass leaking to these alternate classes.
BAND_CONFUSION = {
    "alpha": {1: 0.10, 3: 0.06},
    "beta": {0: 0.08, 2: 0.10},
    "gamma": {2: 0.12, 3: 0.08},
    "theta": {0: 0.14, 1: 0.08},
    "delta": {0: 0.10, 3: 0.12},
}

DATASET_DIFFICULTY = {
    "SEED-IV_reference_scale": 1.0,
    "MAHNOB_reference_scale": 1.25,
}


def stable_seed(*parts: str | int) -> int:
    msg = ":".join(str(p) for p in parts)
    digest = hashlib.sha256(msg.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=axis, keepdims=True), 1e-12)


def child_rng(master: int, *labels: str) -> np.random.Generator:
    return np.random.default_rng(stable_seed(master, *labels))


def fuse_band_probs(band_probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.clip(weights, 0.0, None)
    w = w / np.maximum(w.sum(), 1e-12)
    fused = (w[None, :, None] * band_probs).sum(axis=1)
    row_s = fused.sum(axis=1, keepdims=True)
    return np.where(row_s > 0, fused / np.maximum(row_s, 1e-12), fused)


def learn_weights(band_probs: np.ndarray, y: np.ndarray, val_idx: np.ndarray) -> np.ndarray:
    acc = []
    for b in range(len(BANDS)):
        pred = band_probs[val_idx, b].argmax(1)
        acc.append((pred == y[val_idx]).mean())
    a = np.asarray(acc, dtype=float)
    if a.sum() < 1e-9:
        return np.ones(len(BANDS)) / len(BANDS)
    return a / a.sum()


def verify_fusion(proba: np.ndarray, band_probs: np.ndarray, weights: np.ndarray, tol: float = 1e-9) -> None:
    recalc = fuse_band_probs(band_probs, weights)
    if not np.allclose(proba, recalc, atol=tol, rtol=0):
        d = np.abs(proba - recalc).max()
        raise AssertionError(f"fusion mismatch max delta {d}")
    if not np.all((proba >= 0) & (proba <= 1 + 1e-9)):
        raise AssertionError("probabilities out of [0,1]")
    if not np.allclose(proba.sum(1), 1.0, atol=1e-6):
        raise AssertionError("fused rows do not sum to 1")
    if np.any(proba.argmax(1) != proba.argmax(1)):
        pass
    w = np.clip(weights, 0, None)
    if not np.isclose(w.sum(), 1.0, atol=1e-9) and w.sum() > 0:
        raise AssertionError("weights do not sum to 1")


def _band_logits(
    y: np.ndarray,
    band: str,
    rng: np.random.Generator,
    difficulty: float,
) -> np.ndarray:
    """Class logits without gating (n, 4)."""
    n = len(y)
    logits = np.zeros((n, N_CLASSES), dtype=float)
    scale = BAND_LOGIT_SCALE[band] / difficulty
    noise = BAND_NOISE[band] * difficulty
    confuse = BAND_CONFUSION.get(band, {})
    for i in range(n):
        logits[i] = rng.normal(0, 0.15, N_CLASSES)
        logits[i, y[i]] += scale
        for alt, leak in confuse.items():
            logits[i, alt] += scale * leak
        logits[i] += rng.normal(0, noise, N_CLASSES)
    return logits


def _gate_strength(w: np.ndarray, floor: float = 0.20) -> np.ndarray:
    return floor + (1.0 - floor) * np.clip(w, 0.0, 1.0)


def _logits_to_band_probs(logits: np.ndarray, strength: np.ndarray) -> np.ndarray:
    """Low gate weight attenuates class evidence (approaches softer distribution)."""
    adj = logits * strength[:, None]
    return _softmax(adj, axis=1)


def _compute_gate_quality(episode: np.ndarray, w: np.ndarray, hard: bool | None) -> dict:
    ep = episode > 0.5
    if hard is None:
        sel = np.ones(len(w), dtype=bool)
    elif hard:
        sel = w >= 0.5
    else:
        sel = w >= 0.35
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
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "episode_auc": ep_auc,
        "windows_retained_pct": 100.0 * sel.mean(),
    }


@dataclass
class ModelOutputs:
    y_pred: np.ndarray
    proba: np.ndarray
    band_probs: np.ndarray
    weights_learned: np.ndarray
    weights_equal: np.ndarray
    val_idx: np.ndarray
    folds: list[dict]
    metrics_pooled: dict
    metrics_per_class: list[dict]
    per_band_acc: dict[str, float]


@dataclass
class TeachingWorld:
    dataset: str
    master_seed: int
    n: int
    subject_id: np.ndarray
    trial_id: np.ndarray
    window_id: np.ndarray
    y: np.ndarray
    episode: np.ndarray
    pupil_raw: np.ndarray
    pupil_baselines: dict[str, dict]
    baselines_train: dict[str, float]
    pupil_norm: np.ndarray
    w_hard: np.ndarray
    w_fuzzy: np.ndarray
    w_smooth: np.ndarray
    w_adaptive: np.ndarray
    fold_splits: list[tuple[np.ndarray, np.ndarray]]
    val_idx: np.ndarray
    difficulty: float
    rng: np.random.Generator
    base_logits: np.ndarray  # (n, 5, 4) ungated

    def gate_for(self, model: str) -> np.ndarray:
        if model in ("M0", "M3"):
            return self.w_hard
        if model == "M1":
            return self.w_fuzzy
        if model in ("M2", "M4"):
            return self.w_adaptive
        raise ValueError(model)

    def band_probabilities(self, model: str) -> np.ndarray:
        w = self.gate_for(model)
        strength = _gate_strength(w)
        out = np.zeros_like(self.base_logits)
        for b in range(len(BANDS)):
            out[:, b, :] = _logits_to_band_probs(self.base_logits[:, b, :], strength)
        return out

    def single_band_proba(self, model: str) -> np.ndarray:
        """M0–M2: gate-conditioned band classifier(s) per model role."""
        w = self.gate_for(model)
        strength = _gate_strength(w)
        if model == "M0":
            bi = BANDS.index("alpha")
            return _logits_to_band_probs(self.base_logits[:, bi, :], strength)
        if model == "M1":
            bi = BANDS.index("beta")
            return _logits_to_band_probs(self.base_logits[:, bi, :], strength)
        if model == "M2":
            pa = _logits_to_band_probs(self.base_logits[:, BANDS.index("alpha"), :], strength)
            pg = _logits_to_band_probs(self.base_logits[:, BANDS.index("gamma"), :], strength)
            mix = 0.65 * pa + 0.35 * pg
            mix /= mix.sum(axis=1, keepdims=True)
            return mix
        raise ValueError(model)

    def fusion_outputs(self, model: str) -> ModelOutputs:
        band_probs = self.band_probabilities(model)
        w_eq = np.ones(len(BANDS)) / len(BANDS)
        w_lr = learn_weights(band_probs, self.y, self.val_idx)
        fused_eq = fuse_band_probs(band_probs, w_eq)
        fused = fuse_band_probs(band_probs, w_lr)
        y_pred = fused.argmax(1)
        verify_fusion(fused, band_probs, w_lr)
        folds = self._fold_metrics_fusion(band_probs, model)
        return ModelOutputs(
            y_pred=y_pred,
            proba=fused,
            band_probs=band_probs,
            weights_learned=w_lr,
            weights_equal=w_eq,
            val_idx=self.val_idx,
            folds=folds,
            metrics_pooled=self._pooled_metrics(self.y, y_pred, fused),
            metrics_per_class=self._per_class(self.y, y_pred),
            per_band_acc={BANDS[b]: float((band_probs[:, b].argmax(1) == self.y).mean() * 100) for b in range(5)},
        )

    def classifier_outputs(self, model: str) -> ModelOutputs:
        proba = self.single_band_proba(model)
        y_pred = proba.argmax(1)
        band_probs = self.band_probabilities(model)
        folds = self._fold_metrics(y_pred, proba, model)
        return ModelOutputs(
            y_pred=y_pred,
            proba=proba,
            band_probs=band_probs,
            weights_learned=np.ones(len(BANDS)) / len(BANDS),
            weights_equal=np.ones(len(BANDS)) / len(BANDS),
            val_idx=self.val_idx,
            folds=folds,
            metrics_pooled=self._pooled_metrics(self.y, y_pred, proba),
            metrics_per_class=self._per_class(self.y, y_pred),
            per_band_acc={BANDS[b]: float((band_probs[:, b].argmax(1) == self.y).mean() * 100) for b in range(5)},
        )

    def _pooled_metrics(self, y, y_pred, proba) -> dict:
        return {
            "accuracy_pct": accuracy_score(y, y_pred) * 100,
            "balanced_accuracy_pct": balanced_accuracy_score(y, y_pred) * 100,
            "macro_precision_pct": precision_score(y, y_pred, average="macro", zero_division=0) * 100,
            "macro_recall_pct": recall_score(y, y_pred, average="macro", zero_division=0) * 100,
            "macro_f1_pct": f1_score(y, y_pred, average="macro", zero_division=0) * 100,
            "weighted_f1_pct": f1_score(y, y_pred, average="weighted", zero_division=0) * 100,
            "macro_auc": float(roc_auc_score(y, proba, multi_class="ovr", average="macro")),
        }

    def _per_class(self, y, y_pred) -> list[dict]:
        p, r, f1, sup = precision_recall_fscore_support(y, y_pred, labels=range(4), zero_division=0)
        rows = []
        for c, name in enumerate(CLASSES):
            rows.append({
                "class": name,
                "precision_pct": float(p[c]) * 100,
                "recall_pct": float(r[c]) * 100,
                "f1_pct": float(f1[c]) * 100,
                "support": int(sup[c]),
            })
        return rows

    def _fold_metrics(self, y_pred: np.ndarray, proba: np.ndarray, model: str) -> list[dict]:
        rows = []
        for fi, (tr, te) in enumerate(self.fold_splits):
            rows.append({
                "fold": fi + 1,
                "model": model,
                "accuracy_pct": accuracy_score(self.y[te], y_pred[te]) * 100,
                "macro_f1_pct": f1_score(self.y[te], y_pred[te], average="macro", zero_division=0) * 100,
                "balanced_accuracy_pct": balanced_accuracy_score(self.y[te], y_pred[te]) * 100,
                "n_test": int(len(te)),
            })
        return rows

    def _fold_metrics_fusion(self, band_probs: np.ndarray, model: str) -> list[dict]:
        rows = []
        for fi, (tr, te) in enumerate(self.fold_splits):
            n_val = max(1, int(0.2 * len(tr)))
            sub = tr.copy()
            child_rng(self.master_seed, self.dataset, "fold", fi, model).shuffle(sub)
            v_idx, t_idx = sub[:n_val], sub[n_val:]
            w = learn_weights(band_probs, self.y, v_idx)
            fused_te = fuse_band_probs(band_probs[te], w)
            yhat = fused_te.argmax(1)
            rows.append({
                "fold": fi + 1,
                "model": model,
                "accuracy_pct": accuracy_score(self.y[te], yhat) * 100,
                "macro_f1_pct": f1_score(self.y[te], yhat, average="macro", zero_division=0) * 100,
                "balanced_accuracy_pct": balanced_accuracy_score(self.y[te], yhat) * 100,
                "n_test": int(len(te)),
            })
        return rows

    def gate_quality_table(self) -> dict[str, dict]:
        ep = self.episode
        return {
            "none": _compute_gate_quality(ep, np.ones(len(ep)), None),
            "hard": _compute_gate_quality(ep, self.w_hard, True),
            "fuzzy": _compute_gate_quality(ep, self.w_fuzzy, False),
            "adaptive": _compute_gate_quality(ep, self.w_adaptive, False),
        }


def build_teaching_world(dataset: str, n: int, master_seed: int) -> TeachingWorld:
    difficulty = DATASET_DIFFICULTY[dataset]
    rng = child_rng(master_seed, dataset, "world", n)
    subjects = np.array([f"S{(i % 20) + 1:02d}" for i in range(n)])
    trial_id = np.arange(n) // 8
    window_id = np.arange(n) % 8
    y = np.empty(n, dtype=int)
    for c in range(N_CLASSES):
        slots = np.where(np.arange(n) % N_CLASSES == c)[0]
        if len(slots) < n // N_CLASSES:
            pass
    y = rng.integers(0, N_CLASSES, n)
    episode = (rng.random(n) > (0.50 + 0.05 * (difficulty - 1))).astype(float)

    pupil = np.zeros(n)
    for s in np.unique(subjects):
        m = subjects == s
        base = 3.6 + 0.1 * rng.normal()
        pupil[m] = base + 0.35 * episode[m] + rng.normal(0, 0.09 * difficulty, m.sum())

    train_mask = np.zeros(n, dtype=bool)
    train_mask[: int(0.6 * n)] = True
    v_imputed = impute_gate_values(pupil, trial_id, window_id)
    baselines = participant_baselines(v_imputed, subjects, train_mask)
    pupil_norm = normalize_pupil_feature(v_imputed, subjects, trial_id, baselines)
    z_train = pupil_norm.copy()
    adapt_params = adaptive_fuzzy_params(z_train, subjects, train_mask)

    w_hard = hard_weights(pupil_norm, trial_id, 0.6)
    w_fuzzy = fuzzy_weights(pupil_norm, FUZZY)
    w_smooth = temporal_smooth_weights(w_fuzzy, trial_id, window_id, mode="symmetric", span=1)
    w_adaptive = compute_gate_weights(
        v_imputed, subjects, trial_id, window_id, train_mask, "adaptive",
    )

    base_logits = np.zeros((n, len(BANDS), N_CLASSES))
    for bi, band in enumerate(BANDS):
        base_logits[:, bi, :] = _band_logits(y, band, child_rng(master_seed, dataset, band), difficulty)

    skf = StratifiedKFold(5, shuffle=True, random_state=stable_seed(master_seed, dataset, "folds"))
    fold_splits = list(skf.split(y, y))
    perm = np.arange(n)
    rng.shuffle(perm)
    val_idx = perm[: max(1, int(0.2 * n))]

    base_rows = {}
    for s in np.unique(subjects):
        m = subjects == s
        vals = v_imputed[m]
        base_rows[s] = {
            "subject_id": s,
            "baseline_mean": float(np.mean(vals)),
            "baseline_std": float(np.std(vals) + 1e-6),
            "baseline_window_count": int(m.sum()),
        }

    return TeachingWorld(
        dataset=dataset,
        master_seed=master_seed,
        n=n,
        subject_id=subjects,
        trial_id=trial_id,
        window_id=window_id,
        y=y,
        episode=episode,
        pupil_raw=pupil,
        pupil_baselines=base_rows,
        baselines_train=baselines,
        pupil_norm=pupil_norm,
        w_hard=w_hard,
        w_fuzzy=w_fuzzy,
        w_smooth=w_smooth,
        w_adaptive=w_adaptive,
        fold_splits=fold_splits,
        val_idx=val_idx,
        difficulty=difficulty,
        rng=rng,
        base_logits=base_logits,
    )


def band_accuracy_with_gate(band_probs, y, w, thr):
    sel = w >= thr
    if sel.sum() < 5:
        sel = np.ones(len(y), dtype=bool)
    return {BANDS[b]: float((band_probs[sel, b].argmax(1) == y[sel]).mean() * 100) for b in range(5)}


def gate_ablation_band_table(band_probs, y, w_hard, w_fuzzy):
    return {
        "No gate": band_accuracy_with_gate(band_probs, y, np.ones(len(y)), 0.0),
        "Hard gate (M0)": band_accuracy_with_gate(band_probs, y, w_hard, 0.5),
        "Fuzzy gate (M1)": band_accuracy_with_gate(band_probs, y, w_fuzzy, 0.35),
    }


def fusion_accuracy_for_bands(band_probs, y, bands, weights):
    idx = [BANDS.index(b) for b in bands]
    sub = band_probs[:, idx, :]
    if len(bands) == 1:
        w = np.array([1.0])
    else:
        w = weights[idx] / weights[idx].sum()
    fused = fuse_band_probs(sub, w)
    return float((fused.argmax(1) == y).mean() * 100)


def noise_gate_sweep(world: TeachingWorld, noise_sigmas: list[float]) -> tuple[list[float], dict, dict]:
    rng = child_rng(world.master_seed, "noise")
    gate_f1, emo_acc = {}, {}
    for policy in ("Hard", "Fuzzy", "Adaptive"):
        gate_f1[policy] = []
        emo_acc[policy] = []
    ep = world.episode > 0.5
    for sigma in noise_sigmas:
        pupil = world.pupil_raw + rng.normal(0, sigma, world.n)
        v = impute_gate_values(pupil, world.trial_id, world.window_id)
        z = normalize_pupil_feature(v, world.subject_id, world.trial_id, world.baselines_train)
        wh = hard_weights(z, world.trial_id, 0.6)
        wf = fuzzy_weights(z, FUZZY)
        wa = temporal_smooth_weights(wf, world.trial_id, world.window_id, span=1)
        for policy, w in [("Hard", wh), ("Fuzzy", wf), ("Adaptive", wa)]:
            gq = _compute_gate_quality(ep.astype(float), w, policy == "Hard")
            gate_f1[policy].append(gq["f1"] * 100)
            strength = _gate_strength(w)
            fused = fuse_band_probs(
                np.stack(
                    [_logits_to_band_probs(world.base_logits[:, b], strength) for b in range(5)],
                    axis=1,
                ),
                np.ones(5) / 5,
            )
            emo_acc[policy].append(accuracy_score(world.y, fused.argmax(1)) * 100)
    return noise_sigmas, gate_f1, emo_acc


# Legacy API removed: no post-hoc accuracy adjustment.
def build_fusion_dataset(*_a, **_k):
    raise RuntimeError("build_fusion_dataset removed — use build_teaching_world + fusion_outputs")


def emotion_head_from_fusion(*_a, **_k):
    raise RuntimeError("emotion_head_from_fusion removed — no post-hoc accuracy forcing")
