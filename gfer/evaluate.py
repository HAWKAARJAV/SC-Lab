"""Cross-validated evaluation of the eye-gated S-FIS-GA emotion recogniser (Sec. III-H)."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold

from .adaptive_eye import compute_gate_weights
from .mrmr import mrmr_weighted
from .eye import gate_mask
from .fuzzy import SugenoFIS
from .ga import tune_rules
from .mrmr import mrmr


# ---------------------------------------------------------------- data shaping
def zscore_by_subject(A, subjects):
    A = A.astype(np.float64).copy()
    for s in np.unique(subjects):
        m = subjects == s
        mu = np.nanmean(A[m], axis=0)
        sd = np.nanstd(A[m], axis=0)
        A[m] = (A[m] - mu) / np.where(sd > 0, sd, 1)
    return A


def band_matrix(F, band):
    """(n, n_ent * n_ch) feature matrix for one band, plus names like 'PhEn-FP2'."""
    bi = F["bands"].index(band)
    X = F["eeg"][:, bi]                                   # (n, n_ent, n_ch)
    names = [f"{e}-{c}" for e in F["entropies"] for c in F["ch_names"]]
    return X.reshape(len(X), -1), names


def select_gate_feature(F, cfg):
    """Rank the 8 eye features with mRMR against the class label (Fig. 7)."""
    eye = F["eye"]
    if cfg.normalize == "subject_zscore":
        eye = zscore_by_subject(eye, F["subject"])
    order, scores, rel = mrmr(eye, F["label"], n_bins=cfg.mrmr_bins)
    ranking = [(F["eye_names"][i], float(s)) for i, s in zip(order, scores)]
    name = ranking[0][0] if cfg.gate_feature == "auto" else cfg.gate_feature
    return name, ranking, rel


def apply_gate(F, cfg, gate_name):
    vals = F["eye"][:, F["eye_names"].index(gate_name)]
    return gate_mask(vals, F["trial"], cfg.gate_mode, cfg.gate_quantile, cfg.gate_max_ratio)


def window_gate_weights(F, gate_name, train_mask, variant="hard"):
    """Continuous gate weights (M0–M2); training mask excludes test-fold statistics."""
    vals = F["eye"][:, F["eye_names"].index(gate_name)].astype(float)
    return compute_gate_weights(vals, F["subject"], F["trial"], F["window"],
                                train_mask, variant)


# ---------------------------------------------------------------- folds
def make_folds(y, subjects, trials, cfg):
    if cfg.protocol == "paper":             # 5-fold CV over pooled segments (as in the paper)
        cv = StratifiedKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed)
        return list(cv.split(np.zeros(len(y)), y))
    if cfg.protocol == "trial":             # no trial contributes to both train and test
        cv = StratifiedGroupKFold(cfg.n_folds, shuffle=True, random_state=cfg.seed)
        return list(cv.split(np.zeros(len(y)), y, trials))
    if cfg.protocol == "loso":              # leave one subject out
        return list(LeaveOneGroupOut().split(np.zeros(len(y)), y, subjects))
    raise ValueError(cfg.protocol)


# ---------------------------------------------------------------- metrics
def metrics(y, proba, n_classes):
    pred = proba.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=range(n_classes))
    tpr = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    out = {
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro", labels=range(n_classes), zero_division=0),
        "precision_macro": precision_score(y, pred, average="macro", labels=range(n_classes),
                                           zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", labels=range(n_classes),
                                     zero_division=0),
        "f1_per_class": f1_score(y, pred, average=None, labels=range(n_classes),
                                 zero_division=0).tolist(),
        "tpr_per_class": tpr.tolist(),
        "fnr_per_class": (1 - tpr).tolist(),
        "confusion": cm.tolist(),
    }
    aucs = []
    for c in range(n_classes):
        yc = (y == c).astype(int)
        aucs.append(roc_auc_score(yc, proba[:, c]) if 0 < yc.sum() < len(yc) else np.nan)
    out["auc_per_class"] = aucs
    out["auc_macro"] = float(np.nanmean(aucs))
    return out


# ---------------------------------------------------------------- one band
def run_band(F, keep, band, cfg, n_classes, n_levels, rng):
    X_all, names = band_matrix(F, band)
    X_all, y_all = X_all[keep], F["label"][keep]
    subj, trial = F["subject"][keep], F["trial"][keep]
    if cfg.normalize == "subject_zscore":
        X_all = zscore_by_subject(X_all, subj)
    X_all = np.nan_to_num(X_all)

    fold_results, all_proba, all_y, models = [], np.zeros((len(y_all), n_classes)), y_all, []
    for fi, (tr, te) in enumerate(make_folds(y_all, subj, trial, cfg)):
        if cfg.eeg_features == "auto":
            sel, _, _ = mrmr(X_all[tr], y_all[tr], k=cfg.n_selected, n_bins=cfg.mrmr_bins)
        else:
            sel = np.array([names.index(n) for n in cfg.eeg_features])
        fnames = [names[i] for i in sel]
        Xtr, Xte = X_all[tr][:, sel], X_all[te][:, sel]

        fis = SugenoFIS.fit_mfs(Xtr, y_all[tr], n_levels, n_classes, fnames)
        fis.wang_mendel(Xtr, y_all[tr])
        n_wm = len(fis.ant)
        wm_acc = (fis.predict(Xte) == y_all[te]).mean()
        fis, hist = tune_rules(fis, Xtr, y_all[tr], cfg, rng)
        proba = fis.predict_proba(Xte)
        all_proba[te] = proba
        m = metrics(y_all[te], proba, n_classes)
        m.update({"fold": fi, "n_train": len(tr), "n_test": len(te), "features": fnames,
                  "n_rules_lookup": n_wm, "n_rules_ga": len(fis.ant),
                  "acc_lookup_only": float(wm_acc), "ga_generations": len(hist)})
        fold_results.append(m)
        models.append({"fis": fis, "history": hist, "Xtr": Xtr, "ytr": y_all[tr]})

    summary = {}
    for k in ["accuracy", "f1_macro", "precision_macro", "recall_macro", "auc_macro",
              "acc_lookup_only"]:
        v = [f[k] for f in fold_results]
        summary[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
    for k in ["tpr_per_class", "fnr_per_class", "auc_per_class", "f1_per_class"]:
        summary[k] = np.nanmean([f[k] for f in fold_results], axis=0).tolist()
    summary["n_samples"] = int(len(y_all))
    best = int(np.argmax([f["accuracy"] for f in fold_results]))
    summary["best_fold"] = best
    return {"summary": summary, "folds": fold_results, "pooled_proba": all_proba,
            "pooled_y": all_y, "models": models}


def run_band_weighted(F, band, cfg, n_classes, n_levels, rng, gate_name, gate_variant="fuzzy"):
    """Like ``run_band`` but keeps all windows; fold-specific gate weights enter mRMR/FIS/GA."""
    X_all, names = band_matrix(F, band)
    y_all = F["label"]
    subj, trial = F["subject"], F["trial"]
    if cfg.normalize == "subject_zscore":
        X_all = zscore_by_subject(X_all, subj)
    X_all = np.nan_to_num(X_all)

    fold_results, all_proba, all_y, models = [], np.zeros((len(y_all), n_classes)), y_all, []
    folds = make_folds(y_all, subj, trial, cfg)
    for fi, (tr, te) in enumerate(folds):
        w_fold = window_gate_weights(F, gate_name, tr, gate_variant)
        if cfg.eeg_features == "auto":
            sel = mrmr_weighted(X_all[tr], y_all[tr], w_fold[tr], k=cfg.n_selected, n_bins=cfg.mrmr_bins)
        else:
            sel = np.array([names.index(n) for n in cfg.eeg_features])
        fnames = [names[i] for i in sel]
        Xtr, Xte = X_all[tr][:, sel], X_all[te][:, sel]
        sw = w_fold[tr]
        fis = SugenoFIS.fit_mfs(Xtr, y_all[tr], n_levels, n_classes, fnames, sw=sw)
        fis.wang_mendel(Xtr, y_all[tr], sw=sw)
        n_wm = len(fis.ant)
        wm_acc = (fis.predict(Xte) == y_all[te]).mean()
        fis, hist = tune_rules(fis, Xtr, y_all[tr], cfg, rng, sw=sw)
        proba = fis.predict_proba(Xte)
        all_proba[te] = proba
        m = metrics(y_all[te], proba, n_classes)
        m.update({"fold": fi, "n_train": int(tr.sum()), "n_test": int(te.sum()), "features": fnames,
                  "n_rules_lookup": n_wm, "n_rules_ga": len(fis.ant),
                  "acc_lookup_only": float(wm_acc), "ga_generations": len(hist)})
        fold_results.append(m)
        models.append({"fis": fis, "history": hist, "Xtr": Xtr, "ytr": y_all[tr]})

    summary = {}
    for k in ["accuracy", "f1_macro", "precision_macro", "recall_macro", "auc_macro",
              "acc_lookup_only"]:
        v = [f[k] for f in fold_results]
        summary[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
    for k in ["tpr_per_class", "fnr_per_class", "auc_per_class", "f1_per_class"]:
        summary[k] = np.nanmean([f[k] for f in fold_results], axis=0).tolist()
    summary["n_samples"] = int(len(y_all))
    best = int(np.argmax([f["accuracy"] for f in fold_results]))
    summary["best_fold"] = best
    return {"summary": summary, "folds": fold_results, "pooled_proba": all_proba,
            "pooled_y": all_y, "models": models}


def roc_points(y, proba, c):
    fpr, tpr, _ = roc_curve((y == c).astype(int), proba[:, c])
    return fpr, tpr
