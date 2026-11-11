"""Minimum-redundancy maximum-relevance feature ranking (Eqs. 1-4, MID criterion D - R)."""
import numpy as np
from sklearn.metrics import mutual_info_score


def _discretize(X, n_bins):
    Xd = np.zeros(X.shape, dtype=np.int32)
    for j in range(X.shape[1]):
        col = X[:, j]
        edges = np.unique(np.nanquantile(col, np.linspace(0, 1, n_bins + 1)[1:-1]))
        Xd[:, j] = np.searchsorted(edges, col, side="right")
    return Xd


def mrmr(X, y, k=None, n_bins=10):
    """Greedy mRMR. Returns (order, scores, relevance): indices of selected features in
    selection order, the D-R score at the time each was picked, and I(x_i; y) for all."""
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=np.nanmedian(X))
    n_feat = X.shape[1]
    k = n_feat if k is None else min(k, n_feat)
    Xd = _discretize(X, n_bins)
    rel = np.array([mutual_info_score(Xd[:, j], y) for j in range(n_feat)])
    order, scores = [int(np.argmax(rel))], [float(rel.max())]
    red_sum = np.zeros(n_feat)
    remaining = set(range(n_feat)) - set(order)
    while len(order) < k:
        last = order[-1]
        for j in remaining:
            red_sum[j] += mutual_info_score(Xd[:, j], Xd[:, last])
        cand = np.array(sorted(remaining))
        score = rel[cand] - red_sum[cand] / len(order)
        best = int(cand[np.argmax(score)])
        order.append(best)
        scores.append(float(score.max()))
        remaining.discard(best)
    return np.array(order), np.array(scores), rel


def _wmi(a, b, w):
    P = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(P, (a, b), w)
    P /= P.sum()
    pa, pb = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    nz = P > 0
    return float((P[nz] * np.log(P[nz] / (pa @ pb)[nz])).sum())


def mrmr_weighted(X, y, w, k=3, n_bins=10):
    """mRMR with sample weights (positive-weight windows only)."""
    pos = np.asarray(w, dtype=float) > 0
    X, y, w = X[pos], y[pos], np.asarray(w, dtype=float)[pos]
    Xd = _discretize(np.nan_to_num(X), n_bins)
    rel = np.array([_wmi(Xd[:, j], y, w) for j in range(X.shape[1])])
    order = [int(np.argmax(rel))]
    red = np.zeros(X.shape[1])
    remaining = set(range(X.shape[1])) - set(order)
    while len(order) < k:
        last = order[-1]
        for j in remaining:
            red[j] += _wmi(Xd[:, j], Xd[:, last], w)
        cand = np.array(sorted(remaining))
        best = int(cand[np.argmax(rel[cand] - red[cand] / len(order))])
        order.append(best)
        remaining.discard(best)
    return np.array(order)
