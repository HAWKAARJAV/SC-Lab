"""Vectorised (NumPy) versions of the two slow CPU steps of gfer.fuzzy, same results:

* entropy_cuts   : the greedy information-gain split search, evaluated for all candidate
                   cuts of an interval at once from cumulative class weights;
* wang_mendel    : the lookup-table rule mining, aggregated with np.unique/np.add.at
                   (rules kept in first-occurrence order, as in the reference).
tests/test_gpu_seed.py checks both against gfer.fuzzy.
"""
import numpy as np

from gfer.fuzzy import SugenoFIS, TriangularMFs


def _H(counts):
    """gfer.fuzzy._entropy for every row of counts (k, C)."""
    tot = np.maximum(counts.sum(axis=1, keepdims=True), 1)
    p = counts / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0).sum(axis=1)


def entropy_cuts(x, y, n_levels, n_candidates=64, sw=None):
    sw = np.ones(len(x)) if sw is None else np.asarray(sw, dtype=float)
    pos = sw > 0
    classes = np.unique(y[pos])
    cidx = np.searchsorted(classes, y)
    cand = np.unique(np.quantile(x[pos], np.linspace(0.05, 0.95, n_candidates)))
    N = sw.sum()
    cuts = []
    for _ in range(n_levels - 1):
        bounds = [-np.inf] + sorted(cuts) + [np.inf]
        all_g, all_c = [], []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            cc = cand[(cand > lo) & (cand < hi)]
            if cc.size == 0:
                continue
            m = (x > lo) & (x <= hi) & pos
            g = np.full(cc.size, -1.0)
            if m.sum() >= 2:
                xs, ws, ks = x[m], sw[m], cidx[m]
                o = np.argsort(xs, kind="stable")
                xs, ws, ks = xs[o], ws[o], ks[o]
                W = np.zeros((xs.size, len(classes)))
                W[np.arange(xs.size), ks] = ws
                cum = np.cumsum(W, axis=0)
                tot = cum[-1]
                j = np.searchsorted(xs, cc, side="right")          # samples with x <= c
                ok = (j > 0) & (j < xs.size)
                left = np.where(ok[:, None], cum[np.clip(j - 1, 0, None)], 0)
                right = tot[None, :] - left
                wl, wr, wt = left.sum(1), right.sum(1), tot.sum()
                h = _H(tot[None, :])[0]
                hc = (wl * _H(left) + wr * _H(right)) / wt
                g = np.where(ok, (h - hc) * wt / N, -1.0)
            all_g.append(g); all_c.append(cc)
        if not all_g:
            break
        g, cc = np.concatenate(all_g), np.concatenate(all_c)
        b = int(np.argmax(g))
        if g[b] <= -1:
            break
        cuts.append(cc[b])
    if len(cuts) < n_levels - 1:
        cuts = list(np.quantile(x[pos], np.linspace(0, 1, n_levels + 1)[1:-1]))
    return np.sort(np.array(cuts))


def wang_mendel(fis, X, y, sw=None):
    M = fis.membership_tensor(X)[:, :, :-1]
    levels = M.argmax(axis=2)
    degree = M.max(axis=2).prod(axis=1)
    if sw is not None:
        degree = degree * np.asarray(sw, dtype=float)
    ok = degree > 0
    lv, yy, dd = levels[ok], np.asarray(y)[ok], degree[ok]
    if lv.shape[0] == 0:
        fis.ant, fis.cons, fis.weight = np.zeros((0, X.shape[1]), int), np.zeros(0, int), np.zeros(0)
        return fis
    dims = [m.k for m in fis.mfs]
    code = np.ravel_multi_index(lv.T, dims)
    uniq, first, inv = np.unique(code, return_index=True, return_inverse=True)
    table = np.zeros((uniq.size, fis.n_classes))
    np.add.at(table, (inv, yy), dd)
    order = np.argsort(first, kind="stable")                    # first-occurrence order
    table = table[order]
    fis.ant = lv[first[order]].astype(int)
    fis.cons = table.argmax(axis=1)
    fis.weight = table.max(axis=1) / table.sum(axis=1)
    return fis


def build_fis(X, y, n_levels, n_classes, names, sw):
    """Same as SugenoFIS.fit_mfs(...) followed by .wang_mendel(...), vectorised."""
    sw = np.ones(len(y)) if sw is None else np.asarray(sw, dtype=float)
    pos = sw > 0
    mfs = [TriangularMFs.from_cuts(X[pos, j], entropy_cuts(X[:, j], y, n_levels, sw=sw))
           for j in range(X.shape[1])]
    fis = SugenoFIS(mfs, n_classes, names)
    fis.prior = np.bincount(y, weights=sw, minlength=n_classes) / sw.sum()
    return wang_mendel(fis, X, y, sw)
