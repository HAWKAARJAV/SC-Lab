"""Sugeno fuzzy inference system with rules mined by the lookup-table (Wang-Mendel) method.

Sec. III-G of the paper:
  * each input is discretised into 2/3/4 linguistic levels; split points are found with
    an entropy (information-gain) criterion;
  * triangular membership functions are placed on those levels;
  * one candidate rule is generated per training sample (lookup table); conflicting rules
    (same IF part, different THEN part) are resolved in favour of the higher total weight;
  * zero-order Sugeno inference. Each rule's output is a class indicator, so the
    weighted-average ("centroid") defuzzification yields one score per class; the
    predicted class is the arg-max and the scores are used for ROC/AUC.
"""
import numpy as np

LEVEL_NAMES = {2: ["low", "high"], 3: ["low", "medium", "high"],
               4: ["low", "medium-low", "medium-high", "high"]}
DONT_CARE = -1


# ---------------------------------------------------------------- discretisation
def _entropy(counts):
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return -(p * np.log2(p)).sum()


def entropy_cuts(x, y, n_levels, n_candidates=64, sw=None):
    """Greedy top-down entropy discretisation: repeatedly add the cut point (over all current
    intervals) with the largest weighted information gain until n_levels intervals exist."""
    sw = np.ones(len(x)) if sw is None else np.asarray(sw, dtype=float)
    pos = sw > 0
    classes = np.unique(y[pos])
    cand = np.unique(np.quantile(x[pos], np.linspace(0.05, 0.95, n_candidates)))
    cuts = []
    N = sw.sum()

    def gain(lo, hi, c):
        m = (x > lo) & (x <= hi) & pos
        if m.sum() < 2:
            return -1
        ys, ws = y[m], sw[m]
        lm = x[m] <= c
        if lm.all() or not lm.any():
            return -1
        cnt = lambda v, w: np.array([w[v == k].sum() for k in classes])
        wl, wr, wt = ws[lm].sum(), ws[~lm].sum(), ws.sum()
        h = _entropy(cnt(ys, ws))
        hc = (wl * _entropy(cnt(ys[lm], ws[lm])) + wr * _entropy(cnt(ys[~lm], ws[~lm]))) / wt
        return (h - hc) * wt / N

    for _ in range(n_levels - 1):
        bounds = [-np.inf] + sorted(cuts) + [np.inf]
        best, best_c = -1, None
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            for c in cand[(cand > lo) & (cand < hi)]:
                g = gain(lo, hi, c)
                if g > best:
                    best, best_c = g, c
        if best_c is None:
            break
        cuts.append(best_c)
    if len(cuts) < n_levels - 1:              # fall back to quantiles if entropy finds no split
        cuts = list(np.quantile(x[pos], np.linspace(0, 1, n_levels + 1)[1:-1]))
    return np.sort(np.array(cuts))


class TriangularMFs:
    """k triangular membership functions (shoulders at both ends) for one input."""

    def __init__(self, centers):
        self.c = np.asarray(centers, dtype=float)

    @classmethod
    def from_cuts(cls, x, cuts):
        lo, hi = np.quantile(x, [0.02, 0.98])
        k = len(cuts) + 1
        if k == 2:
            d = max(min(cuts[0] - lo, hi - cuts[0]), 1e-6)
            centers = [cuts[0] - d, cuts[0] + d]          # crossover exactly at the cut
        else:
            mids = [(a + b) / 2 for a, b in zip(cuts[:-1], cuts[1:])]
            centers = [min(lo, cuts[0] - 1e-6)] + mids + [max(hi, cuts[-1] + 1e-6)]
        return cls(np.maximum.accumulate(np.asarray(centers) + np.arange(k) * 1e-9))

    @property
    def k(self):
        return len(self.c)

    def __call__(self, x):
        x = np.asarray(x, dtype=float)[:, None]
        c = self.c
        mu = np.zeros((x.shape[0], self.k))
        for i in range(self.k):
            if i == 0:
                mu[:, i] = np.clip((c[1] - x[:, 0]) / (c[1] - c[0]), 0, 1)
            elif i == self.k - 1:
                mu[:, i] = np.clip((x[:, 0] - c[i - 1]) / (c[i] - c[i - 1]), 0, 1)
            else:
                up = (x[:, 0] - c[i - 1]) / (c[i] - c[i - 1])
                dn = (c[i + 1] - x[:, 0]) / (c[i + 1] - c[i])
                mu[:, i] = np.clip(np.minimum(up, dn), 0, 1)
        return mu


# ---------------------------------------------------------------- the FIS
class SugenoFIS:
    def __init__(self, mfs, n_classes, feature_names=None):
        self.mfs = mfs
        self.n_classes = n_classes
        self.feature_names = feature_names or [f"x{i}" for i in range(len(mfs))]
        self.ant = np.zeros((0, len(mfs)), dtype=int)    # (R, n_in), DONT_CARE allowed
        self.cons = np.zeros(0, dtype=int)
        self.weight = np.zeros(0)
        self.prior = np.ones(n_classes) / n_classes

    @classmethod
    def fit_mfs(cls, X, y, n_levels, n_classes, feature_names=None, sw=None):
        sw = np.ones(len(y)) if sw is None else np.asarray(sw, dtype=float)
        pos = sw > 0
        mfs = [TriangularMFs.from_cuts(X[pos, j], entropy_cuts(X[:, j], y, n_levels, sw=sw))
               for j in range(X.shape[1])]
        fis = cls(mfs, n_classes, feature_names)
        fis.prior = np.bincount(y, weights=sw, minlength=n_classes) / sw.sum()
        return fis

    def membership_tensor(self, X):
        """(n, n_in, k+1); last slot is 1.0 so DONT_CARE (-1) indexes it."""
        k = max(m.k for m in self.mfs)
        M = np.ones((X.shape[0], X.shape[1], k + 1))
        for j, m in enumerate(self.mfs):
            M[:, j, :m.k] = m(X[:, j])
        return M

    @staticmethod
    def firing_from(M, ant):
        """Product t-norm firing strength, (n, R)."""
        if len(ant) == 0:
            return np.zeros((M.shape[0], 0))
        n_in = M.shape[1]
        return M[:, np.arange(n_in)[None, :], ant].prod(axis=2)

    def scores_from(self, F, cons=None, weight=None):
        cons = self.cons if cons is None else cons
        weight = self.weight if weight is None else weight
        S = np.zeros((F.shape[0], self.n_classes))
        if len(cons):
            np.add.at(S.T, cons, (F * weight).T)
        tot = S.sum(axis=1, keepdims=True)
        out = np.where(tot > 0, S / np.where(tot > 0, tot, 1), self.prior)   # unfired -> prior
        return out

    def predict_proba(self, X):
        return self.scores_from(self.firing_from(self.membership_tensor(X), self.ant))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    # ---------------------------------------------------- rule mining (lookup table)
    def wang_mendel(self, X, y, sw=None):
        """One rule per sample: each input takes its max-membership level; degree = product.
        Conflicting rules keep the class with the largest summed degree; the rule weight is
        its confidence (share of that class in the total degree)."""
        M = self.membership_tensor(X)[:, :, :-1]
        levels = M.argmax(axis=2)
        degree = M.max(axis=2).prod(axis=1)
        if sw is not None:
            degree = degree * np.asarray(sw, dtype=float)      # window weight scales rule degree
        table = {}
        for a, c, d in zip(map(tuple, levels), y, degree):
            if d <= 0:
                continue
            table.setdefault(a, np.zeros(self.n_classes))[c] += d
        ant, cons, w = [], [], []
        for a, v in table.items():
            ant.append(a); cons.append(int(v.argmax())); w.append(v.max() / v.sum())
        self.ant, self.cons, self.weight = np.array(ant, dtype=int), np.array(cons), np.array(w)
        return self

    def rule_consequents(self, M, y, ant, sw=None):
        """Data-driven consequent and certainty weight for given antecedents (Ishibuchi)."""
        F = self.firing_from(M, ant)                                   # (n, R)
        Y = np.eye(self.n_classes)[y]                                  # (n, C)
        if sw is not None:
            Y = Y * np.asarray(sw, dtype=float)[:, None]               # weighted firing mass
        mass = F.T @ Y                                                 # (R, C)
        tot = mass.sum(axis=1)
        cons = mass.argmax(axis=1)
        conf = np.where(tot > 0, mass.max(axis=1) / np.where(tot > 0, tot, 1), 0)
        # certainty factor: confidence minus mean confidence of the other classes
        other = np.where(tot > 0, (tot - mass.max(axis=1)) / np.where(tot > 0, tot, 1), 0)
        w = np.clip(conf - other / max(self.n_classes - 1, 1), 0, 1)
        return cons, w, F

    # ---------------------------------------------------- reporting
    def rule_stats(self, X, y):
        """Per-rule coverage (share of samples that fire the rule) and accuracy
        (share of the rule's firing mass that falls on its own class)."""
        F = self.firing_from(self.membership_tensor(X), self.ant)
        cov = (F > 0).mean(axis=0)
        hit = np.array([F[y == c, r].sum() for r, c in enumerate(self.cons)])
        acc = np.where(F.sum(axis=0) > 0, hit / np.maximum(F.sum(axis=0), 1e-12), 0)
        return cov, acc

    def rule_text(self, r, class_names):
        parts = []
        for j, lv in enumerate(self.ant[r]):
            if lv != DONT_CARE:
                parts.append(f"{self.feature_names[j]} is {LEVEL_NAMES[self.mfs[j].k][lv]}")
        cond = " and ".join(parts) if parts else "(always)"
        return f"If {cond} then class is {class_names[self.cons[r]]}"
