"""The four nonlinear entropies of Sec. III-F.2, vectorised over channels.

Every function takes x with shape (n_channels, n_samples) and returns (n_channels,).
"""
from math import factorial, log

import numpy as np


def _embed(x, m, tau):
    n = x.shape[-1] - (m - 1) * tau
    idx = np.arange(n)[:, None] + tau * np.arange(m)[None, :]
    return x[:, idx]                                    # (C, n, m)


def sample_entropy(x, m=5, tau=2, r=0.15):
    """SampEn = -ln(A/B); A, B = matching template pairs of length m+1 and m (Chebyshev <= r*std).

    Processed one channel at a time: the (n, n) distance matrix then fits in CPU cache,
    which is ~4-5x faster than one big (C, n, n) array.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] > 1:
        return np.concatenate([_sample_entropy(x[i:i + 1], m, tau, r) for i in range(x.shape[0])])
    return _sample_entropy(x, m, tau, r)


def _sample_entropy(x, m, tau, r):
    C, N = x.shape
    tol = (r * x.std(axis=1)).astype(np.float32)[:, None, None]
    emb = _embed(x, m + 1, tau)                         # (C, n, m+1), n = N - m*tau
    n = emb.shape[1]
    d = np.zeros((C, n, n), dtype=np.float32)
    for k in range(m):
        np.maximum(d, np.abs(emb[:, :, None, k] - emb[:, None, :, k]), out=d)
    B = ((d <= tol).sum(axis=(1, 2)) - n) / 2
    np.maximum(d, np.abs(emb[:, :, None, m] - emb[:, None, :, m]), out=d)
    A = ((d <= tol).sum(axis=(1, 2)) - n) / 2
    upper = log(n * (n - 1) / 2)                         # value used when A or B is 0
    with np.errstate(divide="ignore", invalid="ignore"):
        se = -np.log(A / B)
    se[(A == 0) | (B == 0)] = upper
    return se


def permutation_entropy(x, m=5, tau=2):
    """PermEn = -sum p(pi) ln p(pi) over the m! ordinal patterns (Eq. 7, natural log)."""
    emb = _embed(np.asarray(x), m, tau)
    C, n, _ = emb.shape
    ranks = np.argsort(np.argsort(emb, axis=2), axis=2)
    code = (ranks * (m ** np.arange(m))).sum(axis=2)    # unique integer per pattern
    K = m ** m
    counts = np.bincount((code + K * np.arange(C)[:, None]).ravel(), minlength=C * K).reshape(C, K)
    p = counts / n
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)


def symbolic_dynamic_entropy(x, m=5, tau=2, c=3):
    """Symbolic dynamic entropy (Eq. 5): state-pattern entropy + state-transition entropy.

    Symbolisation by maximum-entropy (equal-probability) partitioning into c symbols;
    states are embedded symbol vectors (m, tau); the transition is to the symbol that
    follows the state on the same delay lattice.
    """
    x = np.asarray(x)
    C, N = x.shape
    qs = np.quantile(x, np.linspace(0, 1, c + 1)[1:-1], axis=1).T          # (C, c-1)
    sym = (x[:, :, None] > qs[:, None, :]).sum(axis=2)                     # 0..c-1
    n = N - m * tau
    emb = _embed(sym, m, tau)[:, :n]                                        # (C, n, m)
    state = (emb * (c ** np.arange(m))).sum(axis=2)
    nxt = sym[:, m * tau: m * tau + n]
    S = c ** m
    off = np.arange(C)[:, None]
    ps = np.bincount((state + S * off).ravel(), minlength=C * S).reshape(C, S) / n
    joint = np.bincount((state * c + nxt + S * c * off).ravel(),
                        minlength=C * S * c).reshape(C, S, c) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        h_state = -np.where(ps > 0, ps * np.log(ps), 0).sum(axis=1)
        cond = joint / ps[:, :, None]                                        # p(sigma | q)
        h_trans = -np.where(joint > 0, ps[:, :, None] * cond * np.log(cond), 0).sum(axis=(1, 2))
    return h_state + h_trans


def phase_entropy(x, k=5):
    """Phase entropy (Eq. 6) on the second-order difference plot, normalised by log k."""
    x = np.asarray(x, dtype=np.float64)
    d = np.diff(x, axis=1)
    Y, X = d[:, 1:], d[:, :-1]
    theta = np.mod(np.arctan2(Y, X), 2 * np.pi)                             # slope angle 0..2pi
    sector = np.minimum((theta / (2 * np.pi / k)).astype(int), k - 1)
    S = np.stack([(theta * (sector == i)).sum(axis=1) for i in range(k)], axis=1)
    p = S / S.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log(p), 0).sum(axis=1) / log(k)


def all_entropies(x, cfg):
    """Returns (4, C) in the order of config.ENTROPIES: SampEn, PermEn, SymbEn, PhEn."""
    return np.stack([
        sample_entropy(x, cfg.ent_m, cfg.ent_tau, cfg.sampen_r),
        permutation_entropy(x, cfg.ent_m, cfg.ent_tau),
        symbolic_dynamic_entropy(x, cfg.ent_m, cfg.ent_tau, cfg.symben_c),
        phase_entropy(x, cfg.phasen_k),
    ])
