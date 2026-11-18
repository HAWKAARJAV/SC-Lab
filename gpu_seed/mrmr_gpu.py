"""Weighted mRMR (MID criterion) on the GPU; same result as gfer.mrmr.mrmr_weighted."""
import torch


def _discretize(X, n_bins):
    """Per-column quantile bins (same bin boundaries as np.nanquantile + searchsorted 'right')."""
    n = X.shape[0]
    Xs, _ = X.sort(dim=0)
    q = torch.linspace(0, 1, n_bins + 1, device=X.device, dtype=X.dtype)[1:-1]
    pos = q * (n - 1)
    lo, hi = pos.floor().long(), pos.ceil().long()
    frac = (pos - lo.to(X.dtype))[:, None]
    edges = Xs[lo] + (Xs[hi] - Xs[lo]) * frac                   # (n_bins-1, F) linear quantiles
    return (X[:, None, :] >= edges[None, :, :]).sum(dim=1)        # (n, F) bin index, 'right' side


def _wmi_many(A, b, w, nb_a, nb_b):
    """Weighted mutual information between every column of A (n, F) and vector b (n,)."""
    F = A.shape[1]
    idx = (A * nb_b + b[:, None]).T                               # (F, n)
    P = torch.zeros(F, nb_a * nb_b, device=A.device, dtype=torch.float64)
    P.scatter_add_(1, idx, w[None, :].expand(F, -1))
    P = P / P.sum(dim=1, keepdim=True)
    P = P.view(F, nb_a, nb_b)
    pa, pb = P.sum(2, keepdim=True), P.sum(1, keepdim=True)
    ratio = P / (pa * pb).clamp_min(1e-300)
    return torch.where(P > 0, P * torch.log(ratio.clamp_min(1e-300)), torch.zeros_like(P)).sum(dim=(1, 2))


def mrmr_weighted(X, y, w, k=3, n_bins=10, chunk=256):
    """X (n, F), y (n,) int, w (n,) torch tensors on one device. Returns selected column indices."""
    pos = w > 0
    X, y, w = torch.nan_to_num(X[pos].double()), y[pos].long(), w[pos].double()
    const = X.std(dim=0, unbiased=False) < 1e-9
    Xd = torch.cat([_discretize(X[:, i:i + chunk], n_bins) for i in range(0, X.shape[1], chunk)], 1)
    nb = n_bins
    C = int(y.max().item()) + 1
    rel = torch.cat([_wmi_many(Xd[:, i:i + chunk], y, w, nb, C) for i in range(0, X.shape[1], chunk)])
    rel[const] = -float("inf")
    order = [int(rel.argmax())]
    red = torch.zeros_like(rel)
    k = min(k, int((~const).sum()))
    while len(order) < k:
        last = Xd[:, order[-1]]
        red += torch.cat([_wmi_many(Xd[:, i:i + chunk], last, w, nb, nb) for i in range(0, X.shape[1], chunk)])
        score = rel - red / len(order)
        score[order] = -float("inf")
        score[const] = -float("inf")
        order.append(int(score.argmax()))
    return order
