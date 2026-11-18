"""The four entropies of gfer/entropy.py, batched on the GPU with PyTorch.

Every function takes x of shape (B, N) (B independent signals, e.g. windows x channels) and
returns (B,). Results match the NumPy reference implementations in gfer/entropy.py
(tests/test_gpu_seed.py checks this).
"""
import math

import torch


def _embed(x, m, tau):
    """(B, N) -> (B, N - (m-1)*tau, m) delay embedding."""
    return x.unfold(-1, (m - 1) * tau + 1, 1)[..., ::tau]


def sample_entropy(x, m=5, tau=2, r=0.15):
    """SampEn = -ln(A/B), Chebyshev distance <= r*std, self-matches excluded.
    Memory: B * n * n floats per call, so callers should chunk B (see entropy_batch)."""
    x = x.float()
    tol = (r * x.std(dim=1, unbiased=False))[:, None, None]
    emb = _embed(x, m + 1, tau)                                   # (B, n, m+1)
    n = emb.shape[1]
    d = torch.zeros(x.shape[0], n, n, device=x.device)
    for k in range(m):
        torch.maximum(d, (emb[:, :, None, k] - emb[:, None, :, k]).abs(), out=d)
    Bc = ((d <= tol).sum(dim=(1, 2)) - n).double() / 2
    torch.maximum(d, (emb[:, :, None, m] - emb[:, None, :, m]).abs(), out=d)
    Ac = ((d <= tol).sum(dim=(1, 2)) - n).double() / 2
    upper = math.log(n * (n - 1) / 2)
    se = -torch.log(Ac / Bc)
    se = torch.where((Ac == 0) | (Bc == 0), torch.full_like(se, upper), se)
    return se


def permutation_entropy(x, m=5, tau=2):
    emb = _embed(x.double(), m, tau)                              # (B, n, m)
    B, n, _ = emb.shape
    ranks = emb.argsort(dim=2).argsort(dim=2)
    code = (ranks * (m ** torch.arange(m, device=x.device))).sum(dim=2)
    K = m ** m
    counts = torch.zeros(B, K, device=x.device, dtype=torch.float64)
    counts.scatter_add_(1, code, torch.ones_like(code, dtype=torch.float64))
    p = counts / n
    return -(torch.where(p > 0, p * torch.log(p.clamp_min(1e-300)), torch.zeros_like(p))).sum(dim=1)


def symbolic_dynamic_entropy(x, m=5, tau=2, c=3):
    x = x.double()
    B, N = x.shape
    qs = torch.quantile(x, torch.linspace(0, 1, c + 1, device=x.device, dtype=x.dtype)[1:-1], dim=1).T  # (B, c-1)
    sym = (x[:, :, None] > qs[:, None, :]).sum(dim=2)                                    # (B, N)
    n = N - m * tau
    emb = _embed(sym, m, tau)[:, :n]
    state = (emb * (c ** torch.arange(m, device=x.device))).sum(dim=2)                  # (B, n)
    nxt = sym[:, m * tau: m * tau + n]
    S = c ** m
    ps = torch.zeros(B, S, device=x.device, dtype=torch.float64)
    ps.scatter_add_(1, state, torch.ones_like(state, dtype=torch.float64)); ps /= n
    joint = torch.zeros(B, S * c, device=x.device, dtype=torch.float64)
    joint.scatter_add_(1, state * c + nxt, torch.ones_like(state, dtype=torch.float64)); joint /= n
    joint = joint.view(B, S, c)
    h_state = -torch.where(ps > 0, ps * torch.log(ps.clamp_min(1e-300)), torch.zeros_like(ps)).sum(1)
    cond = joint / ps[:, :, None].clamp_min(1e-300)
    h_trans = -torch.where(joint > 0, ps[:, :, None] * cond * torch.log(cond.clamp_min(1e-300)),
                           torch.zeros_like(joint)).sum(dim=(1, 2))
    return h_state + h_trans


def phase_entropy(x, k=5):
    x = x.double()
    d = torch.diff(x, dim=1)
    Y, X = d[:, 1:], d[:, :-1]
    theta = torch.remainder(torch.atan2(Y, X), 2 * math.pi)
    sector = torch.clamp((theta / (2 * math.pi / k)).long(), max=k - 1)
    S = torch.zeros(x.shape[0], k, device=x.device, dtype=torch.float64)
    S.scatter_add_(1, sector, theta)
    p = S / S.sum(dim=1, keepdim=True)
    return -torch.where(p > 0, p * torch.log(p.clamp_min(1e-300)), torch.zeros_like(p)).sum(1) / math.log(k)


def entropy_batch(x, cfg, sampen_chunk=256):
    """(B, N) tensor on the device -> (4, B) float32 in the order SampEn, PermEn, SymbEn, PhEn."""
    m_s = cfg.sampen_m or cfg.ent_m
    se = torch.cat([sample_entropy(x[i:i + sampen_chunk], m_s, cfg.ent_tau, cfg.sampen_r)
                    for i in range(0, x.shape[0], sampen_chunk)])
    return torch.stack([
        se,
        permutation_entropy(x, cfg.ent_m, cfg.ent_tau),
        symbolic_dynamic_entropy(x, cfg.ent_m, cfg.ent_tau, cfg.symben_c),
        phase_entropy(x, cfg.phasen_k),
    ]).float()
