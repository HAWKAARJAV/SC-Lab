"""The GPU (torch) pipeline must reproduce the NumPy reference implementations (runs on CPU torch)."""
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gfer import entropy as E                       # noqa: E402
from gfer.fuzzy import SugenoFIS                    # noqa: E402
from gfer.ga import _evaluate                       # noqa: E402
from gfer.mrmr import mrmr_weighted as mrmr_np      # noqa: E402
from gpu_seed import entropy_gpu as G               # noqa: E402
from gpu_seed.config import GPUConfig               # noqa: E402
from gpu_seed.fis_gpu import GPUFIS                 # noqa: E402
from gpu_seed.mrmr_gpu import mrmr_weighted as mrmr_gpu   # noqa: E402


def _signals(n=12, N=512, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(N) / 128
    x = rng.normal(size=(n, N)) + np.sin(2 * np.pi * rng.uniform(1, 30, (n, 1)) * t)
    return x


@pytest.mark.parametrize("m,tau", [(5, 2), (3, 1)])
def test_entropies_match_numpy(m, tau):
    x = _signals()
    xt = torch.tensor(x)
    np.testing.assert_allclose(G.sample_entropy(xt, m, tau, 0.15).numpy(), E.sample_entropy(x, m, tau, 0.15), rtol=1e-4)
    np.testing.assert_allclose(G.permutation_entropy(xt, m, tau).numpy(), E.permutation_entropy(x, m, tau), rtol=1e-9)
    np.testing.assert_allclose(G.symbolic_dynamic_entropy(xt, m, tau, 3).numpy(),
                               E.symbolic_dynamic_entropy(x, m, tau, 3), rtol=1e-9)
    np.testing.assert_allclose(G.phase_entropy(xt, 5).numpy(), E.phase_entropy(x, 5), rtol=1e-9)


def test_mrmr_matches_numpy():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(600, 40))
    y = rng.integers(0, 4, 600)
    X[:, 7] += y * 1.5; X[:, 21] += (y == 2) * 2.0; X[:, 30] = 0.0       # informative + constant
    w = rng.uniform(0, 1, 600); w[:50] = 0
    ref = list(mrmr_np(X, y, w, k=3, n_bins=10))
    got = mrmr_gpu(torch.tensor(X), torch.tensor(y), torch.tensor(w), k=3, n_bins=10)
    assert got == ref and 30 not in got


def test_batched_ga_fitness_matches_cpu():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 3)); y = rng.integers(0, 4, 400); X[:, 0] += y
    sw = rng.uniform(0.1, 1, 400)
    fis = SugenoFIS.fit_mfs(X, y, 3, 4, ["a", "b", "c"], sw=sw)
    fis.wang_mendel(X, y, sw=sw)
    R, n_in = fis.ant.shape
    cfg = types.SimpleNamespace(ga_w_coverage=0.2, ga_w_size=0.01, ga_balanced=True)
    pop = []
    for _ in range(6):
        act = rng.random(R) < 0.7
        ant = fis.ant.copy()
        mm = rng.random((R, n_in)) < 0.2
        ant[mm] = rng.integers(-1, 3, mm.sum())
        pop.append((act, ant))
    M = fis.membership_tensor(X)
    ref = np.array([_evaluate(fis, M, y, a, an, cfg, sw) for a, an in pop])
    g = GPUFIS(fis, torch.device("cpu"))
    yt = torch.tensor(y); swt = torch.tensor(sw, dtype=torch.float32)
    Yw = torch.nn.functional.one_hot(yt, 4).float() * swt[:, None]
    f, a, c = g.fitness(g.membership(X), yt, Yw, swt, torch.tensor(np.stack([p[0] for p in pop])),
                        torch.tensor(np.stack([p[1] for p in pop])), cfg)
    np.testing.assert_allclose(np.c_[f.numpy(), a.numpy(), c.numpy()], ref, atol=1e-5)


def test_entropy_batch_uses_config():
    cfg = GPUConfig()
    out = G.entropy_batch(torch.tensor(_signals(5)), cfg, sampen_chunk=2)
    assert out.shape == (4, 5) and torch.isfinite(out).all()
