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


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_fast_fis_matches_reference(seed):
    from gfer.fuzzy import entropy_cuts as cuts_ref
    from gpu_seed.fast_fis import build_fis, entropy_cuts
    rng = np.random.default_rng(seed)
    n = 3000
    y = rng.integers(0, 4, n)
    X = rng.normal(size=(n, 3)) + y[:, None] * rng.uniform(0, 0.8, 3)
    sw = rng.uniform(0, 2, n); sw[rng.random(n) < 0.1] = 0
    for j in range(3):
        np.testing.assert_allclose(entropy_cuts(X[:, j], y, 3, sw=sw), cuts_ref(X[:, j], y, 3, sw=sw))
    ref = SugenoFIS.fit_mfs(X, y, 3, 4, ["a", "b", "c"], sw=sw).wang_mendel(X, y, sw=sw)
    got = build_fis(X, y, 3, 4, ["a", "b", "c"], sw)
    np.testing.assert_array_equal(got.ant, ref.ant)
    np.testing.assert_array_equal(got.cons, ref.cons)
    np.testing.assert_allclose(got.weight, ref.weight, rtol=1e-10)
    np.testing.assert_allclose(got.prior, ref.prior)


def test_chunked_fitness_equals_single_pass():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(5000, 4)); y = rng.integers(0, 4, 5000); X[:, 1] += y
    sw = rng.uniform(0.1, 1, 5000)
    fis = SugenoFIS.fit_mfs(X, y, 3, 4, None, sw=sw).wang_mendel(X, y, sw=sw)
    cfg = types.SimpleNamespace(ga_w_coverage=0.2, ga_w_size=0.01, ga_balanced=True)
    g = GPUFIS(fis, torch.device("cpu"))
    yt, swt = torch.tensor(y), torch.tensor(sw, dtype=torch.float32)
    Yw = torch.nn.functional.one_hot(yt, 4).float() * swt[:, None]
    R = len(fis.ant)
    A = torch.tensor(rng.random((5, R)) < 0.7)
    N = torch.tensor(np.stack([fis.ant] * 5))
    M = g.membership(X)
    one = g.fitness(M, yt, Yw, swt, A, N, cfg, max_bytes=1e12)       # single chunk
    many = g.fitness(M, yt, Yw, swt, A, N, cfg, max_bytes=1)         # 1024-window chunks
    for a, b in zip(one, many):
        np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-5)


def test_feature_matrix_improved_pool():
    from gpu_seed.config import GPUConfig
    from gpu_seed.run import feature_matrix
    n = 10
    F = {"eeg": np.zeros((n, 5, 4, 62), np.float32), "de": np.zeros((n, 5, 62), np.float32),
         "eye": np.zeros((n, 31), np.float32), "label": np.zeros(n, int), "bands": ["delta", "theta", "alpha", "beta", "gamma"],
         "entropies": ["SampEn", "PermEn", "SymbEn", "PhEn"], "ch_names": [f"c{i}" for i in range(62)],
         "eye_names": [f"eye{i:02d}" for i in range(31)]}
    X, names = feature_matrix(F, "alpha", GPUConfig())
    assert X.shape == (n, 248) and len(names) == 248
    X, names = feature_matrix(F, "all", GPUConfig(use_de=True, eye_in_classifier=True))
    assert X.shape == (n, 5 * (248 + 62) + 31) and len(names) == X.shape[1]
