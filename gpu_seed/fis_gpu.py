"""Sugeno FIS + genetic rule tuning with the fitness of the whole population evaluated on the GPU.

Same model as gfer.fuzzy / gfer.ga (membership functions and lookup-table rules are built with
gfer.fuzzy on the CPU, which is cheap); the expensive part, scoring every chromosome on every
training window in every generation, is one batched tensor computation here.
"""
import numpy as np
import torch

from gfer.fuzzy import DONT_CARE, SugenoFIS


class GPUFIS:
    def __init__(self, fis, device):
        self.fis, self.device = fis, device
        self.C = fis.n_classes
        self.prior = torch.tensor(fis.prior, device=device, dtype=torch.float32)
        self.k1 = max(m.k for m in fis.mfs) + 1                 # last slot = don't care (1.0)

    def membership(self, X):
        return torch.tensor(self.fis.membership_tensor(X), device=self.device, dtype=torch.float32)

    @staticmethod
    def firing(M, ant):
        """M (n, n_in, k+1), ant (P, R, n_in) long with -1 -> last slot. Returns (P, n, R)."""
        n_in = M.shape[1]
        a = torch.where(ant < 0, torch.full_like(ant, M.shape[2] - 1), ant)       # (P, R, n_in)
        # gather M[:, j, a[p, r, j]] for every p, r, j -> (P, n, R, n_in), product over inputs
        idx = a.permute(0, 2, 1)                                                 # (P, n_in, R)
        out = torch.ones(ant.shape[0], M.shape[0], ant.shape[1], device=M.device)
        for j in range(n_in):
            out = out * M[:, j, :][:, idx[:, j, :]].permute(1, 0, 2)              # (P, n, R)
        return out

    def consequents(self, F, Yw):
        """F (P, n, R), Yw (n, C) weighted one-hot -> cons (P, R), w (P, R)."""
        mass = torch.einsum("pnr,nc->prc", F, Yw)
        tot = mass.sum(2)
        mx, cons = mass.max(2)
        safe = torch.where(tot > 0, tot, torch.ones_like(tot))
        conf = torch.where(tot > 0, mx / safe, torch.zeros_like(tot))
        other = torch.where(tot > 0, (tot - mx) / safe, torch.zeros_like(tot))
        w = (conf - other / max(self.C - 1, 1)).clamp(0, 1)
        return cons, w

    def scores(self, F, cons, w):
        S = torch.einsum("pnr,prc->pnc", F * w[:, None, :], torch.nn.functional.one_hot(cons, self.C).float())
        tot = S.sum(2, keepdim=True)
        return torch.where(tot > 0, S / torch.where(tot > 0, tot, torch.ones_like(tot)), self.prior)

    def fitness(self, M, y, Yw, sw, active, ant, cfg, max_bytes=2.5e8):
        """Batched version of gfer.ga._evaluate. active (P, R) bool, ant (P, R, n_in).
        Processed in chunks of windows so memory stays bounded (two passes: consequents, then scores)."""
        P, R = active.shape
        n = M.shape[0]
        step = max(1024, int(max_bytes / (4 * P * max(R, 1))))
        act = active.float()
        mass = torch.zeros(P, R, self.C, device=M.device)
        for i in range(0, n, step):                                               # pass 1: rule consequents
            F = self.firing(M[i:i + step], ant) * act[:, None, :]
            mass += torch.einsum("pnr,nc->prc", F, Yw[i:i + step])
        tot = mass.sum(2)
        mx, cons = mass.max(2)
        safe = torch.where(tot > 0, tot, torch.ones_like(tot))
        conf = torch.where(tot > 0, mx / safe, torch.zeros_like(tot))
        other = torch.where(tot > 0, (tot - mx) / safe, torch.zeros_like(tot))
        w = (conf - other / max(self.C - 1, 1)).clamp(0, 1) * act
        classes = torch.unique(y)
        hit_c = torch.zeros(P, len(classes), device=M.device)
        hit, covered = torch.zeros(P, device=M.device), torch.zeros(P, device=M.device)
        for i in range(0, n, step):                                               # pass 2: predictions
            F = self.firing(M[i:i + step], ant) * act[:, None, :]
            pred = self.scores(F, cons, w).argmax(2)
            yy, ss = y[i:i + step], sw[i:i + step]
            h = (pred == yy[None, :]).float() * ss[None, :]
            hit += h.sum(1)
            for ci, c in enumerate(classes):
                hit_c[:, ci] += h[:, yy == c].sum(1)
            covered += (((F * w[:, None, :]) > 0).any(2).float() * ss[None, :]).sum(1)
        if cfg.ga_balanced:
            acc = (hit_c / torch.stack([sw[y == c].sum() for c in classes]).clamp_min(1e-12)[None, :]).mean(1)
        else:
            acc = hit / sw.sum()
        cov = covered / sw.sum()
        n_act = act.sum(1)
        fit = acc + cfg.ga_w_coverage * cov - cfg.ga_w_size * n_act / max(R, 1)
        fit = torch.where(n_act > 0, fit, torch.full_like(fit, -1.0))
        return fit, acc, cov


def tune_rules_gpu(fis, X, y, cfg, rng, device, sw=None):
    """GA over (active rule set, antecedents); see gfer/ga.py for the encoding. Returns fis, history."""
    g = GPUFIS(fis, device)
    M = g.membership(X)
    yt = torch.tensor(y, device=device, dtype=torch.long)
    swt = torch.tensor(np.ones(len(y)) if sw is None else sw, device=device, dtype=torch.float32)
    Yw = torch.nn.functional.one_hot(yt, g.C).float() * swt[:, None]
    base = fis.ant.copy()
    R, n_in = base.shape
    ks = np.array([m.k for m in fis.mfs])

    def mutate(active, ant, scale=1.0):
        active, ant = active.copy(), ant.copy()
        flip = rng.random(R) < cfg.ga_mut_active * scale
        active[flip] = ~active[flip]
        m = rng.random((R, n_in)) < cfg.ga_mut_ant * scale
        for r, j in zip(*np.where(m)):
            ant[r, j] = DONT_CARE if rng.random() < 0.4 else rng.integers(0, ks[j])
        return active, ant

    def evaluate(pop):
        A = torch.tensor(np.stack([p[0] for p in pop]), device=device)
        N = torch.tensor(np.stack([p[1] for p in pop]), device=device, dtype=torch.long)
        f, a, c = g.fitness(M, yt, Yw, swt, A, N, cfg)
        return list(zip(f.tolist(), a.tolist(), c.tolist()))

    pop = [(np.ones(R, bool), base.copy())]
    while len(pop) < cfg.ga_pop:
        pop.append(mutate(*pop[0], scale=3.0))
    fits = evaluate(pop)
    history, best_fit, stall = [], -np.inf, 0
    for gen in range(cfg.ga_max_gen):
        order = np.argsort([-f[0] for f in fits])
        gb = fits[order[0]]
        history.append({"gen": gen, "best": gb[0], "acc": gb[1], "coverage": gb[2]})
        if gb[0] > best_fit + 1e-9:
            best_fit, stall = gb[0], 0
        else:
            stall += 1
            if stall >= cfg.ga_patience:
                break
        new = [pop[i] for i in order[:cfg.ga_elite]]

        def pick():
            c = rng.choice(len(pop), cfg.ga_tournament, replace=False)
            return pop[max(c, key=lambda i: fits[i][0])]

        while len(new) < cfg.ga_pop:
            (a1, n1), (a2, n2) = pick(), pick()
            if rng.random() < cfg.ga_cx_rate:
                mask = rng.random(R) < 0.5
                a1 = np.where(mask, a1, a2)
                n1 = np.where(mask[:, None], n1, n2)
            new.append(mutate(a1, n1))
        pop = new
        fits = evaluate(pop)

    best = int(np.argmax([f[0] for f in fits]))
    active, ant = pop[best]
    idx = np.where(active)[0]
    if idx.size == 0:
        return fis, history
    A = torch.tensor(active[idx][None], device=device)
    N = torch.tensor(ant[idx][None], device=device, dtype=torch.long)
    F = g.firing(M, N)
    cons, w = g.consequents(F, Yw)
    cons, w = cons[0].cpu().numpy(), w[0].cpu().numpy()
    seen = {}
    for r in np.where(w > 0)[0]:                    # merge duplicate antecedents (keep higher weight)
        key = tuple(ant[idx][r])
        if key not in seen or w[r] > w[seen[key]]:
            seen[key] = r
    sel = np.array(sorted(seen.values()), dtype=int)
    if sel.size:
        fis.ant, fis.cons, fis.weight = ant[idx][sel], cons[sel], w[sel]
    return fis, history


def predict_proba_gpu(fis, X, device):
    g = GPUFIS(fis, device)
    M = g.membership(X)
    N = torch.tensor(fis.ant[None], device=device, dtype=torch.long)
    F = g.firing(M, N)
    cons = torch.tensor(fis.cons[None], device=device, dtype=torch.long)
    w = torch.tensor(fis.weight[None], device=device, dtype=torch.float32)
    return g.scores(F, cons, w)[0].cpu().numpy()


def build_fis(X, y, n_levels, C, names, sw):
    fis = SugenoFIS.fit_mfs(X, y, n_levels, C, names, sw=sw)
    fis.wang_mendel(X, y, sw=sw)
    return fis
