"""Genetic algorithm that tunes the mined S-FIS rule base (Sec. III-G.2, IV-B).

The paper's chromosome is only shown as a figure (Fig. 6). ASSUMPTION, our encoding:
  chromosome = R rule genes; each gene = (active bit, antecedent level per input with
  "don't care" allowed). Consequent and weight of every rule are recomputed from the
  training data each time (class with the highest firing mass; certainty factor), so the
  GA searches over which rules exist and how general they are.
Initial population: the lookup-table (Wang-Mendel) rules plus mutated copies of them.
Fitness: training accuracy + w_cov * coverage - w_size * (active rules / R)
  (the paper uses accuracy and coverage; the small size term prefers compact rule bases).
Stop: no fitness improvement for `patience` generations, or `max_gen`.
"""
import numpy as np

from .fuzzy import DONT_CARE


def _evaluate(fis, M, y, active, ant, cfg, sw=None):
    idx = np.where(active)[0]
    if idx.size == 0:
        return -1.0, 0.0, 0.0
    cons, w, F = fis.rule_consequents(M, y, ant[idx], sw)
    pred = fis.scores_from(F, cons, w).argmax(axis=1)
    sw_ = np.ones(len(y)) if sw is None else np.asarray(sw, dtype=float)
    acc = (sw_ * (pred == y)).sum() / sw_.sum()                 # weighted accuracy (Eq. 14)
    cov = (sw_ * ((F * w) > 0).any(axis=1)).sum() / sw_.sum()   # weighted coverage
    fit = acc + cfg.ga_w_coverage * cov - cfg.ga_w_size * idx.size / len(active)
    return fit, acc, cov


def tune_rules(fis, X, y, cfg, rng, sw=None):
    M = fis.membership_tensor(X)
    base_ant = fis.ant.copy()
    R, n_in = base_ant.shape
    ks = np.array([m.k for m in fis.mfs])

    def mutate(active, ant, rate_scale=1.0):
        active, ant = active.copy(), ant.copy()
        flip = rng.random(R) < cfg.ga_mut_active * rate_scale
        active[flip] = ~active[flip]
        m = rng.random((R, n_in)) < cfg.ga_mut_ant * rate_scale
        for r, j in zip(*np.where(m)):
            ant[r, j] = DONT_CARE if rng.random() < 0.4 else rng.integers(0, ks[j])
        return active, ant

    pop = [(np.ones(R, bool), base_ant.copy())]
    while len(pop) < cfg.ga_pop:
        pop.append(mutate(*pop[0], rate_scale=3.0))
    fits = [_evaluate(fis, M, y, a, an, cfg, sw) for a, an in pop]

    history, best_fit, stall = [], -np.inf, 0
    for gen in range(cfg.ga_max_gen):
        order = np.argsort([-f[0] for f in fits])
        gen_best = fits[order[0]]
        history.append({"gen": gen, "best": gen_best[0], "mean": float(np.mean([f[0] for f in fits])),
                        "acc": gen_best[1], "coverage": gen_best[2]})
        if gen_best[0] > best_fit + 1e-9:
            best_fit, stall = gen_best[0], 0
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
            if rng.random() < cfg.ga_cx_rate:            # uniform crossover at rule level
                mask = rng.random(R) < 0.5
                a1 = np.where(mask, a1, a2)
                n1 = np.where(mask[:, None], n1, n2)
            new.append(mutate(a1, n1))
        pop = new
        fits = [_evaluate(fis, M, y, a, an, cfg, sw) for a, an in pop]

    best = int(np.argmax([f[0] for f in fits]))
    active, ant = pop[best]
    idx = np.where(active)[0]
    cons, w, _ = fis.rule_consequents(M, y, ant[idx], sw)
    keep = w > 0
    # merge duplicate antecedents produced by mutation (keep the higher weight)
    seen, sel = {}, []
    for r in np.where(keep)[0]:
        key = tuple(ant[idx][r])
        if key not in seen or w[r] > w[seen[key]]:
            seen[key] = r
    sel = np.array(sorted(seen.values()), dtype=int)
    fis.ant, fis.cons, fis.weight = ant[idx][sel], cons[sel], w[sel]
    return fis, history
