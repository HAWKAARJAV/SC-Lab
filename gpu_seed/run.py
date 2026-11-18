"""SEED-IV, GPU: M0 (hard eye gate) vs M1 (fuzzy) vs M2 (adaptive temporal), plus no-gate.

One command does everything and can be re-run safely (finished work is skipped):

    python -m gpu_seed.run --root /path/to/SEED_IV

Steps: inventory -> GPU features (cached) -> experiments (one JSON per task) -> summary.
Results: gpu_seed/results/<run>/   (commit and push this folder)

Protocols
  trial : stratified 5-fold over trials (trial-disjoint), one prediction per trial
  loso  : leave-one-subject-out (15 folds), one prediction per trial
  paper : the paper's protocol, 5-fold over pooled 4-s windows, window-level accuracy
          (optimistic: windows of one trial appear in train and test; for comparison with
          the paper's reported numbers only)
  within: per-person models (subject-dependent): for each subject, 5-fold over that subject's
          72 trials (whole trials held out); the standard SEED-IV setting

Presets: --preset paper (default; the paper's features) or --preset improved (DE + entropy +
eye features pooled over all bands, 4 fuzzy inputs, per-person protocol, logistic-regression
reference). Both reuse the same cached features.
"""
import argparse
import json
import multiprocessing as mp
import os
import platform
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gfer.adaptive_gate import M2_VARIANTS, m2_weights                       # noqa: E402
from gfer.evaluate import zscore_by_subject                                   # noqa: E402
from gfer.gates import gate_weights, normalise_within_subject, normalise_within_trial  # noqa: E402

from .config import PRESETS, GPUConfig                                                 # noqa: E402
from .features import extract_all                                             # noqa: E402
from .fast_fis import build_fis                                              # noqa: E402
from .fis_gpu import predict_proba_gpu, tune_rules_gpu                       # noqa: E402
from .mrmr_gpu import mrmr_weighted                                           # noqa: E402
from .seed4 import CLASS_NAMES, SESSION_LABELS, write_inventory              # noqa: E402

VARIANT_IDS = ["none", "hard", "fuzzy", "adaptive", "adaptive_nosmooth", "adaptive_causal",
               "fuzzy_smooth", "ref_logreg"]                                  # append only (seeds)
BAND_IDS = ["delta", "theta", "alpha", "beta", "gamma", "all"]               # append only (seeds)
NAMES = {"none": "No gate", "hard": "M0 hard", "fuzzy": "M1 fuzzy", "adaptive": "M2 adaptive temporal",
         "adaptive_nosmooth": "M2 adaptive only", "adaptive_causal": "M2 causal", "fuzzy_smooth": "M1+temporal",
         "ref_logreg": "Reference: logistic regression"}


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------ gates (same as ZuCo / gfer)
def window_weights(v, F, variant, group):
    trial, window = F["trial"], F["window"]
    if variant == "none":
        return np.ones(len(v))
    if variant == "hard":                                       # M0: top 40% of each trial
        return gate_weights(normalise_within_trial(v, trial), trial, "hard")
    if variant == "fuzzy":                                      # M1
        return gate_weights(normalise_within_subject(v, group), trial, "fuzzy")
    if variant == "fuzzy_smooth":
        from gfer.adaptive_gate import smooth_within_trial
        return smooth_within_trial(gate_weights(normalise_within_subject(v, group), trial, "fuzzy"),
                                   trial, window, "symmetric")
    if variant in M2_VARIANTS:                                  # M2
        return m2_weights(v, group, trial, window, variant)
    raise ValueError(variant)


# ------------------------------------------------------------------ folds
def folds(F, protocol, seed, n_folds):
    if protocol == "paper":
        sp = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
        return [(a, b) for a, b in sp.split(np.zeros(len(F["label"])), F["label"])]   # window indices
    tids = np.unique(F["trial"])
    tlab = np.array([F["label"][F["trial"] == t][0] for t in tids])
    tsub = np.array([F["subject"][F["trial"] == t][0] for t in tids])
    if protocol == "within":                                     # per-person, whole trials held out
        out = []
        for s_ in np.unique(tsub):
            st, sl = tids[tsub == s_], tlab[tsub == s_]
            for a, b in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(st, sl):
                out.append((np.where(np.isin(F["trial"], st[a]))[0], np.where(np.isin(F["trial"], st[b]))[0]))
        return out
    sp = (StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(tids, tlab) if protocol == "trial"
          else LeaveOneGroupOut().split(tids, tlab, tsub))
    return [(np.where(np.isin(F["trial"], tids[a]))[0], np.where(np.isin(F["trial"], tids[b]))[0]) for a, b in sp]


# ------------------------------------------------------------------ one task
def run_task(F, Xb, names, protocol, variant, band, seed, cfg, device):
    rng = np.random.default_rng(seed * 1000 + VARIANT_IDS.index(variant) * 10 + BAND_IDS.index(band))
    y, trial = F["label"], F["trial"]
    C = len(CLASS_NAMES)
    group = F["session"] if cfg.norm_group == "session" else F["subject"]
    eye = np.nan_to_num(zscore_by_subject(F["eye"], group))
    out = []
    for fi, (tr, te) in enumerate(folds(F, protocol, seed, cfg.n_folds)):
        # gate feature: top eye feature by mRMR on the training windows (paper: "auto")
        dev_eye = torch.tensor(eye[tr], device=device)
        gi = mrmr_weighted(dev_eye, torch.tensor(y[tr], device=device),
                           torch.ones(len(tr), device=device, dtype=torch.float64), k=1, n_bins=cfg.mrmr_bins)[0]
        if variant == "ref_logreg":                              # non-fuzzy reference on all features
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=0.05, max_iter=500, class_weight="balanced")
            clf.fit(Xb[tr], y[tr])
            P = np.zeros((len(te), C)); P[:, clf.classes_] = clf.predict_proba(Xb[te])
            w = np.ones(len(y))
            out.append(_score(protocol, fi, P, w, y, trial, te, {"gate_feature": None, "features": "all",
                                                                 "n_rules": 0, "ga_gen": 0}))
            continue
        w = window_weights(F["eye"][:, gi].astype(float), F, variant, group)
        wtr = w[tr].copy()
        if cfg.ga_balanced:
            cls = np.unique(y[tr])
            cw = {c: wtr.sum() / (len(cls) * max(wtr[y[tr] == c].sum(), 1e-12)) for c in cls}
            wtr = wtr * np.array([cw[c] for c in y[tr]])
        sel = mrmr_weighted(torch.tensor(Xb[tr], device=device), torch.tensor(y[tr], device=device),
                            torch.tensor(wtr, device=device, dtype=torch.float64), k=cfg.n_selected,
                            n_bins=cfg.mrmr_bins)
        Xtr, Xte = Xb[tr][:, sel], Xb[te][:, sel]
        keep = wtr > 0
        fis = build_fis(Xtr[keep], y[tr][keep], cfg.n_levels, C, [names[i] for i in sel], wtr[keep])
        fis, hist = tune_rules_gpu(fis, Xtr[keep], y[tr][keep], cfg, rng, device, sw=wtr[keep])
        P = predict_proba_gpu(fis, Xte, device)
        out.append(_score(protocol, fi, P, w, y, trial, te, {
            "gate_feature": F["eye_names"][gi], "features": [names[i] for i in sel],
            "n_rules": int(len(fis.ant)), "ga_gen": len(hist)}))
    return out


def _score(protocol, fi, P, w, y, trial, te, info):
    info = dict(info, fold=fi)
    if protocol == "paper":                                      # window-level, gate-weighted
        wte = w[te]
        ok = (P.argmax(1) == y[te]).astype(float)
        info["window_acc"] = float((wte * ok).sum() / max(wte.sum(), 1e-12))
        info["n_test_windows"] = int(len(te))
    else:                                                        # one prediction per trial
        preds, wte, tte = [], w[te], trial[te]
        for t in np.unique(tte):
            m = tte == t
            S = (wte[m][:, None] * P[m]).sum(0) / max(wte[m].sum(), 1e-12)
            preds.append((int(t), int(y[te][m][0]), int(S.argmax())))
        info["trial_preds"] = preds
    return info


# ------------------------------------------------------------------ parallel workers
_W = {}
_F_KEYS = ["eye", "subject", "session", "trial", "label", "window"]


def save_shared(F, cfg, shared_dir):
    """Write the arrays every worker needs once; workers memory-map them (shared RAM)."""
    os.makedirs(shared_dir, exist_ok=True)
    group = F["session"] if cfg.norm_group == "session" else F["subject"]
    for k in _F_KEYS:
        np.save(os.path.join(shared_dir, k + ".npy"), F[k])
    all_names = {}
    for b in cfg.bands:
        X, names = feature_matrix(F, b, cfg)
        np.save(os.path.join(shared_dir, f"X_{b}.npy"),
                np.nan_to_num(zscore_by_subject(X, group)).astype(np.float32))
        all_names[b] = names
    with open(os.path.join(shared_dir, "meta.json"), "w") as f:
        json.dump({"names": all_names, "eye_names": F["eye_names"]}, f)


def feature_matrix(F, band, cfg):
    """Candidate features for the classifier: per band (or all bands pooled) the 4 entropies,
    plus the dataset's DE (if use_de), plus the eye features (if eye_in_classifier)."""
    bands = F["bands"] if band == "all" else [band]
    blocks, names = [], []
    for b in bands:
        bi = F["bands"].index(b)
        blocks.append(F["eeg"][:, bi].reshape(len(F["label"]), -1))
        pre = f"{b}:" if band == "all" else ""
        names += [f"{pre}{e}-{c}" for e in F["entropies"] for c in F["ch_names"]]
        if cfg.use_de and F.get("de") is not None:
            blocks.append(F["de"][:, bi])
            names += [f"{pre}DE-{c}" for c in F["ch_names"]]
    if cfg.eye_in_classifier:
        blocks.append(F["eye"])
        names += [f"eye:{n}" for n in F["eye_names"]]
    return np.concatenate(blocks, axis=1).astype(np.float64), names


def _load_shared(shared_dir):
    F = {k: np.load(os.path.join(shared_dir, k + ".npy"), mmap_mode="r") for k in _F_KEYS}
    F = {k: np.asarray(v) for k, v in F.items()}                    # small arrays: copy to RAM
    meta = json.load(open(os.path.join(shared_dir, "meta.json")))
    F["eye_names"] = meta["eye_names"]
    return F, meta["names"]


def _init_worker(shared_dir, cfg, device, threads, out):
    torch.set_num_threads(threads)
    F, names = _load_shared(shared_dir)
    _W.update(F=F, names=names, cfg=cfg, device=torch.device(device), shared=shared_dir, out=out, X={})


def _do_task(task):
    p, v, b, s = task
    W = _W
    if b not in W["X"]:
        W["X"][b] = np.load(os.path.join(W["shared"], f"X_{b}.npy"), mmap_mode="r")
    t1 = time.time()
    res = run_task(W["F"], np.asarray(W["X"][b]), W["names"][b], p, v, b, s, W["cfg"], W["device"])
    path = os.path.join(W["out"], "tasks", f"{p}_{v}_{b}_s{s}.json")
    with open(path + ".tmp", "w") as f:
        json.dump({"protocol": p, "variant": v, "band": b, "seed": s,
                   "elapsed_s": time.time() - t1, "folds": res}, f)
    os.replace(path + ".tmp", path)
    return task, time.time() - t1


def default_workers(device):
    cpus = os.cpu_count() or 4
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info()
        by_vram = int(free / 1.5e9)                                  # ~1.5 GB per worker, safe
        return max(1, min(6, cpus // 2, by_vram))
    return max(1, min(4, cpus // 4))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder that contains the unzipped SEED-IV data")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    ap.add_argument("--run-name", default=None, help="default: 'full' or 'quick'")
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(__file__), "cache"))
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 subjects, 1 seed, 1 band, trial protocol")
    ap.add_argument("--preset", default="paper", choices=list(PRESETS),
                    help="paper (default): the paper's features; improved: DE+entropy+eye, all bands, per-person")
    ap.add_argument("--variants", default=None, help="comma list, default none,hard,fuzzy,adaptive")
    ap.add_argument("--protocols", default=None, help="comma list of trial,loso,paper")
    ap.add_argument("--bands", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--fs", type=float, default=None, help="EEG sampling rate; default inferred")
    ap.add_argument("--labels", default=None, help="JSON {\"1\": [...24], \"2\": [...], \"3\": [...]}")
    ap.add_argument("--sampen-chunk", type=int, default=None, help="signals per SampEn GPU batch (default auto); "
                    "e.g. 64 if the GPU runs out of memory")
    ap.add_argument("--workers", type=int, default=0, help="experiments run in parallel (default auto, ~4 on an "
                    "8 GB GPU); lower to 2 if you see CUDA out-of-memory")
    ap.add_argument("--cpu-workers", type=int, default=0, help="processes that filter EEG during feature "
                    "extraction (default: CPU cores - 2, max 8)")
    a = ap.parse_args()

    cfg = GPUConfig(preset=a.preset)
    for k, v in PRESETS[a.preset].items():
        setattr(cfg, k, v)
    if a.variants: cfg.variants = a.variants.split(",")
    if a.protocols: cfg.protocols = a.protocols.split(",")
    if a.bands: cfg.bands = a.bands.split(",")
    if a.seeds: cfg.seeds = [int(s) for s in a.seeds.split(",")]
    if a.sampen_chunk: cfg.sampen_chunk = a.sampen_chunk
    if a.quick:
        cfg.protocols, cfg.seeds = (["within", "trial"] if a.preset == "improved" else ["trial"]), [0]
        cfg.bands = ["all"] if a.preset == "improved" else ["alpha"]
    labels = SESSION_LABELS
    if a.labels:
        labels = {int(k): v for k, v in json.load(open(a.labels)).items()}
    run = a.run_name or {("paper", False): "full", ("paper", True): "quick",
                         ("improved", False): "improved", ("improved", True): "quick_improved"}[(a.preset, a.quick)]
    out = os.path.join(a.out, run)
    os.makedirs(os.path.join(out, "tasks"), exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "a", encoding="utf-8")

    def log(msg):
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:                         # old Windows consoles
            print(msg.encode("ascii", "replace").decode(), flush=True)
        logf.write(msg + "\n"); logf.flush()

    device = pick_device(a.device)
    log(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')}  device={device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
        + f"  torch={torch.__version__}  python={platform.python_version()}  os={platform.platform()}")
    if device.type == "cpu":
        log("WARNING: no GPU found; this will be very slow. Check that the CUDA build of PyTorch is installed.")

    sessions, inv = write_inventory(a.root, os.path.join(a.out, "dataset_inventory.txt"), labels)
    log(inv.split("\n\n")[0])
    if not sessions:
        log("ERROR: no SEED-IV EEG/eye session pairs found under --root. See dataset_inventory.txt.")
        sys.exit(1)
    subjects = None
    if a.quick:
        subjects = sorted({s["subject"] for s in sessions})[:2]
        log(f"quick mode: subjects {subjects}")
    t0 = time.time()
    cpu_workers = a.cpu_workers or max(1, min(8, (os.cpu_count() or 4) - 2))
    F = extract_all(a.root, cfg, a.cache, device, labels, a.fs, subjects, log, cpu_workers)
    log(f"features: {F['eeg'].shape[0]} windows, {len(F['trial_ids'])} trials, "
        f"{len(set(F['subject']))} subjects  [{time.time() - t0:.0f}s]")
    with open(os.path.join(out, "trials.json"), "w") as f:
        json.dump({"trial_ids": F["trial_ids"],
                   "subject": [F["subject"][F["trial"] == t][0] for t in range(len(F["trial_ids"]))],
                   "label": [int(F["label"][F["trial"] == t][0]) for t in range(len(F["trial_ids"]))],
                   "class_names": CLASS_NAMES, "config": cfg.to_dict(), "device": str(device)}, f)

    order = {"loso": 0, "within": 1, "paper": 2, "trial": 3}     # longest tasks first
    tasks = [(p, v, b, s) for p in sorted(cfg.protocols, key=lambda x: order.get(x, 3))
             for s in cfg.seeds for v in cfg.variants for b in cfg.bands]
    todo = [t for t in tasks if not os.path.exists(os.path.join(out, "tasks", f"{t[0]}_{t[1]}_{t[2]}_s{t[3]}.json"))]
    workers = max(1, min(a.workers or default_workers(device), len(todo) or 1))
    log(f"{len(tasks)} tasks ({len(todo)} to run) on {workers} parallel workers: protocols={cfg.protocols} "
        f"variants={cfg.variants} bands={cfg.bands} seeds={cfg.seeds}")
    shared = os.path.join(a.cache, "shared_" + run)
    save_shared(F, cfg, shared)
    del F
    threads = max(1, (os.cpu_count() or 4) // workers)
    t2, done = time.time(), 0

    def report(task, el):
        nonlocal done
        done += 1
        p, v, b, s = task
        rate = (time.time() - t2) / done
        log(f"  [{done}/{len(todo)}] {p:5s} {NAMES[v]:22s} {b:5s} seed {s}  {el:.0f}s  "
            f"(elapsed {(time.time() - t0) / 60:.1f} min, ~{rate * (len(todo) - done) / 60:.0f} min left)")

    if workers == 1:
        _init_worker(shared, cfg, str(device), threads, out)
        for t in todo:
            report(*_do_task(t))
    elif todo:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker,
                                 initargs=(shared, cfg, str(device), threads, out)) as ex:
            for fut in as_completed([ex.submit(_do_task, t) for t in todo]):
                report(*fut.result())
    from .summarize import summarize
    summarize(out)
    log(f"DONE. Results in {out}  (report.md, summary.csv, figures). Total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
