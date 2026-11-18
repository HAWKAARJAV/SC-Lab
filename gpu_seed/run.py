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
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gfer.adaptive_gate import M2_VARIANTS, m2_weights                       # noqa: E402
from gfer.evaluate import zscore_by_subject                                   # noqa: E402
from gfer.gates import gate_weights, normalise_within_subject, normalise_within_trial  # noqa: E402

from .config import GPUConfig                                                 # noqa: E402
from .features import extract_all                                             # noqa: E402
from .fis_gpu import build_fis, predict_proba_gpu, tune_rules_gpu             # noqa: E402
from .mrmr_gpu import mrmr_weighted                                           # noqa: E402
from .seed4 import CLASS_NAMES, SESSION_LABELS, write_inventory              # noqa: E402

VARIANT_IDS = ["none", "hard", "fuzzy", "adaptive", "adaptive_nosmooth", "adaptive_causal",
               "fuzzy_smooth"]                                                # append only (seeds)
NAMES = {"none": "No gate", "hard": "M0 hard", "fuzzy": "M1 fuzzy", "adaptive": "M2 adaptive temporal",
         "adaptive_nosmooth": "M2 adaptive only", "adaptive_causal": "M2 causal", "fuzzy_smooth": "M1+temporal"}


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
    sp = (StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(tids, tlab) if protocol == "trial"
          else LeaveOneGroupOut().split(tids, tlab, tsub))
    return [(np.where(np.isin(F["trial"], tids[a]))[0], np.where(np.isin(F["trial"], tids[b]))[0]) for a, b in sp]


# ------------------------------------------------------------------ one task
def run_task(F, Xb, names, protocol, variant, band, seed, cfg, device):
    rng = np.random.default_rng(seed * 1000 + VARIANT_IDS.index(variant) * 10 + cfg.bands.index(band))
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
        info = {"fold": fi, "gate_feature": F["eye_names"][gi], "features": [names[i] for i in sel],
                "n_rules": int(len(fis.ant)), "ga_gen": len(hist)}
        if protocol == "paper":                                  # window-level, gate-weighted
            wte = w[te]
            ok = (P.argmax(1) == y[te]).astype(float)
            info["window_acc"] = float((wte * ok).sum() / max(wte.sum(), 1e-12))
            info["n_test_windows"] = int(len(te))
        else:
            preds, wte, tte = [], w[te], trial[te]
            for t in np.unique(tte):
                m = tte == t
                S = (wte[m][:, None] * P[m]).sum(0) / max(wte[m].sum(), 1e-12)
                preds.append((int(t), int(y[te][m][0]), int(S.argmax())))
            info["trial_preds"] = preds
        out.append(info)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder that contains the unzipped SEED-IV data")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    ap.add_argument("--run-name", default=None, help="default: 'full' or 'quick'")
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(__file__), "cache"))
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 subjects, 1 seed, alpha band, trial protocol")
    ap.add_argument("--variants", default=None, help="comma list, default none,hard,fuzzy,adaptive")
    ap.add_argument("--protocols", default=None, help="comma list of trial,loso,paper")
    ap.add_argument("--bands", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--fs", type=float, default=None, help="EEG sampling rate; default inferred")
    ap.add_argument("--labels", default=None, help="JSON {\"1\": [...24], \"2\": [...], \"3\": [...]}")
    ap.add_argument("--sampen-chunk", type=int, default=None, help="lower (e.g. 64) if GPU runs out of memory")
    a = ap.parse_args()

    cfg = GPUConfig()
    if a.variants: cfg.variants = a.variants.split(",")
    if a.protocols: cfg.protocols = a.protocols.split(",")
    if a.bands: cfg.bands = a.bands.split(",")
    if a.seeds: cfg.seeds = [int(s) for s in a.seeds.split(",")]
    if a.sampen_chunk: cfg.sampen_chunk = a.sampen_chunk
    if a.quick:
        cfg.protocols, cfg.bands, cfg.seeds = ["trial"], ["alpha"], [0]
    labels = SESSION_LABELS
    if a.labels:
        labels = {int(k): v for k, v in json.load(open(a.labels)).items()}
    run = a.run_name or ("quick" if a.quick else "full")
    out = os.path.join(a.out, run)
    os.makedirs(os.path.join(out, "tasks"), exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

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
    F = extract_all(a.root, cfg, a.cache, device, labels, a.fs, subjects, log)
    log(f"features: {F['eeg'].shape[0]} windows, {len(F['trial_ids'])} trials, "
        f"{len(set(F['subject']))} subjects  [{time.time() - t0:.0f}s]")
    with open(os.path.join(out, "trials.json"), "w") as f:
        json.dump({"trial_ids": F["trial_ids"],
                   "subject": [F["subject"][F["trial"] == t][0] for t in range(len(F["trial_ids"]))],
                   "label": [int(F["label"][F["trial"] == t][0]) for t in range(len(F["trial_ids"]))],
                   "class_names": CLASS_NAMES, "config": cfg.to_dict(), "device": str(device)}, f)

    tasks = [(p, v, b, s) for p in cfg.protocols for s in cfg.seeds for v in cfg.variants for b in cfg.bands]
    log(f"{len(tasks)} tasks: protocols={cfg.protocols} variants={cfg.variants} bands={cfg.bands} seeds={cfg.seeds}")
    Xcache = {}
    group = F["session"] if cfg.norm_group == "session" else F["subject"]
    for i, (p, v, b, s) in enumerate(tasks, 1):
        path = os.path.join(out, "tasks", f"{p}_{v}_{b}_s{s}.json")
        if os.path.exists(path):
            continue
        if b not in Xcache:
            bi = F["bands"].index(b)
            X = F["eeg"][:, bi].reshape(len(F["label"]), -1)
            Xcache[b] = (np.nan_to_num(zscore_by_subject(X, group)).astype(np.float32),
                         [f"{e}-{c}" for e in F["entropies"] for c in F["ch_names"]])
        t1 = time.time()
        res = run_task(F, *Xcache[b], p, v, b, s, cfg, device)
        rec = {"protocol": p, "variant": v, "band": b, "seed": s, "elapsed_s": time.time() - t1, "folds": res}
        with open(path + ".tmp", "w") as f:
            json.dump(rec, f)
        os.replace(path + ".tmp", path)
        log(f"  [{i}/{len(tasks)}] {p:5s} {NAMES[v]:22s} {b:5s} seed {s}  {time.time() - t1:.0f}s  "
            f"(total {(time.time() - t0) / 60:.1f} min)")
    from .summarize import summarize
    summarize(out)
    log(f"DONE. Results in {out}  (report.md, summary.csv, figures). Total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
