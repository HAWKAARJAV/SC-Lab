"""Per-trial features: CPU filtering + wavelet (gfer.signal), GPU entropies, SEED-IV eye features.

Speed: several CPU worker processes load and filter sessions in parallel while the main process
keeps the GPU busy with the entropies. Cached per trial in <cache>/<key>/<trial_id>.npz, so an
interrupted run resumes where it stopped.
"""
import hashlib
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import torch

from gfer.signal import BAND_ORDER, preprocess, wavelet_bands

from .entropy_gpu import entropy_batch
from .seed4 import SEED_CHANNELS, discover, iter_trials, load_de

ENTROPIES = ["SampEn", "PermEn", "SymbEn", "PhEn"]
_KEYS = ["bandpass", "filter_order", "notch_hz", "notch_q", "common_average_ref", "target_fs",
         "wavelet", "wavelet_level", "window_s", "ent_m", "ent_tau", "sampen_m", "sampen_r",
         "symben_c", "phasen_k"]


def cache_key(cfg, extra=""):
    d = {k: getattr(cfg, k) for k in _KEYS}
    d["extra"] = extra
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]


def auto_sampen_chunk(cfg, device):
    """Signals per SampEn batch: as many as fit in ~40% of free GPU memory."""
    if cfg.sampen_chunk:
        return cfg.sampen_chunk
    if device.type != "cuda":
        return 256
    m = (cfg.sampen_m or cfg.ent_m)
    n = int(round(cfg.window_s * cfg.target_fs)) - m * cfg.ent_tau
    free, _ = torch.cuda.mem_get_info(device)
    per_signal = n * n * 4 * 3                       # distance matrix + temporaries
    return int(np.clip(0.4 * free / per_signal, 32, 4096))


def prepare_trial(tr, cfg):
    """CPU part: filter, resample, wavelet bands, cut into windows -> (n_win*5*C, win) float32."""
    x = preprocess(tr["eeg"], tr["fs"], cfg)
    win = int(round(cfg.window_s * cfg.target_fs))
    n_win = min(x.shape[1] // win, tr["eye"].shape[0])
    bands = wavelet_bands(x, cfg.wavelet, cfg.wavelet_level)
    segs = np.stack([np.stack([bands[b][:, w * win:(w + 1) * win] for b in BAND_ORDER])
                     for w in range(n_win)])                             # (n_win, 5, C, win), float64 as in gfer
    return segs, tr["eye"][:n_win].astype(np.float32)


def _prepare_session(args):
    """Filter one session; each trial's windows go to a temporary .npy file (only the paths are
    sent back: a whole session is ~1.5 GB, too large for a Windows inter-process pipe)."""
    ses, labels, fs, cfg, skip, tmpdir = args
    out = []
    for tr in iter_trials(ses, labels, fs, cfg.window_s):
        if tr["trial_id"] in skip:
            continue
        segs, eye = prepare_trial(tr, cfg)
        path = os.path.join(tmpdir, tr["trial_id"] + ".npy")
        np.save(path, segs)
        out.append((tr["trial_id"], path, eye, tr["label"], tr["fs"]))
    return out


def _gpu_entropies(segs, cfg, device, chunk):
    n_win, nb, C, win = segs.shape
    xt = torch.from_numpy(segs.reshape(-1, win)).to(device)
    ent = entropy_batch(xt, cfg, chunk).cpu().numpy()                   # (4, n_win*5*C)
    return ent.reshape(4, n_win, nb, C).transpose(1, 2, 0, 3).astype(np.float32)   # (n_win, 5, 4, C)


def extract_all(root, cfg, cache_dir, device, labels, fs=None, subjects=None, log=print, cpu_workers=1):
    sessions, _ = discover(root)
    if subjects:
        sessions = [s for s in sessions if s["subject"] in subjects]
    cdir = os.path.join(cache_dir, cache_key(cfg, extra=f"fs={fs}"))
    os.makedirs(cdir, exist_ok=True)
    chunk = auto_sampen_chunk(cfg, device)
    tid = lambda s, t: f"{s['subject']}_s{s['session']}_t{t:02d}"
    cached = {f[:-4] for f in os.listdir(cdir) if f.endswith(".npz")}
    tmpdir = os.path.join(cdir, "_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    for f in os.listdir(tmpdir):                       # leftovers from an interrupted run
        os.remove(os.path.join(tmpdir, f))
    jobs = [(s, labels, fs, cfg, {tid(s, t) for t in range(1, 25) if tid(s, t) in cached}, tmpdir)
            for s in sessions if any(tid(s, t) not in cached for t in range(1, 25))]
    log(f"features: {len(sessions)} sessions, {len(jobs)} still to compute; "
        f"{cpu_workers} CPU workers, SampEn batch {chunk} on {device}")
    t0, done = time.time(), 0

    def consume(results):
        nonlocal done
        for trial_id, path, eye, label, f in results:
            segs = np.load(path)
            eeg = _gpu_entropies(segs, cfg, device, chunk)
            np.savez(os.path.join(cdir, trial_id + ".npz"), eeg=eeg, eye=eye, label=label, fs=f)
            del segs
            os.remove(path)
        done += 1
        log(f"  features: session {done}/{len(jobs)} done  [{time.time() - t0:.0f}s]")

    if jobs and cpu_workers > 1:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(cpu_workers, mp_context=ctx) as ex:
            pending, it = set(), iter(jobs)
            for j in it:                                   # keep at most cpu_workers sessions in flight
                pending.add(ex.submit(_prepare_session, j))
                if len(pending) >= cpu_workers:
                    fin, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in fin:
                        consume(f.result())
            for f in pending:
                consume(f.result())
    else:
        for j in jobs:
            consume(_prepare_session(j))

    rows = []
    for ses in sessions:
        for t in range(1, 25):
            p = os.path.join(cdir, tid(ses, t) + ".npz")
            if os.path.exists(p):
                rows.append((ses, t, tid(ses, t), p))
    eeg, eye, subj, sess, trial, label, win, tids, des = [], [], [], [], [], [], [], [], []
    de_cache, have_de = {}, all(r[0].get("de") for r in rows)
    for ti, (ses, tnum, t_id, p) in enumerate(rows):
        s, se = ses["subject"], ses["session"]
        z = np.load(p)
        n = z["eeg"].shape[0]
        eeg.append(z["eeg"]); eye.append(z["eye"])
        if have_de:                                   # dataset DE features, aligned to the same windows
            key = (s, se)
            if key not in de_cache:
                de_cache.clear(); de_cache[key] = load_de(ses)
            d = de_cache[key].get(tnum)
            arr = np.full((n, 5, z["eeg"].shape[-1]), np.nan, np.float32)
            if d is not None:
                m = min(n, d.shape[0]); arr[:m] = d[:m, :, :arr.shape[2]]
            des.append(arr)
        subj += [s] * n; sess += [f"{s}_s{se}"] * n; trial += [ti] * n
        label += [int(z["label"])] * n; win += list(range(n)); tids.append(t_id)
    n_ch = eeg[0].shape[-1]
    if have_de:
        log(f"DE features loaded from eeg_feature_smooth for {len(des)} trials")
    return {"eeg": np.concatenate(eeg), "eye": np.concatenate(eye),
            "de": np.concatenate(des) if have_de else None,
            "subject": np.array(subj), "session": np.array(sess), "trial": np.array(trial),
            "label": np.array(label), "window": np.array(win), "trial_ids": tids,
            "ch_names": SEED_CHANNELS if n_ch == 62 else [f"ch{i}" for i in range(n_ch)],
            "bands": BAND_ORDER, "entropies": ENTROPIES,
            "eye_names": [f"eye{i:02d}" for i in range(eye[0].shape[1])]}
