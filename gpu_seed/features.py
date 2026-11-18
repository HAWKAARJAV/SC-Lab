"""Per-trial features: CPU filtering + wavelet (gfer.signal), GPU entropies, SEED-IV eye features.

Cached per trial in <cache>/<key>/<trial_id>.npz, so an interrupted run resumes where it stopped.
"""
import hashlib
import json
import os
import time

import numpy as np
import torch

from gfer.signal import BAND_ORDER, preprocess, wavelet_bands

from .entropy_gpu import entropy_batch
from .seed4 import SEED_CHANNELS, discover, iter_trials

ENTROPIES = ["SampEn", "PermEn", "SymbEn", "PhEn"]
_KEYS = ["bandpass", "filter_order", "notch_hz", "notch_q", "common_average_ref", "target_fs",
         "wavelet", "wavelet_level", "window_s", "ent_m", "ent_tau", "sampen_m", "sampen_r",
         "symben_c", "phasen_k"]


def cache_key(cfg, extra=""):
    d = {k: getattr(cfg, k) for k in _KEYS}
    d["extra"] = extra
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]


def trial_features(tr, cfg, device):
    x = preprocess(tr["eeg"], tr["fs"], cfg)                      # (C, n) at target_fs
    win = int(round(cfg.window_s * cfg.target_fs))
    n_win = min(x.shape[1] // win, tr["eye"].shape[0])           # align EEG and eye windows
    bands = wavelet_bands(x, cfg.wavelet, cfg.wavelet_level)
    C = x.shape[0]
    segs = np.stack([np.stack([bands[b][:, w * win:(w + 1) * win] for b in BAND_ORDER])
                     for w in range(n_win)])                      # (n_win, 5, C, win)
    xt = torch.from_numpy(np.ascontiguousarray(segs.reshape(-1, win))).to(device)
    ent = entropy_batch(xt, cfg, cfg.sampen_chunk).cpu().numpy()  # (4, n_win*5*C)
    eeg = ent.reshape(4, n_win, len(BAND_ORDER), C).transpose(1, 2, 0, 3)   # (n_win, 5, 4, C)
    return eeg.astype(np.float32), tr["eye"][:n_win].astype(np.float32)


def extract_all(root, cfg, cache_dir, device, labels, fs=None, subjects=None, log=print):
    sessions, _ = discover(root)
    if subjects:
        sessions = [s for s in sessions if s["subject"] in subjects]
    cdir = os.path.join(cache_dir, cache_key(cfg, extra=f"fs={fs}"))
    os.makedirs(cdir, exist_ok=True)
    rows, t0, done = [], time.time(), 0
    n_total = sum(1 for _ in sessions) * 24
    for ses in sessions:
        todo = []
        for t in range(1, 25):
            tid = f"{ses['subject']}_s{ses['session']}_t{t:02d}"
            if not os.path.exists(os.path.join(cdir, tid + ".npz")):
                todo.append(t)
        if todo:
            for tr in iter_trials(ses, labels, fs, cfg.window_s):
                p = os.path.join(cdir, tr["trial_id"] + ".npz")
                if os.path.exists(p):
                    continue
                eeg, eye = trial_features(tr, cfg, device)
                np.savez(p, eeg=eeg, eye=eye, label=tr["label"], fs=tr["fs"])
        for t in range(1, 25):
            tid = f"{ses['subject']}_s{ses['session']}_t{t:02d}"
            p = os.path.join(cdir, tid + ".npz")
            if os.path.exists(p):
                rows.append((ses["subject"], ses["session"], tid, p))
        done += 24
        log(f"  features {done}/{n_total} trials  [{time.time() - t0:.0f}s]")
    eeg, eye, subj, sess, trial, label, win, tids = [], [], [], [], [], [], [], []
    for ti, (s, se, tid, p) in enumerate(rows):
        z = np.load(p)
        n = z["eeg"].shape[0]
        eeg.append(z["eeg"]); eye.append(z["eye"])
        subj += [s] * n; sess += [f"{s}_s{se}"] * n; trial += [ti] * n
        label += [int(z["label"])] * n; win += list(range(n)); tids.append(tid)
    n_ch = eeg[0].shape[-1]
    return {"eeg": np.concatenate(eeg), "eye": np.concatenate(eye),
            "subject": np.array(subj), "session": np.array(sess), "trial": np.array(trial),
            "label": np.array(label), "window": np.array(win), "trial_ids": tids,
            "ch_names": SEED_CHANNELS if n_ch == 62 else [f"ch{i}" for i in range(n_ch)],
            "bands": BAND_ORDER, "entropies": ENTROPIES,
            "eye_names": [f"eye{i:02d}" for i in range(eye[0].shape[1])]}
