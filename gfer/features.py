"""Per-trial feature extraction with on-disk caching.

For every trial and every non-overlapping 4-s window we compute
  * 8 eye features                         -> eye   (n_windows, 8)
  * 4 entropies x n_ch channels x 5 bands  -> eeg   (n_windows, 5, 4, n_ch)
for ALL windows. Gating is applied later as a row mask, so gate settings can be changed
(or switched off for the ablation) without recomputing entropies.
"""
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .config import ENTROPIES, EYE_FEATURES
from .entropy import all_entropies
from .eye import eye_window_features
from .signal import BAND_ORDER, preprocess, wavelet_bands

_FEATURE_KEYS = ["bandpass", "filter_order", "notch_hz", "notch_q", "common_average_ref",
                 "target_fs", "wavelet", "wavelet_level", "window_s", "ent_m", "ent_tau",
                 "sampen_r", "symben_c", "phasen_k"]


def _cache_key(cfg, data_id):
    """Depends on the feature settings AND the dataset, so two datasets with the same
    subject/trial names never share cached features."""
    d = {k: getattr(cfg, k) for k in _FEATURE_KEYS}
    d["data"] = data_id
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _trial_fingerprint(t):
    """Cheap content hash of a trial (shape + a strided sample of the EEG + label)."""
    h = hashlib.md5()
    h.update(f"{t.subject}|{t.trial_id}|{t.label}|{t.eeg.shape}|{t.eeg_fs}".encode())
    h.update(np.ascontiguousarray(t.eeg[:, ::997]).tobytes())
    h.update(np.ascontiguousarray(t.pupil[::97]).tobytes())
    return h.hexdigest()


def trial_features(trial, cfg):
    x = preprocess(trial.eeg, trial.eeg_fs, cfg)
    fs = cfg.target_fs
    win = int(round(cfg.window_s * fs))
    n_win = x.shape[1] // win
    bands = wavelet_bands(x, cfg.wavelet, cfg.wavelet_level)
    eeg = np.zeros((n_win, len(BAND_ORDER), len(ENTROPIES), x.shape[0]), dtype=np.float32)
    for w in range(n_win):
        for bi, band in enumerate(BAND_ORDER):
            eeg[w, bi] = all_entropies(bands[band][:, w * win:(w + 1) * win], cfg)
    eye = eye_window_features(trial.pupil, trial.fixations, n_win, cfg.window_s)
    return eeg, eye


def _worker(args):
    trial, cfg, path = args
    eeg, eye = trial_features(trial, cfg)
    np.savez(path, eeg=eeg, eye=eye)
    return path


def extract_all(trials, cfg, cache_dir):
    """Compute (or load cached) features for all trials. Returns a dict of stacked arrays."""
    data_id = hashlib.md5("".join(_trial_fingerprint(t) for t in trials).encode()).hexdigest()
    cdir = os.path.join(cache_dir, _cache_key(cfg, data_id))
    os.makedirs(cdir, exist_ok=True)
    paths = [os.path.join(cdir, f"{t.subject}__{t.trial_id}.npz") for t in trials]
    todo = [(t, cfg, p) for t, p in zip(trials, paths) if not os.path.exists(p)]
    if todo:
        n_jobs = cfg.n_jobs or os.cpu_count()
        print(f"Extracting features for {len(todo)} trials on {n_jobs} processes "
              f"(cache: {cdir})", flush=True)
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for i, _ in enumerate(ex.map(_worker, todo), 1):
                if i % max(1, len(todo) // 10) == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} trials", flush=True)

    eeg, eye, subj, trial_ix, label, win_ix = [], [], [], [], [], []
    for ti, (t, p) in enumerate(zip(trials, paths)):
        z = np.load(p)
        n = z["eeg"].shape[0]
        eeg.append(z["eeg"]); eye.append(z["eye"])
        subj += [t.subject] * n
        trial_ix += [ti] * n
        label += [t.label] * n
        win_ix += list(range(n))
    return {
        "eeg": np.concatenate(eeg), "eye": np.concatenate(eye),
        "subject": np.array(subj), "trial": np.array(trial_ix),
        "label": np.array(label), "window": np.array(win_ix),
        "ch_names": list(trials[0].ch_names), "bands": BAND_ORDER,
        "entropies": ENTROPIES, "eye_names": EYE_FEATURES,
    }
