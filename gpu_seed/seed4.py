"""SEED-IV discovery and loading (works with the official layout and Kaggle re-uploads).

Official SEED-IV layout (BCMI lab):
    eeg_raw_data/{1,2,3}/<subject>_<date>.mat      variables <prefix>_eeg1 ... <prefix>_eeg24, (62, samples)
    eye_feature_smooth/{1,2,3}/<subject>_<date>.mat variables eye_1 ... eye_24, (31, windows) of 4-s windows
    (eeg_feature_smooth/, eye_raw_data/ are not needed)

Instead of trusting folder names, every .mat file under --root is inspected:
  * a file with variables matching *_eeg<k> is a raw-EEG session file,
  * a file with variables matching eye_<k> is an eye-feature session file,
  * the session is the nearest parent folder named 1, 2 or 3; the subject is the number
    before the first "_" of the file name.
EEG and eye files are paired by (session, file name). The EEG sampling rate is inferred from
the number of samples per 4-s eye window (snapped to 200/250/500/1000 Hz) unless --fs is given.

Labels (SEED-IV ReadMe): 0 neutral, 1 sad, 2 fear, 3 happy; the same order for every subject
within a session. Override with --labels <json> if your copy's ReadMe differs; the inventory
step prints any ReadMe it finds so you can check.
"""
import json
import os
import re
from collections import defaultdict

import numpy as np

CLASS_NAMES = ["neutral", "sad", "fear", "happy"]
SESSION_LABELS = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}
SEED_CHANNELS = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5",
    "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4",
    "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"]
EEG_VAR = re.compile(r"^[A-Za-z0-9]+_eeg(\d+)$")
EYE_VAR = re.compile(r"^eye_(\d+)$")


def mat_vars(path):
    """{name: shape} without loading the data (MATLAB v5 via scipy, v7.3 via h5py)."""
    import h5py
    import scipy.io as sio
    try:
        if h5py.is_hdf5(path):
            with h5py.File(path, "r") as f:
                return {k: tuple(f[k].shape[::-1]) for k in f.keys() if not k.startswith("#")}
        return {n: tuple(s) for n, s, _ in sio.whosmat(path)}
    except Exception as e:                                       # unreadable / truncated file
        return {"__error__": str(e)}


def load_vars(path, names):
    import h5py
    import scipy.io as sio
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as f:
            return {n: np.array(f[n]).T for n in names}
    d = sio.loadmat(path, variable_names=list(names))
    return {n: d[n] for n in names}


def _session_of(path, root):
    for part in reversed(os.path.relpath(path, root).split(os.sep)[:-1]):
        if part in ("1", "2", "3"):
            return int(part)
    return None


def discover(root):
    """Return (sessions, report). sessions: list of dicts with subject, session, eeg, eye paths."""
    eeg, eye, other, report = {}, {}, [], []
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            if not fn.lower().endswith(".mat"):
                continue
            p = os.path.join(dp, fn)
            v = mat_vars(p)
            ses = _session_of(p, root)
            n_eeg = sum(bool(EEG_VAR.match(k)) for k in v)
            n_eye = sum(bool(EYE_VAR.match(k)) for k in v)
            key = (ses, fn)
            if n_eeg >= 1 and ses is not None:
                eeg[key] = (p, v)
            elif n_eye >= 1 and ses is not None:
                eye[key] = (p, v)
            else:
                other.append((p, sorted(v)[:6]))
    sessions = []
    for key in sorted(eeg, key=lambda k: (k[0], k[1])):
        ses, fn = key
        subj = fn.split("_")[0]
        if key not in eye:
            report.append(f"WARNING no eye-feature file for EEG {eeg[key][0]}")
            continue
        sessions.append({"subject": f"S{int(subj):02d}" if subj.isdigit() else subj, "session": ses,
                         "eeg": eeg[key][0], "eye": eye[key][0], "file": fn,
                         "eeg_vars": eeg[key][1], "eye_vars": eye[key][1]})
    report.append(f"raw-EEG session files: {len(eeg)}; eye-feature session files: {len(eye)}; "
                  f"paired sessions: {len(sessions)}; other .mat files: {len(other)}")
    for p, v in other[:15]:
        report.append(f"  other: {os.path.relpath(p, root)}  vars={v}")
    return sessions, report


def infer_fs(n_samples, n_windows, win_s=4.0):
    raw = n_samples / max(n_windows, 1) / win_s
    return min((200.0, 250.0, 500.0, 1000.0), key=lambda f: abs(f - raw)), raw


def iter_trials(ses, labels=SESSION_LABELS, fs=None, win_s=4.0):
    """Yield one dict per trial: eeg (62, n) float32, fs, eye (n_win, n_feat), label, ids."""
    eeg_names = {int(EEG_VAR.match(k).group(1)): k for k in ses["eeg_vars"] if EEG_VAR.match(k)}
    eye_names = {int(EYE_VAR.match(k).group(1)): k for k in ses["eye_vars"] if EYE_VAR.match(k)}
    common = sorted(set(eeg_names) & set(eye_names))
    E = load_vars(ses["eeg"], [eeg_names[t] for t in common])
    Y = load_vars(ses["eye"], [eye_names[t] for t in common])
    shapes = [Y[eye_names[t]].shape for t in common]
    # feature axis = the axis whose size is the same for every trial
    feat_axis = 0 if len({s[0] for s in shapes}) == 1 else 1
    for t in common:
        x = np.asarray(E[eeg_names[t]], dtype=np.float32)
        if x.shape[0] > x.shape[1]:
            x = x.T                                                  # -> (channels, samples)
        ey = np.asarray(Y[eye_names[t]], dtype=np.float64)
        ey = ey.T if feat_axis == 0 else ey                         # -> (windows, features)
        f, raw = (fs, None) if fs else infer_fs(x.shape[1], ey.shape[0], win_s)
        yield {"subject": ses["subject"], "session": ses["session"], "trial": t,
               "trial_id": f"{ses['subject']}_s{ses['session']}_t{t:02d}",
               "label": int(labels[ses["session"]][t - 1]), "eeg": x, "fs": float(f),
               "fs_raw_estimate": raw, "eye": ey}


def write_inventory(root, out_path, labels=SESSION_LABELS):
    sessions, report = discover(root)
    L = [f"SEED-IV inventory for {root}", ""] + report + [""]
    subj = defaultdict(list)
    for s in sessions:
        subj[s["subject"]].append(s["session"])
    L.append(f"subjects: {len(subj)} -> " + ", ".join(f"{k}:{sorted(v)}" for k, v in sorted(subj.items())))
    if sessions:
        s0 = sessions[0]
        L.append(f"example EEG file: {s0['eeg']}")
        L.append("  vars: " + ", ".join(f"{k}{list(v)}" for k, v in list(s0['eeg_vars'].items())[:4]) + " ...")
        L.append(f"example eye file: {s0['eye']}")
        L.append("  vars: " + ", ".join(f"{k}{list(v)}" for k, v in list(s0['eye_vars'].items())[:4]) + " ...")
        tr = next(iter_trials(s0, labels))
        L.append(f"first trial: eeg {tr['eeg'].shape}, eye {tr['eye'].shape}, fs used {tr['fs']} "
                 f"(estimate {tr['fs_raw_estimate']:.1f} Hz), label {CLASS_NAMES[tr['label']]}")
        if tr["eeg"].shape[0] != 62:
            L.append(f"WARNING: expected 62 EEG channels, found {tr['eeg'].shape[0]}")
    for dp, _, fns in os.walk(root):                              # show any ReadMe for the labels
        for fn in fns:
            if "readme" in fn.lower() and fn.lower().endswith((".txt", ".md")):
                L += ["", f"--- {os.path.join(dp, fn)} ---", open(os.path.join(dp, fn), errors="replace").read()[:4000]]
    L += ["", "labels used (0 neutral, 1 sad, 2 fear, 3 happy):", json.dumps(labels)]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(L) + "\n")
    return sessions, "\n".join(L)
