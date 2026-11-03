"""Dataset-independent trial format.

Every dataset loader (demo, SEED-IV, MAHNOB-HCI, ZuCo) converts its files into a list
of `Trial` objects. The rest of the pipeline only sees this format.

On disk (the "npz" format, used by the demo generator):
    <root>/meta.json                       {"class_names": [...], "eeg_fs": ..., ...}
    <root>/<subject>/<trial_id>.npz        arrays: eeg, pupil, fixations; scalars: label
"""
import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Trial:
    subject: str
    trial_id: str
    label: int
    eeg: np.ndarray          # (n_channels, n_samples), microvolts
    eeg_fs: float
    ch_names: List[str]
    pupil: np.ndarray        # (n, 3): time_s, diameter_x, diameter_y  (NaN during blinks)
    fixations: np.ndarray    # (n, 2): onset_s, duration_ms


def save_npz(trial: Trial, root: str):
    d = os.path.join(root, trial.subject)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(
        os.path.join(d, f"{trial.trial_id}.npz"),
        eeg=trial.eeg.astype(np.float32), pupil=trial.pupil.astype(np.float32),
        fixations=trial.fixations.astype(np.float32), label=trial.label,
    )


def load_npz(root: str) -> (List[Trial], dict):
    with open(os.path.join(root, "meta.json")) as f:
        meta = json.load(f)
    trials = []
    for subj in sorted(os.listdir(root)):
        sd = os.path.join(root, subj)
        if not os.path.isdir(sd):
            continue
        for fn in sorted(os.listdir(sd)):
            if not fn.endswith(".npz"):
                continue
            z = np.load(os.path.join(sd, fn))
            trials.append(Trial(
                subject=subj, trial_id=fn[:-4], label=int(z["label"]),
                eeg=z["eeg"], eeg_fs=meta["eeg_fs"], ch_names=meta["ch_names"],
                pupil=z["pupil"], fixations=z["fixations"],
            ))
    return trials, meta


def load_dataset(kind: str, root: str):
    if kind in ("npz", "demo"):
        return load_npz(root)
    raise ValueError(f"No loader for dataset kind '{kind}' yet "
                     "(add one that returns List[Trial], meta).")
