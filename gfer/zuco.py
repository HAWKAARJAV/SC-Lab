"""ZuCo 1.0 Task 1 (sentiment reading) -> gfer npz trials.

    python -m gfer.zuco --src data/zuco --out data/zuco_npz [--channels 1020|all] [--subjects ZAB,ZDM]

One trial = one sentence; label = sentence sentiment (negative / neutral / positive).
This is an ADAPTATION: ZuCo labels are the sentiment of the text being read, not an
induced emotion as in SEED-IV / MAHNOB-HCI.

Per block (SR1..SR8; the EEG files are named gip_/oip_ and SR/SNR inconsistently):
  * EEG (EEGLAB struct, MATLAB v7.3, 500 Hz, 105 ch): band-pass 1-60 Hz + 50 Hz notch on the
    whole continuous block (avoids edge artefacts on 3-7 s sentences), resample to 128 Hz,
    then cut each sentence [onset, offset).
  * Eye tracking (EyeLink, 500 Hz): pupil area -> diameter proxy sqrt(area) (0 = blink -> NaN);
    fixations from eyeevent.fixations (duration = endtime - latency; the stored column is broken).
  * Markers: 10/11 = sentence on/off, 12/13 = control sentence on/off (followed by a question).
    The 400 sentences are presented in a fixed order, so the k-th onset marker of the session
    is sentence_labels_task1.csv row k (sorted by id). Verified: all 46 control markers fall on
    rows flagged as control.
EEG and eye data are each segmented with their own copy of the markers, so no cross-clock sync
is needed.
"""
import argparse
import glob
import json
import os
import re

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from fractions import Fraction
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

from .data import Trial, save_npz

CLASS_NAMES = ["negative", "neutral", "positive"]
ONSET, OFFSET = (10, 12), (11, 13)
EEG_FS, TARGET_FS = 500.0, 128.0

# Approximate 10-20 positions on the EGI HydroCel-128 net (EGI technical note). Used for the
# fast "1020" channel subset; names keep the electrode number so nothing is hidden.
HC128_1020 = {"Fp1": "E22", "Fp2": "E9", "F7": "E33", "F3": "E24", "Fz": "E11", "F4": "E124",
              "F8": "E122", "T7": "E45", "C3": "E36", "Cz": "Cz", "C4": "E104", "T8": "E108",
              "P7": "E58", "P3": "E52", "Pz": "E62", "P4": "E92", "P8": "E96", "O1": "E70",
              "Oz": "E75", "O2": "E83"}
HC128_1020.pop("Cz")      # Cz is the EGI recording reference: flat (SD = 0) in ZuCo


# ------------------------------------------------------------------ labels
def sentence_labels(meta_dir):
    """Label per presentation position (0..399). NaN-shifted CSV rows are repaired."""
    L = pd.read_csv(os.path.join(meta_dir, "sentiment_labels_task1.csv"), sep=";", comment="#")
    L = L.sort_values("sentence_id").reset_index(drop=True)
    lab = L["sentiment_label"].copy()
    bad = lab.isna()
    # a ';' inside a few sentences pushes the label into the "control" column
    lab[bad] = pd.to_numeric(L.loc[bad, "control"], errors="coerce")
    return L["sentence_id"].to_numpy(), lab.to_numpy()


# ------------------------------------------------------------------ files
def block_files(subj_dir, subj):
    out = {}
    for b in range(1, 9):
        eeg = [f for f in glob.glob(os.path.join(subj_dir, f"*ip_{subj}_S*R{b}_EEG.mat"))
               if re.search(rf"_(SR|SNR){b}_EEG\.mat$", f)]
        et = os.path.join(subj_dir, f"{subj}_SR{b}_corrected_ET.mat")
        if eeg and os.path.exists(et):
            out[b] = (sorted(eeg)[0], et)
    return out


def _h5str(f, ref):
    return "".join(chr(c) for c in np.array(f[ref]).ravel()).strip()


def _read_eeg_block_v5(path):
    """A few ZuCo files are MATLAB v5 instead of v7.3 (HDF5)."""
    E = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["EEG"]
    ev = np.atleast_1d(E.event)
    typ = []
    for e in ev:
        try:
            typ.append(int(str(e.type).strip()))
        except ValueError:
            typ.append(-1)
    lat = np.array([float(e.latency) - 1 for e in ev])
    ch = [str(c.labels).strip() for c in np.atleast_1d(E.chanlocs)]
    return np.asarray(E.data, dtype=np.float64), float(E.srate), ch, np.array(typ), lat


def read_eeg_block(path):
    if not h5py.is_hdf5(path):
        return _read_eeg_block_v5(path)
    with h5py.File(path, "r") as f:
        E = f["EEG"]
        fs = float(np.array(E["srate"]).ravel()[0])
        x = np.array(E["data"], dtype=np.float64).T              # (channels, samples)
        cl = E["chanlocs"]["labels"]
        ch = [_h5str(f, cl[i][0]) for i in range(cl.shape[0])]
        ev = E["event"]
        typ, lat = [], []
        for i in range(ev["type"].shape[0]):
            o = f[ev["type"][i][0]]
            a = np.array(o).ravel()
            t = _h5str(f, ev["type"][i][0]) if o.dtype == np.uint16 else str(int(a[0]))
            try:
                typ.append(int(t))
            except ValueError:
                typ.append(-1)
            lat.append(float(np.array(f[ev["latency"][i][0]]).ravel()[0]) - 1)   # 1-based
    return x, fs, ch, np.array(typ), np.array(lat)


def read_et_block(path):
    d = sio.loadmat(path, squeeze_me=True)
    data = d["data"]                                          # TIME(ms), gaze x, gaze y, pupil area
    fx = d["eyeevent"]["fixations"].item()["data"].item()
    fix = np.atleast_2d(fx)[:, :2] if np.size(fx) else np.zeros((0, 2))
    return data, fix, d["event"][:, 1].astype(int), d["event"][:, 0].astype(float)


def _segments(types, times):
    """[(onset, offset)] in presentation order from 10/11 and 12/13 marker pairs."""
    seg, cur = [], None
    for t, x in zip(types, times):
        if t in ONSET:
            cur = x
        elif t in OFFSET and cur is not None:
            seg.append((cur, x)); cur = None
    return seg


def preprocess_block(x, fs):
    x = x - x.mean(axis=1, keepdims=True)
    sos = butter(4, [1.0, 60.0], btype="bandpass", fs=fs, output="sos")
    x = sosfiltfilt(sos, x, axis=1)
    b, a = iirnotch(50.0, 30.0, fs=fs)
    x = filtfilt(b, a, x, axis=1)
    fr = Fraction(TARGET_FS / fs).limit_denominator(1000)
    return resample_poly(x, fr.numerator, fr.denominator, axis=1)


# ------------------------------------------------------------------ convert
def convert_subject(src, subj, out, labels, n_before, ch_keep, min_s=2.0):
    files = block_files(os.path.join(src, "preprocessed", subj), subj)
    n_trials, report = 0, []
    for b, (eeg_path, et_path) in sorted(files.items()):
        try:
            x, fs, ch, etyp, elat = read_eeg_block(eeg_path)
            et, fix, ttyp, ttime = read_et_block(et_path)
        except Exception as e:                                # truncated download etc.
            report.append(f"block {b}: SKIP ({type(e).__name__}: {e})"); continue
        idx = [ch.index(c) for c in ch_keep] if ch_keep else list(range(len(ch)))
        x = preprocess_block(x[idx], fs)
        eseg, tseg = _segments(etyp, elat), _segments(ttyp, ttime)
        if len(eseg) != len(tseg) or len(eseg) != n_before[b]:
            report.append(f"block {b}: SKIP marker mismatch eeg={len(eseg)} et={len(tseg)} "
                          f"expected={n_before[b]}"); continue
        base = sum(n_before[k] for k in range(1, b))          # presentation index offset
        t_et = et[:, 0]
        area = et[:, 3].astype(float); area[area <= 0] = np.nan
        diam = np.sqrt(area)
        kept = 0
        for k, ((e0, e1), (t0, t1)) in enumerate(zip(eseg, tseg)):
            pos = base + k
            lab = labels[pos]
            if not np.isfinite(lab) or (e1 - e0) / fs < min_s:
                continue
            a, z = int(round(e0 * TARGET_FS / fs)), int(round(e1 * TARGET_FS / fs))
            m = (t_et >= t0) & (t_et < t1)
            pupil = np.c_[(t_et[m] - t0) / 1000.0, diam[m], diam[m]]
            fm = (fix[:, 0] >= t0) & (fix[:, 0] < t1)
            fixations = np.c_[(fix[fm, 0] - t0) / 1000.0, fix[fm, 1] - fix[fm, 0]]
            save_npz(Trial(subject=subj, trial_id=f"B{b}_S{pos + 1:03d}", label=int(lab) + 1,
                           eeg=x[:, a:z], eeg_fs=TARGET_FS, ch_names=[ch[i] for i in idx],
                           pupil=pupil, fixations=fixations), out)
            kept += 1
        n_trials += kept
        report.append(f"block {b}: {kept} sentences")
    return n_trials, report, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/zuco")
    ap.add_argument("--out", default="data/zuco_npz")
    ap.add_argument("--channels", default="1020", choices=["1020", "all"])
    ap.add_argument("--subjects", default=None, help="comma list; default = every folder in src")
    ap.add_argument("--window-s", type=float, default=1.0,
                    help="analysis window stored in meta.json (sentences last 3-7 s)")
    a = ap.parse_args()
    ids, labels = sentence_labels(os.path.join(a.src, "meta"))
    # sentences per block (fixed presentation order; identical for every subject)
    n_before = {1: 49, 2: 50, 3: 50, 4: 50, 5: 51, 6: 49, 7: 50, 8: 51}
    assert sum(n_before.values()) == len(labels) == 400
    ch_keep = list(HC128_1020.values()) if a.channels == "1020" else None
    pre = os.path.join(a.src, "preprocessed")
    subs = a.subjects.split(",") if a.subjects else sorted(
        d for d in os.listdir(pre) if os.path.isdir(os.path.join(pre, d)))
    os.makedirs(a.out, exist_ok=True)
    summary = {}
    for s in subs:
        n, rep, _ = convert_subject(a.src, s, a.out, labels, n_before, ch_keep)
        summary[s] = {"n_trials": n, "blocks": rep}
        print(f"{s}: {n} trials  | " + "; ".join(rep), flush=True)
    ch_names = ([f"{k}({v})" for k, v in HC128_1020.items()] if ch_keep else None)
    if ch_names is None:                                      # all channels: take from a trial
        first = next(s for s in subs if summary[s]["n_trials"])
        f0 = next(iter(block_files(os.path.join(pre, first), first).values()))[0]
        ch_names = read_eeg_block(f0)[2]
    meta = {"dataset": "zuco1_task1_sentiment", "class_names": CLASS_NAMES, "eeg_fs": TARGET_FS,
            "eye_fs": 500.0, "ch_names": ch_names, "window_s": a.window_s, "prefiltered": True,
            # 1-s windows = 128 samples: the paper's m=5, tau=2 leaves SampEn with no template
            # matches in 57-94% of alpha/beta/gamma windows (constant fallback value).
            "config": {"ent_m": 3, "ent_tau": 1, "sampen_m": 2, "ga_balanced": True},
            "note": "REAL EEG (ZuCo 1.0 task 1). Labels = text sentiment, not induced emotion. "
                    "pupil columns = sqrt(pupil area), x and y identical.",
            "subjects": summary}
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"-> {a.out}  ({sum(v['n_trials'] for v in summary.values())} trials)")


if __name__ == "__main__":
    main()
