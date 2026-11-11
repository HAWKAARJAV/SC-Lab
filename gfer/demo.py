"""Synthetic SEED-IV-shaped demo dataset, for checking that the pipeline runs end to end.

THIS IS NOT REAL DATA. Results on it say nothing about emotion recognition; they only show
the code works and that each stage behaves as designed.

Design (so that every stage has something to find):
  * 62-channel EEG (SEED-IV montage) at 200 Hz: 1/f background + posterior alpha + 50 Hz
    line noise + per-channel noise, mixed through a subject-specific spatial matrix.
  * Each trial has a few "emotional episodes". During an episode:
      - pupil diameter (Y more than X) dilates  -> the eye gate can find the episode;
      - fixation durations change by class;
      - channels around T7, T8, Fp2 receive a class-specific band-limited component whose
        bandwidth / burstiness differs by class -> entropy features differ by class.
        Strength per band: alpha > beta > gamma > theta > delta (mirrors the paper).
    Outside episodes the EEG carries no class information, so gating should help.
  * Distractor pupil dilations with no EEG change make the gate imperfect.
"""
import argparse
import json
import os

import numpy as np

from .data import Trial, save_npz

SEED_IV_CHANNELS = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1",
    "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ",
    "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2",
]
# 32-channel BIOPAC montage (MAHNOB-HCI teaching layout; names normalised to match SEED-style caps)
MAHNOB_CHANNELS = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "CZ",
    "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "PZ", "P4", "P8", "PO3", "PO4", "O1",
    "OZ", "O2", "AF3", "AF4",
]
CLASS_NAMES = ["sad", "happy", "fear", "neutral"]          # SEED-IV order differs; label ids are ours

# channels that carry the emotional signature (paper's selected channels and neighbours)
TARGET_CHANNELS = {"T7": 1.0, "T8": 1.0, "FP2": 1.0, "FT7": 0.6, "TP7": 0.6,
                   "FT8": 0.6, "TP8": 0.6, "FP1": 0.5, "AF4": 0.5}
# band -> (centre Hz, half-width Hz, strength); bands match db4 level-4 DWT at 128 Hz
BANDS = {"delta": (2.5, 1.5, 0.25), "theta": (6.0, 2.0, 0.4), "alpha": (11.0, 3.0, 1.0),
         "beta": (23.0, 7.0, 0.8), "gamma": (44.0, 12.0, 0.6)}
# class -> (relative bandwidth, amplitude, burst modulation depth)
CLASS_SIG = {0: (0.08, 1.2, 0.0),   # sad: narrow-band, regular  -> low entropy
             1: (0.45, 1.0, 0.0),   # happy: medium bandwidth
             2: (0.95, 1.2, 0.9),   # fear: broadband, bursty
             3: (0.0, 0.0, 0.0)}    # neutral: nothing added
PUPIL_DILATION = {0: 0.40, 1: 0.50, 2: 0.65, 3: 0.35}      # mm, added to Y during episodes
FIX_FACTOR = {0: 1.35, 1: 1.10, 2: 0.80, 3: 1.00}          # fixation-duration multiplier


def _pink(n, rng):
    f = np.fft.rfftfreq(n)
    spec = (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return x / x.std()


def _bandnoise(n, fs, lo, hi, rng):
    f = np.fft.rfftfreq(n, 1 / fs)
    spec = rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)
    spec[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(spec, n)
    s = x.std()
    return x / s if s > 0 else x


def _smooth_mask(n, fs, episodes, ramp_s=0.5):
    m = np.zeros(n)
    t = np.arange(n) / fs
    for a, b in episodes:
        up = np.clip((t - a) / ramp_s, 0, 1)
        down = np.clip((b - t) / ramp_s, 0, 1)
        m = np.maximum(m, np.minimum(up, down))
    return m


def _episodes(dur, rng, n_ep, len_range=(6, 10), start=4.0):
    eps = []
    for _ in range(200):
        if len(eps) == n_ep:
            break
        L = rng.uniform(*len_range)
        a = rng.uniform(start, dur - L)
        if all(b0 + 1 < a or a + L + 1 < a0 for a0, b0 in eps):
            eps.append((a, a + L))
    return sorted(eps)


def _graded_intensity(dur, rng, knot_s=1.0, smooth_s=3.0, slope=1.5):
    """Smooth emotional-intensity curve in (0, 1): Gaussian-smoothed white noise at 1-s knots,
    standardised and passed through a logistic. Returns (knot_times, intensity_at_knots)."""
    tk = np.arange(0, dur + knot_s, knot_s)
    raw = rng.standard_normal(len(tk) + 40)
    g = np.exp(-0.5 * (np.arange(-20, 21) * knot_s / smooth_s) ** 2)
    sm = np.convolve(raw, g / g.sum(), mode="same")[20:20 + len(tk)]
    sm = (sm - sm.mean()) / (sm.std() + 1e-12)
    return tk, 1.0 / (1.0 + np.exp(-slope * 2.0 * sm))


def _common_average_ref(eeg: np.ndarray) -> np.ndarray:
    return eeg - eeg.mean(axis=0, keepdims=True)


def _target_weights(ch_names):
    return {ch: w for ch, w in TARGET_CHANNELS.items() if ch in ch_names}


def make_trial(subject, trial_id, label, subj_params, rng, ch_names=None, dur=60.0, fs=200.0,
               eye_fs=60.0, signal_strength=1.0, graded=False, common_average_ref=False):
    ch_names = ch_names or SEED_IV_CHANNELS
    n = int(dur * fs)
    C = len(ch_names)
    mix, pupil_base, gain = subj_params["mix"], subj_params["pupil_base"], subj_params["gain"]
    targets = _target_weights(ch_names)

    # --- background EEG ---
    src = np.stack([_pink(n, rng) for _ in range(mix.shape[1])])
    eeg = mix @ src * 15.0
    post = np.array([1.0 if ch.startswith(("O", "PO", "P", "CB")) else 0.3 for ch in ch_names])
    eeg += np.outer(post, _bandnoise(n, fs, 9, 11, rng)) * 8.0
    eeg += rng.standard_normal((C, n)) * 3.0
    t = np.arange(n) / fs
    eeg += 4.0 * np.sin(2 * np.pi * 50 * t + rng.uniform(0, 6.28))[None, :]   # line noise
    eeg += rng.normal(0, 20, (C, 1))                                           # DC offsets

    # --- episodes ---
    tk, ik = np.array([0.0]), np.array([0.0])
    episodes = _episodes(dur, rng, n_ep=3)
    distractors = _episodes(dur, rng, n_ep=2, len_range=(3, 5))
    emask = _smooth_mask(n, fs, episodes)
    if graded:
        # GRADED mode: emotional intensity varies continuously in (0, 1) instead of on/off episodes.
        # The EEG class component, pupil dilation and fixation change all scale with intensity.
        tk, ik = _graded_intensity(dur, rng)
        emask = np.interp(t, tk, ik)
        episodes = []                                   # ground truth: intervals with intensity > 0.5
        above = ik > 0.5
        i = 0
        while i < len(tk):
            if above[i]:
                j = i
                while j + 1 < len(tk) and above[j + 1]:
                    j += 1
                if tk[j] > tk[i]:
                    episodes.append((float(tk[i]), float(tk[j])))
                i = j + 1
            else:
                i += 1

    bw, amp, burst = CLASS_SIG[label]
    if amp > 0:
        for ch, w in targets.items():
            ci = ch_names.index(ch)
            for band, (fc, hw, strength) in BANDS.items():
                half = max(hw * bw, 0.15)
                comp = _bandnoise(n, fs, fc - half, fc + half, rng)
                if burst > 0:
                    comp *= 1 + burst * np.sin(2 * np.pi * rng.uniform(2, 4) * t)
                eeg[ci] += emask * comp * amp * strength * w * gain * 6.0 * signal_strength

    if common_average_ref:
        eeg = _common_average_ref(eeg)

    # --- pupil (mm) ---
    ne = int(dur * eye_fs)
    te = np.arange(ne) / eye_fs
    drift = np.convolve(rng.standard_normal(ne), np.ones(120) / 120, mode="same") * 0.6
    ep_e = np.interp(te, tk, ik) if graded else _smooth_mask(ne, eye_fs, episodes, ramp_s=1.5)
    dis_e = _smooth_mask(ne, eye_fs, distractors, ramp_s=1.0)
    dil = PUPIL_DILATION[label] * rng.uniform(0.85, 1.15)
    py = pupil_base + drift + ep_e * dil + dis_e * 0.25 + rng.normal(0, 0.03, ne)
    px = pupil_base - 0.05 + drift + ep_e * dil * 0.6 + dis_e * 0.2 + rng.normal(0, 0.03, ne)
    for _ in range(int(dur / 4)):                       # blinks -> NaN
        b = rng.integers(0, ne - 12)
        py[b:b + 9] = np.nan
        px[b:b + 9] = np.nan
    pupil = np.column_stack([te, px, py])

    # --- fixations ---
    fix, tt = [], rng.uniform(0, 0.3)
    while tt < dur:
        if graded:
            mean = 260.0 * (1.0 + (FIX_FACTOR[label] - 1.0) * float(np.interp(tt, tk, ik)))
        else:
            in_ep = any(a <= tt <= b for a, b in episodes)
            mean = 260.0 * (FIX_FACTOR[label] if in_ep else 1.0)
        d = rng.gamma(4.0, mean / 4.0)
        fix.append((tt, d))
        tt += d / 1000.0 + rng.uniform(0.02, 0.06)
    fixations = np.array(fix)

    tr = Trial(subject, trial_id, label, eeg.astype(np.float32), fs, list(ch_names), pupil, fixations)
    if graded:
        return tr, episodes, [float(v) for v in ik]
    return tr, episodes


def generate(out, n_subjects=5, trials_per_class=3, dur=60.0, seed=0, signal_strength=1.0, graded=False,
             ch_names=None, dataset_name="synthetic-demo", common_average_ref=False, eeg_fs=200.0):
    ch_names = ch_names or SEED_IV_CHANNELS
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    C = len(ch_names)
    truth = {}
    for s in range(n_subjects):
        subj = f"S{s + 1:02d}"
        params = {"mix": rng.normal(0, 1, (C, 8)) / np.sqrt(8),
                  "pupil_base": rng.uniform(3.0, 4.2), "gain": rng.uniform(0.8, 1.2)}
        labels = np.repeat(np.arange(4), trials_per_class)
        rng.shuffle(labels)
        for i, lab in enumerate(labels):
            res = make_trial(subj, f"T{i + 1:02d}", int(lab), params, rng, ch_names=ch_names, dur=dur,
                             fs=eeg_fs, signal_strength=signal_strength, graded=graded,
                             common_average_ref=common_average_ref)
            tr, eps = res[0], res[1]
            save_npz(tr, out)
            truth[f"{subj}/{tr.trial_id}"] = {"label": int(lab), "episodes": eps}
            if graded:
                truth[f"{subj}/{tr.trial_id}"]["intensity_1hz"] = res[2]
        print(f"  {subj}: {len(labels)} trials", flush=True)
    meta = {"dataset": dataset_name, "class_names": CLASS_NAMES, "eeg_fs": eeg_fs,
            "eye_fs": 60.0, "ch_names": ch_names, "line_hz": 50.0, "signal_strength": signal_strength,
            "seed": seed, "graded": graded, "common_average_ref": common_average_ref,
            "synthetic": True,
            "note": "SYNTHETIC teaching demo — NOT the official SEED-IV or MAHNOB-HCI release."}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(out, "ground_truth_episodes.json"), "w") as f:
        json.dump(truth, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the synthetic demo dataset")
    ap.add_argument("--out", default="data/demo")
    ap.add_argument("--subjects", type=int, default=5)
    ap.add_argument("--trials-per-class", type=int, default=3)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--signal-strength", type=float, default=1.0,
                    help="scale of the class-specific EEG component (1 = hard, 3 = easy)")
    ap.add_argument("--graded", action="store_true",
                    help="continuous emotional intensity instead of on/off episodes")
    a = ap.parse_args()
    generate(a.out, a.subjects, a.trials_per_class, a.duration, a.seed, a.signal_strength, a.graded)
