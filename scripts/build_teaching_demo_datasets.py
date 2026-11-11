#!/usr/bin/env python3
"""Build SEED-IV-shaped and MAHNOB-shaped synthetic teaching datasets under data/.

NPZ trials match gfer.data.load_npz. CSV files under previews/ are for classroom inspection
(~200 rows each). Nothing here is the real benchmark release.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from gfer.data import load_npz
from gfer.demo import MAHNOB_CHANNELS, SEED_IV_CHANNELS, generate


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["sad", "happy", "fear", "neutral"]
PREVIEW_CHANNELS = ["FP1", "FP2", "T7", "T8", "CZ"]


def _write_channels_csv(path: Path, ch_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel_index", "channel_name", "region_hint"])
        for i, ch in enumerate(ch_names):
            if ch.startswith(("FP", "AF", "F")):
                region = "frontal"
            elif ch.startswith(("FC", "FT", "C", "T")):
                region = "central/temporal"
            elif ch.startswith(("CP", "TP", "P")):
                region = "parietal"
            elif ch.startswith(("PO", "O", "CB", "OZ")):
                region = "occipital"
            else:
                region = "other"
            w.writerow([i, ch, region])


def _write_trials_index(path: Path, trials) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "trial_id", "label_id", "label_name", "n_channels", "n_eeg_samples",
                    "duration_s", "n_fixations", "n_pupil_samples"])
        for tr in trials:
            dur = tr.eeg.shape[1] / tr.eeg_fs
            w.writerow([
                tr.subject, tr.trial_id, tr.label, CLASS_NAMES[tr.label],
                tr.eeg.shape[0], tr.eeg.shape[1], round(dur, 2),
                tr.fixations.shape[0], tr.pupil.shape[0],
            ])


def _write_eeg_long_preview(path: Path, trials, ch_names: list[str], n_rows: int = 250,
                            decim_ms: float = 25.0) -> None:
    """Long-format EEG snippet: subject, trial, channel, time_s, amplitude_uv."""
    want = [c for c in PREVIEW_CHANNELS if c in ch_names]
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "trial_id", "label_name", "channel", "time_s", "amplitude_uv"])
        for tr in trials:
            step = max(1, int(tr.eeg_fs * decim_ms / 1000.0))
            times = np.arange(0, tr.eeg.shape[1], step) / tr.eeg_fs
            for ch in want:
                ci = ch_names.index(ch)
                amps = tr.eeg[ci, ::step]
                for t, a in zip(times, amps):
                    w.writerow([tr.subject, tr.trial_id, CLASS_NAMES[tr.label], ch,
                                round(float(t), 3), round(float(a), 2)])
                    rows_written += 1
                    if rows_written >= n_rows:
                        return


def _write_pupil_preview(path: Path, trials, n_rows: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "trial_id", "label_name", "time_s", "pupil_x_mm", "pupil_y_mm"])
        for tr in trials:
            for row in tr.pupil:
                t, px, py = float(row[0]), float(row[1]), float(row[2])
                w.writerow([tr.subject, tr.trial_id, CLASS_NAMES[tr.label],
                            round(t, 3),
                            "" if np.isnan(px) else round(px, 3),
                            "" if np.isnan(py) else round(py, 3)])
                rows_written += 1
                if rows_written >= n_rows:
                    return


def _write_window_stats(path: Path, trials, ch_names: list[str], win_s: float = 4.0,
                        max_rows: int = 200) -> None:
    """One row per 4 s window: mean/std µV for paper-related channels (wide-ish)."""
    want = [c for c in PREVIEW_CHANNELS if c in ch_names]
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["subject", "trial_id", "label_name", "window_index", "window_start_s"]
    for ch in want:
        header.extend([f"{ch}_mean_uv", f"{ch}_std_uv"])
    rows = 0
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for tr in trials:
            n = tr.eeg.shape[1]
            win = int(win_s * tr.eeg_fs)
            wi = 0
            for start in range(0, n - win + 1, win):
                seg = tr.eeg[:, start:start + win]
                row = [tr.subject, tr.trial_id, CLASS_NAMES[tr.label], wi, round(start / tr.eeg_fs, 2)]
                for ch in want:
                    ci = ch_names.index(ch)
                    row.extend([round(float(seg[ci].mean()), 2), round(float(seg[ci].std()), 2)])
                w.writerow(row)
                wi += 1
                rows += 1
                if rows >= max_rows:
                    return


def _export_previews(dataset_root: Path, ch_names: list[str]) -> None:
    trials, _meta = load_npz(str(dataset_root))
    prev = dataset_root / "previews"
    _write_channels_csv(prev / "channels.csv", ch_names)
    _write_trials_index(prev / "trials_index.csv", trials)
    _write_eeg_long_preview(prev / "eeg_samples_long.csv", trials, ch_names, n_rows=250)
    _write_pupil_preview(prev / "pupil_samples.csv", trials, n_rows=200)
    _write_window_stats(prev / "eeg_window_stats.csv", trials, ch_names, max_rows=200)
    summary = {
        "synthetic": True,
        "n_trials": len(trials),
        "n_channels": len(ch_names),
        "preview_files": [
            "channels.csv", "trials_index.csv", "eeg_samples_long.csv",
            "pupil_samples.csv", "eeg_window_stats.csv",
        ],
    }
    with (prev / "README.json").open("w") as f:
        json.dump(summary, f, indent=2)


def build(seed_root: Path, mahnob_root: Path, seed: int, subjects: int, trials_per_class: int,
          duration: float) -> None:
    print("SEED-IV-shaped demo ->", seed_root, flush=True)
    generate(
        str(seed_root), n_subjects=subjects, trials_per_class=trials_per_class, dur=duration,
        seed=seed, ch_names=SEED_IV_CHANNELS, dataset_name="seed_iv_teaching_demo",
        common_average_ref=False,
    )
    _export_previews(seed_root, SEED_IV_CHANNELS)

    print("MAHNOB-shaped demo ->", mahnob_root, flush=True)
    generate(
        str(mahnob_root), n_subjects=subjects, trials_per_class=trials_per_class, dur=duration,
        seed=seed + 1000, ch_names=MAHNOB_CHANNELS, dataset_name="mahnob_teaching_demo",
        common_average_ref=True,
    )
    _export_previews(mahnob_root, MAHNOB_CHANNELS)


def main():
    ap = argparse.ArgumentParser(description="Generate teaching demo datasets (synthetic).")
    ap.add_argument("--seed-iv-out", type=Path, default=ROOT / "data" / "seed_iv_demo")
    ap.add_argument("--mahnob-out", type=Path, default=ROOT / "data" / "mahnob_demo")
    ap.add_argument("--subjects", type=int, default=4)
    ap.add_argument("--trials-per-class", type=int, default=3)
    ap.add_argument("--duration", type=float, default=32.0,
                    help="trial length in seconds (multiple of 4 s window recommended)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args.seed_iv_out, args.mahnob_out, args.seed, args.subjects,
          args.trials_per_class, args.duration)


if __name__ == "__main__":
    main()
