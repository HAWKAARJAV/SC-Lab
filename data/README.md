# Teaching demo datasets (synthetic)

This folder holds **synthetic** EEG + eye-tracking data shaped like the public benchmarks used in the SC-Lab project. These files are **not** the official [SEED-IV](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html) or [MAHNOB-HCI](https://mahnob-db.eu/hci/) releases and must not be cited as benchmark results.

| Folder | Channels | Preprocessing hint |
|--------|----------|-------------------|
| `seed_iv_demo/` | 62 (SEED-IV montage names) | No common-average reference |
| `mahnob_demo/` | 32 (MAHNOB-style montage) | Common-average reference applied in the generator |

## Layout

Each dataset follows `gfer.data.load_npz`:

- `meta.json` — class names, sampling rate, channel list, synthetic flag
- `<subject>/<trial_id>.npz` — `eeg` (µV), `pupil`, `fixations`, `label`
- `previews/` — CSV files (~100–250 rows) for classroom inspection without loading full trials

## Regenerate

From the repository root (with your Python environment active):

```bash
python scripts/build_teaching_demo_datasets.py
```

Optional flags: `--subjects 4 --trials-per-class 3 --duration 32 --seed 42`

## Load in code

```python
from gfer.data import load_dataset

seed_trials = load_dataset("npz", root="data/seed_iv_demo")
mahnob_trials = load_dataset("npz", root="data/mahnob_demo")
```

Or use the legacy demo path via `load_dataset("demo", root="data/demo")` after running `python -m gfer.demo --out data/demo`.
