"""All tunable parameters in one place.

Defaults follow Farhadi Sedehi et al., IEEE Access 2024 (DOI 10.1109/ACCESS.2024.3430934).
Where the paper is silent or ambiguous the choice is marked ASSUMPTION.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class Config:
    # ---------------- EEG preprocessing (Sec. III-F) ----------------
    bandpass: tuple = (1.0, 60.0)        # 4th-order Butterworth
    filter_order: int = 4
    notch_hz: float = 50.0               # power line (China / Europe)
    notch_q: float = 30.0
    common_average_ref: bool = False     # paper: MAHNOB only
    target_fs: float = 128.0             # downsample

    # ---------------- Wavelet (Sec. III-F.1) ----------------
    wavelet: str = "db4"
    wavelet_level: int = 4               # at 128 Hz: A4 0-4, D4 4-8, D3 8-16, D2 16-32, D1 32-64 Hz

    # ---------------- Segmentation ----------------
    window_s: float = 4.0                # 4-s segments for both eye and EEG

    # ---------------- Entropies (Sec. III-F.2) ----------------
    ent_m: int = 5                       # embedding dimension (SampEn, PermEn, SymbEn)
    ent_tau: int = 2                     # time delay
    sampen_r: float = 0.15               # x std
    symben_c: int = 3                    # alphabet size
    phasen_k: int = 5                    # SODP sectors

    # ---------------- Eye gate (Sec. III-E, IV-A) ----------------
    gate_feature: str = "auto"           # "auto" = top-ranked eye feature by mRMR
    gate_mode: str = "quantile"          # "quantile" | "max_ratio" | "none"
    # ASSUMPTION: paper says "maximum of this feature is used as threshold"; read literally
    # that keeps one window per trial. We keep windows whose value is in the top part of the trial.
    gate_quantile: float = 0.6           # keep windows >= this within-trial quantile
    gate_max_ratio: float = 0.9          # for gate_mode="max_ratio"

    # ---------------- Feature normalisation ----------------
    # ASSUMPTION: paper silent. Per-subject z-scoring of entropy features (unsupervised).
    normalize: str = "subject_zscore"    # "subject_zscore" | "none"

    # ---------------- EEG feature selection ----------------
    n_selected: int = 3                  # paper: 3 inputs to the S-FIS
    # "auto" = mRMR on the training fold; or a fixed list such as the paper's
    # ["PhEn-FP2", "SampEn-T8", "PermEn-T7"]
    eeg_features: object = "auto"
    mrmr_bins: int = 10

    # ---------------- Fuzzy system (Sec. III-G) ----------------
    n_levels: List[int] = field(default_factory=lambda: [2, 3, 4])   # paper picked 3

    # ---------------- GA (Sec. III-G.2, IV-B) ----------------
    ga_pop: int = 27                     # paper: population 27
    ga_max_gen: int = 500                # paper's run stopped at 450
    ga_patience: int = 10                # stop after 10 generations with no improvement
    ga_tournament: int = 3
    ga_elite: int = 2
    ga_cx_rate: float = 0.8
    ga_mut_active: float = 0.05
    ga_mut_ant: float = 0.05
    ga_w_coverage: float = 0.2           # fitness = acc + w_cov*coverage - w_size*frac_rules
    ga_w_size: float = 0.01
    n_top_rules: int = 10

    # ---------------- Evaluation (Sec. III-H) ----------------
    protocol: str = "paper"              # "paper" (5-fold over segments) | "trial" | "loso"
    n_folds: int = 5
    seed: int = 0
    bands: List[str] = field(default_factory=lambda: ["delta", "theta", "alpha", "beta", "gamma"])
    n_jobs: int = 0                      # 0 = all cores

    def to_dict(self):
        return asdict(self)


ENTROPIES = ["SampEn", "PermEn", "SymbEn", "PhEn"]
EYE_FEATURES = [
    "fix_dur_mean", "fix_dur_std", "fix_freq", "fix_dur_max",
    "pupil_x_mean", "pupil_x_std", "pupil_y_mean", "pupil_y_std",
]
