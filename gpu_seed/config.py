"""Settings for the GPU SEED-IV run. Defaults follow the paper (Farhadi Sedehi et al., 2024)."""
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class GPUConfig:
    # preprocessing (paper Sec. III-F)
    bandpass: tuple = (1.0, 60.0)
    filter_order: int = 4
    notch_hz: float = 50.0
    notch_q: float = 30.0
    common_average_ref: bool = False
    target_fs: float = 128.0
    prefiltered: bool = False
    wavelet: str = "db4"
    wavelet_level: int = 4
    window_s: float = 4.0                 # SEED-IV eye features are per 4-s window
    # entropies: paper values; 4-s windows at 128 Hz = 512 samples, enough for m=5
    ent_m: int = 5
    ent_tau: int = 2
    sampen_m: Optional[int] = None
    sampen_r: float = 0.15
    symben_c: int = 3
    phasen_k: int = 5
    # classifier
    n_selected: int = 3
    n_levels: int = 3
    mrmr_bins: int = 10
    ga_pop: int = 27
    ga_max_gen: int = 500
    ga_patience: int = 10
    ga_tournament: int = 3
    ga_elite: int = 2
    ga_cx_rate: float = 0.8
    ga_mut_active: float = 0.05
    ga_mut_ant: float = 0.05
    ga_w_coverage: float = 0.2
    ga_w_size: float = 0.01
    ga_balanced: bool = True              # balanced-accuracy fitness + class-balanced weights
    # experiment
    variants: List[str] = field(default_factory=lambda: ["none", "hard", "fuzzy", "adaptive"])
    protocols: List[str] = field(default_factory=lambda: ["trial", "loso", "paper"])
    bands: List[str] = field(default_factory=lambda: ["delta", "theta", "alpha", "beta", "gamma"])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    n_folds: int = 5
    norm_group: str = "session"           # z-scoring / gate scaling per "session" or "subject"
    sampen_chunk: int = 0                 # signals per SampEn GPU batch; 0 = auto from free GPU memory
    # feature pool for the classifier (the "improved" preset turns these on)
    use_de: bool = False                  # add the dataset's DE features (eeg_feature_smooth, de_LDS)
    eye_in_classifier: bool = False       # let the classifier also select eye-movement features
    preset: str = "paper"


    def to_dict(self):
        return asdict(self)


PRESETS = {
    # the paper's setup: 4 entropies per band, 3 fuzzy inputs, one band at a time
    "paper": {},
    # honest improvements fixed in advance (no tuning on test data): per-person protocol,
    # DE + entropy + eye features pooled over all bands, 4 fuzzy inputs, logistic-regression reference
    "improved": {"use_de": True, "eye_in_classifier": True, "bands": ["all"], "n_selected": 4,
                 "protocols": ["within", "trial", "loso", "paper"],
                 "variants": ["none", "hard", "fuzzy", "adaptive", "ref_logreg"]},
}
