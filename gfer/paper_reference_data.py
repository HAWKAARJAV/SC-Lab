"""Transcribed values from Farhadi Sedehi et al., IEEE Access 2024 (DOI 10.1109/ACCESS.2024.3430934).

Source PDF: ``Recognition_of_emotional_categories_using_mined_fu (1).pdf`` (repo root).
Do NOT mix these records with illustrative teaching data.
"""
from __future__ import annotations

PAPER = {
    "citation": {
        "title": "Recognition of emotional categories using mined fuzzy rules from "
                 "electroencephalogram signals with gated eye track data approach",
        "authors": "Javid Farhadi Sedehi, Nader Jafarnia Dabanloo, Keivan Maghooli, Ali Sheikhani",
        "journal": "IEEE Access",
        "year": 2024,
        "doi": "10.1109/ACCESS.2024.3430934",
        "source_file": "Recognition_of_emotional_categories_using_mined_fu (1).pdf",
    },
    "eye_features_order": [
        "mean_pupil_diameter_Y",
        "mean_pupil_diameter_X",
        "mean_fixation_duration",
        "std_pupil_diameter_Y",
        "std_pupil_diameter_X",
        "fixation_frequency",
        "max_fixation_duration",
        "std_fixation_duration",
    ],
    "eye_mrmr_top_score_approximate": {
        "SEED-IV": 0.0157272,
        "MAHNOB-HCI": 0.0202258,
        "note": "Approximate — read from Fig. 7 bar chart; not stated numerically in text.",
    },
    "eeg_features_selected": ["PhEn-FP2", "SampEn-T8", "PermEn-T7"],
    "ga": {
        "population_size": 27,
        "stopping_generation_seed_iv_alpha": 450,
        "stopping_patience_generations": 10,
        "elite": None,
        "crossover_rate": None,
        "mutation_rate": None,
        "note": "Elite/crossover/mutation not numerically specified in results section.",
    },
    "evaluation": {"protocol": "5-fold CV", "metrics": ["accuracy", "TPR", "FNR", "AUC"]},
    "frequency_bands_paper_hz": {
        "delta": "1-4",
        "theta": "4-8",
        "alpha": "8-13",
        "beta": "13-25",
        "gamma": "25-60",
    },
    "implementation_wavelet_note": (
        "Repo DWT at 128 Hz (4-level db4) yields dyadic bands "
        "0-4, 4-8, 8-16, 16-32, 32-64 Hz — not identical to paper table."
    ),
    "reported_accuracy_percent": {
        "SEED-IV": {"alpha": 94.42, "beta": 93.02, "gamma": 89.80, "theta": None, "delta": None},
        "MAHNOB-HCI": {"alpha": 85.20, "beta": 82.92, "gamma": 79.05, "theta": None, "delta": None},
        "note_theta_delta": "Paper states lower accuracies; numeric values not given in excerpted results.",
    },
    "auc_summary_approximate": {
        "SEED-IV": {"per_class": ">0.9", "alpha_beta": "~0.99", "gamma": "~0.98", "theta": "~0.95", "delta": "~0.90"},
        "MAHNOB-HCI": {"per_class": ">0.9", "alpha_theta": "~0.97", "beta": "~0.95", "gamma": "~0.94", "delta": "~0.93"},
        "note": "Qualitative from Sec. IV text / Figs. 11 & 13 — not per-class numeric transcription.",
    },
}

# Tables 2–6 (SEED-IV rule sets) transcribed from PDF pp. 11–12.
RULE_SETS = {
    "delta": [
        (1, "If PhEn-FP2 is medium and SampEn-T8 is medium and PermEn-T7 is high", "neutral"),
        (2, "If PhEn-FP2 is medium and SampEn-T8 is low and PermEn-T7 is medium", "sad"),
        (3, "If PhEn-FP2 is medium and SampEn-T8 is high", "neutral"),
        (4, "If PhEn-FP2 is low and PermEn-T7 is low", "sad"),
        (5, "If PhEn-FP2 is medium and SampEn-T8 is medium and PermEn-T7 is medium", "neutral"),
        (6, "If PhEn-FP2 is low and SampEn-T8 is medium and PermEn-T7 is medium", "fear"),
        (7, "If PhEn-FP2 is high and PermEn-T7 is high", "happy"),
        (8, "If SampEn-T8 is high and PermEn-T7 is high", "sad"),
        (9, "If PhEn-FP2 is low and SampEn-T8 is high", "sad"),
        (10, "If SampEn-T8 is medium", "happy"),
    ],
    "theta": [
        (1, "If PhEn-FP2 is medium and SampEn-T8 is medium", "sad"),
        (2, "If PhEn-FP2 is medium and SampEn-T8 is low and PermEn-T7 is low", "sad"),
        (3, "If PhEn-FP2 is medium and SampEn-T8 is high and PermEn-T7 is medium", "neutral"),
        (4, "If PhEn-FP2 is medium and SampEn-T8 is low and PermEn-T7 is high", "happy"),
        (5, "If PhEn-FP2 is low and PermEn-T7 is high", "neutral"),
        (6, "If PhEn-FP2 is low and SampEn-T8 is high and PermEn-T7 is low", "neutral"),
        (7, "If PhEn-FP2 is medium and PermEn-T7 is medium", "happy"),
        (8, "If SampEn-T8 is high", "neutral"),
        (9, "If PhEn-FP2 is medium and SampEn-T8 is medium and PermEn-T7 is medium", "happy"),
        (10, "If PhEn-FP2 is high and SampEn-T8 is low and PermEn-T7 is medium", "neutral"),
    ],
    "alpha": [
        (1, "If PhEn-FP2 is high and SampEn-T8 is medium and PermEn-T7 is low", "happy"),
        (2, "If PhEn-FP2 is low and PermEn-T7 is high", "happy"),
        (3, "If PhEn-FP2 is medium and SampEn-T8 is medium and PermEn-T7 is medium", "fear"),
        (4, "If PhEn-FP2 is medium and SampEn-T8 is medium and PermEn-T7 is low", "happy"),
        (5, "If PhEn-FP2 is low and SampEn-T8 is low and PermEn-T7 is medium", "sad"),
        (6, "If PhEn-FP2 is medium and SampEn-T8 is high and PermEn-T7 is low", "fear"),
        (7, "If PhEn-FP2 is low and SampEn-T8 is medium and PermEn-T7 is medium", "happy"),
        (8, "If PhEn-FP2 is medium and SampEn-T8 is medium", "sad"),
        (9, "If PhEn-FP2 is high and PermEn-T7 is medium", "fear"),
        (10, "If PhEn-FP2 is low and SampEn-T8 is medium and PermEn-T7 is low", "neutral"),
    ],
    "beta": [
        (1, "If PhEn-FP2 is high and SampEn-T8 is medium and PermEn-T7 is low", "happy"),
        (2, "If PhEn-FP2 is low and PermEn-T7 is high", "happy"),
        (3, "If PhEn-FP2 is medium and SampEn-T8 is medium and PermEn-T7 is medium", "fear"),
        (4, "If PhEn-FP2 is medium and SampEn-T8 is medium and PermEn-T7 is low", "happy"),
        (5, "If PhEn-FP2 is low and SampEn-T8 is low and PermEn-T7 is medium", "sad"),
        (6, "If PhEn-FP2 is medium and SampEn-T8 is high and PermEn-T7 is low", "fear"),
        (7, "If PhEn-FP2 is low and SampEn-T8 is medium and PermEn-T7 is medium", "happy"),
        (8, "If PhEn-FP2 is medium and SampEn-T8 is medium", "sad"),
        (9, "If PhEn-FP2 is high and SampEn-T8 is low and PermEn-T7 is medium", "fear"),
        (10, "If PhEn-FP2 is low and SampEn-T8 is medium and PermEn-T7 is low", "neutral"),
    ],
    "gamma": [
        (1, "If PhEn-FP2 is low and Perm-FP2 is high", "happy"),
        (2, "If PhEn-FP2 is high and SampEn-T8 is medium and PermEn-T7 is low", "happy"),
        (3, "If PhEn-FP2 is medium and SampEn-T8 is medium and PermEn-T7 is medium", "fear"),
        (4, "If PhEn-FP2 is medium and SampEn-T8 is medium and PermEn-T7 is low", "happy"),
        (5, "If PhEn-FP2 is low and SampEn-T8 is low and PermEn-T7 is medium", "sad"),
        (6, "If PhEn-FP2 is medium and SampEn-T8 is high and PermEn-T7 is low", "fear"),
        (7, "If PhEn-FP2 is low and SampEn-T8 is medium and PermEn-T7 is medium", "happy"),
        (8, "If PhEn-FP2 is medium and SampEn-T8 is medium", "sad"),
        (9, "If PhEn-FP2 is low and SampEn-T8 is medium and PermEn-T7 is low", "neutral"),
        (10, "If PhEn-FP2 is high and SampEn-T8 is low and PermEn-T7 is high", "fear"),
    ],
}

TABLE_7_RULES = [
    {"emotion": "sadness", "rule": "If PhEn-FP2 is medium and SampEn-T8 is low and PermEn-T7 is medium",
     "coverage_pct": 57, "accuracy_pct": 75.78},
    {"emotion": "sadness", "rule": "If PhEn-FP2 is low and SampEn-T8 is low and PermEn-T7 is medium",
     "coverage_pct": 44, "accuracy_pct": 73.90},
    {"emotion": "sadness", "rule": "If phT7 Is medium, sampT8 Medium",
     "coverage_pct": 43, "accuracy_pct": 72.45},
    {"emotion": "happiness", "rule": "If PhEn-FP2 is high and PermEn-T7 is high",
     "coverage_pct": 58, "accuracy_pct": 82.65},
    {"emotion": "happiness", "rule": "If PhEn-FP2 is medium and SampEn-T8 is low and PermEn-T7 is high",
     "coverage_pct": 41, "accuracy_pct": 84.25},
    {"emotion": "happiness", "rule": "If PhEn-FP2 is low and PeFP2 is high",
     "coverage_pct": 40, "accuracy_pct": 79.65},
    {"emotion": "happiness", "rule": "If PhEn-FP2 is high and SampEn-T8 is medium and PermEn-T7 is low",
     "coverage_pct": 30, "accuracy_pct": 75.45},
    {"emotion": "fear", "rule": "If PhEn-T7 is Low, sampEn-T8 Medium, PHFP2 is Medium",
     "coverage_pct": 55, "accuracy_pct": 72.29},
    {"emotion": "fear", "rule": "If PhEn-T7 is medium, sampT8 Medium, PHFP2 is Medium",
     "coverage_pct": 52, "accuracy_pct": 70.42},
    {"emotion": "neutrality", "rule": "If PhEn-FP2 is medium and SampEn-T8 is medium and PermEn-T7 is high",
     "coverage_pct": 55, "accuracy_pct": 75.23},
    {"emotion": "neutrality", "rule": "If PhEn-FP2 is medium and SampEn-T8 is high and PermEn-T7 is medium",
     "coverage_pct": 45, "accuracy_pct": 70.45},
    {"emotion": "neutrality", "rule": "If PhEn-FP2 is low and SampEn-T8 is medium and PermEn-T7 is low",
     "coverage_pct": 43, "accuracy_pct": 74.25},
]

TABLE_8_COMPARISON = [
    {"ref": "[60]", "dataset": "MAHNOB-HCI", "method": "PSD, RFE, GNB", "evaluation": "10-fold CV",
     "classes": "arousal, valence", "accuracy": "66.7 (arousal), 80 (valence)"},
    {"ref": "[61]", "dataset": "MAHNOB-HCI", "method": "Power spectrum, SVM", "evaluation": "LOSO CV",
     "classes": "arousal, valence", "accuracy": "55.72 (valence), 60.23 (arousal)"},
    {"ref": "[62]", "dataset": "MAHNOB-HCI", "method": "SP, KNN, SVM", "evaluation": "LOSO CV",
     "classes": "arousal, valence", "accuracy": "63 (valence), 65.1 (arousal)"},
    {"ref": "[63]", "dataset": "SEED-IV", "method": "Granger causality, Transfer entropy, KNN",
     "evaluation": "LOSO CV", "classes": "sad, happy, fear, neutral", "accuracy": "90"},
    {"ref": "[24]", "dataset": "SEED-IV", "method": "Multi-branch EEG and eye feature fusion",
     "evaluation": "5-fold CV", "classes": "sad, happy, fear, neutral", "accuracy": "87.32"},
    {"ref": "[7]", "dataset": "SEED-IV", "method": "PSD, DE; eye + multimodal deep learning, FLF",
     "evaluation": "Hold out CV", "classes": "happy, sad, fear, neutral", "accuracy": "85.11"},
    {"ref": "[27]", "dataset": "SEED-IV", "method": "1-D convolution; eye SOFNN", "evaluation": "5-fold CV",
     "classes": "sad, happy, fear, neutral", "accuracy": "86.27"},
    {"ref": "Ours (2024)", "dataset": "SEED-IV, MAHNOB-HCI",
     "method": "Wavelet transform, entropies, mRMR, S-FIS-GA", "evaluation": "5-fold CV",
     "classes": "4 (sad, happy, fear, neutral)", "accuracy": "94.42 (SEED-IV), 85.20 (MAHNOB-HCI)"},
]

CLASSES = ["sadness", "happiness", "fear", "neutrality"]
