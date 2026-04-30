"""Centralized benchmark configuration — single source of truth for paths and hyperparameters"""

from typing import Dict, Any
from pathlib import Path

# --- Paths ----------------------------------------------------------------

def _find_project_root() -> Path:
    current = Path(__file__).resolve().parent  # src/
    for p in [current, current.parent, current.parent.parent]:
        if (p / "SubModules").exists():
            return p
    raise FileNotFoundError("Cannot find project root (SubModules directory)")

PROJECT_ROOT = _find_project_root()
SRC_ROOT = PROJECT_ROOT / "src"
ADBENCH_ROOT = PROJECT_ROOT / "SubModules" / "ADBench"
ADBENCH_DATASETS = ADBENCH_ROOT / "adbench" / "datasets" / "Classical"
RESULTS_DIR = SRC_ROOT / "results"

# --- Dataset --------------------------------------------------------------

DATASET_CONFIG = {
    "name": "bnp",
    "path": ADBENCH_DATASETS / "bnp_tabular_labeled.npz.zip",
    "test_all_datasets": False,
}

EVAL_DATASETS = {
    "annthyroid": ADBENCH_DATASETS / "2_annthyroid.npz",
    "Ionosphere": ADBENCH_DATASETS / "18_Ionosphere.npz",
    "letter":     ADBENCH_DATASETS / "20_letter.npz",
    "cardio":     ADBENCH_DATASETS / "6_cardio.npz",
    "satellite":  ADBENCH_DATASETS / "30_satellite.npz",
    "fault":      ADBENCH_DATASETS / "12_fault.npz",
    "bnp":        ADBENCH_DATASETS / "bnp_tabular_labeled.npz.zip"
}

# --- Model training -------------------------------------------------------

MODEL_CONFIGS = {
    "prenet":  {"total_epochs": 100, "batch_size": 256},
    "deepsad": {"total_epochs": 100, "pretrain": True, "ae_epochs": 20, "batch_size": 128},
    "devnet":  {"total_epochs": 3, "batch_size": 256, "nb_batch": 5, "network_depth": 2},
    "xgbod":   {"total_epochs": 3},
}

# --- Scenarios ------------------------------------------------------------

TEST_SCENARIOS = {
    "single_model_baseline": {
        "description": "Single PReNet baseline (no collaboration)",
        "models": ["prenet"],
        "warmup_epochs": 0, "max_chapters": 1, "eval_interval": 1,
    },
    "normal_collaborative": {
        "description": "Collaborative training with strict confidence filtering",
        "models": ["prenet", "deepsad"],
        "warmup_epochs": 1, "max_chapters": 3,
        "anomaly_threshold": 0.5,
        "confidence_threshold_low": 0.05, "confidence_threshold_high": 0.95,
        "eval_interval": 1,
    },
    "catastrophic_collaborative": {
        "description": "Inverted-label stress test to verify pseudo-labeling impact",
        "models": ["prenet", "deepsad"],
        "warmup_epochs": 0, "max_chapters": 3,
        "anomaly_threshold": 0.5,
        "confidence_threshold_low": 1.0, "confidence_threshold_high": 0.0,
        "eval_interval": 1, "inverted_labels": True,
    },
    "semisupervised_mode_a": {
        "description": "Semi-supervised: overwrite all labels (preserve_labeled=False)",
        "models": ["prenet", "deepsad"],
        "warmup_epochs": 1, "max_chapters": 3,
        "labeled_ratio": 0.1, "preserve_labeled": False, "eval_interval": 1,
    },
    "semisupervised_mode_b": {
        "description": "Semi-supervised: preserve labeled (preserve_labeled=True)",
        "models": ["prenet", "deepsad"],
        "warmup_epochs": 1, "max_chapters": 3,
        "labeled_ratio": 0.1, "preserve_labeled": True, "eval_interval": 1,
    },
}

# --- Cross-validation defaults --------------------------------------------

CV_DEFAULTS = {
    "seed": 58,
    "n_trials": 3,
    "warmup_epochs": 20,
    "max_chapters": 15,
    "epochs_per_chapter": 3,
    "gru_start_chapter": 3,
    "train_split": 0.60,
    "val_test_split": 0.50,
    "semi_label_ratio": 0.10,
    "semi_stratified": True,
    "preserve_labeled": True,
}

# --- Strategy -------------------------------------------------------------

STRATEGY_CONFIG = {
    "max_chapters": 3,
    "patience": 3,
    "patience_threshold": 0.001,
}

# --- Evaluation -----------------------------------------------------------

EVALUATION_THRESHOLDS = {
    "catastrophic_degradation_min": 0.2,
    "collaboration_improvement_min": 0.01,
}

# --- Logging --------------------------------------------------------------

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": ["console", "file"],
    "log_file": "benchmark_results.log",
}

# --- Accessors ------------------------------------------------------------

def get_scenario(name: str) -> Dict[str, Any]:
    if name not in TEST_SCENARIOS:
        raise KeyError(f"Scenario '{name}' not found. Available: {list(TEST_SCENARIOS.keys())}")
    return TEST_SCENARIOS[name]

def get_model_config(name: str) -> Dict[str, Any]:
    if name not in MODEL_CONFIGS:
        raise KeyError(f"Model '{name}' not found. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[name].copy()

def get_all_scenarios() -> Dict[str, str]:
    return {k: v["description"] for k, v in TEST_SCENARIOS.items()}
