"""
Centralized benchmark configurations for MVP tests.

Follows industrial standards:
- Single source of truth for hyperparameters
- Type-safe configuration dictionaries
- Easy to modify/extend without code changes
"""

from typing import Dict, Any
from pathlib import Path

# ============================================================================
# PATH CONFIGURATION (auto-detect project root)
# ============================================================================

def _find_project_root() -> Path:
    """Find project root by looking for SubModules directory."""
    current = Path(__file__).resolve().parent  # src/
    for parent in [current, current.parent, current.parent.parent]:
        if (parent / "SubModules").exists():
            return parent
    raise FileNotFoundError("Could not find project root (SubModules directory)")

PROJECT_ROOT = _find_project_root()
SRC_ROOT = PROJECT_ROOT / "src"
ADBENCH_ROOT = PROJECT_ROOT / "SubModules" / "ADBench"
ADBENCH_DATASETS = ADBENCH_ROOT / "adbench" / "datasets" / "Classical"
RESULTS_DIR = SRC_ROOT / "results"

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATASET_CONFIG = {
    "name": "annthyroid",
    "path": ADBENCH_DATASETS / "2_annthyroid.npz",
    #"path": ADBENCH_DATASETS / "12_fault.npz",
    "test_all_datasets": False,  # Set to True to smoke-test all ADBench datasets
}

# Available datasets for EVAL benchmarks
EVAL_DATASETS = {
    "annthyroid": ADBENCH_DATASETS / "2_annthyroid.npz",
    "Ionosphere": ADBENCH_DATASETS / "18_Ionosphere.npz",
    "letter": ADBENCH_DATASETS / "20_letter.npz",
    "cardio": ADBENCH_DATASETS / "6_cardio.npz",
    "satellite": ADBENCH_DATASETS / "30_satellite.npz",
    "fault" : ADBENCH_DATASETS / "12_fault.npz"
}

# ============================================================================
# MODEL TRAINING CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = {
    "prenet": {
        "total_epochs": 100,
        "batch_size": 256,
    },
    "deepsad": {
        "total_epochs": 100,
        "pretrain": True,
        "ae_epochs": 20,
        "batch_size": 128,
    },
    "devnet": {
        "total_epochs": 3,
        "batch_size": 256,
        "nb_batch": 5,
        "network_depth": 2,
    },
    "xgbod": {
        "total_epochs": 3,
    },
}

# ============================================================================
# TEST SCENARIOS
# ============================================================================

TEST_SCENARIOS = {
    "single_model_baseline": {
        "description": "Single model baseline (PReNet) without collaboration",
        "models": ["prenet"],
        "warmup_epochs": 0,
        "max_chapters": 1,
        "eval_interval": 1,
    },
    
    "normal_collaborative": {
        "description": "Normal collaborative training with strict confidence filtering",
        "models": ["prenet", "deepsad"],
        "warmup_epochs": 1,
        "max_chapters": 3,
        "anomaly_threshold": 0.5,
        "confidence_threshold_low": 0.05,    # Only normals with score < 0.05
        "confidence_threshold_high": 0.95,   # Only anomalies with score > 0.95
        "eval_interval": 1,
    },
    
    "catastrophic_collaborative": {
        "description": "Catastrophic training (inverted labels) to verify pseudo-labeling impact",
        "models": ["prenet", "deepsad"],
        "warmup_epochs": 0,  # No clean warmup phase
        "max_chapters": 3,
        "anomaly_threshold": 0.5,
        "confidence_threshold_low": 1.0,     # Accept all as normals
        "confidence_threshold_high": 0.0,    # Accept all as anomalies
        "eval_interval": 1,
        "inverted_labels": True,  # Flag for custom exchange()
    },
    
    "semisupervised_mode_a": {
        "description": "Semi-supervised: overwrite all labels (preserve_labeled=False)",
        "models": ["prenet", "deepsad"],
        "warmup_epochs": 1,
        "max_chapters": 3,
        "labeled_ratio": 0.1,  # 10% labeled samples
        "preserve_labeled": False,
        "eval_interval": 1,
    },
    
    "semisupervised_mode_b": {
        "description": "Semi-supervised: preserve labeled samples (preserve_labeled=True)",
        "models": ["prenet", "deepsad"],
        "warmup_epochs": 1,
        "max_chapters": 3,
        "labeled_ratio": 0.1,  # 10% labeled samples
        "preserve_labeled": True,
        "eval_interval": 1,
    },
}

# ============================================================================
# CONVERGENCE STRATEGY
# ============================================================================

STRATEGY_CONFIG = {
    "max_chapters": 3,
    "patience": 3,  # Stop after N chapters without improvement
    "patience_threshold": 0.001,  # Minimum improvement threshold
}

# ============================================================================
# EVALUATION THRESHOLDS
# ============================================================================

EVALUATION_THRESHOLDS = {
    "catastrophic_degradation_min": 0.2,  # Catastrophic mode must degrade by >=0.2 AUC
    "collaboration_improvement_min": 0.01,  # Collaboration should improve AUC by >=0.01
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": ["console", "file"],  # Console + file logging
    "log_file": "benchmark_results.log",
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_scenario(scenario_name: str) -> Dict[str, Any]:
    """
    Get test scenario configuration by name.
    
    Args:
        scenario_name: Key in TEST_SCENARIOS dict
    
    Returns:
        Configuration dictionary for the scenario
    
    Raises:
        KeyError: If scenario not found
    """
    if scenario_name not in TEST_SCENARIOS:
        available = list(TEST_SCENARIOS.keys())
        raise KeyError(f"Scenario '{scenario_name}' not found. Available: {available}")
    return TEST_SCENARIOS[scenario_name]


def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get model training configuration by name.
    
    Args:
        model_name: Key in MODEL_CONFIGS dict
    
    Returns:
        Configuration dictionary for the model
    
    Raises:
        KeyError: If model not found
    """
    if model_name not in MODEL_CONFIGS:
        available = list(MODEL_CONFIGS.keys())
        raise KeyError(f"Model '{model_name}' not found. Available: {available}")
    return MODEL_CONFIGS[model_name].copy()


def get_all_scenarios() -> Dict[str, str]:
    """
    Get all available test scenarios with descriptions.
    
    Returns:
        Dict mapping scenario names to descriptions
    """
    return {name: cfg["description"] for name, cfg in TEST_SCENARIOS.items()}
