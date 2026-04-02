# CoBench: Collaborative Anomaly Detection Benchmark

A research-grade benchmark for **collaborative semi-supervised anomaly detection**. Multiple heterogeneous models (PReNet, DeepSAD, DevNet, XGBOD) train together by exchanging pseudo-labels on unlabeled data, improving detection beyond solo baselines.

Built on top of [ADBench](https://github.com/Minqi824/ADBench) (tabular anomaly detection). Graph support via [GADBench](https://github.com/FuanKahunaGod/GADBench) and [PyGOD](https://github.com/pygod-team/pygod) is planned.

## Repository Structure

```
CoBench/
├── src/
│   ├── base.py                 # Abstract classes: Model, Data, CoLearning, Strategy
│   ├── main.py                 # CLI entry point
│   ├── colearner.py            # CoLearner variants (SimpleCoLearner, CoLearnerVal, DelayedRecurrentCoLearner)
│   ├── strategy.py             # Convergence strategies (Plateau, Adaptive, RecurrentPlateau)
│   ├── recurrentmodels.py      # GRU judge over stacked detector embeddings
│   ├── runner.py               # Cross-validation runner, summary printing, plot generation
│   ├── results_io.py           # CSV export (stats + raw per-trial)
│   ├── benchmark_config.py     # Paths, hyperparameters, dataset registry
│   ├── utils.py                # Seed, model factory, embedding extraction, data validation
│   ├── alpha.ipynb             # Development / testing notebook
│   └── baselines/
│       ├── adbench/
│       │   ├── ModelWrapperADBench.py   # PReNet, DeepSAD, DevNet, XGBOD wrappers
│       │   └── data_loader.py           # ClassicalADBenchData (NPZ loader + splits)
│       └── gadbench/                    # GADBench wrappers (in progress)
├── configs/                    # Configuration files (detectors, etc.)
├── SubModules/                 # External benchmarks (git submodules, unmodified)
│   ├── ADBench/
│   └── GADBench/
└── docs/
    └── README_API.md           # API reference for developers
```

## Core Concepts

### Model (`base.py`)
All detectors implement:
- `train(epochs)` -- incremental training with pseudo-labels from peers
- `fit()` -- standalone training for solo baselines
- `predict_scores(use_train, use_val)` -- anomaly scores in [0, 1]
- `get_embeddings()` -- intermediate representations for the recurrent judge
- `get_loss(use_val)` -- optional, for early stopping

### Data (`base.py` + `data_loader.py`)
- Loads ADBench `.npz` datasets, splits into train/val/test (stratified, seeded)
- Partitions train into labeled / unlabeled for semi-supervised learning
- `resolve_labels(policy)` handles unlabeled samples: `unlabeled_as_normal` (default), `unlabeled_as_is`, `unlabeled_as_removed`
- Scaler fit on train only (no test leakage)

### CoLearning (`colearner.py`)
Orchestrates multi-model training in chapters:
1. **Warmup** -- pre-train all models independently
2. **Exchange** -- each model scores unlabeled data, sends high-confidence pseudo-labels to peers
3. **Arbitration** -- when senders disagree, a pluggable arbiter (default: max confidence) resolves conflicts
4. **Train** -- each model trains on updated pseudo-labels
5. **Converge** -- strategy decides when to stop

Variants: `CoLearnerVal` (validation-based monitoring), `DelayedRecurrentCoLearner` (adds GRU judge after N chapters).

### Strategy (`strategy.py`)
- `PlateauStrategy` -- stop when a metric plateaus for N chapters
- `AdaptivePlateauStrategy` -- patience decays after each stop
- `RecurrentPlateauStrategy` -- monitors ensemble before GRU starts, then monitors GRU

### Recurrent Judge (`recurrentmodels.py`)
A GRU that learns from stacked detector embeddings. Trained after a configurable delay to let detectors stabilize first.

## Quick Start

```bash
# Solo + collaborative comparison on annthyroid
python src/main.py --models prenet deepsad --n_trials 3 --data_to_test annthyroid

# With GRU recurrent judge
python src/main.py --models prenet deepsad --recurrent_judge gru --data_to_test annthyroid

# Multiple datasets
python src/main.py --models prenet deepsad --data_to_test annthyroid cardio satellite
```

Key CLI flags: `--warmup_epochs`, `--max_chapters`, `--epochs_per_chapter`, `--available_ratio_of_data` (labeled fraction), `--colearning_strategy`, `--patience`.

## Evaluation

| Comparison | What it measures |
|---|---|
| Solo vs Collaborative | Does pseudo-label exchange improve AUC/AP? |
| Collaborative vs Collaborative+GRU | Does a recurrent ensemble add value? |
| Cross-validation (N trials) | Statistical robustness (mean +/- std) |

Primary metrics: ROC-AUC and Average Precision on test set. Monitoring: validation AUC per chapter, exchange precision, loss curves.

## Adding a New Model

1. Subclass `Model` in `src/baselines/`
2. Implement `train(epochs)`, `fit()`, `predict_scores()`, `get_embeddings()`
3. Register in `get_model_detector_dict()` in `ModelWrapperADBench.py`
4. Add hyperparameters to `MODEL_CONFIGS` in `benchmark_config.py`

## References

- **ADBench**: [GitHub](https://github.com/Minqi824/ADBench) -- Tabular anomaly detection (57 datasets, 30 algorithms)
- **GADBench**: [GitHub](https://github.com/FuanKahunaGod/GADBench) -- Graph anomaly detection
- **PyGOD**: [GitHub](https://github.com/pygod-team/pygod) -- PyTorch Geometric-based detectors
