# CoBench API Reference

Welcome Mohammed! This document should walk you through the main interfaces so you can extend or debug CoBench.

## Model (`src/base.py`)

Every detector inherits from `Model`. The key methods:

| Method | Purpose |
|---|---|
| `train(epochs)` | Run N training epochs using current pseudo-labels |
| `fit()` | Full standalone training (for solo baselines) |
| `predict_scores(indexes, use_train, use_val)` | Anomaly scores in [0,1] on test/train/val |
| `get_embeddings(indexes, use_train)` | Internal representations for the GRU judge |
| `get_loss(use_val)` | Latest train or val loss (optional, used for early stopping) |

A chapter is a training step of `epochs` epochs.

The `y_train` property automatically resolves labels: pseudo-labels > semi-supervised > original, respecting `preserve_labeled`.

## Data (`src/base.py` + `src/baselines/adbench/data_loader.py`)

- `Data` is the abstract base. `ClassicalADBenchData` is the concrete implementation for ADBench `.npz` files.
- On init: loads data, splits train/val/test, creates labeled/unlabeled partition.
- `resolve_labels(policy)` maps unlabeled (-1) samples per policy (`unlabeled_as_normal`, `unlabeled_as_is`, `unlabeled_as_removed`).
- Scaler is fit on train only -- no test leakage.

## CoLearning (`src/base.py` + `src/colearner.py`)

The collaboration loop lives here. Hierarchy:

```
CoLearning (abstract)
  └── SimpleCoLearner         -- bidirectional pseudo-label exchange
        ├── CoLearnerVal      -- adds validation-based monitoring
        └── RecurrentCoLearner -- adds GRU judge training
              └── DelayedRecurrentCoLearner -- delays GRU until chapter N
  └── SingleModel             -- solo baseline (no collaboration)
```

### Pseudo-Label Flow
1. At each chapter, each model scores the unlabeled training data (`predict_scores(use_train=True)`)
2. High-confidence anomalies and normals are proposed to peer models
3. When multiple senders disagree on a sample, the arbiter picks the most confident vote
4. Models retrain on their updated pseudo-labels in the next chapter

### Thresholds
Confidence thresholds control which pseudo-labels get sent. Priority: per-pair override > per-model default > global CoLearner default.

## Strategy (`src/strategy.py`)

| Strategy | Behavior |
|---|---|
| `PlateauStrategy` | Stops when a metric plateaus for `patience` chapters |
| `AdaptivePlateauStrategy` | Same, but patience shrinks after each stop |
| `RecurrentPlateauStrategy` | Monitors ensemble metric before GRU starts, then switches to GRU metric |

All strategies implement `should_continue(metrics, chapter)` and `reset()`.

## Recurrent Judge (`src/recurrentmodels.py`)

`GRURecurrentModel` takes stacked detector embeddings (shape: `[n_samples, n_detectors, emb_dim]`) and predicts anomaly scores. Trained inside `DelayedRecurrentCoLearner` after `recurrent_start_chapter`.

## Model Wrappers (`src/baselines/adbench/ModelWrapperADBench.py`)

Currently available: `PReNetWrapper`, `DeepSADWrapper`, `DevNetWrapper`, `XGBODWrapper`.

When writing a new wrapper:
- `predict_scores()` must support `use_val=True` for validation scoring
- `get_loss(use_val=True)` is recommended for models with loss tracking
- Scores must be normalized to [0, 1]

## Runner (`src/runner.py`)

`run_cv()` orchestrates cross-validation: for each trial it trains solo baselines, a collaborative learner, and optionally a GRU judge, collecting AUC/AP per method.

## Config (`src/benchmark_config.py`)

Single source of truth for paths, dataset registry, model hyperparameters, and CV defaults. Add new datasets to `EVAL_DATASETS` and new model configs to `MODEL_CONFIGS`.

## Extension Checklist

- [ ] Reproduce solo baseline AUC before adding collaboration
- [ ] Graph model compatibility (GADBench/PyGOD wrappers)
- [ ] More ADBench model wrappers