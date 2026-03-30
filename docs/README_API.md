# CoBench API Notes

This document is written for our dear Adam and hopefully one day those who want to extend or debug CoBench. It summarizes the global API

## Core Interfaces (src/base.py)

### Model (abstract)
- `train(epoch)`: run _epoch_ training epochs (uses current pseudo-labels if present)
- `fit()`: full training for single-model baselines.
- `predict_scores(indexes=None, use_train=False, use_val=False)`: return anomaly scores in [0,1]; use_train=True scores training split; use_val=True scores validation split.
- `get_embeddings(data_or_indexes=None, use_train=True)`: return embeddings for ensemble/recurrent models
- `get_loss(use_val=False)`: **optional design rule** — return latest train or val loss if model tracks losses; used for early stopping. Return None if unavailable.

### Data (abstract)
- Provides X/Y and global indexes
- Maintains per-model pseudo_labels_by_model and pseudo_label_confidence.
- get_pseudo_labels(model_name): retrieve the current pseudo-label vector for a model.
- update_pseudo_labels(model_name, indexes, labels, confidence=None): update labels at provided indices.

### Strategy (abstract + SimpleStrategy)
- should_continue(history): convergence check; SimpleStrategy uses chapter-level AUC plateau with patience

### CoLearning / CoLearner
- Warmup: optional pretraining for each model (train(epoch)).
- exchange(): each model predicts on training data (predict_scores(use_train=True)), selects anomalies/normals via get_anomaly_indexes / get_normal_indexes, and sends pseudo-labels to peers.
- cotrain(recurrent_model=None, eval_interval=1): main collaborative loop with history logging; evaluates ensemble and per-model AUC on test set.

## Wrappers (src/baselines/adbench/ModelWrapperADBench.py)
- PReNetWrapper, XGBODWrapper, DeepSADWrapper, DevNetWrapper
- **Design rule for new wrappers**:
  - `predict_scores()` must support `use_val=True` for validation scoring (required)
  - `get_loss(use_val=True)` should be implemented for models with loss tracking (optional but recommended)
  - Never cache predictions based on data identity in a way that breaks with use_val swaps
- TensorFlow import is sus as of today

## Data Wrapper (src/baselines/adbench/data_loader.py)
- ClassicalADBenchData handles .npz datasets, train/test split, normalization, and index bookkeeping.
- Exposes X_train, X_test, y_test and caches them; train_indexes/test_indexes align with local splits.

## Pseudo-Label Flow
1) Model scores training data (predict_scores(use_train=True)) during training phase.
2) CoLearner computes anomaly/normal indices based on anomaly_threshold.
3) send_anomalies / send_normals update pseudo_labels_by_model with confidence tracking.
4) Models consume updated pseudo-labels in their train(epoch) implementations.

## Evaluation Path
- Ensemble AUC: mean of model scores on test set; evaluated every chapter
- Per-model AUC: each model’s test scores versus y_test
- Validation AUC: scored via `predict_scores(use_val=True)` for monitoring early stopping (no test leakage)
- RNN AUC: Recurrent judge's scores versus ensemble/mean normal model scores
- Same goes for AP and Rec@k

## Extension Checklist
**** Assure we get the same results as solo baslines using train_epoch
*** Graph compatibility 
*** More models (ADBench, GADBench, classic models)