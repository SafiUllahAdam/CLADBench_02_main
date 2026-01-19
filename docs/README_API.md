# CoBench API Notes

This document is written for our dear Adam and hopefully one day those who want to extend or debug CoBench. It summarizes the global API

## Core Interfaces (src/base.py)

### Model (abstract)
- train(epoch): run epoch training epochs (uses current pseudo-labels if present)
- fit(): full training for single-model baselines.
- predict_scores(indexes=None, use_train=False): return anomaly scores in [0, 1]; when use_train=True, score the training split.
- get_embeddings(data_or_indexes=None): return embeddings for downstream ensemble/recurrent models

### Data (abstract)
- Provides X_train, X_test, y_test accessors and global index bookkeeping.
- Maintains per-model pseudo_labels_by_model and pseudo_label_confidence.
- get_pseudo_labels(model_name): retrieve the current pseudo-label vector for a model.
- update_pseudo_labels(model_name, indexes, labels, confidence=None): update labels at provided indices.

### Strategy (abstract + SimpleStrategy)
- should_continue(history): convergence check; SimpleStrategy uses chapter-level AUC plateau with patience.
- get_weights(history): optional weighting; SimpleStrategy returns None.

### CoLearning / CoLearner
- Warmup: optional pretraining for each model (train(epoch)).
- exchange(): each model predicts on training data (predict_scores(use_train=True)), selects anomalies/normals via get_anomaly_indexes / get_normal_indexes, and sends pseudo-labels to peers.
- cotrain(recurrent_model=None, eval_interval=1): main collaborative loop with history logging; evaluates ensemble and per-model AUC on test set.

### SingleModel
- Thin wrapper to train a single model with the same interface; defaults to SimpleStrategy to avoid abstract strategy errors.

## Wrappers (src/baselines/adbench/ModelWrapperADBench.py)
- PReNetWrapper, XGBODWrapper, DeepSADWrapper, DevNetWrapper implement Model.
- All wrappers honor predict_scores(use_train=True/False) to choose X_train vs X_test and accept optional indexes for subset scoring.

## Data Wrapper (src/baselines/adbench/data_loader.py)
- ClassicalADBenchData handles .npz datasets, train/test split, normalization, and index bookkeeping.
- Exposes X_train, X_test, y_test and caches them; train_indexes/test_indexes align with local splits.

## Pseudo-Label Flow
1) Model scores training data (predict_scores(use_train=True)).
2) CoLearner computes anomaly/normal indices based on anomaly_threshold.
3) send_anomalies / send_normals update pseudo_labels_by_model for peers with confidence tracking.
4) Models consume updated pseudo-labels in their train(epoch) implementations.

## Evaluation Path
- Ensemble AUC: mean of model scores on test set; evaluated every chapter
- Per-model AUC: each model’s test scores versus y_test
- RNN AUC: Recurrent judge's scores versus ensemble/mean normal model scores
- Same goes for AP and Rec@k 

## Extension Checklist
- Implement Model methods and ensure predict_scores supports use_train flag.
- If adding new data types, subclass Data to expose train/test splits and pseudo-label storage.
- Register new wrappers under baselines/* as appropriate and ensure compatible configs under configs/detectors/.
