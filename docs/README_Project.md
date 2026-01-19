# CoBench Project Overview

CoBench is a collaborative anomaly detection benchmark that unifies heterogeneous detectors from ADBench (tabular/CV/NLP), GADBench (graphs), and PyGOD (graph GNNs) under a single training API. It focuses on letting models teach each other via pseudo-label exchange, yielding stronger detectors than single-model baselines.

## What’s Inside
- src/base.py: core abstractions (Model, Data, Strategy, CoLearner, SingleModel).
- src/baselines/: wrappers that adapt external benchmarks to the unified API.
- src/mvp.ipynb: end-to-end notebook with real data/models and collaborative tests.
- configs/detectors/: hyperparameters per detector.
- data/: storage for prepared datasets.
- results/: metrics and logs.

## How It Works (high level)
1) Load data through a Data wrapper that standardizes splits and indices.
2) Wrap detectors so they all expose train(), fit(), predict_scores(), get_embeddings().
3) Warmup (optional): each model trains alone.
4) Exchange loop: models score the training set, share pseudo-labels (anomalies/normals) with confidence, then train another epoch on the updated labels.
5) Evaluate ensemble and per-model AUC on the test set each chapter; stop via Strategy.

## Why It’s Useful
- Apples-to-apples comparison across very different anomaly detectors.
- Demonstrates collaborative gains over single-model baselines.
- Extensible: add new detectors or data types by implementing the unified interfaces.

## Current Status
- Core API and ADBench wrappers implemented (PReNet, XGBOD, DeepSAD, DevNet).
- Collaborative loop tested in mvp.ipynb; exchange and cotrain paths exercised.
- SingleModel baseline fixed to default to SimpleStrategy.

## How to Run (short form)
- Install deps from SubModules/ADBench/requirements.txt (plus PyOD for XGBOD).
- Use src/mvp.ipynb to run warmup + collaborative tests on provided datasets.
- For scripts, import from src/base.py and the ADBench wrappers, construct CoLearner, and call cotrain().
