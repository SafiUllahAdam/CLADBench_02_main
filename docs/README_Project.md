# CoBench Project Overview

CoBench is a collaborative anomaly detection benchmark that unifies heterogeneous detectors from ADBench (tabular/CV/NLP), GADBench (graphs), and PyGOD (graph GNNs) under a single training API. It focuses on letting models teach each other via pseudo-label exchange, yielding stronger detectors than single-model baselines.

## Code structure
- src/base.py: core abstractions (Model, Data, Strategy, CoLearner, SingleModel).
- src/baselines/: wrappers that adapt external benchmarks to the unified API.
- src/mvp.ipynb: end-to-end notebook with real data/models and collaborative tests.
- configs/detectors/: hyperparameters per detector.
- data/: storage for prepared datasets.
- results/: metrics and logs.

## Why It’s Useful
- Direct comparison across different benchmarks.
- Demonstrates collaborative gains over single-model baselines.
- Extensible: add new detectors or data types by implementing the unified interfaces.

## Current Status
- Core API and few ADBench wrappers implemented (PReNet, XGBOD, DeepSAD, DevNet).
- Collaborative loop tested but not fully approved especially the recurrent model training

## How to Run (short form)
- Install deps from SubModules/ADBench/requirements.txt (plus PyOD for XGBOD).
- Use src/mvp.ipynb to run warmup + collaborative tests on provided datasets.
- For scripts, import from src/base.py and the ADBench wrappers, construct CoLearner, and call cotrain().
