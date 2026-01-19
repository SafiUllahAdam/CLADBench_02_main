# CoBench: Collaborative Anomaly Detection Benchmark

A unified benchmark suite for anomaly detection that combines **ADBench** (tabular/image/NLP anomalies) and **GADBench** (graph anomalies) into a single collaborative learning framework.

## Vision

CoBench enables **collaborative anomaly detection** where multiple heterogeneous models (classical, deep learning, GNN-based) train together through pseudo-label exchange, improving detection performance beyond single-model baselines.

## Architecture Overview

```
CoBench/
├── src/                          # Core implementation
│   ├── base.py                   # Abstract architecture (Model, Data, CoLearning interfaces)
│   ├── main.py                   # Benchmark orchestration pipeline
│   ├── ADWrapperutils.py         # Utilities for model/data wrappers
│   ├── baselines/                # Model & data adapters for SubModules
│   │   ├── adbench/              # ADBench wrapper (tabular, CV, NLP models)
│   │   ├── gadbench/             # GADBench wrapper (graph models)
│   │   └── pygod/                # PyGOD wrapper (graph detector suite)
│   └── utils/                    # Shared utilities
├── configs/                      # Configuration files
│   └── detectors/                # Model-specific hyperparameters
├── data/                         # Dataset storage & preprocessing
│   ├── adbench/                  # Tabular/image/NLP datasets
│   ├── gadbench/                 # Graph datasets
│   └── pygod/                    # Graph anomaly datasets
├── SubModules/                   # External benchmarks (unmodified)
│   ├── ADBench/                  # Tabular & deep learning anomaly detection
│   ├── GADBench/                 # Graph anomaly detection
│   └── pygod/                    # PyTorch Geometric-based detectors
└── results/                      # Experiment outputs & logs
```

## Core Concepts

### 1. **Model Interface** (`base.py`)
All detector implementations inherit from `Model` with two training paradigms:
- **`train(epoch)`**: Single epoch of collaborative training (receives pseudo-labels from other models)
- **`fit()`**: Standalone training for single-model baseline comparison
- **`get_embeddings(data_or_indexes)`**: Extract intermediate representations for embedding aggregation
- **`predict_scores(indexes)`**: Generate anomaly scores for pseudo-labeling

### 2. **Data Management** (`Data` class)
Handles dataset-level operations:
- Global index management across train/test splits
- Pseudo-label storage and updates
- Mask management for supervised/unsupervised partitions
- Train/validation/test split generation

### 3. **Collaborative Learning** (`CoLearning` base class)
Orchestrates multi-model training:
- **Warmup phase**: Pre-train all models independently for N epochs
- **Exchange phase**: Aggregate embeddings and pseudo-labels across models
- **Joint training**: Each model trains on updated pseudo-labels from peers
- **Convergence**: Strategy-driven stopping condition

### 4. **Concrete Implementations**
- **`CoLearner`**: Multi-model collaborative training with asynchronous pseudo-label exchange
- **`SingleCoLearner`**: Single-model baseline (no collaboration)
- **`RecurrentModel`**: Post-training model that learns from aggregated embeddings (ensemble learning)

## Workflow

```
┌─────────────────────────────────────────────────────┐
│ 1. Load Dataset (ADBench/GADBench/PyGOD)            │
│    ↓ Data wrapper normalizes to common interface     │
└─────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────┐
│ 2. Instantiate Models (ADBench/GADBench/PyGOD)      │
│    ↓ Model wrappers implement Model interface       │
└─────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────┐
│ 3. Create CoLearner with Strategy                   │
│    ↓ Warmup: Pre-train all models                   │
└─────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────┐
│ 4. Main Loop (Chapter-based Exchange)               │
│    ├─ Exchange: Aggregate embeddings/pseudo-labels  │
│    ├─ Train: Each model trains on new labels        │
│    └─ Converge: Check convergence criteria          │
└─────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────┐
│ 5. Evaluate & Report Results                        │
│    ├─ ROC-AUC, Precision, Recall metrics            │
│    └─ Compare single-model vs collaborative results │
└─────────────────────────────────────────────────────┘
```

## Key Design Features

### Unification across Benchmarks
- **Common Model Interface**: All detectors (tabular, image, NLP, graph) implement `train(epoch)` and `get_embeddings()`
- **Modular Wrappers**: Data and model adapters in `baselines/` convert ADBench, GADBench, PyGOD to unified API
- **Flexible Data Handling**: Support heterogeneous input types (numpy arrays, graphs, text) through Data wrapper

### Collaborative Learning
- **Pseudo-Label Exchange**: Models improve each other's training data through inter-model label sharing
- **Embedding Aggregation**: Extract intermediate embeddings to create ensemble-like recurrent learners
- **Asynchronous Training**: Strategy class manages models at different convergence speeds
- **Epoch-level Granularity**: `train(epoch)` enables fine-grained control vs traditional `fit()` one-shot training

### Benchmarking Clarity
- **Single vs Collaborative**: `SingleCoLearner` vs `CoLearner` for fair baseline comparison
- **Configurable Combinations**: Test any subset of {ADBench, GADBench, PyGOD} models
- **Multi-trial Stability**: Run N trials with different random seeds
- **Reproducible**: Seed management, time tracking, metric logging

## Quick Start for LLM Integration

When implementing new models or adapters:

1. **Implement Model Interface**
   ```python
   class MyDetector(Model):
       def train(self, epoch):
           # Update on current epoch with pseudo-labels
           pass
       def get_embeddings(self, data_or_indexes):
           # Return intermediate representation (numpy array or tensor)
           return embeddings
       def predict_scores(self, indexes):
           # Return anomaly scores [0, 1] range
           return scores
   ```

2. **Create Data Wrapper** (if needed)
   ```python
   class MyDataWrapper(Data):
       def pseudo_label(self, scores):
           # Convert scores to binary labels
           pass
   ```

3. **Add to Baseline Registry**
   ```python
   # In baselines/__init__.py
   model_detector_dict = {
       'my_detector': MyDetector,
       ...
   }
   ```

4. **Use in Benchmark**
   ```python
   # main.py automatically discovers and benchmarks your model
   ```

## Development Guidelines

### For Model Developers
- Focus on `train()` and `get_embeddings()` methods
- Ensure `predict_scores()` returns consistent 1D array of anomaly scores
- Store model state to enable epoch-level training (vs one-shot `fit()`)

### For Integration Engineers
- Each benchmark (ADBench/GADBench/PyGOD) has dedicated wrapper in `baselines/`
- Data wrappers normalize different input formats (graphs, tables, images)
- Model wrappers expose uniform training interface

### For Benchmark Runners
- Modify `main.py` for dataset/model combinations
- Configure hyperparameters in `configs/detectors/`
- Strategy class in `base.py` governs convergence behavior

## File Descriptions

| File/Folder | Purpose |
|---|---|
| `base.py` | Abstract base classes: Model, Data, CoLearning, RecurrentModel, Strategy |
| `main.py` | Benchmark orchestration: dataset loading, model instantiation, co-training loop |
| `ADWrapperutils.py` | Common utilities for model/data wrappers |
| `baselines/base.py` | Base wrapper classes for benchmark adapters |
| `baselines/adbench/` | Wraps ADBench models to Model interface |
| `baselines/gadbench/` | Wraps GADBench models to Model interface |
| `baselines/pygod/` | Wraps PyGOD graph detectors to Model interface |
| `configs/detectors/` | Hyperparameter configs per model per dataset |
| `data/` | Downloaded/preprocessed datasets organized by benchmark |
| `results/` | Output logs, metrics, trained model checkpoints |

## Contributing Models

To add a new anomaly detector:

1. Implement `Model` in appropriate `baselines/*/` folder
2. Override `train(epoch)`, `fit()`, `get_embeddings()`, `predict_scores()`
3. Register in `model_detector_dict`
4. Add hyperparameter config to `configs/detectors/{model_name}.yaml`
5. Test with `main.py` on small dataset

## References

- **ADBench**: [GitHub](https://github.com/Minqi824/ADBench) - Tabular, image, NLP anomalies
- **GADBench**: [GitHub](https://github.com/FuanKahunaGod/GADBench) - Graph anomalies
- **PyGOD**: [GitHub](https://github.com/pygod-team/pygod) - Graph neural network detectors
