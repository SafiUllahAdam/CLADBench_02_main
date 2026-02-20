# CoBench Summary

CoBench is a collaborative semi-supervised anomaly detection benchmark for tabular hospital data. It standardizes datasets, model wrappers, and co-training strategies to compare solo vs collaborative performance using AUC-based metrics.

## Architecture

```
Data (load → split → semi-supervised partition → resolve_labels)
  ↓
Model (fit / train / predict_scores / get_embeddings)
  ↓
CoLearning (warmup → [exchange → arbitrate → train] × chapters → evaluate)
  ↓
Strategy (convergence control: plateau, adaptive, recurrent-aware)
```

### Data (`base.py` → `Data`, `data_loader.py` → `ClassicalADBenchData`)
- Loads tabular datasets (ADBench `.npz` format), splits into train/val/test (stratified, seeded)
- Within train: partitions into labeled / unlabeled via `generate_semisupervised_split` (configurable ratio, stratification, anomaly caps, guardrails for extreme imbalance)
- **Unlabeled policy** — centralized via `resolve_labels(policy)`:
  - `unlabeled_as_normal` (default): `-1 → 0`
  - `unlabeled_as_is`: keeps `-1` for natively semi-supervised models
  - `unlabeled_as_removed`: supervised-only on labeled subset
- Preprocessing: scaler fit on full train (including unlabeled), transform on val/test — no leakage
- Validation checks in `utils.py`: split integrity, no overlap, stratification drift, leakage, policy compliance

### Model (`base.py` → `Model`)
- Unified API: `fit()`, `train(epochs)`, `predict_scores()`, `get_embeddings()`, optional `get_loss()`
- `y_train` property: cascades pseudo-labels → `resolve_labels()` → original, respecting `preserve_labeled`
- Wrappers: PReNet, DeepSAD, DevNet, XGBOD (in `ModelWrapperADBench.py`)
- Models never resolve unlabeled labels themselves — the Data layer handles it

### CoLearning (`base.py` → `CoLearning`, `colearner.py` → `SimpleCoLearner`, `CoLearnerVal`)
- Multi-model pseudo-label exchange on unlabeled samples
- Configurable per-pair thresholds, per-model defaults, global fallbacks
- **Pseudo-label arbitration**: when multiple senders propose conflicting labels for the same sample, a pluggable arbiter (`default_arbiter` = max-confidence) resolves the conflict. Metadata (`PseudoLabelProposal`: sender, label, confidence) is stored for traceability. Hook supports future policies (weighted voting, model-priority, etc.)
- `CoLearnerVal`: evaluates on validation AUC per chapter, tracks per-model val loss history

### RecurrentModel (`base.py` → `RecurrentModel`, `recurrentmodels.py` → `GRURecurrentModel`)
- GRU judge trained on stacked/concatenated detector embeddings
- Delayed start (configurable chapter), evaluated alongside detector ensemble
- `RecurrentCoLearner` / `DelayedRecurrentCoLearner` in `colearner.py`

### Strategy (`strategy.py`)
- `PlateauStrategy`: stop when metric plateaus for N chapters
- `AdaptivePlateauStrategy`: patience decays each time stopping triggers
- `RecurrentPlateauStrategy`: fallback to ensemble before GRU starts, then monitors GRU metric

## Benchmark Flow

| Eval | Description |
|------|-------------|
| EVAL 1 | Solo baselines per dataset → test AUC |
| EVAL 2 | Collaborative learners → compare against solo AUC, report deltas |
| EVAL 2b | Training dynamics: losses, exchange precision, per-chapter validation AUC |
| EVAL 3 | Collaborative + recurrent ensemble |

Notebooks: `alpha.ipynb` (dev/testing), `complete_analysis.ipynb` (full benchmark)

### Cross-Validation
Runs N trials (default 5, probably 10 for the real bench) with incremented seeds for statistical robustness. Each trial shares the same data split across solo / collab / GRU for fair comparison.

Per trial:
1. Solo baselines — `fit()` once, record AUC & AP
2. Collaborative (`CoLearnerVal` + `PlateauStrategy`) — cotrain, record per-model + ensemble AUC & AP
3. GRU recurrent judge (`DelayedRecurrentCoLearner`) — cotrain with delayed start, record GRU AUC & AP

Output: `cv_results` dict with per-trial AUC/AP arrays, loss histories, chapter histories, and seeds.

### Visualization (`alpha.ipynb` cells 12–13)

**Loss dynamics (cell 12)** — grid of (model × config) subplots:
- NaN-padded aggregation handles variable-length training histories
- Mean ± std ribbon (solid region when majority of trials active, dotted + faded when sparse)
- Individual trial traces as thin transparent lines
- Epoch distribution annotation box per subplot
- Panel labels (a, b, c, …) for publication referencing

**AUC & AP** — two figures:
- *Figure 1(a)*: collaborative validation AUC over chapters (per-model + ensemble + Judge) with solo baselines as dashed horizontal lines. Ribbon = IQR (Q25–Q75).
- *Figure 1(b)*: ΔAUC (collab − solo) per chapter per model; zero line = no gain.
- *Figure 2*: final test-set AUC & AP bar charts. Bars = mean, error bars = IQR, dots = individual trial values. Methods: solo, collab, ensemble, GRU judge.

### Results Export (`alpha.ipynb` cell 14)

**Filename convention**: `{id}.{n_models}.{recurrent}.{colearner}.{strategy}.{models}.csv`
- `id`: auto-incremented integer (zero-padded to 3 digits)
- `n_models`: number of detector models
- `recurrent`: 1 if GRU judge present, 0 otherwise
- `colearner` / `strategy`: class names (e.g. `CoLearnerVal.PlateauStrategy`)
- `models`: dash-joined model names (e.g. `prenet-deepsad-gru`)

Two files per run:
- **Stats CSV** (`{id}.….csv`): row 0 is `#META:{json}` with full config (seeds, splits, hyperparams, timestamp). Body has summary stats per method (mean, std, min, Q25, median, Q75, max, n_trials) for AUC, AP, and Δ(collab−solo).
- **Raw CSV** (`{id}.….raw.csv`): one row per trial × method with individual AUC & AP values. Same `#META` header.

Reload example:
```python
import json, csv, pandas as pd
with open("003.2.1.CoLearnerVal.PlateauStrategy.prenet-deepsad-gru.csv") as f:
    meta = json.loads(f.readline().removeprefix("#META:"))
    df = pd.read_csv(f)
```

## Metrics
- **Primary**: AUROC on test set (solo vs collaborative)
- **Secondary**: AUPRC (Average Precision, for rare anomalies), Precision@k
- **Monitoring**: validation AUC per chapter, exchange precision, per-model loss curves
- **Cross-validation**: mean ± std and IQR across N seeded trials

## Status
- ✅ Tabular AD (ADBench) with flexible data extraction
- ✅ Semi-supervised split generation (labeled/unlabeled + label preservation)
- ✅ Centralized unlabeled-label policy (`resolve_labels`) — no model-level hacks
- ✅ Pseudo-label arbitration with pluggable strategy and metadata traceability
- ✅ Multi-model pseudo-label exchange with per-pair routing
- ✅ Data validation checks (splits, leakage, policy compliance)
- ✅ GRU Recurrent Ensemble (delayed start, stacked embeddings)
- ⏳ Graph AD (GADBench/PyGOD)

## Submodules
- ADBench: 57 datasets, 30 algorithms
- GADBench: graph AD benchmark
- PyGOD: PyTorch Geometric detectors