# CoBench -- Internal Onboarding Guide

*For new contributors. Based on actual codebase inspection (March 2026).*

---

## 1. High-Level Purpose

CoBench is a **collaborative semi-supervised anomaly detection benchmark** designed for hospital-grade research. It unifies detectors from ADBench (tabular/CV/NLP), GADBench (graph), and PyGOD (graph GNNs) under a single API so they can be evaluated both **solo** and **collaboratively**.

### Design philosophy

The benchmark does not use the original `fit()` API of integrated methods. Instead, it enforces:

- **`train(epochs)`** -- incremental training, one or more epochs at a time. This is the backbone of the co-learning loop, which alternates between pseudo-label exchange and training. A traditional `fit()` trains the whole model at once and cannot be interrupted mid-training to receive new labels.
- **`get_loss(use_val=False)`** -- optional but recommended. Returns the latest train or validation loss. Used by early-stopping strategies and for monitoring dynamics.
- **`predict_scores()`** -- returns anomaly scores in `[0, 1]`. These scores drive pseudo-label generation during the exchange phase.

Why this matters: the co-learning loop calls `train(1)` repeatedly, injecting pseudo-labels between epochs. If a model only supports `fit()`, it cannot participate in collaborative learning. Every wrapper must decompose training into epoch-level increments.

### What the benchmark produces

For each dataset x trial, it compares:

| Evaluation    | Description                                              |
|---------------|----------------------------------------------------------|
| Solo baseline | Each detector trained alone via `fit()`                  |
| Collaborative | Models exchange pseudo-labels via `CoLearnerVal.cotrain()` |
| Collaborative + GRU | Same as above, with a recurrent judge on stacked embeddings |

Primary metrics: **ROC-AUC** and **Average Precision** on the held-out test set. Deltas (collab minus solo) quantify the collaboration gain.

---

## 2. Global Architecture

### Folder structure (what actually exists)

```
CoBench/
  src/
    base.py                            # Abstract classes: Model, Data, Strategy, CoLearning, RecurrentModel
    colearner.py                       # Concrete co-learning loops: SimpleCoLearner, CoLearnerVal,
                                       #   RecurrentCoLearner, DelayedRecurrentCoLearner, SingleModel
    strategy.py                        # Convergence strategies: PlateauStrategy, AdaptivePlateauStrategy,
                                       #   RecurrentPlateauStrategy
    recurrentmodels.py                 # GRURecurrentModel (PyTorch GRU judge)
    benchmark_config.py                # All paths, dataset registry, hyperparams, CV defaults
    main.py                            # CLI entrypoint for running the benchmark
    runner.py                          # Cross-validation loop (run_cv), summary printing, plot dispatch
    results_io.py                      # CSV export of CV results (stats + raw per-trial)
    utils.py                           # create_models, set_seed, validate_data, extract_embeddings_auto
    ADWrapperUtils.py                  # Legacy model registry (generate_AD_dictionary); not used in main flow
    baselines/
      adbench/
        ModelWrapperADBench.py         # PReNetWrapper, DeepSADWrapper, DevNetWrapper, XGBODWrapper
                                       #   + get_model_detector_dict() registry
        data_loader.py                 # ClassicalADBenchData (Data subclass for .npz datasets)
      gadbench/                        # Placeholder (empty files); graph support not yet implemented
    results/                           # Auto-numbered CSV output files (stats + raw)
    texgad/                            # Placeholder (empty files); future GADBench CLI
  SubModules/                          # Unmodified clones of ADBench, GADBench, PyGOD
  docs/                                # README_Project.md, README_API.md, NEXT_STEPS.md
  AGENTS.md                            # LLM agent instructions and class reference
  SUMMARY.md                           # Architecture summary
  CLAUDE.md                            # Coding rules for AI assistants
```

### Module responsibilities

| Module | Role | Layer |
|--------|------|-------|
| `base.py` | All abstract interfaces (Model, Data, Strategy, CoLearning, RecurrentModel) + PseudoLabelProposal + default_arbiter | **Core API** |
| `colearner.py` | Concrete co-learning loops; pseudo-label exchange orchestration; evaluation | **Experiment logic** |
| `strategy.py` | Convergence/early-stopping policies | **Experiment logic** |
| `recurrentmodels.py` | GRU judge model (a RecurrentModel implementation) | **Model** |
| `ModelWrapperADBench.py` | Wrappers adapting ADBench detectors to the Model API | **Wrapper** |
| `data_loader.py` | ClassicalADBenchData: .npz loading, splitting, normalization | **Data handling** |
| `benchmark_config.py` | Paths, dataset list, model hyperparams, CV defaults | **Config** |
| `main.py` | CLI argument parsing, validation, dispatch to `run_cv` | **Script** |
| `runner.py` | Cross-validation loop, solo+collab+GRU per trial, summary | **Script** |
| `results_io.py` | CSV export (stats + raw) with encoded filenames + metadata | **Output** |
| `utils.py` | Seed management, model factory, data validation, embedding extraction | **Utility** |
| `ADWrapperUtils.py` | Legacy ADBench model dictionary (not called by main flow) | **Legacy** |

---

## 3. Critical Files -- Do Not Modify Unless Strictly Necessary

### `src/base.py`

**Why it is critical:** Every class in the system inherits from or depends on the abstract interfaces defined here. The `Model` base class defines the `y_train` property (the pseudo-label resolution cascade), the `set_pseudo_labels` / `clear_pseudo_labels` methods, and the abstract method signatures that all wrappers must implement.

**What breaks if you change it carelessly:**
- Changing the `y_train` property logic (lines 78-104) will silently corrupt label resolution for every model. Pseudo-labels may overwrite ground-truth, or unlabeled samples may be treated incorrectly.
- Modifying `_extract_data()` (lines 70-76) changes what data every model receives at construction.
- Altering the `CoLearning` base class changes the pseudo-label exchange machinery (`_propose_labels`, `_finalize_pseudo_labels`, `send_anomalies`, `send_normals`) that all co-learner variants rely on.
- Changing `Data._init_semisupervised()` affects the labeled/unlabeled partition for every experiment.

### `src/colearner.py`

**Why it is critical:** Contains the core training loops. `CoLearnerVal.cotrain()` is the primary experimental loop used in production. `SimpleCoLearner.exchange()` implements the pseudo-label exchange protocol.

**What breaks if you change it carelessly:**
- Altering the exchange logic changes which pseudo-labels models receive -- any bug here silently corrupts all collaborative results.
- Changing evaluation order or metric computation invalidates benchmark comparisons.
- The `SingleModel` class is the solo baseline reference; any change to it makes solo-vs-collab deltas meaningless.

### `src/benchmark_config.py`

**Why it is critical:** Single source of truth for all paths (PROJECT_ROOT, ADBENCH_ROOT, dataset paths), hyperparameters (MODEL_CONFIGS), and CV defaults. Every module imports from here.

**What breaks if you change it carelessly:**
- Changing `EVAL_DATASETS` paths breaks data loading.
- Changing `MODEL_CONFIGS` alters default hyperparameters across all experiments, making results non-reproducible relative to prior runs.
- Changing `CV_DEFAULTS` changes the default benchmark configuration.

### `src/utils.py` -- validation functions

**Why it is critical:** `validate_splits()`, `check_no_test_leakage()`, and `check_unlabeled_hidden()` are the safety nets that catch data integrity bugs before training starts.

**What breaks if you change it carelessly:**
- Relaxing or removing assertions can allow leaky or malformed data to pass silently, producing invalid benchmark results.

---

## 4. Safe Modification Zones

### Adding a new model wrapper

**Where:** `src/baselines/adbench/ModelWrapperADBench.py`

**How:**
1. Create a new class inheriting from `Model` (imported from `base`).
2. Implement: `train(epochs)`, `fit()`, `predict_scores(indexes, use_train, use_val)`, `get_embeddings(indexes, use_train)`.
3. Optionally implement `get_loss(use_val)` for early-stopping support.
4. Register the class in `get_model_detector_dict()` at the top of the file.
5. Add default hyperparameters in `benchmark_config.py` under `MODEL_CONFIGS`.

**Template (follow existing wrappers):**
```python
class MyModelWrapper(Model):
    def __init__(self, train_config: dict, model_config: dict, data: dict):
        defaults = {"seed": 42, "total_epochs": 50, "batch_size": 128}
        config = {**defaults, **(train_config or {})}
        super().__init__(train_config=config, model_config=model_config, data=data)
        # Initialize your detector here
        self._train_loss_history = []
        self._val_loss_history = []

    def fit(self) -> None:
        remaining = self.train_config["total_epochs"] - self._current_epoch
        if remaining > 0:
            self.train(remaining)
        self._fitted = True

    def train(self, epochs: int = 1) -> None:
        y_train = self.y_train  # MUST use this property, not y_train_original
        for _ in range(epochs):
            # ... train one epoch ...
            self._current_epoch += 1
            self._fitted = True

    def predict_scores(self, indexes=None, use_train=False, use_val=False) -> np.ndarray:
        # Select data source
        if use_train: X = self.X_train
        elif use_val: X = self.X_val
        else: X = self.X_test
        X_batch = X if indexes is None else X[indexes]
        # ... compute scores ...
        # Normalize to [0, 1]
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-8:
            scores = (scores - s_min) / (s_max - s_min)
        else:
            scores = np.full_like(scores, 0.5)
        return scores

    def get_embeddings(self, indexes=None, use_train=True) -> np.ndarray:
        # Return internal representations
        ...
```

### Adding a new convergence strategy

**Where:** `src/strategy.py`

Inherit from `Strategy` (from `base.py`). Implement `should_continue(model_metrics, chapter)` and `reset()`. Register in `runner.py`'s `STRATEGY_REGISTRY` dict.

### Adding a new dataset

**Where:** `src/benchmark_config.py` (add to `EVAL_DATASETS` dict)

The dataset must be a `.npz` file with either `{X, y}` or `{X_train, y_train, X_test, y_test}` keys. `ClassicalADBenchData` handles both formats.

### Adding a new recurrent model

**Where:** Create a new file or add to `src/recurrentmodels.py`

Inherit from `RecurrentModel` (from `base.py`). Implement `train(aggregated_embeddings, labels, epochs)`, `predict_scores(aggregated_embeddings)`, and optionally `get_loss(aggregated_embeddings, labels)`.

### Adding a new co-learner variant

**Where:** `src/colearner.py`

Inherit from `SimpleCoLearner` or `CoLearning`. Override `cotrain()` and/or `exchange()`. Keep the same metric dict structure so strategies and results export work.

---

## 5. Mandatory Coding and Usage Rules

### Interface constraints

1. **`train(epochs)` must use `self.y_train`**, never `self.y_train_original`. The `y_train` property resolves pseudo-labels on top of semi-supervised labels. Bypassing it breaks collaborative learning.

2. **`predict_scores()` must return values in `[0, 1]`**. All downstream logic (threshold comparisons, AUC computation, ensemble averaging) depends on this normalization. Use min-max normalization as a final step.

3. **`predict_scores()` must support `use_train=True` and `use_val=True`**. The exchange phase calls `predict_scores(use_train=True)` to score training data for pseudo-label generation. `CoLearnerVal` calls `predict_scores(use_val=True)` for validation AUC monitoring.

4. **`get_loss(use_val=True)` should return a float or None**. Strategies use this for early stopping. If your model does not track loss, return `None`.

5. **`fit()` must call `train(remaining_epochs)`** to fill the total_epochs budget. This ensures the solo baseline uses the same underlying training code.

6. **Scores must be finite.** `np.nan` or `np.inf` in scores will crash AUC computation (`_compute_auc_or_raise` in `colearner.py`).

### Naming conventions

- Model wrapper classes: `{ModelName}Wrapper` (e.g., `PReNetWrapper`)
- Registry keys: lowercase model name (e.g., `"prenet"`, `"deepsad"`)
- Loss history attributes: `_train_loss_history` (list of floats), `_val_loss_history` (list of floats)
- Last loss attributes: `_last_train_loss`, `_last_val_loss`

### Experiment reproducibility

- Always use `set_seed(seed)` from `utils.py` before creating data or models. It sets Python, NumPy, and PyTorch seeds.
- CV trials use `seed + trial_idx` for per-trial determinism.
- Do not use `torch.cuda.is_available()` to change model architecture; only use it for device placement.

### Results and output

- Results files follow the naming convention `{id}.{n_models}.{recurrent}.{colearner}.{strategy}.{models}.csv`
- Never manually name result files; use `results_io.export_cv_results()` which auto-increments the ID.
- CSV files have a `#META:` JSON header line containing full experiment configuration.

### Coding style (from CLAUDE.md)

- As few functions as possible, each clear and effective.
- Maximum 1-line comments per function.
- No speculative abstractions; no unused code.

---

## 6. Hidden Assumptions and Invariants

### Data format invariants

- **X arrays are `float32`, y arrays are `int32`** with values in `{0, 1}` for labels, `-1` for unlabeled.
- **`n_train`, `n_test`, `n_val`** must match the `.shape[0]` of the corresponding X arrays. `validate_splits()` asserts this.
- **`labeled_indexes` and `unlabeled_indexes` must be disjoint and their union must equal `range(n_train)`.**
- **`semisupervised_labels[labeled_indexes]` must equal `y_train_original[labeled_indexes]`.**
- **`semisupervised_labels[unlabeled_indexes]` must be `-1`.**

### Pseudo-label lifecycle

1. At the start of each chapter, `exchange()` calls `clear_pseudo_labels()` on all models, wiping previous pseudo-labels.
2. Each sender model scores training data via `predict_scores(use_train=True)`.
3. High-confidence anomalies (score > `confidence_threshold_high`) and normals (score < `confidence_threshold_low`) are proposed to receiver models via `_propose_labels()`.
4. `_finalize_pseudo_labels()` calls the `pseudo_label_arbiter` to resolve conflicts when multiple senders propose labels for the same sample. Default: pick the proposal with highest confidence.
5. Resolved labels are stored in `model._pseudo_labels` (an `np.ndarray` of shape `(n_train,)` with `-1` for no proposal).
6. When a model's `y_train` property is accessed, pseudo-labels overlay on top of `resolve_labels()` output. If `preserve_labeled=True`, ground-truth labeled samples are protected from overwrite.
7. The `_y_train_cache` is invalidated whenever `set_pseudo_labels()` or `clear_pseudo_labels()` is called.

### Threshold resolution priority

For pseudo-label transfer thresholds, the priority order is:
1. Per-pair override: `transfer_thresholds[(sender_idx, receiver_idx)][kind]`
2. Per-sender default: `sender.default_confidence_high` / `sender.default_confidence_low`
3. Global CoLearner default: `self.confidence_threshold_high` / `self.confidence_threshold_low`

### Score normalization

All `predict_scores()` implementations must return values in `[0, 1]`. The standard normalization pattern used across all wrappers:
```python
s_min, s_max = scores.min(), scores.max()
if s_max - s_min > 1e-8:
    scores = (scores - s_min) / (s_max - s_min)
else:
    scores = np.full_like(scores, 0.5)
```
This is a min-max rescaling within the batch. Consequently, scores are **relative within a batch**, not absolute. A score of 0.9 on test data does not mean the same as 0.9 on train data. The `anomaly_threshold` (default `0.5`) is applied to training-set scores only.

### Embedding aggregation for the GRU

- `"concat"`: concatenate embeddings from all models along axis=1 -> shape `(n_samples, sum_dims)`. The GRU treats this as a single-step sequence.
- `"stack"`: pad embeddings to equal dimension, stack along axis=1 -> shape `(n_samples, n_detectors, dim)`. The GRU processes this as a multi-step sequence (one step per detector).
- `"mean"`: element-wise mean -> shape `(n_samples, max_dim)`.
- The `DelayedRecurrentCoLearner` defaults to `"stack"` and the `n_detectors` parameter of `GRURecurrentModel` must match `len(models)`.

### Scaler fitting

`MinMaxScaler` is fit on **training data only** (including unlabeled). Test and validation data are `transform()`-ed. `check_no_test_leakage()` verifies `scaler.n_samples_seen_ == n_train`.

### Strategy `metric_key`

Strategy classes look up a key in the `metrics` dict returned by the co-learning evaluation step. The dict always contains `"model_0"`, `"model_1"`, ..., `"ensemble"`, and (after GRU starts) `"gru"`. If your strategy monitors a key that doesn't exist in the metrics dict, `PlateauStrategy.should_continue()` will raise a `ValueError`.

---

## 7. Common Failure Modes

### 1. Using `y_train_original` instead of `y_train` in `train()`

**Symptom:** Model trains on ground-truth labels; collaborative learning has no effect; solo and collab AUC are identical.
**Root cause:** The model bypasses the pseudo-label resolution cascade.
**Fix:** Always use `self.y_train` in `train()`. This property resolves `pseudo_labels -> semisupervised_labels -> y_train_original` with proper caching.

### 2. Scores not normalized to [0, 1]

**Symptom:** `_compute_auc_or_raise` may succeed (AUC is rank-based), but pseudo-label exchange breaks. Thresholds like `confidence_threshold_high=0.98` assume scores are in [0,1]. Raw scores in [-100, 100] would cause all samples to be labeled as anomalies.
**Root cause:** Missing min-max normalization in `predict_scores()`.
**Fix:** Apply the standard normalization block (see section 6).

### 3. Not implementing `predict_scores(use_val=True)`

**Symptom:** `CoLearnerVal.cotrain()` crashes or returns wrong validation AUC. May silently score test data instead.
**Root cause:** The wrapper ignores the `use_val` parameter.
**Fix:** Implement the three-way dispatch: `if use_train: X = self.X_train; elif use_val: X = self.X_val; else: X = self.X_test`.

### 4. Forgetting to increment `_current_epoch`

**Symptom:** `fit()` trains indefinitely because `remaining = total_epochs - _current_epoch` never decreases.
**Root cause:** `train()` does not increment `self._current_epoch` after each epoch.
**Fix:** Add `self._current_epoch += 1` in the epoch loop.

### 5. Forgetting to set `_fitted = True`

**Symptom:** `predict_scores()` calls `fit()` internally (the guard `if not self._fitted: self.fit()` triggers), causing double training.
**Root cause:** `train()` does not set `self._fitted = True`.
**Fix:** Set `self._fitted = True` at the end of each epoch in `train()`.

### 6. `RecurrentPlateauStrategy.recurrent_start_chapter` mismatched with `DelayedRecurrentCoLearner.recurrent_start_chapter`

**Symptom:** Strategy expects GRU metrics at chapter N but they don't appear until chapter M, causing the strategy to use the fallback metric longer (or shorter) than intended.
**Root cause:** The two values must be set to the same integer.
**Fix:** Pass the same `gru_start_chapter` value to both.

### 7. Modifying Data objects after model construction

**Symptom:** Models hold stale references to `X_train`, `y_train_original`, etc.
**Root cause:** `Model.__init__` calls `_extract_data(data)` which copies attribute references at construction time. If the Data object's arrays are later replaced (not mutated in-place), models won't see the change.
**Fix:** Don't replace Data arrays after constructing models. If you must, reconstruct the models.

### 8. Duplicate line in `colearner.py`

**Note:** Line 200 in `colearner.py` has a duplicated method signature (`def _collect_embeddings` appears twice). This is a known copy-paste artifact. It currently works because Python uses the second definition, but be aware if editing that area.

---

## 8. Practical Workflow

### Before modifying anything

1. **Read these files first, in this order:**
   - `base.py` -- understand the abstract API (Model, Data, Strategy, CoLearning)
   - `ModelWrapperADBench.py` -- study `PReNetWrapper` as the reference implementation
   - `colearner.py` -- understand `SimpleCoLearner.exchange()` and `CoLearnerVal.cotrain()`
   - `benchmark_config.py` -- know the defaults and paths
   - `AGENTS.md` -- coding rules specific to this project

2. **Run the benchmark once to see it work:**
   ```bash
   cd src/
   python main.py --models prenet deepsad --data_to_test annthyroid --n_trials 1
   ```
   This runs 1 trial with PReNet+DeepSAD on annthyroid. Check the terminal output for solo/collab AUC and deltas.

3. **Inspect a result file** to understand the output format:
   ```bash
   head -5 src/results/001.*.csv
   ```

### When adding a new model

1. Write the wrapper class in `ModelWrapperADBench.py`, following `PReNetWrapper` as template.
2. Register it in `get_model_detector_dict()`.
3. Add default hyperparameters in `benchmark_config.py` under `MODEL_CONFIGS`.
4. Test solo first:
   ```bash
   python main.py --models your_model --data_to_test annthyroid --n_trials 1
   ```
5. Test collaborative:
   ```bash
   python main.py --models your_model deepsad --data_to_test annthyroid --n_trials 1
   ```
6. Verify that collab AUC is different from solo AUC (if not, pseudo-labels are probably not reaching your model -- check that `train()` uses `self.y_train`).

### When modifying existing logic

1. Run the benchmark before your change and save the output.
2. Make your change.
3. Run the benchmark again with the same seed and parameters.
4. Compare AUC/AP values. If they changed, understand why.

---

## 9. Checklist Before Merging

- [ ] All abstract methods from `Model` are implemented: `train()`, `fit()`, `predict_scores()`, `get_embeddings()`
- [ ] `train()` uses `self.y_train` (not `y_train_original`)
- [ ] `predict_scores()` returns values in `[0, 1]` for all inputs (train, val, test)
- [ ] `predict_scores()` handles `use_train=True` and `use_val=True`
- [ ] `_current_epoch` is incremented in each training epoch
- [ ] `_fitted` is set to `True` after training
- [ ] Solo baseline runs without error: `python main.py --models your_model --n_trials 1`
- [ ] Collaborative run produces different AUC than solo (if not, explain why)
- [ ] No `np.nan` or `np.inf` in output scores
- [ ] Loss tracking works (if implemented): `get_loss()` and `get_loss(use_val=True)` return valid floats
- [ ] Results CSV is correctly generated in `src/results/`
- [ ] No hardcoded paths; all paths use `benchmark_config.py`
- [ ] Seeds are respected (running twice with the same seed gives the same result)
- [ ] No unused imports or dead code introduced
- [ ] Code follows the project style: minimal functions, max 1-line comments

---

## 10. Open Questions and Ambiguous Parts

### Confirmed facts

- The benchmark currently only supports **tabular** data via `ClassicalADBenchData`. Graph support (GADBench/PyGOD) is not implemented; the `baselines/gadbench/` files are empty placeholders.
- The CLI entry point is `main.py` which calls `runner.run_cv()`. The notebook `alpha.ipynb` is a parallel development/testing environment.
- `ADWrapperUtils.py` (`generate_AD_dictionary`) is a legacy file. The actual model registry is `get_model_detector_dict()` in `ModelWrapperADBench.py`. The legacy file is not called by any current code path.
- DevNet requires TensorFlow; its import is guarded. If TF is not installed, `devnet` is simply absent from the registry.
- XGBOD requires PyOD; its import is also guarded.

### Inferred assumptions (not explicitly documented)

- **Exchange happens only on `unlabeled_indexes`** (when `keep_truth=True`, which is the default). This means models only exchange pseudo-labels for samples where the ground truth is hidden.
- **`anomaly_threshold` (default 0.5) is applied to min-max normalized training scores.** Because normalization is per-batch, this threshold is effectively "above the median score" in many cases, not a calibrated probability.
- **The GRU judge is evaluated on test embeddings at the end of each trial** (in `runner._run_gru_trial`), but the collection uses `embeddings_use_train=False` by monkey-patching the attribute after training. This is fragile.
- **`_y_train_cache` is invalidated by `set_pseudo_labels` and `clear_pseudo_labels`, but not by changes to `data.semisupervised_labels` or `data.resolve_labels`**. If external code modifies semi-supervised labels after model construction, the cache may be stale.

### Known fragilities

- **Duplicate method signature** on line 200 of `colearner.py`: `_collect_embeddings` is defined twice with the same signature. Python uses the second definition silently. This should be cleaned up.
- **`_test_ds_cache_id` in `DeepSADWrapper`** caches by `id(X_batch)`. Since `id()` is the memory address, the cache breaks correctly on new arrays but might give false cache hits if a deleted array's memory is reused. In practice this is unlikely but worth noting.
- **`texgad/` directory** contains only `todo` placeholder files. No graph CLI exists yet.
- **`configs/detectors/` and `data/` directories** referenced in `README.md` do not exist. All hyperparameters are in `benchmark_config.py` and all datasets are loaded from `SubModules/ADBench/`.
- **No automated tests exist.** Validation is manual via CLI runs and notebook execution.

### Open design questions

- How will graph data (node features + adjacency matrices) be integrated into the `Data` abstract class, which currently assumes flat `X_train`/`X_test` arrays?
- Should `predict_scores()` use batch-level or dataset-level min-max normalization? The current approach (per-call normalization) means scores are not comparable across different calls.
- Should the `Data` class own the pseudo-labels (as its `pseudo_labels_by_model` dict suggests in the base class) or should models own them (as the current implementation in `Model._pseudo_labels` does)? Currently the `Data` pseudo-label dict exists but is not used by the main flow.

---

## Appendix: Minimum Reading List

| Priority | File | What to understand |
|----------|------|--------------------|
| 1 | `base.py` | Abstract API, `y_train` resolution, pseudo-label flow |
| 2 | `ModelWrapperADBench.py` | How a wrapper implements the API (use `PReNetWrapper` as reference) |
| 3 | `colearner.py` | `exchange()`, `cotrain()`, the chapter loop |
| 4 | `benchmark_config.py` | All paths, defaults, and hyperparameters |
| 5 | `runner.py` | CV loop, solo/collab/GRU trial structure |
| 6 | `strategy.py` | Early stopping logic |
| 7 | `data_loader.py` | How datasets are loaded and split |
| 8 | `utils.py` | Data validation, embedding extraction |
| 9 | `AGENTS.md` | Project coding rules and class reference |
