# AGENTS

Hospital-grade collaborative anomaly detection benchmark. Compares solo vs collaborative semi-supervised AD using AUC and AP. Test notebook: `alpha.ipynb`. Production benchmark: `complete_analysis.ipynb` (CLI-driven).

Don't change API unless asked. Don't add functions unless necessary. Short clear functions, 1-line comments max. Be rigorous.

## Quality guidelines :
- The main benchmark should preferably only call functions defined in base.py
- If a researcher wants to work on my benchmark and add custom code (model, recurrent model, data, colearning strategy), they only need to write a new class with the abstract classes written in base.py, always ask yourself if the code you're building may impact the first two mentioned points
- A-rank open source research paper quality code
- ideal scenario : main.py only calls functions defined in abstract classes in base.py

## Core classes
- `Data` (base.py): train/val/test splits, labeled/unlabeled indices, semi-supervised labels, pseudo-label store.
- `Model` (ModelWrapperADBench.py): `fit()`, `train(epochs)`, `predict_scores()`, optional `get_embeddings()`, `get_loss()`. Scores in $[0,1]$. Exposes `_train_loss_history`, `_val_loss_history`.
- `GRURecurrentModel` (recurrentmodels.py): judge model on stacked detector embeddings. API: `train(embeddings, labels, epochs)`, `predict_scores(embeddings)`, `get_loss(embeddings, labels)`, loss histories. Val loss computed by colearner via `_compute_recurrent_val_loss()`.
- `CoLearnerVal` (colearner.py): chapters × epochs loop; exchanges high-confidence pseudo-labels on unlabeled split at each chapter start.
- `DelayedRecurrentCoLearner` (colearner.py): extends `CoLearnerVal`; activates GRU after `recurrent_start_chapter`. Uses `embeddings_aggregate="stack"`. Exposes `_collect_embeddings(indexes)`.
- `Strategy` (strategy.py): `PlateauStrategy`, `AdaptivePlateauStrategy`, `RecurrentPlateauStrategy` (monitors `fallback_key` then switches to `recurrent_key` once GRU starts).
- `benchmark_config.py`: all path/config constants (`PROJECT_ROOT`, `DATASET_CONFIG`, etc.).
- `utils.py`: `create_models`, `validate_data`, `extract_embeddings_auto`.

## alpha.ipynb flow
1. Setup: seeds, paths/config imports, benchmark classes, and split/semi-supervised knobs.
2. Data init + checks: build `ClassicalADBenchData`, run `validate_data`, print labeled/unlabeled stats.
3. Model selection: define detector list (`prenet`, `deepsad`) and recurrent model (`gru`).
4. TEST 1 (solo sanity): train one detector for a few epochs, report test ROC-AUC (+ loss when available).
5. TEST 2 (collab sanity): solo baselines vs `CoLearnerVal` + `PlateauStrategy`; run co-training and print per-model/ensemble AUC deltas.
6. TEST 3 (pseudo-label audit): print pseudo-label counts and proportions per detector.
7. Recurrent judge test: run `DelayedRecurrentCoLearner` + `GRURecurrentModel` + `RecurrentPlateauStrategy`, then evaluate GRU test AUC.
8. Quick visuals: chapter AUC curves, solo-vs-collab bars, and loss mosaic across experiments.
9. Cross-validation benchmark: run `N_TRIALS` seeds; store solo/collab/GRU AUC, AP, losses, and per-chapter histories in `cv_results`.
10. CV publication plots: (a) loss ribbons, (b) chapter AUC + ΔAUC dynamics, (c) final test AUC/AP bars with IQR + trial scatter.
11. Export artifacts: save summary CSV + raw per-trial CSV in `src/results/` with encoded filename metadata.
12. Save figures: write dynamics/bar/loss PNGs using the same results stem.

## Metrics
- Test: ROC-AUC and AP, solo vs collaborative vs collaborative + Recurrent judge.
- Validation AUC for per-chapter monitoring; exchange precision for pseudo-label quality.
- CV: mean ± std over N_TRIALS, collab − solo deltas per model.

## Agent notes
- GRU embeddings: pad to same dim, stack axis=1 → shape `(n_samples, n_detectors, dim)`.
- `RecurrentPlateauStrategy.recurrent_start_chapter` must match `DelayedRecurrentCoLearner.recurrent_start_chapter`.
