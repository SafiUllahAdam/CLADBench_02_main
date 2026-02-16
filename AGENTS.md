# AGENTS

## Project summary
CoBench is a collaborative anomaly detection benchmark. It is built for a hospital and uses standardizes datasets, model wrappers, and co-training strategies to compare solo vs collaborative performance using AUC-based metrics.

Preliminary test notebook (check if compile, if the code runs): alpha.ipynb
Real notebook (do we achieve state of the art): complete_analysis.ipynb

## Core concepts (from base.py)
- `Data`: unified dataset interface (train/test/val splits, labeled/unlabeled indices, semi-supervised labels, pseudo-label stores).
- `Model`(ModelWrapperADBench.py): unified model API (`fit`, `train (epochs)`, `predict_scores`, optionnal`get_embeddings`, optional `get_loss`). Models consume `Data` and expose anomaly scores in $[0,1]$. The get_embeddings function is yet to be normalized. 
- RecurrentModel (recurrentmodels.py): yet to be built
- `CoLearning `(colearner.py): coordinates multi-model training with pseudo-label exchange and strategy-driven stopping (see `SimpleCoLearner`, `CoLearnerVal`). A training session has several chapters, a chapter has several epochs. At the beginning of the chapter high-confidence pseudo-labels are exchanged on the unlabled part of the training set, then each model trains a few epochs.
- `Strategy`(strategy.py): controls convergence based on metrics (e.g., plateau strategies).
Don't change API unless asked
Don't add functions unless necessary
Write short and clear functions, comments are 1 line max

The complete_analysis notebook is a base for the benchmark that shall be used through CLI 

Be rigorous and double check the logic of the code

## Benchmark flow (from complete_analysis.ipynb)
1) **EVAL 1**: Train solo baselines per dataset; report test AUC.
2) **EVAL 2**: Train collaborative learners; compare against solo AUC and report deltas.
3) **EVAL 2b**: Inspect training dynamics (losses, exchange precision, per-chapter validation AUC).
4) **EVAL 3**: Train collaborative learners; add the collaborative ensemble learning. 

## Metrics
- Primary: ROC-AUC on test set (solo vs collaborative).
- Validation AUC is used for per-chapter monitoring in `CoLearnerVal`.
- Exchange precision tracks pseudo-label quality on unlabeled samples.

## Notes for agents
- Use the `Model` API to keep wrappers consistent.
- Prefer validation splits for early stopping/strategy decisions.
