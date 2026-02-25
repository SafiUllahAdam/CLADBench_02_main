import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from base import CoLearning, Data, Model, Strategy, RecurrentModel
from utils import extract_embeddings_auto

logger = logging.getLogger(__name__)

class SimpleCoLearner(CoLearning):
    """Multi-model collaborative learning with bidirectional pseudo-label exchange."""
    
    def __init__(self, models: List[Model], data: Data, strategy: Strategy,
                 warmup_epochs: int = 10, max_chapters: int = 10,
                 anomaly_threshold: float = 0.5, confidence_threshold_low: float = 0.02,
                 confidence_threshold_high: float = 0.98, keep_truth: bool = True,
                 transfer_thresholds: Optional[Dict] = None,
                 pseudo_label_arbiter=None):
        super().__init__(models, data, strategy, warmup_epochs, max_chapters,
                         anomaly_threshold, confidence_threshold_low, confidence_threshold_high,
                         transfer_thresholds=transfer_thresholds,
                         pseudo_label_arbiter=pseudo_label_arbiter)
        self.keep_truth = keep_truth  # If True, only exchange on unlabeled; if False, exchange on all
        for model in models:
            model.clear_pseudo_labels()
        self.model_scores: Dict[str, np.ndarray] = {f"model_{i}": None for i in range(len(models))}
        self.ensemble_scores: Optional[np.ndarray] = None
        self.ensemble_disagreement: Optional[np.ndarray] = None
        self.exchange_history: List[Dict] = []

    def exchange(self) -> None:
        """Exchange pseudo-labels across all (sender, receiver) pairs."""
        for model in self.models:
            model.clear_pseudo_labels()

        exchange_indexes = self.data.unlabeled_indexes if self.keep_truth else np.arange(self.data.n_train)
        if self.keep_truth and (exchange_indexes is None or len(exchange_indexes) == 0):
            logger.warning("[Exchange] No unlabeled_indexes set - check Data labeled_ratio")
            return

        n_models = len(self.models)
        y_true = self.data.y_train_original
        exchange_stats = {
            "keep_truth": self.keep_truth,
            "n_exchange_candidates": len(exchange_indexes),
            "n_anomalies": 0, "n_normals": 0,
            "correct_anomalies": 0, "correct_normals": 0,
            "per_pair": {},
        }

        # Score once per sender, then dispatch to each receiver
        for s in range(n_models):
            scores = self.models[s].predict_scores(use_train=True)
            self.model_scores[f"model_{s}"] = scores

            candidate_scores = scores[exchange_indexes]
            anomaly_mask = candidate_scores > self.anomaly_threshold
            anomaly_idxs = exchange_indexes[anomaly_mask]
            normal_idxs = exchange_indexes[~anomaly_mask]

            for r in range(n_models):
                if r == s:
                    continue
                n_sent_anom = self.send_anomalies(s, r, anomaly_idxs, scores)
                n_sent_norm = self.send_normals(s, r, normal_idxs, scores)
                pair_stats = {"n_anomalies": n_sent_anom, "n_normals": n_sent_norm}
                exchange_stats["per_pair"][(s, r)] = pair_stats
                exchange_stats["n_anomalies"] += n_sent_anom
                exchange_stats["n_normals"] += n_sent_norm

            if y_true is not None:
                exchange_stats["correct_anomalies"] += int(np.sum(y_true[anomaly_idxs] == 1))
                exchange_stats["correct_normals"] += int(np.sum(y_true[normal_idxs] == 0))

        na, nn = exchange_stats["n_anomalies"], exchange_stats["n_normals"]
        ca, cn = exchange_stats["correct_anomalies"], exchange_stats["correct_normals"]
        exchange_stats["anomaly_precision"] = ca / na if na else 0.0
        exchange_stats["normal_precision"] = cn / nn if nn else 0.0
        exchange_stats["overall_precision"] = (ca + cn) / max(1, na + nn)
        self.exchange_history.append(exchange_stats)

        # Resolve conflicts via arbiter and write final pseudo-labels
        self._finalize_pseudo_labels()

        all_scores = np.array(list(self.model_scores.values()))
        self.ensemble_scores = np.mean(all_scores, axis=0)
        self.ensemble_disagreement = np.std(all_scores, axis=0)

    def _warmup(self) -> None:
        logger.info(f"[Warmup] Training {len(self.models)} models for {self.warmup_epochs} epochs...")
        for epoch in range(self.warmup_epochs):
            for model in self.models:
                model.train(1)
            if (epoch + 1) % max(1, self.warmup_epochs // 3) == 0:
                logger.info(f"  Epoch {epoch + 1}/{self.warmup_epochs}")

    def _evaluate(self, y_true: np.ndarray) -> Dict[str, float]:
        scores_list: List[np.ndarray] = []
        metrics: Dict[str, float] = {}
        for i, model in enumerate(self.models):
            sc = model.predict_scores()
            scores_list.append(sc)
            try:
                metrics[f"model_{i}"] = roc_auc_score(y_true, sc)
            except Exception:
                metrics[f"model_{i}"] = 0.5
        try:
            metrics["ensemble"] = roc_auc_score(y_true, np.mean(scores_list, axis=0))
        except Exception:
            metrics["ensemble"] = 0.5
        return metrics

    def _log(self, chapter: int, metrics: Dict[str, float]) -> None:
        logger.info(f"  Chapter {chapter + 1}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    def cotrain(self, eval_interval: int = 1) -> Dict[str, List[float]]:
        history = {"warmup": [], "chapters": []}
        self._warmup()
        logger.info(f"[Collaborative] Running up to {self.max_chapters} chapters...")
        for chapter in range(self.max_chapters):
            self.exchange()
            for model in self.models:
                model.train(1)
            metrics = self._evaluate(self.data.y_test)
            if (chapter + 1) % eval_interval == 0:
                self._log(chapter, metrics)
            history["chapters"].append(metrics)
            if self.strategy and not self.strategy.should_continue(metrics, chapter):
                logger.info(f"  → Converged at chapter {chapter + 1}")
                break
        logger.info("[Done] Collaborative training complete")
        return history


class RecurrentCoLearner(SimpleCoLearner):
    """Co-learner that also trains a recurrent model on aggregated embeddings."""

    def __init__(
        self,
        models: List[Model],
        data: Data,
        strategy: Strategy,
        recurrent_model: RecurrentModel,
        warmup_epochs: int = 10,
        max_chapters: int = 10,
        keep_truth: bool = True,
        embeddings_layer: Optional[str] = None,
        embeddings_batch_size: int = 1024,
        embeddings_use_train: bool = True,
        embeddings_use_unlabeled: bool = False,
        embeddings_aggregate: str = "concat",
        **kwargs,
    ):
        super().__init__(
            models,
            data,
            strategy,
            warmup_epochs=warmup_epochs,
            max_chapters=max_chapters,
            keep_truth=keep_truth,
            **kwargs,
        )
        self.recurrent_model = recurrent_model
        self.embeddings_layer = embeddings_layer
        self.embeddings_batch_size = embeddings_batch_size
        self.embeddings_use_train = embeddings_use_train
        self.embeddings_use_unlabeled = embeddings_use_unlabeled
        self.embeddings_aggregate = embeddings_aggregate

    def _get_embedding_indexes(self) -> Optional[np.ndarray]:
        if self.embeddings_use_train:
            if self.embeddings_use_unlabeled and self.data.unlabeled_indexes is not None:
                return self.data.unlabeled_indexes
            return np.arange(self.data.n_train)
        return None

    def _resolve_recurrent_labels(self, indexes: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Get labels for recurrent training: resolved base + pseudo-labels from arbiter."""
        if hasattr(self.data, 'resolve_labels'):
            labels = self.data.resolve_labels(policy="unlabeled_as_is")
        elif getattr(self.data, "semisupervised_labels", None) is not None:
            labels = self.data.semisupervised_labels.copy()
        elif getattr(self.data, "y_train_original", None) is not None:
            labels = self.data.y_train_original.copy()
        else:
            return None

        # Merge pseudo-labels via majority vote, respecting preserve_labeled
        pseudo_list = [m._pseudo_labels for m in self.models if m._pseudo_labels is not None]
        if pseudo_list:
            pseudo_array = np.array(pseudo_list)
            mask = np.any(pseudo_array != -1, axis=0)
            if getattr(self.data, 'preserve_labeled', False):
                labeled_idx = getattr(self.data, 'labeled_indexes', None)
                if labeled_idx is not None and len(labeled_idx) > 0:
                    mask[labeled_idx] = False
            for i in np.where(mask)[0]:
                votes = pseudo_array[:, i][pseudo_array[:, i] != -1]
                labels[i] = int(np.round(np.mean(votes)))

        return labels if indexes is None else labels[indexes]

    def _collect_embeddings(self, indexes: Optional[np.ndarray], use_train: Optional[bool] = None) -> np.ndarray:
        # Collect one embedding view per detector model
        use_train = self.embeddings_use_train if use_train is None else use_train
        embeddings = []
        for model in self.models:
            emb = extract_embeddings_auto(
                model,
                indexes=indexes,
                use_train=use_train,
                layer=self.embeddings_layer,
                batch_size=self.embeddings_batch_size,
            )
            emb = np.asarray(emb)
            if emb.ndim == 1:
                emb = emb.reshape(-1, 1)
            logger.info(f"[{model.__class__.__name__}] Embeddings shape: {emb.shape}")
            embeddings.append(emb)

        if not embeddings:
            raise ValueError("No embeddings were collected from models.")

        if self.embeddings_aggregate == "concat":
            # Keep all model features side-by-side
            return np.concatenate(embeddings, axis=1)
        if self.embeddings_aggregate == "mean":
            # Average aligned features when dimensions match
            return np.mean(np.stack(embeddings, axis=0), axis=0)
        if self.embeddings_aggregate == "stack":
            # Pad to a common size so each model becomes one sequence step
            max_dim = max(emb.shape[1] for emb in embeddings)
            padded = []
            for emb in embeddings:
                if emb.shape[1] < max_dim:
                    pad = np.zeros((emb.shape[0], max_dim - emb.shape[1]), dtype=emb.dtype)
                    emb = np.concatenate([emb, pad], axis=1)
                padded.append(emb)

            shapes = [emb.shape for emb in padded]
            if len(set(shapes)) > 1:
                raise ValueError(f"Embeddings have mismatched shapes for stacking: {shapes}. "
                                 f"All models must have same n_samples. Check that all models use same data split.")
            
            return np.stack(padded, axis=1)

        raise ValueError(f"Unsupported embeddings_aggregate: {self.embeddings_aggregate}")

    def _train_recurrent(self, chapter: int) -> None:
        # Train recurrent model epoch-by-epoch; embed once, cache for val loss
        indexes = self._get_embedding_indexes()
        labels = self._resolve_recurrent_labels(indexes)
        if labels is None:
            logger.warning("[Recurrent] Labels not available; skipping recurrent training.")
            return
        aggregated_embeddings = self._collect_embeddings(indexes)
        val_emb = self._collect_val_embeddings()  # cache once per chapter
        n_epochs = getattr(self.recurrent_model, "num_epochs", 1)
        for _ in range(n_epochs):
            self.recurrent_model.train(aggregated_embeddings, labels, epochs=1)
            self._compute_recurrent_val_loss_cached(val_emb)

    def _collect_val_embeddings(self) -> Optional[np.ndarray]:
        # Extract fresh val embeddings by temporarily swapping test → val in each model
        if self.data.X_val is None or self.data.y_val is None:
            return None
        saved = []
        for m in self.models:
            saved.append((getattr(m, "X_test", None), getattr(m, "y_test", None)))
            m.X_test, m.y_test = m.X_val, self.data.y_val
        try:
            return self._collect_embeddings(indexes=None, use_train=False)
        finally:
            for m, (xt, yt) in zip(self.models, saved):
                m.X_test, m.y_test = xt, yt

    def _compute_recurrent_val_loss(self) -> None:
        # Compute val loss with fresh embeddings and append to recurrent model history
        if not getattr(self.recurrent_model, "_fitted", False):
            return
        val_emb = self._collect_val_embeddings()
        if val_emb is None:
            return
        val_loss = self.recurrent_model.get_loss(val_emb, self.data.y_val)
        if val_loss is not None:
            self.recurrent_model._val_loss_history.append(val_loss)

    def _compute_recurrent_val_loss_cached(self, val_emb: Optional[np.ndarray]) -> None:
        # Reuse pre-collected val embeddings to avoid redundant extraction
        if not getattr(self.recurrent_model, "_fitted", False) or val_emb is None:
            return
        val_loss = self.recurrent_model.get_loss(val_emb, self.data.y_val)
        if val_loss is not None:
            self.recurrent_model._val_loss_history.append(val_loss)

    def cotrain(self, eval_interval: int = 1) -> Dict[str, List[float]]:
        history = {"warmup": [], "chapters": []}
        self._warmup()
        logger.info(f"[Collaborative] Running up to {self.max_chapters} chapters...")
        for chapter in range(self.max_chapters):
            self.exchange()
            for model in self.models:
                model.train(1)
            self._train_recurrent(chapter)
            metrics = self._evaluate(self.data.y_test)
            if (chapter + 1) % eval_interval == 0:
                self._log(chapter, metrics)
            history["chapters"].append(metrics)
            if self.strategy and not self.strategy.should_continue(metrics, chapter):
                logger.info(f"  → Converged at chapter {chapter + 1}")
                break
        logger.info("[Done] Collaborative training complete")
        return history


class DelayedRecurrentCoLearner(RecurrentCoLearner):
    """Recurrent co-learner that delays recurrent training until later chapters.

    Uses validation-based early stopping (like CoLearnerVal) to avoid test leakage.
    Supports epochs_per_chapter for parity with CoLearnerVal.
    """

    def __init__(
        self,
        models: List[Model],
        data: Data,
        strategy: Strategy,
        recurrent_model: RecurrentModel,
        recurrent_start_chapter: int = 3,
        epochs_per_chapter: int = 1,
        **kwargs,
    ):
        super().__init__(
            models,
            data,
            strategy,
            recurrent_model,
            **kwargs,
        )
        self.recurrent_start_chapter = recurrent_start_chapter
        self.epochs_per_chapter = epochs_per_chapter
        self.val_loss_history = {i: [] for i in range(len(models))}

    def _evaluate_with_gru(self, y_val: np.ndarray, chapter: int) -> Dict[str, float]:
        """Evaluate detectors + GRU judge on validation set."""
        val_scores_list = []
        metrics = {}
        for i, model in enumerate(self.models):
            sc = _predict_scores_on_val(model, self.data)
            val_scores_list.append(sc)
            try:
                metrics[f"model_{i}"] = _compute_auc_or_raise(y_val, sc)
            except ValueError:
                metrics[f"model_{i}"] = 0.5
        try:
            metrics["ensemble"] = _compute_auc_or_raise(
                y_val, np.mean(val_scores_list, axis=0))
        except ValueError:
            metrics["ensemble"] = 0.5
        if chapter >= self.recurrent_start_chapter and self.recurrent_model._fitted:
            val_emb = self._collect_val_embeddings()
            if val_emb is not None:
                try:
                    gru_scores = self.recurrent_model.predict_scores(val_emb)
                    metrics["gru"] = roc_auc_score(y_val, gru_scores)
                except Exception:
                    metrics["gru"] = 0.5
        return metrics

    def cotrain(self, eval_interval: int = 1) -> Dict[str, List[float]]:
        history = {"warmup": [], "chapters": []}
        y_val = self.data.y_val
        if y_val is None or len(y_val) == 0:
            raise ValueError("Validation set required for DelayedRecurrentCoLearner.")

        # Warmup with val loss tracking
        logger.info(f"[Warmup] Training {len(self.models)} models for {self.warmup_epochs} epochs...")
        for epoch in range(self.warmup_epochs):
            for model in self.models:
                model.train(1)
            for i, model in enumerate(self.models):
                val_loss = _compute_val_loss(model, self.data)
                if val_loss is not None:
                    self.val_loss_history[i].append(val_loss)
            if (epoch + 1) % max(1, self.warmup_epochs // 3) == 0:
                logger.info(f"  Epoch {epoch + 1}/{self.warmup_epochs}")

        logger.info(f"[Collaborative] Running up to {self.max_chapters} chapters "
                    f"({self.epochs_per_chapter} epochs/chapter)...")
        for chapter in range(self.max_chapters):
            self.exchange()
            for ep in range(self.epochs_per_chapter):
                for model in self.models:
                    model.train(1)
                for i, model in enumerate(self.models):
                    val_loss = _compute_val_loss(model, self.data)
                    if val_loss is not None:
                        self.val_loss_history[i].append(val_loss)
            if chapter >= self.recurrent_start_chapter:
                self._train_recurrent(chapter)
            metrics = self._evaluate_with_gru(y_val, chapter)
            if (chapter + 1) % eval_interval == 0:
                self._log(chapter, metrics)
            history["chapters"].append(metrics)
            if self.strategy and not self.strategy.should_continue(metrics, chapter):
                logger.info(f"  → Converged at chapter {chapter + 1}")
                break
        logger.info("[Done] Collaborative training complete")
        return history

class SingleModel(CoLearning):
    """Single-model baseline for comparison."""
    
    def __init__(self, model: Model, data: Data, strategy: Optional[Strategy] = None, warmup_epochs: int = 0):
        super().__init__([model], data, None, warmup_epochs, 1)
        self.model = model
        self.test_scores: Optional[np.ndarray] = None

    def exchange(self) -> None:
        pass

    def cotrain(self, eval_interval: int = 1) -> Dict[str, float]:
        logger.info("Training single model...")
        self.model.fit()
        self.test_scores = self.model.predict_scores()
        try:
            auc = roc_auc_score(self.data.y_test, self.test_scores)
        except Exception:
            raise RuntimeError("Can't calculate the auc")
        return {"model_0": auc}
    



def _compute_auc_or_raise(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if y_true.shape[0] != scores.shape[0]:
        raise ValueError("AUC requires y_true and scores to have the same length.")

    mask = np.isin(y_true, [0, 1])
    y = y_true[mask]
    s = scores[mask]

    if y.size == 0:
        raise ValueError("AUC requires at least one labeled sample.")
    if np.unique(y).size < 2:
        raise ValueError("AUC requires both normal and anomaly labels.")
    if not np.all(np.isfinite(s)):
        raise ValueError("AUC requires finite scores only.")

    return roc_auc_score(y, s)

def _predict_scores_on_val(model, data):
    X_val, y_val = data.X_val, data.y_val
    if hasattr(model, "X_test") and hasattr(model, "y_test"):
        X_test_orig, y_test_orig = model.X_test, model.y_test
        model.X_test, model.y_test = X_val, y_val
        try:
            scores = model.predict_scores()
        finally:
            model.X_test, model.y_test = X_test_orig, y_test_orig
    else:
        scores = model.predict_scores()
    return scores

def _compute_val_loss(model, data):
    """Compute validation loss; returns None if unavailable."""
    if hasattr(model, "get_loss"):
        loss = model.get_loss(use_val=True)
        if loss is not None:
            return loss
    return None



class CoLearnerVal(SimpleCoLearner):
    def __init__(self, *args, epochs_per_chapter: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs_per_chapter = epochs_per_chapter
        # Track validation loss history per model during training
        self.val_loss_history = {}  # {model_idx: [loss_per_epoch]}
    
    def cotrain(self, eval_interval: int = 1):
        history = {"warmup": [], "chapters": []}
        y_val = self.data.y_val
        if y_val is None or len(y_val) == 0:
            raise ValueError("Validation set required for CoLearnerVal.")
        
        # Initialize validation loss tracking
        for i in range(len(self.models)):
            self.val_loss_history[i] = []

        logger.info(f"[Warmup] Training {len(self.models)} models for {self.warmup_epochs} epochs...")
        for epoch in range(self.warmup_epochs):
            for model in self.models:
                model.train(1)
            for i, model in enumerate(self.models):
                val_loss = _compute_val_loss(model, self.data)
                if val_loss is not None:
                    self.val_loss_history[i].append(val_loss)
            if (epoch + 1) % max(1, self.warmup_epochs // 3) == 0:
                logger.info(f"  Epoch {epoch + 1}/{self.warmup_epochs}")

        logger.info(f"[Collaborative] Running up to {self.max_chapters} chapters ({self.epochs_per_chapter} epochs/chapter)...")
        for chapter in range(self.max_chapters):
            self.exchange()
            
            for ep in range(self.epochs_per_chapter):
                for model in self.models:
                    model.train(1)
                for i, model in enumerate(self.models):
                    val_loss = _compute_val_loss(model, self.data)
                    if val_loss is not None:
                        self.val_loss_history[i].append(val_loss)

            val_scores_list = []
            model_metrics = {}

            for i, model in enumerate(self.models):
                model_val_scores = _predict_scores_on_val(model, self.data)
                val_scores_list.append(model_val_scores)
                try:
                    model_auc = _compute_auc_or_raise(y_val, model_val_scores)
                except ValueError as exc:
                    raise ValueError(f"Invalid validation AUC for model_{i}: {exc}") from exc
                model_metrics[f"model_{i}"] = model_auc

            ensemble_scores = np.mean(np.array(val_scores_list), axis=0)
            try:
                ensemble_auc = _compute_auc_or_raise(y_val, ensemble_scores)
            except ValueError as exc:
                raise ValueError(f"Invalid validation AUC for ensemble: {exc}") from exc
            model_metrics["ensemble"] = ensemble_auc

            if (chapter + 1) % eval_interval == 0:
                metric_str = ", ".join(f"{k}={v:.4f}" for k, v in model_metrics.items())
                logger.info(f"  Chapter {chapter + 1}: {metric_str}")

            history["chapters"].append(model_metrics)

            if self.strategy is not None and not self.strategy.should_continue(model_metrics, chapter):
                logger.info(f"  → Converged at chapter {chapter + 1}")
                break

        logger.info("[Done] Collaborative training complete")
        return history