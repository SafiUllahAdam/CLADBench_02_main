import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from base import CoLearning, Data, Model, Strategy, RecurrentModel
from utils import extract_embeddings_auto

logger = logging.getLogger(__name__)


def _compute_auc_or_raise(y_true, scores):
    y, s = np.asarray(y_true), np.asarray(scores)
    if y.shape[0] != s.shape[0]:
        raise ValueError("y_true/scores length mismatch")
    mask = np.isin(y, [0, 1])
    y, s = y[mask], s[mask]
    if y.size == 0 or np.unique(y).size < 2:
        raise ValueError("AUC needs both classes present")
    if not np.all(np.isfinite(s)):
        raise ValueError("AUC needs finite scores")
    return roc_auc_score(y, s)


def _predict_scores_on_val(model, data):
    """Score validation set directly via use_val parameter."""
    return model.predict_scores(use_val=True)


def _compute_val_loss(model, data):
    if hasattr(model, "get_loss"):
        return model.get_loss(use_val=True)
    return None


class SimpleCoLearner(CoLearning):
    """Collaborative learning with bidirectional pseudo-label exchange."""

    def __init__(self, models: List[Model], data: Data, strategy: Strategy,
                 warmup_epochs: int = 10, max_chapters: int = 10,
                 anomaly_threshold: float = 0.5, confidence_threshold_low: float = 0.02,
                 confidence_threshold_high: float = 0.98, keep_truth: bool = True,
                 transfer_thresholds: Optional[Dict] = None, pseudo_label_arbiter=None):
        super().__init__(models, data, strategy, warmup_epochs, max_chapters,
                         anomaly_threshold, confidence_threshold_low, confidence_threshold_high,
                         transfer_thresholds=transfer_thresholds,
                         pseudo_label_arbiter=pseudo_label_arbiter)
        self.keep_truth = keep_truth
        for m in models:
            m.clear_pseudo_labels()
        self.model_scores: Dict[str, np.ndarray] = {f"model_{i}": None for i in range(len(models))}
        self.ensemble_scores: Optional[np.ndarray] = None
        self.ensemble_disagreement: Optional[np.ndarray] = None
        self.exchange_history: List[Dict] = []

    def exchange(self) -> None:
        for m in self.models:
            m.clear_pseudo_labels()

        exch_idx = self.data.unlabeled_indexes if self.keep_truth else np.arange(self.data.n_train)
        if self.keep_truth and (exch_idx is None or len(exch_idx) == 0):
            logger.warning("[Exchange] No unlabeled_indexes set")
            return

        n = len(self.models)
        y_true = self.data.y_train_original
        stats = {"keep_truth": self.keep_truth, "n_exchange_candidates": len(exch_idx),
                 "n_anomalies": 0, "n_normals": 0,
                 "correct_anomalies": 0, "correct_normals": 0, "per_pair": {}}

        for s in range(n):
            scores = self.models[s].predict_scores(use_train=True)
            self.model_scores[f"model_{s}"] = scores
            cand = scores[exch_idx]
            anom_idx = exch_idx[cand > self.anomaly_threshold]
            norm_idx = exch_idx[cand <= self.anomaly_threshold]

            for r in range(n):
                if r == s:
                    continue
                na = self.send_anomalies(s, r, anom_idx, scores)
                nn = self.send_normals(s, r, norm_idx, scores)
                stats["per_pair"][(s, r)] = {"n_anomalies": na, "n_normals": nn}
                stats["n_anomalies"] += na
                stats["n_normals"] += nn

            if y_true is not None:
                stats["correct_anomalies"] += int(np.sum(y_true[anom_idx] == 1))
                stats["correct_normals"] += int(np.sum(y_true[norm_idx] == 0))

        na, nn = stats["n_anomalies"], stats["n_normals"]
        ca, cn = stats["correct_anomalies"], stats["correct_normals"]
        stats["anomaly_precision"] = ca / na if na else 0.0
        stats["normal_precision"] = cn / nn if nn else 0.0
        stats["overall_precision"] = (ca + cn) / max(1, na + nn)
        self.exchange_history.append(stats)

        self._finalize_pseudo_labels()
        all_sc = np.array(list(self.model_scores.values()))
        self.ensemble_scores = np.mean(all_sc, axis=0)
        self.ensemble_disagreement = np.std(all_sc, axis=0)

    def _warmup(self) -> None:
        logger.info(f"[Warmup] {len(self.models)} models x {self.warmup_epochs} epochs")
        for ep in range(self.warmup_epochs):
            for m in self.models:
                m.train(1)
            if (ep + 1) % max(1, self.warmup_epochs // 3) == 0:
                logger.info(f"  Epoch {ep + 1}/{self.warmup_epochs}")

    def _evaluate(self, y_true: np.ndarray) -> Dict[str, float]:
        scores, metrics = [], {}
        for i, m in enumerate(self.models):
            sc = m.predict_scores()
            scores.append(sc)
            try:
                metrics[f"model_{i}"] = roc_auc_score(y_true, sc)
            except Exception:
                metrics[f"model_{i}"] = 0.5
        try:
            metrics["ensemble"] = roc_auc_score(y_true, np.mean(scores, axis=0))
        except Exception:
            metrics["ensemble"] = 0.5
        return metrics

    def _log(self, chapter: int, metrics: Dict[str, float]) -> None:
        logger.info(f"  Ch {chapter + 1}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    def cotrain(self, eval_interval: int = 1) -> Dict[str, List[float]]:
        history = {"warmup": [], "chapters": []}
        self._warmup()
        logger.info(f"[Collab] Up to {self.max_chapters} chapters")
        for ch in range(self.max_chapters):
            self.exchange()
            for m in self.models:
                m.train(1)
            metrics = self._evaluate(self.data.y_test)
            if (ch + 1) % eval_interval == 0:
                self._log(ch, metrics)
            history["chapters"].append(metrics)
            if self.strategy and not self.strategy.should_continue(metrics, ch):
                logger.info(f"  Converged at chapter {ch + 1}")
                break
        logger.info("[Done]")
        return history


class RecurrentCoLearner(SimpleCoLearner):
    """Also trains a recurrent judge on aggregated embeddings."""

    def __init__(self, models: List[Model], data: Data, strategy: Strategy,
                 recurrent_model: RecurrentModel,
                 warmup_epochs: int = 10, max_chapters: int = 10, keep_truth: bool = True,
                 embeddings_layer: Optional[str] = None, embeddings_batch_size: int = 1024,
                 embeddings_use_train: bool = True, embeddings_use_unlabeled: bool = False,
                 embeddings_aggregate: str = "concat", **kwargs):
        super().__init__(models, data, strategy, warmup_epochs=warmup_epochs,
                         max_chapters=max_chapters, keep_truth=keep_truth, **kwargs)
        self.recurrent_model = recurrent_model
        self.embeddings_layer = embeddings_layer
        self.embeddings_batch_size = embeddings_batch_size
        self.embeddings_use_train = embeddings_use_train
        self.embeddings_use_unlabeled = embeddings_use_unlabeled
        self.embeddings_aggregate = embeddings_aggregate

    def _get_embedding_indexes(self) -> Optional[np.ndarray]:
        if not self.embeddings_use_train:
            return None
        if self.embeddings_use_unlabeled and self.data.unlabeled_indexes is not None:
            return self.data.unlabeled_indexes
        return np.arange(self.data.n_train)

    def _resolve_recurrent_labels(self, indexes: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Base labels + majority-vote pseudo-labels for recurrent training."""
        if hasattr(self.data, 'resolve_labels'):
            labels = self.data.resolve_labels(policy="unlabeled_as_is")
        elif getattr(self.data, "semisupervised_labels", None) is not None:
            labels = self.data.semisupervised_labels.copy()
        elif getattr(self.data, "y_train_original", None) is not None:
            labels = self.data.y_train_original.copy()
        else:
            return None

        pseudo_list = [m._pseudo_labels for m in self.models if m._pseudo_labels is not None]
        if pseudo_list:
            pa = np.array(pseudo_list)
            mask = np.any(pa != -1, axis=0)
            if getattr(self.data, 'preserve_labeled', False):
                labeled = getattr(self.data, 'labeled_indexes', None)
                if labeled is not None and len(labeled):
                    mask[labeled] = False
            for i in np.where(mask)[0]:
                votes = pa[:, i][pa[:, i] != -1]
                labels[i] = int(np.round(np.mean(votes)))

        return labels if indexes is None else labels[indexes]

    def _collect_embeddings(self, indexes: Optional[np.ndarray],
                            use_train: Optional[bool] = None, use_val: bool = False) -> np.ndarray:
        use_train = self.embeddings_use_train if use_train is None else use_train
        embeddings = []
        for m in self.models:
            emb = extract_embeddings_auto(m, indexes=indexes, use_train=use_train, use_val=use_val,
                                          layer=self.embeddings_layer,
                                          batch_size=self.embeddings_batch_size)
            emb = np.atleast_2d(np.asarray(emb))
            logger.info(f"[{m.__class__.__name__}] Embeddings: {emb.shape}")
            embeddings.append(emb)
        if not embeddings:
            raise ValueError("No embeddings collected")

        agg = self.embeddings_aggregate
        if agg == "concat":
            return np.concatenate(embeddings, axis=1)
        if agg == "mean":
            return np.mean(np.stack(embeddings, axis=0), axis=0)
        if agg == "stack":
            max_d = max(e.shape[1] for e in embeddings)
            padded = [np.pad(e, ((0, 0), (0, max_d - e.shape[1]))) if e.shape[1] < max_d else e
                      for e in embeddings]
            shapes = {p.shape for p in padded}
            if len(shapes) > 1:
                raise ValueError(f"Shape mismatch for stacking: {shapes}")
            return np.stack(padded, axis=1)
        raise ValueError(f"Unsupported embeddings_aggregate: {agg}")

    def _collect_val_embeddings(self) -> Optional[np.ndarray]:
        if self.data.X_val is None or self.data.y_val is None:
            return None
        return self._collect_embeddings(indexes=None, use_train=False, use_val=True)

    def _compute_recurrent_val_loss(self) -> None:
        if not getattr(self.recurrent_model, "_fitted", False):
            return
        val_emb = self._collect_val_embeddings()
        if val_emb is None:
            return
        val_loss = self.recurrent_model.get_loss(val_emb, self.data.y_val)
        if val_loss is not None:
            self.recurrent_model._val_loss_history.append(val_loss)

    def _compute_recurrent_val_loss_cached(self, val_emb: Optional[np.ndarray]) -> None:
        if not getattr(self.recurrent_model, "_fitted", False) or val_emb is None:
            return
        val_loss = self.recurrent_model.get_loss(val_emb, self.data.y_val)
        if val_loss is not None:
            self.recurrent_model._val_loss_history.append(val_loss)

    def _train_recurrent(self, chapter: int) -> None:
        indexes = self._get_embedding_indexes()
        labels = self._resolve_recurrent_labels(indexes)
        if labels is None:
            logger.warning("[Recurrent] No labels; skipping")
            return
        agg_emb = self._collect_embeddings(indexes)
        val_emb = self._collect_val_embeddings()
        for _ in range(getattr(self.recurrent_model, "num_epochs", 1)):
            self.recurrent_model.train(agg_emb, labels, epochs=1)
            self._compute_recurrent_val_loss_cached(val_emb)

    def cotrain(self, eval_interval: int = 1) -> Dict[str, List[float]]:
        history = {"warmup": [], "chapters": []}
        self._warmup()
        logger.info(f"[Collab] Up to {self.max_chapters} chapters")
        for ch in range(self.max_chapters):
            self.exchange()
            for m in self.models:
                m.train(1)
            self._train_recurrent(ch)
            metrics = self._evaluate(self.data.y_test)
            if (ch + 1) % eval_interval == 0:
                self._log(ch, metrics)
            history["chapters"].append(metrics)
            if self.strategy and not self.strategy.should_continue(metrics, ch):
                logger.info(f"  Converged at chapter {ch + 1}")
                break
        logger.info("[Done]")
        return history


class DelayedRecurrentCoLearner(RecurrentCoLearner):
    """Delays GRU training until recurrent_start_chapter. Val-based early stopping."""

    def __init__(self, models: List[Model], data: Data, strategy: Strategy,
                 recurrent_model: RecurrentModel, recurrent_start_chapter: int = 3,
                 epochs_per_chapter: int = 1, **kwargs):
        super().__init__(models, data, strategy, recurrent_model, **kwargs)
        self.recurrent_start_chapter = recurrent_start_chapter
        self.epochs_per_chapter = epochs_per_chapter
        self.val_loss_history = {i: [] for i in range(len(models))}

    def _evaluate_with_gru(self, y_val: np.ndarray, chapter: int) -> Dict[str, float]:
        val_scores, metrics = [], {}
        for i, m in enumerate(self.models):
            sc = _predict_scores_on_val(m, self.data)
            val_scores.append(sc)
            try:
                metrics[f"model_{i}"] = _compute_auc_or_raise(y_val, sc)
            except ValueError:
                metrics[f"model_{i}"] = 0.5
        try:
            metrics["ensemble"] = _compute_auc_or_raise(y_val, np.mean(val_scores, axis=0))
        except ValueError:
            metrics["ensemble"] = 0.5

        if chapter >= self.recurrent_start_chapter and self.recurrent_model._fitted:
            val_emb = self._collect_val_embeddings()
            if val_emb is not None:
                try:
                    metrics["gru"] = roc_auc_score(y_val, self.recurrent_model.predict_scores(val_emb))
                except Exception:
                    metrics["gru"] = 0.5
        return metrics

    def _warmup_with_val(self) -> None:
        logger.info(f"[Warmup] {len(self.models)} models x {self.warmup_epochs} epochs")
        for ep in range(self.warmup_epochs):
            for m in self.models:
                m.train(1)
            for i, m in enumerate(self.models):
                vl = _compute_val_loss(m, self.data)
                if vl is not None:
                    self.val_loss_history[i].append(vl)
            if (ep + 1) % max(1, self.warmup_epochs // 3) == 0:
                logger.info(f"  Epoch {ep + 1}/{self.warmup_epochs}")

    def cotrain(self, eval_interval: int = 1) -> Dict[str, List[float]]:
        history = {"warmup": [], "chapters": []}
        y_val = self.data.y_val
        if y_val is None or len(y_val) == 0:
            raise ValueError("Validation set required for DelayedRecurrentCoLearner")

        self._warmup_with_val()

        logger.info(f"[Collab] Up to {self.max_chapters} chapters ({self.epochs_per_chapter} ep/ch)")
        for ch in range(self.max_chapters):
            self.exchange()
            for _ in range(self.epochs_per_chapter):
                for m in self.models:
                    m.train(1)
                for i, m in enumerate(self.models):
                    vl = _compute_val_loss(m, self.data)
                    if vl is not None:
                        self.val_loss_history[i].append(vl)
            if ch >= self.recurrent_start_chapter:
                self._train_recurrent(ch)
            metrics = self._evaluate_with_gru(y_val, ch)
            if (ch + 1) % eval_interval == 0:
                self._log(ch, metrics)
            history["chapters"].append(metrics)
            if self.strategy and not self.strategy.should_continue(metrics, ch):
                logger.info(f"  Converged at chapter {ch + 1}")
                break
        logger.info("[Done]")
        return history


class SingleModel(CoLearning):
    """Single-model baseline (no collaboration)."""

    def __init__(self, model: Model, data: Data, strategy: Optional[Strategy] = None,
                 warmup_epochs: int = 0):
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
            return {"model_0": roc_auc_score(self.data.y_test, self.test_scores)}
        except Exception:
            raise RuntimeError("Cannot compute AUC")


class CoLearnerVal(SimpleCoLearner):
    """SimpleCoLearner with validation-based monitoring (no test leakage)."""

    def __init__(self, *args, epochs_per_chapter: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs_per_chapter = epochs_per_chapter
        self.val_loss_history = {}

    def cotrain(self, eval_interval: int = 1):
        history = {"warmup": [], "chapters": []}
        y_val = self.data.y_val
        if y_val is None or len(y_val) == 0:
            raise ValueError("Validation set required for CoLearnerVal")

        for i in range(len(self.models)):
            self.val_loss_history[i] = []

        logger.info(f"[Warmup] {len(self.models)} models x {self.warmup_epochs} epochs")
        for ep in range(self.warmup_epochs):
            for m in self.models:
                m.train(1)
            for i, m in enumerate(self.models):
                vl = _compute_val_loss(m, self.data)
                if vl is not None:
                    self.val_loss_history[i].append(vl)
            if (ep + 1) % max(1, self.warmup_epochs // 3) == 0:
                logger.info(f"  Epoch {ep + 1}/{self.warmup_epochs}")

        logger.info(f"[Collab] Up to {self.max_chapters} chapters ({self.epochs_per_chapter} ep/ch)")
        for ch in range(self.max_chapters):
            self.exchange()
            for _ in range(self.epochs_per_chapter):
                for m in self.models:
                    m.train(1)
                for i, m in enumerate(self.models):
                    vl = _compute_val_loss(m, self.data)
                    if vl is not None:
                        self.val_loss_history[i].append(vl)

            val_scores, metrics = [], {}
            for i, m in enumerate(self.models):
                sc = _predict_scores_on_val(m, self.data)
                val_scores.append(sc)
                try:
                    metrics[f"model_{i}"] = _compute_auc_or_raise(y_val, sc)
                except ValueError as e:
                    raise ValueError(f"Invalid val AUC for model_{i}: {e}") from e
            try:
                metrics["ensemble"] = _compute_auc_or_raise(y_val, np.mean(val_scores, axis=0))
            except ValueError as e:
                raise ValueError(f"Invalid val AUC for ensemble: {e}") from e

            if (ch + 1) % eval_interval == 0:
                self._log(ch, metrics)
            history["chapters"].append(metrics)

            if self.strategy is not None and not self.strategy.should_continue(metrics, ch):
                logger.info(f"  Converged at chapter {ch + 1}")
                break

        logger.info("[Done]")
        return history
