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
    """Score the validation set"""
    return model.predict_scores(use_val=True)


def _compute_val_loss(model, data):
    if hasattr(model, "get_loss"):
        return model.get_loss(use_val=True)
    return None


class SimpleCoLearner(CoLearning):
    """Collaborative learning with pseudo-label exchange between all model pairs"""

    def __init__(self, models: List[Model], data: Data, strategy: Strategy,
                 warmup_epochs: int = 10, max_chapters: int = 10,
                 anomaly_threshold: float = 0.5, confidence_threshold_low: float = 0.10,
                 confidence_threshold_high: float = 0.90, keep_truth: bool = True,
                 transfer_thresholds: Optional[Dict] = None, pseudo_label_arbiter=None):
        super().__init__(
            models,
            data,
            strategy,
            warmup_epochs,
            max_chapters,
            anomaly_threshold,
            confidence_threshold_low,
            confidence_threshold_high,
            transfer_thresholds=transfer_thresholds,
            pseudo_label_arbiter=pseudo_label_arbiter,
        )
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
        stats = {
            "keep_truth": self.keep_truth,
            "n_exchange_candidates": len(exch_idx),
            "n_anomalies": 0,
            "n_normals": 0,
            "correct_anomalies": 0,
            "correct_normals": 0,
            "per_pair": {},
        }

        for s in range(n):
            scores = self.models[s].predict_scores(use_train=True)
            self.model_scores[f"model_{s}"] = scores
            cand = scores[exch_idx]
            anom_idx = exch_idx[cand > self.confidence_threshold_high]
            norm_idx = exch_idx[cand <= self.confidence_threshold_low]

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
    """Adds a GRU judge trained on detector embeddings"""

    def __init__(self, models: List[Model], data: Data, strategy: Strategy,
                 recurrent_model: RecurrentModel,
                 warmup_epochs: int = 10, max_chapters: int = 10, keep_truth: bool = True,
                 embeddings_layer: Optional[str] = None, embeddings_batch_size: int = 1024,
                 embeddings_use_train: bool = True, embeddings_use_unlabeled: bool = False,
                 embeddings_aggregate: str = "concat", **kwargs):
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
        if not self.embeddings_use_train:
            return None
        if self.embeddings_use_unlabeled and self.data.unlabeled_indexes is not None:
            return self.data.unlabeled_indexes
        return np.arange(self.data.n_train)

    def _resolve_recurrent_labels(self, indexes: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Merge base labels with majority-vote pseudo-labels for the GRU"""
        if hasattr(self.data, "resolve_labels"):
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
            if getattr(self.data, "preserve_labeled", False):
                labeled = getattr(self.data, "labeled_indexes", None)
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
            emb = extract_embeddings_auto(
                m,
                indexes=indexes,
                use_train=use_train,
                use_val=use_val,
                layer=self.embeddings_layer,
                batch_size=self.embeddings_batch_size,
            )
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
            padded = [
                np.pad(e, ((0, 0), (0, max_d - e.shape[1]))) if e.shape[1] < max_d else e
                for e in embeddings
            ]
            shapes = {p.shape for p in padded}
            if len(shapes) > 1:
                raise ValueError(f"Shape mismatch for stacking: {shapes}")
            return np.stack(padded, axis=1)
        raise ValueError(f"Unsupported embeddings_aggregate: {agg}")

    def _collect_val_embeddings(self) -> Optional[np.ndarray]:
        if self.data.X_val is None or self.data.y_val is None:
            return None
        return self._collect_embeddings(indexes=None, use_train=False, use_val=True)

    def _compute_recurrent_val_loss_cached(self, val_emb: Optional[np.ndarray]) -> None:
        if not getattr(self.recurrent_model, "_fitted", False) or val_emb is None:
            return
        y_val = self.data.y_val[:len(val_emb)]
        val_loss = self.recurrent_model.get_loss(val_emb, y_val)
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
    """Starts GRU training only after a fixed chapter"""

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
    """Solo baseline without collaboration"""

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
    """Collaborative learner monitored on validation AUC"""

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


class BasicAdaptiveCoLearner(CoLearnerVal):
    """TExGAD-style triple exchange with separate purification"""

    def __init__(self, *args, purification_ratio: float = 0.20,
                 purification_ratio_anomaly: Optional[float] = None,
                 purification_ratio_normal: Optional[float] = None,
                 specialist_forgetting: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.models) != 3:
            raise ValueError("BasicAdaptiveCoLearner requires exactly 3 models: UD, SD, DD")

        self.purification_ratio = purification_ratio
        self.purification_ratio_anomaly = (
            purification_ratio if purification_ratio_anomaly is None else purification_ratio_anomaly
        )
        self.purification_ratio_normal = (
            purification_ratio if purification_ratio_normal is None else purification_ratio_normal
        )
        self.specialist_forgetting = specialist_forgetting
        self.model_scores = {f"model_{i}": None for i in range(len(self.models))}
        self.ensemble_scores = None
        self.ensemble_disagreement = None
        self.exchange_history = []

    def _resolve_roles(self):
        """Read each model's declared role attribute"""
        role_to_idx = {"ud": None, "sd": None, "dd": None}
        for i, m in enumerate(self.models):
            role = getattr(m, "role", None)
            if role in role_to_idx and role_to_idx[role] is None:
                role_to_idx[role] = i

        missing = [r for r, v in role_to_idx.items() if v is None]
        if missing:
            raise ValueError(
                f"Missing role(s) {missing}; each wrapper must define `role` as 'ud', 'sd', or 'dd'"
            )
        return role_to_idx

    def _purify_indexes(self, indexes, scores, label):
        """Keep the most confident pseudo-labels"""
        if indexes is None or len(indexes) == 0:
            return indexes

        ratio = self.purification_ratio_anomaly if label == 1 else self.purification_ratio_normal
        keep_n = max(1, int(np.ceil(len(indexes) * ratio)))
        conf = scores[indexes]
        order = np.argsort(-conf) if label == 1 else np.argsort(conf)
        return indexes[order[:keep_n]]

    def _adaptive_ensemble_scores(self, scores_list):
        """Confidence-weighted ensemble scoring"""
        scores_array = np.asarray(scores_list, dtype=float)
        if scores_array.ndim != 2:
            raise ValueError("scores_list must produce a 2D array of shape (n_models, n_samples)")
        confidence = np.abs(scores_array - 0.5) * 2.0
        weighted = confidence * (scores_array ** 2)
        denom = np.sum(confidence, axis=0)
        denom[denom == 0.0] = 1.0
        return np.sum(weighted, axis=0) / denom

    def _apply_specialist_forgetting(self, exch_idx, stats):
        """Keep each anomaly pseudo-label for one receiver only"""
        if not self.specialist_forgetting:
            return

        score_mat = np.vstack([
            self.model_scores[f"model_{i}"] for i in range(len(self.models))
        ])

        owned, forgotten = 0, 0

        for idx in np.asarray(exch_idx, dtype=int):
            receivers = []

            for r, m in enumerate(self.models):
                pseudo = getattr(m, "_pseudo_labels", None)
                if pseudo is not None and pseudo[idx] == 1:
                    receivers.append(r)

            if len(receivers) <= 1:
                continue

            owner = receivers[int(np.argmax(score_mat[receivers, idx]))]
            owned += 1

            for r in receivers:
                if r == owner:
                    continue
                self.models[r]._pseudo_labels[idx] = -1
                if hasattr(self.models[r], "_y_train_cache"):
                    self.models[r]._y_train_cache = None
                forgotten += 1

        stats["specialist_owned_anomalies"] = owned
        stats["specialist_forgotten_anomaly_labels"] = forgotten

    def _send_pair(self, sender, receiver, exch_idx, stats):
        """Send purified pseudo-labels from sender to receiver"""
        sc = self.model_scores[f"model_{sender}"]
        anom = exch_idx[sc[exch_idx] >= self.confidence_threshold_high]
        norm = exch_idx[sc[exch_idx] <= self.confidence_threshold_low]

        anom = self._purify_indexes(anom, sc, label=1)
        norm = self._purify_indexes(norm, sc, label=0)

        pair_stats = {
            "n_anomalies": len(anom),
            "n_normals": len(norm),
        }

        y_true = getattr(self.data, "y_train_original", None)
        if y_true is not None:
            pair_stats["tp"] = int(np.sum(y_true[anom] == 1))
            pair_stats["fp"] = int(np.sum(y_true[anom] == 0))
            pair_stats["tn"] = int(np.sum(y_true[norm] == 0))
            pair_stats["fn"] = int(np.sum(y_true[norm] == 1))
            pair_stats["anomaly_precision"] = pair_stats["tp"] / max(1, pair_stats["tp"] + pair_stats["fp"])
            pair_stats["normal_precision"] = pair_stats["tn"] / max(1, pair_stats["tn"] + pair_stats["fn"])

            for key in ["tp", "fp", "tn", "fn"]:
                stats[key] = stats.get(key, 0) + pair_stats[key]

        na = self.send_anomalies(sender, receiver, anom, sc)
        nn = self.send_normals(sender, receiver, norm, sc)

        pair_stats["n_anomalies"] = na
        pair_stats["n_normals"] = nn
        stats["per_pair"][(sender, receiver)] = pair_stats
        stats["n_anomalies"] += na
        stats["n_normals"] += nn

    def exchange(self) -> None:
        for m in self.models:
            m.clear_pseudo_labels()

        exch_idx = self.data.unlabeled_indexes if self.keep_truth else np.arange(self.data.n_train)
        if self.keep_truth and (exch_idx is None or len(exch_idx) == 0):
            logger.warning("[Exchange] No unlabeled_indexes set")
            return

        roles = self._resolve_roles()
        ud, sd, dd = roles["ud"], roles["sd"], roles["dd"]

        stats = {
            "keep_truth": self.keep_truth,
            "n_exchange_candidates": len(exch_idx),
            "per_pair": {},
            "n_anomalies": 0,
            "n_normals": 0,
        }

        for i, m in enumerate(self.models):
            self.model_scores[f"model_{i}"] = m.predict_scores(use_train=True)

        for sender, receivers in [(ud, [sd, dd]), (sd, [ud, dd]), (dd, [ud, sd])]:
            for receiver in receivers:
                self._send_pair(sender, receiver, exch_idx, stats)

        self._finalize_pseudo_labels()
        self._apply_specialist_forgetting(exch_idx, stats)
        self.exchange_history.append(stats)

        all_sc = np.array(list(self.model_scores.values()))
        self.ensemble_scores = self._adaptive_ensemble_scores(all_sc)
        self.ensemble_disagreement = np.std(all_sc, axis=0)

    def cotrain(self, eval_interval: int = 1):
        history = {"warmup": [], "chapters": []}
        y_val = self.data.y_val
        if y_val is None or len(y_val) == 0:
            raise ValueError("Validation set required for BasicAdaptiveCoLearner")

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
                metrics["ensemble"] = _compute_auc_or_raise(y_val, self._adaptive_ensemble_scores(val_scores))
            except ValueError as e:
                raise ValueError(f"Invalid val AUC for adaptive ensemble: {e}") from e

            if (ch + 1) % eval_interval == 0:
                self._log(ch, metrics)
            history["chapters"].append(metrics)

            if self.strategy is not None and not self.strategy.should_continue(metrics, ch):
                logger.info(f"  Converged at chapter {ch + 1}")
                break

        logger.info("[Done]")
        return history


class DelayedRecurrentBasicAdaptiveCoLearner(BasicAdaptiveCoLearner):
    """TExGAD triple exchange + adaptive delayed GRU + Context-rich GRU inputs"""

    def __init__(self, models: List[Model], data: Data, strategy: Strategy,
                 recurrent_model: RecurrentModel, recurrent_start_chapter: int = 3,
                 epochs_per_chapter: int = 1,
                 embeddings_layer: Optional[str] = None, embeddings_batch_size: int = 1024,
                 embeddings_use_train: bool = True, embeddings_use_unlabeled: bool = False,
                 embeddings_aggregate: str = "stack",
                 adaptive_start_window: int = 3,
                 adaptive_start_tolerance: float = 0.003,
                 adaptive_max_wait_chapter: Optional[int] = None,
                 recurrent_include_raw: bool = False,
                 recurrent_include_scores: bool = False,
                 final_score_blend_grid: Optional[List[float]] = None,
                 recurrent_token_sequence: bool = False,   
                 **kwargs):
        super().__init__(models=models, data=data, strategy=strategy, **kwargs)
        self.recurrent_model = recurrent_model
        self.recurrent_start_chapter = recurrent_start_chapter
        self.epochs_per_chapter = epochs_per_chapter
        self.embeddings_layer = embeddings_layer
        self.embeddings_batch_size = embeddings_batch_size
        self.embeddings_use_train = embeddings_use_train
        self.embeddings_use_unlabeled = embeddings_use_unlabeled
        self.embeddings_aggregate = embeddings_aggregate
        self.adaptive_start_window = adaptive_start_window
        self.adaptive_start_tolerance = adaptive_start_tolerance
        self.adaptive_max_wait_chapter = adaptive_max_wait_chapter
        self.recurrent_include_raw = recurrent_include_raw
        self.recurrent_include_scores = recurrent_include_scores
        self.recurrent_token_sequence = recurrent_token_sequence

        self._recurrent_started = False
        self._recurrent_start_actual_chapter = None
        self._val_ensemble_auc_history = []
        self._pseudo_context_history = []
        self.final_score_blend_grid = final_score_blend_grid or [0.0, 0.25, 0.5, 0.75, 1.0] 
        self._best_blend_alpha = 1.0                                                        
        self._best_blend_val_auc = -float("inf")                                            

    def _get_embedding_indexes(self) -> Optional[np.ndarray]:
        if not self.embeddings_use_train:
            return None
        if self.embeddings_use_unlabeled and self.data.unlabeled_indexes is not None:
            return self.data.unlabeled_indexes
        return np.arange(self.data.n_train)

# NEW collects X_train/X_val/X_test.
    def _collect_raw_features(self, indexes: Optional[np.ndarray],
                              use_train: bool, use_val: bool) -> np.ndarray:
        """Return raw features aligned with recurrent embeddings"""
        if use_val:
            raw = self.data.X_val
        elif use_train:
            raw = self.data.X_train
            if indexes is not None:
                raw = raw[np.asarray(indexes)]
        else:
            raw = self.data.X_test

        if raw is None:
            raise ValueError("Raw features are unavailable for recurrent input")

        return np.atleast_2d(np.asarray(raw, dtype=np.float32))
# NEW This builds: Raw1, E1, S1, Raw2, E2, S2, Raw3, E3, S3     Shape becomes:[n_samples, 9, token_dim]
    def _pad_features(self, x: np.ndarray, width: int) -> np.ndarray:
        """Pad features to a fixed width"""
        if x.shape[1] > width:
            raise ValueError(f"Cannot pad width {x.shape[1]} to smaller width {width}")
        if x.shape[1] == width:
            return x
        return np.pad(x, ((0, 0), (0, width - x.shape[1])))

    def _token_marker(self, n: int, token_type: int, role: int) -> np.ndarray:
        """Build token type and detector role markers"""
        marker = np.zeros((n, 6), dtype=np.float32)
        marker[:, token_type] = 1.0
        marker[:, 3 + role] = 1.0
        return marker

    def _collect_token_sequence_input(self, raw, embeddings, scores_list) -> np.ndarray:
        """Build repeated raw/embedding/score token sequence"""
        if raw is None:
            raise ValueError("recurrent_token_sequence=True requires recurrent_include_raw=True")
        if any(scores is None for scores in scores_list):
            raise ValueError("recurrent_token_sequence=True requires recurrent_include_scores=True")

        n = raw.shape[0]
        body_d = max([raw.shape[1], 1] + [e.shape[1] for e in embeddings])
        tokens = []

        for role_idx, (emb, scores) in enumerate(zip(embeddings, scores_list)):
            raw_token = np.concatenate(
                [self._pad_features(raw, body_d), self._token_marker(n, 0, role_idx)],
                axis=1,
            )
            emb_token = np.concatenate(
                [self._pad_features(emb, body_d), self._token_marker(n, 1, role_idx)],
                axis=1,
            )
            score_token = np.concatenate(
                [self._pad_features(scores, body_d), self._token_marker(n, 2, role_idx)],
                axis=1,
            )

            tokens.extend([raw_token, emb_token, score_token])

        recurrent_input = np.stack(tokens, axis=1)
        logger.info(f"[TokenSequenceInput] shape={recurrent_input.shape}")
        return recurrent_input


# builds each GRU step as [raw | padded embedding | score], then stacks to [n_samples, 3, feat_dim]
    def _collect_embeddings(self, indexes: Optional[np.ndarray],
                            use_train: Optional[bool] = None, use_val: bool = False) -> np.ndarray:
        use_train = self.embeddings_use_train if use_train is None else use_train
        raw = self._collect_raw_features(indexes, use_train, use_val) if self.recurrent_include_raw else None

        embeddings, scores_list = [], []
        roles = self._resolve_roles()
        ordered_models = [self.models[roles["ud"]], self.models[roles["sd"]], self.models[roles["dd"]]]

        for m in ordered_models:
            emb = extract_embeddings_auto(
                m,
                indexes=indexes,
                use_train=use_train,
                use_val=use_val,
                layer=self.embeddings_layer,
                batch_size=self.embeddings_batch_size,
            )
            emb = np.atleast_2d(np.asarray(emb, dtype=np.float32))
            embeddings.append(emb)

            if self.recurrent_include_scores:
                if use_val:
                    scores = m.predict_scores(use_val=True)
                elif use_train:
                    scores = m.predict_scores(use_train=True)
                    if indexes is not None:
                        scores = scores[np.asarray(indexes)]
                else:
                    scores = m.predict_scores()

                scores = np.asarray(scores, dtype=np.float32).reshape(-1, 1)
                if scores.shape[0] != emb.shape[0]:
                    raise ValueError(f"Score/features length mismatch: scores={scores.shape}, emb={emb.shape}")
                scores_list.append(scores)
            else:
                scores_list.append(None)

            if raw is not None and raw.shape[0] != emb.shape[0]:
                raise ValueError(f"Raw/features length mismatch: raw={raw.shape}, emb={emb.shape}")

            logger.info(f"[{m.__class__.__name__}] Embeddings: {emb.shape}")

# NEW 
        if not embeddings:
            raise ValueError("No recurrent input collected")

        if self.recurrent_token_sequence:
            if self.embeddings_aggregate != "stack":
                raise ValueError("recurrent_token_sequence=True supports only embeddings_aggregate='stack'")
            return self._collect_token_sequence_input(raw, embeddings, scores_list)

        agg = self.embeddings_aggregate

        if agg == "stack":
            max_emb_d = max(e.shape[1] for e in embeddings)
            steps = []

            for emb, scores in zip(embeddings, scores_list):
                padded_emb = (
                    np.pad(emb, ((0, 0), (0, max_emb_d - emb.shape[1])))
                    if emb.shape[1] < max_emb_d else emb
                )

                parts = []
                if raw is not None:
                    parts.append(raw)
                parts.append(padded_emb)
                if scores is not None:
                    parts.append(scores)

                step = np.concatenate(parts, axis=1)
                steps.append(step)

            shapes = {s.shape for s in steps}
            if len(shapes) > 1:
                raise ValueError(f"Shape mismatch for stacking: {shapes}")

            recurrent_input = np.stack(steps, axis=1)
            logger.info(f"[RecurrentInput] shape={recurrent_input.shape}")
            return recurrent_input

        steps = []
        for emb, scores in zip(embeddings, scores_list):
            parts = []
            if raw is not None:
                parts.append(raw)
            parts.append(emb)
            if scores is not None:
                parts.append(scores)
            steps.append(np.concatenate(parts, axis=1))

        if agg == "concat":
            return np.concatenate(steps, axis=1)
        if agg == "mean":
            return np.mean(np.stack(steps, axis=0), axis=0)

        raise ValueError(f"Unsupported embeddings_aggregate: {agg}")

    def _collect_val_embeddings(self) -> Optional[np.ndarray]:
        if self.data.X_val is None or self.data.y_val is None:
            return None
        return self._collect_embeddings(indexes=None, use_train=False, use_val=True)

    def _compute_recurrent_val_loss_cached(self, val_emb: Optional[np.ndarray]) -> None:
        if not getattr(self.recurrent_model, "_fitted", False) or val_emb is None:
            return
        y_val = self.data.y_val[:len(val_emb)]
        val_loss = self.recurrent_model.get_loss(val_emb, y_val)
        if val_loss is not None:
            self.recurrent_model._val_loss_history.append(val_loss)

    def _current_pseudo_context_labels(self) -> np.ndarray:
        """Resolve this chapter's pseudo-labels by model majority vote"""
        labels = np.full(self.data.n_train, -1, dtype=int)
        pseudo_list = [m._pseudo_labels for m in self.models if m._pseudo_labels is not None]

        if not pseudo_list:
            return labels

        pa = np.array(pseudo_list)
        skip = np.zeros(self.data.n_train, dtype=bool)

        if getattr(self.data, "preserve_labeled", False):
            labeled = getattr(self.data, "labeled_indexes", None)
            if labeled is not None and len(labeled):
                skip[labeled] = True

        mask = np.any(pa != -1, axis=0) & ~skip

        for i in np.where(mask)[0]:
            votes = pa[:, i][pa[:, i] != -1]
            n0 = int(np.sum(votes == 0))
            n1 = int(np.sum(votes == 1))

            if n1 > n0:
                labels[i] = 1
            elif n0 > n1:
                labels[i] = 0
            else:
                labels[i] = -1

        return labels
    
    def _record_pseudo_context(self) -> None:
        """No-op: GRU uses current chapter pseudo-labels only"""
        return
 
    def _resolve_context_recurrent_targets(self, indexes: Optional[np.ndarray]):
        """Use real labels plus current chapter pseudo-labels"""
        if hasattr(self.data, "resolve_labels"):
            labels = self.data.resolve_labels(policy="unlabeled_as_is")
        elif getattr(self.data, "semisupervised_labels", None) is not None:
            labels = self.data.semisupervised_labels.copy()
        elif getattr(self.data, "y_train_original", None) is not None:
            labels = self.data.y_train_original.copy()
        else:
            return None, None

        labels = np.asarray(labels).copy()

        current = self._current_pseudo_context_labels()
        mask = current != -1

        if getattr(self.data, "preserve_labeled", False):
            labeled = getattr(self.data, "labeled_indexes", None)
            if labeled is not None and len(labeled):
                mask[labeled] = False

        labels[mask] = current[mask]

        if indexes is None:
            keep = np.where(labels != -1)[0]
        else:
            indexes = np.asarray(indexes)
            keep = indexes[labels[indexes] != -1]

        logger.info(f"[PseudoContext] kept {len(keep)} recurrent labels from current chapter only")
        return keep, labels[keep]

    def _train_recurrent(self, chapter: int) -> None:
        indexes = self._get_embedding_indexes()
        indexes, labels = self._resolve_context_recurrent_targets(indexes)

        if labels is None or indexes is None or len(indexes) == 0:
            logger.warning("[Recurrent] No stable labels; skipping")
            return

        agg_emb = self._collect_embeddings(indexes)
        val_emb = self._collect_val_embeddings()

        for _ in range(getattr(self.recurrent_model, "num_epochs", 1)):
            self.recurrent_model.train(agg_emb, labels, epochs=1)
            self._compute_recurrent_val_loss_cached(val_emb)

    def _mark_recurrent_started(self, chapter: int, reason: str) -> None:
        self._recurrent_started = True
        self._recurrent_start_actual_chapter = chapter
        if self.strategy is not None and hasattr(self.strategy, "recurrent_start_chapter"):
            self.strategy.recurrent_start_chapter = chapter
        logger.info(f"[Recurrent] {reason} at chapter {chapter + 1}")

    def _should_start_recurrent(self, chapter: int) -> bool:
        """Check if validation ensemble AUC is stable enough to start GRU"""
        if self._recurrent_started:
            return False
        if chapter < self.recurrent_start_chapter:
            return False

        if self.adaptive_max_wait_chapter is not None and chapter >= self.adaptive_max_wait_chapter:
            self._mark_recurrent_started(chapter, "Force-start")
            return True

        if len(self._val_ensemble_auc_history) < self.adaptive_start_window:
            return False

        recent = self._val_ensemble_auc_history[-self.adaptive_start_window:]
        if max(recent) - min(recent) <= self.adaptive_start_tolerance:
            self._mark_recurrent_started(chapter, "Adaptive start")
            return True

        return False

    def _evaluate_base_models(self, y_val: np.ndarray) -> Dict[str, float]:
        val_scores, metrics = [], {}
        for i, m in enumerate(self.models):
            sc = _predict_scores_on_val(m, self.data)
            val_scores.append(sc)
            try:
                metrics[f"model_{i}"] = _compute_auc_or_raise(y_val, sc)
            except ValueError:
                metrics[f"model_{i}"] = 0.5

        try:
            metrics["ensemble"] = _compute_auc_or_raise(y_val, self._adaptive_ensemble_scores(val_scores))
        except ValueError:
            metrics["ensemble"] = 0.5

        return metrics

    # def _evaluate_with_gru(self, y_val: np.ndarray, chapter: int) -> Dict[str, float]:
    #     metrics = self._evaluate_base_models(y_val)

    #     if self._recurrent_started and getattr(self.recurrent_model, "_fitted", False):
    #         val_emb = self._collect_val_embeddings()
    #         if val_emb is not None:
    #             try:
    #                 metrics["gru"] = roc_auc_score(y_val, self.recurrent_model.predict_scores(val_emb))
    #             except Exception:
    #                 metrics["gru"] = 0.5

    #     return metrics
    
    def _evaluate_with_gru(self, y_val: np.ndarray, chapter: int) -> Dict[str, float]:
        metrics = self._evaluate_base_models(y_val)

        if self._recurrent_started and getattr(self.recurrent_model, "_fitted", False):
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


# NEW evening

    def _select_final_blend(self, y_val: np.ndarray) -> None:
        """Select final GRU/ensemble blend by validation AUC"""
        if not self._recurrent_started or not getattr(self.recurrent_model, "_fitted", False):
            return

        val_emb = self._collect_val_embeddings()
        if val_emb is None:
            return

        gru_scores = self.recurrent_model.predict_scores(val_emb)
        ensemble_scores = self._adaptive_ensemble_scores(
            [_predict_scores_on_val(m, self.data) for m in self.models]
        )

        best_alpha = 1.0
        best_auc = _compute_auc_or_raise(y_val, gru_scores)

        for alpha in self.final_score_blend_grid:
            blended = alpha * gru_scores + (1.0 - alpha) * ensemble_scores
            try:
                auc = _compute_auc_or_raise(y_val, blended)
            except ValueError:
                continue

            if auc > best_auc:
                best_alpha = float(alpha)
                best_auc = float(auc)

        self._best_blend_alpha = best_alpha
        self._best_blend_val_auc = best_auc
        logger.info(
            f"[Recurrent] Selected final blend alpha={best_alpha:.2f}, "
            f"val AUC={best_auc:.4f}"
        )


    def predict_final_scores(self) -> np.ndarray:
        """Return validation-selected final GRU/ensemble scores on test data"""
        if not getattr(self.recurrent_model, "_fitted", False):
            raise RuntimeError("Recurrent model must be fitted before final scoring")

        test_emb = self._collect_embeddings(indexes=None, use_train=False)
        gru_scores = self.recurrent_model.predict_scores(test_emb)
        ensemble_scores = self._adaptive_ensemble_scores(
            [m.predict_scores() for m in self.models]
        )

        alpha = self._best_blend_alpha
        return alpha * gru_scores + (1.0 - alpha) * ensemble_scores


    def cotrain(self, eval_interval: int = 1):
        history = {"warmup": [], "chapters": []}
        y_val = self.data.y_val
        if y_val is None or len(y_val) == 0:
            raise ValueError("Validation set required for DelayedRecurrentBasicAdaptiveCoLearner")

        for i in range(len(self.models)):
            self.val_loss_history[i] = []

        self._warmup_with_val()
        logger.info(f"[Collab] Up to {self.max_chapters} chapters ({self.epochs_per_chapter} ep/ch)")

        for ch in range(self.max_chapters):
            self.exchange()
            self._record_pseudo_context()

            for _ in range(self.epochs_per_chapter):
                for m in self.models:
                    m.train(1)
                for i, m in enumerate(self.models):
                    vl = _compute_val_loss(m, self.data)
                    if vl is not None:
                        self.val_loss_history[i].append(vl)

            metrics = self._evaluate_base_models(y_val)
            self._val_ensemble_auc_history.append(metrics["ensemble"])

            # if self._should_start_recurrent(ch) or self._recurrent_started:
            #     self._train_recurrent(ch)
            #     metrics = self._evaluate_with_gru(y_val, ch)

            # if (ch + 1) % eval_interval == 0:
            #     self._log(ch, metrics)

            # history["chapters"].append(metrics)
            
            if self._should_start_recurrent(ch) or self._recurrent_started:
                self._train_recurrent(ch)
                metrics = self._evaluate_with_gru(y_val, ch)
                if "gru" in metrics:
                    improved = self.recurrent_model.update_best(metrics["gru"])
                    if improved:
                        logger.info(
                            f"[Recurrent] New best val AUC={metrics['gru']:.4f} "
                            f"at chapter {ch + 1}"
                        )

            if (ch + 1) % eval_interval == 0:
                self._log(ch, metrics)

            history["chapters"].append(metrics)

            if self.strategy is not None and not self.strategy.should_continue(metrics, ch):
                waiting = (
                    not self._recurrent_started
                    and self.adaptive_max_wait_chapter is not None
                    and ch < self.adaptive_max_wait_chapter
                )
                if waiting:
                    logger.info(
                        f"  Holding convergence until recurrent start "
                        f"by chapter {self.adaptive_max_wait_chapter + 1}"
                    )
                else:
                    logger.info(f"  Converged at chapter {ch + 1}")
                    break

        # logger.info("[Done]")
        # return history
# 2nd iteration
        # if self._recurrent_started and self.recurrent_model.restore_best():
        #     logger.info(
        #         f"[Recurrent] Restored best validation checkpoint "
        #         f"AUC={self.recurrent_model._best_val_auc:.4f}"
        #     )

        # logger.info("[Done]")
        # return history
        
        if self._recurrent_started:
            if self.recurrent_model.restore_best():
                logger.info(
                    f"[Recurrent] Restored best validation checkpoint "
                    f"AUC={self.recurrent_model._best_val_auc:.4f}"
                )
            self._select_final_blend(y_val)

        logger.info("[Done]")
        return history