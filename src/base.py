from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import logging
import numpy as np

logger = logging.getLogger(__name__)

UNLABELED_POLICIES = ("unlabeled_as_normal", "unlabeled_as_is", "unlabeled_as_removed")


@dataclass
class PseudoLabelProposal:
    sender_idx: int
    label: int          # 0 or 1
    confidence: float   # distance from threshold


def default_arbiter(proposals: List[PseudoLabelProposal]) -> Optional[int]:
    if not proposals:
        return None
    return max(proposals, key=lambda p: p.confidence).label


class Model(ABC):
    """Base anomaly detector with incremental training and pseudo-label support."""

    def __init__(self, train_config: Optional[Dict] = None,
                 model_config: Optional[Dict] = None,
                 data: Optional['Data'] = None):
        self.train_config = train_config or {}
        self.model_config = model_config or {}
        self.data = data
        self._fitted = False
        self._current_epoch = 0
        self._pseudo_labels = None
        self._pseudo_label_meta: Optional[Dict[int, PseudoLabelProposal]] = None
        self._y_train_cache: Optional[np.ndarray] = None

        # Per-model transfer thresholds (fallback before CoLearner defaults)
        self.default_confidence_high: Optional[float] = None
        self.default_confidence_low: Optional[float] = None

        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.y_train_original: Optional[np.ndarray] = None

        if data is not None:
            self._extract_data(data)

    @staticmethod
    def _get(obj, *keys, required=True):
        for k in keys:
            val = getattr(obj, k, None)
            if val is not None:
                return val
            try:
                val = obj[k]
                if val is not None:
                    return val
            except (KeyError, TypeError, AttributeError):
                pass
        if required:
            raise KeyError(f"Missing: {keys}")
        return None

    def _extract_data(self, data) -> None:
        self.X_train = self._get(data, 'X_train')
        self.X_test = self._get(data, 'X_test')
        self.X_val = self._get(data, 'X_val', required=False)
        self.y_test = self._get(data, 'y_test')
        self.y_val = self._get(data, 'y_val', required=False)
        self.y_train_original = self._get(data, 'y_train_original', 'y_train')

    @property
    def y_train(self) -> np.ndarray:
        """Resolved labels: pseudo > semisupervised > original. Cached & read-only."""
        if self._y_train_cache is not None:
            return self._y_train_cache

        if self.data is not None and hasattr(self.data, 'resolve_labels'):
            base = self.data.resolve_labels()
        elif self.data is not None and getattr(self.data, 'semisupervised_labels', None) is not None:
            base = self.data.semisupervised_labels.copy()
        else:
            base = self.y_train_original.copy() if self.y_train_original is not None else None
        if base is None:
            return None

        # Overlay pseudo-labels, protecting ground-truth labeled samples
        if self._pseudo_labels is not None:
            mask = self._pseudo_labels != -1
            if getattr(self.data, 'preserve_labeled', False):
                labeled = getattr(self.data, 'labeled_indexes', None)
                if labeled is not None and len(labeled):
                    mask[labeled] = False
            base[mask] = self._pseudo_labels[mask]

        base.flags.writeable = False
        self._y_train_cache = base
        return self._y_train_cache

    def set_pseudo_labels(self, labels: np.ndarray) -> None:
        self._pseudo_labels = labels.copy()
        self._y_train_cache = None

    def clear_pseudo_labels(self) -> None:
        self._pseudo_labels = None
        self._pseudo_label_meta = None
        self._y_train_cache = None

    @abstractmethod
    def train(self, epochs: int = 1) -> None: ...

    @abstractmethod
    def fit(self) -> None: ...

    @abstractmethod
    def predict_scores(self, indexes: Optional[np.ndarray] = None,
                       use_train: bool = False, use_val: bool = False) -> np.ndarray:
        """Return anomaly scores in [0,1]. use_train=True scores training data; use_val=True scores validation data."""

    @abstractmethod
    def get_embeddings(self, indexes: Optional[np.ndarray] = None,
                       use_train: bool = True) -> np.ndarray: ...

    def get_loss(self, use_val: bool = False) -> Optional[float]:
        return None


class RecurrentModel(ABC):
    """Sequence model over aggregated detector embeddings."""

    @abstractmethod
    def train(self, aggregated_embeddings: np.ndarray, labels: np.ndarray,
              epochs: int = 1) -> None: ...

    @abstractmethod
    def predict_scores(self, aggregated_embeddings: np.ndarray) -> np.ndarray: ...

    def get_loss(self, aggregated_embeddings: np.ndarray,
                 labels: np.ndarray) -> Optional[float]:
        return None


class Data(ABC):
    """Dataset with train/val/test splits and labeled/unlabeled partitioning."""

    def __init__(self, dataset, train_test_split: float = 0.8, random_state: int = 42,
                 preserve_labeled: bool = False, data_type: str = "tabular",
                 val_test_split: float = 0.0, labeled_ratio: float = 0.1,
                 stratified: bool = True, max_anomalies: Optional[int] = None,
                 anomaly_ratio: Optional[float] = None,
                 unlabeled_policy: str = "unlabeled_as_normal"):
        self.dataset = dataset
        self.train_test_split = train_test_split
        self.val_test_split = val_test_split
        self.random_state = random_state
        self.preserve_labeled = preserve_labeled
        self.data_type = data_type
        self.unlabeled_policy = unlabeled_policy

        self.n_samples = self.n_train = self.n_test = self.n_val = None
        self.train_indexes = self.test_indexes = self.val_indexes = None

        self.labeled_indexes = self.unlabeled_indexes = None
        self.labeled_ratio = None
        self.semisupervised_labels = None
        self.pseudo_labels_by_model: Dict[str, np.ndarray] = {}
        self.pseudo_label_confidence: Dict[str, np.ndarray] = {}
        self.y_train_original = None

        self._load()
        self._init_semisupervised(labeled_ratio, stratified, max_anomalies, anomaly_ratio)

    @abstractmethod
    def _load(self) -> None:
        """Must set X_train, X_test, y_test, y_train_original, n_train, n_test."""

    def _init_semisupervised(self, labeled_ratio: float, stratified: bool,
                             max_anomalies: Optional[int],
                             anomaly_ratio: Optional[float]) -> None:
        if self.y_train_original is None:
            raise ValueError("_load() must set y_train_original")
        if self.unlabeled_policy not in UNLABELED_POLICIES:
            raise ValueError(f"Unknown policy '{self.unlabeled_policy}', use {UNLABELED_POLICIES}")

        from sklearn.model_selection import train_test_split

        self.labeled_ratio = labeled_ratio
        rng = np.random.RandomState(self.random_state)
        normal_idx = np.where(self.y_train_original == 0)[0]
        anomaly_idx = np.where(self.y_train_original == 1)[0]

        if max_anomalies is not None or anomaly_ratio is not None:
            n_vis = (min(max_anomalies, len(anomaly_idx)) if max_anomalies
                     else max(1, int(len(anomaly_idx) * anomaly_ratio)))
            vis_anom = rng.choice(anomaly_idx, size=n_vis, replace=False)
            n_norm = max(1, max(2, int(self.n_train * labeled_ratio)) - n_vis)
            vis_norm = rng.choice(normal_idx, size=min(n_norm, len(normal_idx)), replace=False)
            self.labeled_indexes = np.sort(np.concatenate([vis_norm, vis_anom]))
        else:
            n_labeled = max(2, int(self.n_train * labeled_ratio))
            self.labeled_indexes, _ = train_test_split(
                np.arange(self.n_train), train_size=n_labeled,
                stratify=self.y_train_original if stratified else None,
                random_state=self.random_state)
            self.labeled_indexes = np.sort(self.labeled_indexes)

        self.unlabeled_indexes = np.sort(np.setdiff1d(np.arange(self.n_train), self.labeled_indexes))

        # Inject missing class if needed (e.g. tiny labeled set has no anomalies)
        for target, pool, name in [(1, anomaly_idx, "anomaly"), (0, normal_idx, "normal")]:
            if np.sum(self.y_train_original[self.labeled_indexes] == target) == 0 and len(pool):
                logger.warning(f"No labeled {name}s - injecting 1")
                self.labeled_indexes = np.sort(np.union1d(self.labeled_indexes, rng.choice(pool, 1)))
                self.unlabeled_indexes = np.sort(np.setdiff1d(np.arange(self.n_train), self.labeled_indexes))

        self.semisupervised_labels = np.full(self.n_train, -1, dtype=np.int32)
        self.semisupervised_labels[self.labeled_indexes] = self.y_train_original[self.labeled_indexes]

        n_a = np.sum(self.y_train_original[self.labeled_indexes] == 1)
        n_n = np.sum(self.y_train_original[self.labeled_indexes] == 0)
        logger.info(f"[Semi-sup] Labeled={len(self.labeled_indexes)} ({n_n}N/{n_a}A), "
                    f"Unlabeled={len(self.unlabeled_indexes)}, Policy={self.unlabeled_policy}")

    def resolve_labels(self, policy: str = None) -> np.ndarray:
        """Return training labels with unlabeled (-1) resolved per policy."""
        policy = policy or self.unlabeled_policy
        labels = self.semisupervised_labels if self.semisupervised_labels is not None else self.y_train_original
        if labels is None:
            raise ValueError("No training labels available")
        out = labels.copy()
        if policy == "unlabeled_as_normal":
            out[out == -1] = 0
        elif policy in ("unlabeled_as_is", "unlabeled_as_removed"):
            pass
        else:
            raise ValueError(f"Unknown policy: {policy}")
        return out


class Strategy(ABC):

    @abstractmethod
    def should_continue(self, model_metrics: Dict[str, float], chapter: int) -> bool: ...

    @abstractmethod
    def reset(self): ...


class CoLearning(ABC):
    """Multi-model collaborative AD with pseudo-label exchange."""

    def __init__(self, models: List[Model], data: Data, strategy: Optional[Strategy],
                 warmup_epochs: int = 10, max_chapters: int = 10,
                 anomaly_threshold: float = 0.5,
                 confidence_threshold_low: float = 0.02,
                 confidence_threshold_high: float = 0.98,
                 transfer_thresholds: Optional[Dict] = None,
                 pseudo_label_arbiter: Optional[Callable] = None):
        self.models = models
        self.data = data
        self.strategy = strategy
        self.warmup_epochs = warmup_epochs
        self.max_chapters = max_chapters
        self.anomaly_threshold = anomaly_threshold
        self.confidence_threshold_low = confidence_threshold_low
        self.confidence_threshold_high = confidence_threshold_high
        self.transfer_thresholds = transfer_thresholds or {}
        self.training_history: List[Dict] = []
        self.pseudo_label_arbiter = pseudo_label_arbiter or default_arbiter

    def _resolve_threshold(self, sender_idx: int, receiver_idx: int, kind: str) -> Optional[float]:
        # Priority: pair override > sender default > colearner default
        pair = (sender_idx, receiver_idx)
        if pair in self.transfer_thresholds and kind in self.transfer_thresholds[pair]:
            return self.transfer_thresholds[pair][kind]
        sender = self.models[sender_idx]
        if kind == "high":
            return sender.default_confidence_high or self.confidence_threshold_high
        return sender.default_confidence_low or self.confidence_threshold_low

    def _propose_labels(self, sender_idx: int, receiver_idx: int,
                        indexes: np.ndarray, scores: np.ndarray,
                        label: int, kind: str) -> int:
        if len(indexes) == 0:
            return 0
        threshold = self._resolve_threshold(sender_idx, receiver_idx, kind)
        if threshold is None:
            return 0

        s = scores[indexes]
        mask = (s > threshold) if kind == "high" else (s < threshold)
        confidences = np.abs(s[mask] - threshold)
        idxs = indexes[mask]
        if len(idxs) == 0:
            return 0

        receiver = self.models[receiver_idx]
        if receiver._pseudo_label_meta is None:
            receiver._pseudo_label_meta = {}
        for j, idx in enumerate(idxs):
            prop = PseudoLabelProposal(sender_idx=sender_idx, label=label,
                                       confidence=float(confidences[j]))
            receiver._pseudo_label_meta.setdefault(int(idx), []).append(prop)
        return len(idxs)

    def _finalize_pseudo_labels(self) -> None:
        for receiver in self.models:
            if receiver._pseudo_label_meta is None:
                continue
            receiver._pseudo_labels = np.full(self.data.n_train, -1, dtype=np.int32)
            resolved = {}
            for idx, proposals in receiver._pseudo_label_meta.items():
                decision = self.pseudo_label_arbiter(proposals)
                if decision is not None:
                    receiver._pseudo_labels[idx] = decision
                    resolved[idx] = max(proposals, key=lambda p: p.confidence)
            receiver._pseudo_label_meta = resolved
            receiver._y_train_cache = None

    def send_anomalies(self, sender_idx, receiver_idx, indexes, scores) -> int:
        return self._propose_labels(sender_idx, receiver_idx, indexes, scores, label=1, kind="high")

    def send_normals(self, sender_idx, receiver_idx, indexes, scores) -> int:
        return self._propose_labels(sender_idx, receiver_idx, indexes, scores, label=0, kind="low")

    @abstractmethod
    def exchange(self) -> None: ...

    @abstractmethod
    def cotrain(self, eval_interval: int = 1) -> Dict[str, List[float]]: ...
