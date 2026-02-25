from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

UNLABELED_POLICIES = ("unlabeled_as_normal", "unlabeled_as_is", "unlabeled_as_removed")


@dataclass
class PseudoLabelProposal:
    """One model's proposed label for one sample."""
    sender_idx: int
    label: int          # 0 or 1
    confidence: float   # distance from threshold (higher = more confident)


def default_arbiter(proposals: List[PseudoLabelProposal]) -> Optional[int]:
    """Pick the proposal with highest confidence (threshold-margin). Returns 0/1 or None."""
    if not proposals:
        return None
    best = max(proposals, key=lambda p: p.confidence)
    return best.label


class Model(ABC):
    """Base class for anomaly detection models supporting collaborative learning.
    
    Accepts data as either:
    - A Data subclass instance (preferred): Uses object attributes directly
    - A dict-like object: Uses dictionary-style access for backwards compatibility
    """
    
    def __init__(self, train_config: Optional[Dict] = None, model_config: Optional[Dict] = None, 
                 data: Optional['Data'] = None):
        self.train_config = train_config or {}
        self.model_config = model_config or {}
        self.data = data
        self._fitted = False
        self._current_epoch = 0
        self._pseudo_labels = None
        self._pseudo_label_meta: Optional[Dict[int, PseudoLabelProposal]] = None  # idx → winning proposal
        self._y_train_cache: Optional[np.ndarray] = None
        
        # Per-model default transfer thresholds (fallback before CoLearner defaults)
        self.default_confidence_high: Optional[float] = None
        self.default_confidence_low: Optional[float] = None
        
        # Initialize data attributes
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.y_train_original: Optional[np.ndarray] = None
        
        if data is not None:
            self._extract_data(data)
    
    def _get_value(self, data, *keys, required: bool = True):
        """Extract value from data trying multiple keys (attribute then dict access)."""
        for key in keys:
            if hasattr(data, key) and getattr(data, key) is not None:
                return getattr(data, key)
            if hasattr(data, '__getitem__'):
                try:
                    val = data[key]
                    if val is not None:
                        return val
                except (KeyError, TypeError):
                    pass
            if hasattr(data, 'get') and data.get(key) is not None:
                return data.get(key)
        if required:
            raise KeyError(f"Could not find any of {keys} in data")
        return None
    
    def _extract_data(self, data) -> None:
        self.X_train = self._get_value(data, 'X_train')
        self.X_test = self._get_value(data, 'X_test')
        self.X_val = self._get_value(data, 'X_val', required=False)
        self.y_test = self._get_value(data, 'y_test')
        self.y_val = self._get_value(data, 'y_val', required=False)
        self.y_train_original = self._get_value(data, 'y_train_original', 'y_train')
            
    @property
    def y_train(self) -> np.ndarray:
        """Cached training labels: pseudo_labels → resolved semisupervised_labels → original. Read-only."""
        if self._y_train_cache is not None:
            return self._y_train_cache

        if self.data is not None and hasattr(self.data, 'resolve_labels'):
            base_labels = self.data.resolve_labels()
        elif self.data is not None and getattr(self.data, 'semisupervised_labels', None) is not None:
            base_labels = self.data.semisupervised_labels.copy()
        else:
            base_labels = self.y_train_original.copy() if self.y_train_original is not None else None

        if base_labels is None:
            return None

        if self._pseudo_labels is not None:
            mask = self._pseudo_labels != -1
            if self.data is not None and getattr(self.data, 'preserve_labeled', False):
                labeled_idx = getattr(self.data, 'labeled_indexes', None)
                if labeled_idx is not None and len(labeled_idx) > 0:
                    mask[labeled_idx] = False
            base_labels[mask] = self._pseudo_labels[mask]

        base_labels.flags.writeable = False
        self._y_train_cache = base_labels
        return self._y_train_cache
    
    def set_pseudo_labels(self, labels: np.ndarray) -> None:
        self._pseudo_labels = labels.copy()
        self._y_train_cache = None  # invalidate
    
    def clear_pseudo_labels(self) -> None:
        self._pseudo_labels = None
        self._pseudo_label_meta = None
        self._y_train_cache = None  # invalidate
    
    @abstractmethod
    def train(self, epochs: int = 1) -> None:
        """Train for 'epochs' epochs using current data and pseudo-labels"""
        pass 

    @abstractmethod
    def fit(self) -> None:
        """Train to convergence"""
        pass

    @abstractmethod
    def predict_scores(self, indexes: Optional[np.ndarray] = None, use_train: bool = False) -> np.ndarray:
        """Return anomaly scores [0,1] where higher = more anomalous."""
        pass 

    @abstractmethod
    def get_embeddings(self, indexes: Optional[np.ndarray] = None, use_train: bool = True) -> np.ndarray:
        """Extract learned representations for collaborative learning."""
        pass
    
    def get_loss(self, use_val: bool = False) -> Optional[float]:
        return None


class RecurrentModel(ABC):
    """Ensemble model trained on aggregated embeddings (future project)."""
    
    @abstractmethod
    def train(self, aggregated_embeddings: np.ndarray, labels: np.ndarray, epochs: int = 1) -> None:
        pass

    @abstractmethod
    def predict_scores(self, aggregated_embeddings: np.ndarray) -> np.ndarray:
        pass

    def get_loss(self, aggregated_embeddings: np.ndarray, labels: np.ndarray) -> Optional[float]:
        return None


class Data(ABC):
    """Unified dataset interface managing indices, splits, and pseudo-labels."""
    
    def __init__(self, dataset, train_test_split: float = 0.8, random_state: int = 42, 
                 preserve_labeled: bool = False, data_type: str = "tabular", 
                 val_test_split: float = 0.0,
                 labeled_ratio: float = 0.1, stratified: bool = True,
                 max_anomalies: Optional[int] = None, anomaly_ratio: Optional[float] = None,
                 unlabeled_policy: str = "unlabeled_as_normal"):
        self.dataset = dataset
        self.train_test_split = train_test_split
        self.val_test_split = val_test_split
        self.random_state = random_state
        self.preserve_labeled = preserve_labeled
        self.data_type = data_type
        self.unlabeled_policy = unlabeled_policy
        
        self.n_samples = None
        self.n_train = None
        self.n_test = None
        self.n_val = None
        self.train_indexes = None
        self.test_indexes = None
        self.val_indexes = None
        
        self.labeled_indexes = None
        self.unlabeled_indexes = None
        self.labeled_ratio = None
        self.semisupervised_labels = None
        
        self.pseudo_labels_by_model: Dict[str, np.ndarray] = {}
        self.pseudo_label_confidence: Dict[str, np.ndarray] = {}
        self.y_train_original = None
        
        # Load data (subclass) then partition train into labeled/unlabeled
        self._load()
        self._init_semisupervised(labeled_ratio, stratified, max_anomalies, anomaly_ratio)
    
    @abstractmethod
    def _load(self) -> None:
        """Load data and create train/test/val splits. Must set X_train, X_test, y_test, y_train_original, n_train, n_test."""
        pass
    
    def _init_semisupervised(self, labeled_ratio: float, stratified: bool,
                             max_anomalies: Optional[int], anomaly_ratio: Optional[float]) -> None:
        """Partition train set into labeled/unlabeled for semi-supervised learning."""
        if self.y_train_original is None:
            raise ValueError("_load() must set y_train_original first")
        if self.unlabeled_policy not in UNLABELED_POLICIES:
            raise ValueError(f"Unknown policy '{self.unlabeled_policy}', choose from {UNLABELED_POLICIES}")
    
        from sklearn.model_selection import train_test_split
        
        self.labeled_ratio = labeled_ratio
        rng = np.random.RandomState(self.random_state)
        normal_idx = np.where(self.y_train_original == 0)[0]
        anomaly_idx = np.where(self.y_train_original == 1)[0]
        
        if max_anomalies is not None or anomaly_ratio is not None:
            n_vis = min(max_anomalies, len(anomaly_idx)) if max_anomalies else max(1, int(len(anomaly_idx) * anomaly_ratio))
            vis_anom = rng.choice(anomaly_idx, size=n_vis, replace=False)
            n_total = max(2, int(self.n_train * labeled_ratio))
            n_norm = max(1, n_total - n_vis)
            vis_norm = rng.choice(normal_idx, size=min(n_norm, len(normal_idx)), replace=False)
            self.labeled_indexes = np.sort(np.concatenate([vis_norm, vis_anom]))
            self.unlabeled_indexes = np.sort(np.setdiff1d(np.arange(self.n_train), self.labeled_indexes))
        else:
            n_labeled = max(2, int(self.n_train * labeled_ratio))
            self.labeled_indexes, self.unlabeled_indexes = train_test_split(
                np.arange(self.n_train), train_size=n_labeled,
                stratify=self.y_train_original if stratified else None, random_state=self.random_state)
            self.labeled_indexes = np.sort(self.labeled_indexes)
            self.unlabeled_indexes = np.sort(self.unlabeled_indexes)
    
        # Guardrail: ensure at least 1 anomaly and 1 normal labeled
        n_anom = np.sum(self.y_train_original[self.labeled_indexes] == 1)
        n_norm = np.sum(self.y_train_original[self.labeled_indexes] == 0)
        if n_anom == 0 and len(anomaly_idx) > 0:
            logger.warning("No labeled anomalies — injecting 1 random anomaly into labeled set")
            pick = rng.choice(anomaly_idx, 1)
            self.labeled_indexes = np.sort(np.union1d(self.labeled_indexes, pick))
            self.unlabeled_indexes = np.sort(np.setdiff1d(np.arange(self.n_train), self.labeled_indexes))
        if n_norm == 0 and len(normal_idx) > 0:
            logger.warning("No labeled normals — injecting 1 random normal into labeled set")
            pick = rng.choice(normal_idx, 1)
            self.labeled_indexes = np.sort(np.union1d(self.labeled_indexes, pick))
            self.unlabeled_indexes = np.sort(np.setdiff1d(np.arange(self.n_train), self.labeled_indexes))

        self.semisupervised_labels = np.full(self.n_train, -1, dtype=np.int32)
        self.semisupervised_labels[self.labeled_indexes] = self.y_train_original[self.labeled_indexes]
        
        n_anom = np.sum(self.y_train_original[self.labeled_indexes] == 1)
        n_norm = np.sum(self.y_train_original[self.labeled_indexes] == 0)
        logger.info(f"[Semi-supervised] Labeled: {len(self.labeled_indexes)} ({n_norm} normal, {n_anom} anomaly), "
                    f"Unlabeled: {len(self.unlabeled_indexes)}, Policy: {self.unlabeled_policy}")

    def resolve_labels(self, policy: str = None) -> np.ndarray:
        """Return training labels with unlabeled (-1) resolved per policy."""
        policy = policy or getattr(self, 'unlabeled_policy', 'unlabeled_as_normal')
        labels = self.semisupervised_labels if self.semisupervised_labels is not None else self.y_train_original
        if labels is None:
            raise ValueError("No training labels available")
        out = labels.copy()
        if policy == "unlabeled_as_normal":
            out[out == -1] = 0
        elif policy == "unlabeled_as_is":
            pass
        elif policy == "unlabeled_as_removed":
            pass  # caller uses labeled_indexes to subset X
        else:
            raise ValueError(f"Unknown policy: {policy}")
        return out
    
class Strategy(ABC):
    """Convergence strategy for handling asynchronous model training."""
    
    @abstractmethod
    def should_continue(self, model_metrics: Dict[str, float], chapter: int) -> bool:
        """Determine if training should continue."""
        pass
    @abstractmethod
    def reset(self):
        """reset the should_continue status."""


class CoLearning(ABC):
    """Base class for multi-model collaborative anomaly detection """

    def __init__(self, models: List[Model], data: Data, strategy: Optional[Strategy],
                 warmup_epochs: int = 10, max_chapters: int = 10,
                 anomaly_threshold: float = 0.5, confidence_threshold_low: float = 0.02,
                 confidence_threshold_high: float = 0.98,
                 transfer_thresholds: Optional[Dict] = None,
                 pseudo_label_arbiter: Optional[Callable[[List[PseudoLabelProposal]], Optional[int]]] = None):
        self.models = models
        self.data = data
        self.strategy = strategy
        self.warmup_epochs = warmup_epochs
        self.max_chapters = max_chapters
        self.anomaly_threshold = anomaly_threshold
        self.confidence_threshold_low = confidence_threshold_low
        self.confidence_threshold_high = confidence_threshold_high
        self.transfer_thresholds = transfer_thresholds or {} # example : {(0,1) : {"high":None, "low": 0.02,model 0 sends no anomalies to 1
        self.training_history: List[Dict] = []               #             (0,2) : {"high":0.6, "low":0.02}} model 0 sends score>0.6 as anomalies to 2
        self.pseudo_label_arbiter = pseudo_label_arbiter or default_arbiter

    def _resolve_threshold(self, sender_idx: int, receiver_idx: int, kind: str) -> Optional[float]:
        """Resolve threshold: pair override > sender model default > colearner default | None = disabled"""
        pair = (sender_idx, receiver_idx)
        if pair in self.transfer_thresholds:
            pair_cfg = self.transfer_thresholds[pair]
            if kind in pair_cfg:
                return pair_cfg[kind]  # None here means explicitly disabled
        sender = self.models[sender_idx]
        if kind == "high":
            model_val = sender.default_confidence_high
            return model_val if model_val is not None else self.confidence_threshold_high
        model_val = sender.default_confidence_low
        return model_val if model_val is not None else self.confidence_threshold_low

    def _propose_labels(self, sender_idx: int, receiver_idx: int,
                        indexes: np.ndarray, scores: np.ndarray,
                        label: int, kind: str) -> int:
        """Accumulate proposals for receiver; arbiter resolves conflicts later. Returns count proposed."""
        if len(indexes) == 0:
            return 0
        threshold = self._resolve_threshold(sender_idx, receiver_idx, kind)
        if threshold is None:
            return 0
        if kind == "high":
            mask = scores[indexes] > threshold
            confidences = scores[indexes][mask] - threshold
        else:
            mask = scores[indexes] < threshold
            confidences = threshold - scores[indexes][mask]
        idxs = indexes[mask]
        if len(idxs) == 0:
            return 0
        receiver = self.models[receiver_idx]
        if receiver._pseudo_label_meta is None:
            receiver._pseudo_label_meta = {}
        for j, idx in enumerate(idxs):
            proposal = PseudoLabelProposal(sender_idx=sender_idx, label=label, confidence=float(confidences[j]))
            receiver._pseudo_label_meta.setdefault(int(idx), []).append(proposal)
        return len(idxs)

    def _finalize_pseudo_labels(self) -> None:
        """Resolve accumulated proposals via arbiter and write final pseudo-labels."""
        for receiver in self.models:
            if receiver._pseudo_label_meta is None:
                continue
            receiver._pseudo_labels = np.full(self.data.n_train, -1, dtype=np.int32)
            resolved_meta = {}
            for idx, proposals in receiver._pseudo_label_meta.items():
                decision = self.pseudo_label_arbiter(proposals)
                if decision is not None:
                    receiver._pseudo_labels[idx] = decision
                    best = max(proposals, key=lambda p: p.confidence)
                    resolved_meta[idx] = best
            receiver._pseudo_label_meta = resolved_meta
            receiver._y_train_cache = None  # invalidate after direct write

    def send_anomalies(self, sender_idx: int, receiver_idx: int,
                       indexes: np.ndarray, scores: np.ndarray) -> int:
        """Propose high-confidence anomaly pseudo-labels from sender to receiver."""
        return self._propose_labels(sender_idx, receiver_idx, indexes, scores, label=1, kind="high")

    def send_normals(self, sender_idx: int, receiver_idx: int,
                     indexes: np.ndarray, scores: np.ndarray) -> int:
        """Propose high-confidence normal pseudo-labels from sender to receiver."""
        return self._propose_labels(sender_idx, receiver_idx, indexes, scores, label=0, kind="low")

    @abstractmethod
    def exchange(self) -> None:
        pass

    @abstractmethod
    def cotrain(self, eval_interval: int = 1) -> Dict[str, List[float]]:
        pass



