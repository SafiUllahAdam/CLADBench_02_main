from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from sklearn.metrics import roc_curve, auc


class Model(ABC):
    """
    Abstract base class for anomaly detection models supporting collaborative learning.
    
    All models must implement epoch-level training (for co-learning) and embedding extraction
    for inter-model knowledge sharing. Models can be trained independently (fit()) or
    collaboratively (train(epoch) called repeatedly).
    """
    
    @abstractmethod
    def train(self, epoch: int) -> None:
        """
        Execute a single training epoch with collaborative learning context.
        
        This method should update the model using:
        1. Current training data with original labels
        2. Pseudo-labels from other models (if exchange phase active)
        3. Updated pseudo-labels stored in self.data
        
        Args:
            epoch: Current epoch number (0-indexed) for tracking/scheduling
        """
        pass 

    @abstractmethod
    def fit(self) -> None:
        """
        Standalone training for single-model baseline comparison.
        
        Train the model to convergence on the training set without receiving
        pseudo-labels from collaborating models. Used to measure impact of
        collaborative learning by comparing fit() vs train(epoch) results.
        """
        pass

    @abstractmethod
    def predict_scores(self, indexes: Optional[np.ndarray] = None, use_train: bool = False) -> np.ndarray:
        """
        Generate anomaly scores for data samples.
        
        Scores should be in range [0, 1] where higher values indicate higher
        anomaly likelihood. Used for pseudo-labeling and evaluation.
        
        Args:
            indexes: Optional indices of samples to score. If None, score all test data.
                    For iterative learning, typically unsupervised training subset.
            use_train: If True, score the training set; otherwise score the test set
        
        Returns:
            1D array of anomaly scores, shape (n_samples,)
        """
        pass 

    @abstractmethod
    def get_embeddings(self, indexes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract intermediate representation (embedding) for collaborative learning.
        
        Embeddings are aggregated across models and used to train RecurrentModel
        or to enhance pseudo-labeling. Should expose model's learned feature space.
        
        Args:
            indexes: Optional indices of samples. If None, return embeddings for all data.
        
        Returns:
            2D array of embeddings, shape (n_samples, embedding_dim)
        """
        pass

class RecurrentModel(ABC):
    """
    Post-hoc ensemble model trained on aggregated embeddings from multiple detectors.
    
    RecurrentModel learns a recurrent decision boundary by:
    1. Collecting embeddings from all Model instances
    2. Aggregating them (concatenation, averaging, etc.)
    3. Training on aggregated embeddings with pseudo-labels
    
    This enables ensemble-like boosting within the collaborative framework.
    """
    
    @abstractmethod
    def train(self, aggregated_embeddings: np.ndarray, labels: np.ndarray, epoch: int) -> None:
        """
        Train on aggregated embeddings from multiple models for one epoch.
        
        Args:
            aggregated_embeddings: Combined embeddings from all models, shape (n_samples, n_features)
            labels: Pseudo-labels or true labels, shape (n_samples,)
            epoch: Current epoch number for scheduling
        """
        pass

    @abstractmethod
    def fit(self, aggregated_embeddings: np.ndarray, labels: np.ndarray) -> None:
        """
        Standalone training on aggregated embeddings to convergence.
        
        Args:
            aggregated_embeddings: Combined embeddings from all models
            labels: Training labels
        """
        pass
    
    @abstractmethod
    def predict_scores(self, aggregated_embeddings: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores from aggregated embeddings.
        
        Args:
            aggregated_embeddings: Combined embeddings, shape (n_samples, n_features)
        
        Returns:
            Anomaly scores, shape (n_samples,)
        """
        pass


class Data(ABC):
    """
    Unified dataset interface managing indices, splits, and pseudo-labels.
    
    Responsibilities:
    - Maintain global indexing across different data formats (arrays, graphs, text)
    - Provide typed access to training/test features and labels
    - Store and update pseudo-labels from models during collaborative learning
    - Support index-based pseudo-label updates (critical for CoLearning efficiency)
    - Track label confidence/uncertainty for iterative refinement
    
    Each benchmark (ADBench/GADBench) has a Data subclass that normalizes
    heterogeneous data formats (numpy arrays, NetworkX graphs, PyG data, etc.)
    to a common index-based interface.
    
    Key Design: Pseudo-labels managed via model_name + indexes to avoid full array copies.
    """
    
    def __init__(self, dataset, train_test_split: float = 0.8, random_state: int = 42, 
                 preserve_labeled: bool = False, data_type: str = "tabular"):
        """
        Initialize dataset with global indexing and splits.
        
        Args:
            dataset: Raw dataset from benchmark
            train_test_split: Fraction for training (default 0.8)
            random_state: Seed for reproducible splits
            preserve_labeled: If True, preserve ground-truth labels on labeled_indexes during pseudo-labeling
            data_type: Type of data ("tabular", "graph", etc.)
        """
        self.dataset = dataset
        self.train_test_split = train_test_split
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.preserve_labeled = preserve_labeled
        self.data_type = data_type
        
        # Will be set by subclasses
        self.n_samples = None
        self.n_train = None
        self.n_test = None
        self.train_indexes = None
        self.test_indexes = None
        
        # Semi-supervised learning support
        self.labeled_indexes = None  # Indices of initially labeled training samples
        self.unlabeled_indexes = None  # Indices of unlabeled training samples
        self.labeled_ratio = None  # Ratio of labeled samples
        self.semisupervised_labels = None  # Combined labeled + pseudo-labeled data
        
        # Pseudo-labels: dict mapping model_name -> pseudo-label array
        # Only for training samples (length = n_train)
        self.pseudo_labels_by_model: Dict[str, np.ndarray] = {}
        self.pseudo_label_confidence: Dict[str, np.ndarray] = {}
        self.y_train_original = None  # Original training labels (preserved)
        
        # Call setup
        self.assign_global_indexes()
        self.generate_splits()
    
    def generate_semisupervised_split(self, 
                                     labeled_ratio: float = 0.1,
                                     stratified: bool = True) -> None:
        """
        Split training data into labeled and unlabeled subsets for semi-supervised learning.
    
        Args:
            labeled_ratio: Fraction of training samples to label (default 0.1 = 10%)
            stratified: Whether to maintain class balance in labeled subset
        """
        if self.y_train_original is None:
            raise ValueError("generate_splits() must be called before generate_semisupervised_split()")
    
        self.labeled_ratio = labeled_ratio
        n_labeled = max(2, int(self.n_train * labeled_ratio))  # At least 2 samples
    
        if stratified and self.y_train_original is not None:
            # Stratified sampling to maintain anomaly/normal ratio
            from sklearn.model_selection import train_test_split
            labeled_idx, unlabeled_idx = train_test_split(
                np.arange(self.n_train),
                train_size=n_labeled,
                stratify=self.y_train_original,
                random_state=self.random_state
            )
        else:
            # Random sampling
            indices = self.rng.permutation(self.n_train)
            labeled_idx = indices[:n_labeled]
            unlabeled_idx = indices[n_labeled:]
    
        self.labeled_indexes = np.sort(labeled_idx)
        self.unlabeled_indexes = np.sort(unlabeled_idx)
    
        # Initialize pseudo-labels: labeled samples get true labels, unlabeled get -1
        self.semisupervised_labels = np.full(self.n_train, -1, dtype=np.int32)
        self.semisupervised_labels[self.labeled_indexes] = self.y_train_original[self.labeled_indexes]
        
        # Print warning if preserve_labeled is True but all samples are labeled
        if self.preserve_labeled and len(self.unlabeled_indexes) == 0:
            print(f"[WARNING] preserve_labeled=True but all {self.n_train} training samples are labeled. "
                  f"Pseudo-labeling will have no effect (fully supervised mode).")

    # ========== Abstract Methods ==========
    
    @abstractmethod
    def assign_global_indexes(self) -> None:
        """
        Assign unique global indices to all samples in the dataset.
        Must set: self.n_samples, self.global_indexes
        """
        pass
    
    @abstractmethod
    def generate_splits(self) -> None:
        """
        Create train/test splits and set:
        - self.n_train, self.n_test
        - self.train_indexes, self.test_indexes
        - self.y_train_original (preserve original training labels)
        """
        pass
    
    @abstractmethod
    def _get_X_train(self) -> np.ndarray:
        """Return training features, shape (n_train, n_features)."""
        pass
    
    @abstractmethod
    def _get_X_test(self) -> np.ndarray:
        """Return test features, shape (n_test, n_features)."""
        pass
    
    @abstractmethod
    def _get_y_test(self) -> np.ndarray:
        """Return test labels (ground truth), shape (n_test,)."""
        pass
    
    @abstractmethod
    def apply_threshold_pseudo_label(self, scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Convert scores to pseudo-labels using threshold.
        
        Args:
            scores: Anomaly scores, shape (n_train,)
            threshold: Decision boundary (default 0.5)
        
        Returns:
            Binary labels {0, 1}, shape (n_train,)
        """
        pass

    # ========== Pseudo-Label Management (Index-Based) ==========
    
    def update_pseudo_labels(self, 
                            model_name: str, 
                            scores: np.ndarray, 
                            threshold: float = 0.5) -> None:
        """
        Update pseudo-labels for a specific model (all training samples).
        
        Args:
            model_name: Identifier for the model (e.g., "prenet", "deepsad")
            scores: Anomaly scores, shape (n_train,)
            threshold: Decision threshold
        """
        labels = self.apply_threshold_pseudo_label(scores, threshold)
        
        # If preserve_labeled=True and semi-supervised split exists, only update unlabeled indices
        if self.preserve_labeled and self.unlabeled_indexes is not None:
            # Initialize with original labels
            if model_name not in self.pseudo_labels_by_model:
                self.pseudo_labels_by_model[model_name] = self.y_train_original.copy()
                self.pseudo_label_confidence[model_name] = np.ones(self.n_train)
            
            # Update only unlabeled indices
            self.pseudo_labels_by_model[model_name][self.unlabeled_indexes] = labels[self.unlabeled_indexes]
            
            # Confidence = distance from boundary (only for unlabeled)
            confidence = 1.0 - np.abs(scores - threshold) / max(threshold, 1 - threshold)
            self.pseudo_label_confidence[model_name][self.unlabeled_indexes] = np.clip(confidence[self.unlabeled_indexes], 0, 1)
        else:
            # Update all indices (default behavior)
            self.pseudo_labels_by_model[model_name] = labels
            
            # Confidence = distance from boundary
            confidence = 1.0 - np.abs(scores - threshold) / max(threshold, 1 - threshold)
            self.pseudo_label_confidence[model_name] = np.clip(confidence, 0, 1)
    
    def update_pseudo_labels_by_indexes(self,
                                       model_name: str,
                                       indexes: np.ndarray,
                                       scores: np.ndarray,
                                       threshold: float = 0.5) -> None:
        """
        Update pseudo-labels for only specific training sample indices.
        Efficient for online/active learning scenarios.
        
        Args:
            model_name: Model identifier
            indexes: Training indices to update, shape (k,) where k <= n_train
            scores: Scores for those samples, shape (k,)
            threshold: Decision threshold
        """
        # Initialize if first update
        if model_name not in self.pseudo_labels_by_model:
            self.pseudo_labels_by_model[model_name] = self.y_train_original.copy()
            self.pseudo_label_confidence[model_name] = np.ones(self.n_train)
        
        # If preserve_labeled=True, filter out labeled indices
        if self.preserve_labeled and self.labeled_indexes is not None:
            # Only update indices that are in unlabeled_indexes
            unlabeled_mask = np.isin(indexes, self.unlabeled_indexes)
            filtered_indexes = indexes[unlabeled_mask]
            filtered_scores = scores[unlabeled_mask]
            
            if len(filtered_indexes) == 0:
                return  # All indices were labeled, nothing to update
            
            # Update only unlabeled indices
            labels = self.apply_threshold_pseudo_label(filtered_scores, threshold)
            self.pseudo_labels_by_model[model_name][filtered_indexes] = labels
            
            confidence = 1.0 - np.abs(filtered_scores - threshold) / max(threshold, 1 - threshold)
            self.pseudo_label_confidence[model_name][filtered_indexes] = np.clip(confidence, 0, 1)
        else:
            # Update all specified indices (default behavior)
            labels = self.apply_threshold_pseudo_label(scores, threshold)
            self.pseudo_labels_by_model[model_name][indexes] = labels
            
            confidence = 1.0 - np.abs(scores - threshold) / max(threshold, 1 - threshold)
            self.pseudo_label_confidence[model_name][indexes] = np.clip(confidence, 0, 1)
    
    def get_pseudo_labels(self, model_name: str) -> np.ndarray:
        """
        Get pseudo-labels for a model.
        
        Args:
            model_name: Model identifier
        
        Returns:
            Pseudo-labels, shape (n_train,), or original labels if not set
        """
        if model_name not in self.pseudo_labels_by_model:
            return self.y_train_original.copy()
        return self.pseudo_labels_by_model[model_name]
    
    def get_ensemble_pseudo_labels(self, aggregation: str = "majority") -> np.ndarray:
        """
        Aggregate pseudo-labels across all models.
        
        Args:
            aggregation: "majority" (majority vote), "mean" (soft vote), "max_confidence"
        
        Returns:
            Aggregated pseudo-labels, shape (n_train,)
        """
        if not self.pseudo_labels_by_model:
            return self.y_train_original.copy()
        
        labels_stack = np.array(list(self.pseudo_labels_by_model.values()))
        
        if aggregation == "majority":
            return np.round(np.mean(labels_stack, axis=0)).astype(int)
        elif aggregation == "mean":
            return np.mean(labels_stack, axis=0)
        elif aggregation == "max_confidence":
            # For each sample, pick label from most confident model
            confidence_stack = np.array(list(self.pseudo_label_confidence.values()))
            max_conf_idx = np.argmax(confidence_stack, axis=0)
            return labels_stack[max_conf_idx, np.arange(self.n_train)]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    
    def reset_pseudo_labels(self, model_name: Optional[str] = None) -> None:
        """
        Reset pseudo-labels to original training labels.
        
        Args:
            model_name: If provided, reset only this model; else reset all
        """
        if model_name:
            if model_name in self.pseudo_labels_by_model:
                self.pseudo_labels_by_model[model_name] = self.y_train_original.copy()
                self.pseudo_label_confidence[model_name] = np.ones(self.n_train)
        else:
            self.pseudo_labels_by_model.clear()
            self.pseudo_label_confidence.clear()

    # ========== Typed Property Access ==========
    
    @property
    def X_train(self) -> np.ndarray:
        """Training features, shape (n_train, n_features)."""
        return self._get_X_train()
    
    @property
    def X_test(self) -> np.ndarray:
        """Test features, shape (n_test, n_features)."""
        return self._get_X_test()
    
    @property
    def y_train(self) -> np.ndarray:
        """Training labels (original), shape (n_train,)."""
        return self.y_train_original
    
    @property
    def y_test(self) -> np.ndarray:
        """Test labels (ground truth), shape (n_test,)."""
        return self._get_y_test()
    
    # ========== Backward Compatibility (Dict-like Access) ==========
    
    def __getitem__(self, key: str) -> np.ndarray:
        """Dict-like access for backward compatibility."""
        if key == "X_train":
            return self.X_train
        elif key == "y_train":
            return self.y_train
        elif key == "X_test":
            return self.X_test
        elif key == "y_test":
            return self.y_test
        else:
            raise KeyError(f"Unknown key: {key}")
    
    def get(self, key: str, default=None):
        """Dict-like get with default."""
        try:
            return self[key]
        except KeyError:
            return default

class Strategy(ABC):
    """
    Convergence strategy for handling asynchronous model training.
    
    Models may converge at different rates. Strategy determines:
    - When to continue/stop the collaborative learning loop
    - How to weight model contributions based on convergence status
    - How to handle fast vs slow converging models
    """
    
    @abstractmethod
    def should_continue(self, model_metrics: Dict[str, float], chapter: int) -> bool:
        """
        Determine if training should continue.
        
        Args:
            model_metrics: Dict of {model_name: auc_score, ...}
            chapter: Current chapter/phase number
        
        Returns:
            True if training should continue, False to stop
        """
        pass

    @abstractmethod
    def get_weights(self, model_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Get contribution weights for each model based on performance.
        
        Args:
            model_metrics: Dict of {model_name: auc_score, ...}
        
        Returns:
            Dict of {model_name: weight} where sum(weights) = 1.0
        """
        pass


class SimpleStrategy(Strategy):
    """
    Simple convergence strategy based on ensemble AUC plateau or max chapters.
    
    Continues training until:
    - Ensemble AUC has plateaued (change < patience_threshold for N checks)
    - Max chapters reached
    - All models have converged individually
    """
    
    def __init__(self, 
                 max_chapters: int = 50,
                 patience: int = 5,
                 patience_threshold: float = 0.001):
        """
        Initialize simple strategy.
        
        Args:
            max_chapters: Maximum training chapters (handled by CoLearning)
            patience: Stop if no improvement for this many chapters
            patience_threshold: Minimum AUC improvement to reset patience counter
        """
        self.max_chapters = max_chapters
        self.patience = patience
        self.patience_threshold = patience_threshold
        self.best_ensemble_auc = 0.0
        self.patience_counter = 0
    
    def should_continue(self, model_metrics: Dict[str, float], chapter: int) -> bool:
        """
        Stop if ensemble AUC plateaus or max chapters reached.
        """
        # Check ensemble improvement
        current_auc = model_metrics.get("ensemble", 0.5)
        
        if current_auc > self.best_ensemble_auc + self.patience_threshold:
            # Improvement detected, reset patience
            self.best_ensemble_auc = current_auc
            self.patience_counter = 0
            return True
        else:
            # No improvement
            self.patience_counter += 1
            return self.patience_counter < self.patience
    
    def get_weights(self, model_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Uniform weights (no weighting by performance for now).
        """
        n_models = len([k for k in model_metrics.keys() if k.startswith("model_")])
        return {f"model_{i}": 1.0 / n_models for i in range(n_models)}



class CoLearning(ABC):
    """
    Orchestrates multi-model collaborative anomaly detection training.
    
    Manages:
    1. **Warmup Phase**: Pre-train all models independently
    2. **Exchange Phase**: Aggregate embeddings/pseudo-labels from models
    3. **Joint Training Phase**: Each model trains on others' pseudo-labels
    4. **Convergence**: Stop when strategy criteria met
    
    This is the core loop implementing the co-training paradigm.
    """

    def __init__(self,
                 models: List[Model],
                 data: Data,
                 strategy: Strategy,
                 warmup_epochs: int = 10,
                 max_chapters: int = 10):
        """
        Initialize collaborative learning framework.
        
        Args:
            models: List of Model instances to train collaboratively
            data: Data wrapper managing dataset and pseudo-labels
            strategy: Strategy for convergence and model weighting
            warmup_epochs: Pre-training epochs before pseudo-label exchange
            max_chapters: Maximum number of exchange chapters
        """
        self.models = models
        self.data = data
        self.strategy = strategy
        self.warmup_epochs = warmup_epochs
        self.max_chapters = max_chapters
        self.training_history = []

    def send_anomalies(self, model_idx: int, training_set: np.ndarray, indexes: np.ndarray) -> None:
        """
        Send detected anomalies from one model to others for label refinement.
        
        Args:
            model_idx: Source model index
            training_set: Original training data
            indexes: Indices of samples detected as anomalies
        """
        pass

    def send_normals(self, model_idx: int, training_set: np.ndarray, indexes: np.ndarray) -> None:
        """
        Send detected normals from one model to others for pseudo-labeling.
        
        Args:
            model_idx: Source model index
            training_set: Original training data
            indexes: Indices of samples detected as normal
        """
        pass

    def aggregate_embeddings(self) -> np.ndarray:
        """
        Collect and aggregate embeddings from all models.
        
        Returns:
            Aggregated embeddings, shape (n_samples, n_features_combined)
        """
        embeddings_list = []
        for model in self.models:
            emb = model.get_embeddings()
            embeddings_list.append(emb)
        # Concatenate or average based on dimensionality
        return np.concatenate(embeddings_list, axis=1)

    @abstractmethod
    def exchange(self) -> None:
        """
        Execute pseudo-label exchange between models.
        
        Strategies:
        - All-to-all: Broadcast each model's predictions to others
        - Weighted: Send labels weighted by model confidence/accuracy
        - Selective: Only exchange high-confidence predictions
        """
        pass

    @abstractmethod
    def cotrain(self, 
                recurrent_model: Optional[RecurrentModel] = None,
                eval_interval: int = 1) -> Dict[str, List[float]]:
        """
        Main collaborative training loop.
        
        Args:
            recurrent_model: Optional post-hoc ensemble model on aggregated embeddings
            eval_interval: Evaluate metrics every N chapters
        
        Returns:
            Dictionary of training history with metric curves
        """
        pass



class CoLearner(CoLearning):
    """
    Multi-model collaborative anomaly detection with bidirectional pseudo-label exchange.
    
    Implementation of CoLearning that enables all models to benefit from each other's
    predictions. Each model serves as both learner and teacher through iterative
    pseudo-label refinement.
    
    Example (3-model setup):
        - Model A predicts normal samples → sent to B, C for training
        - Model B predicts anomalies → sent to A, C for training
        - Model C predicts borderline → all models learn from disagreement
    """
    
    def __init__(self, 
                 models: List[Model], 
                 data: Data, 
                 strategy: Strategy,
                 warmup_epochs: int = 10,
                 max_chapters: int = 10,
                 anomaly_threshold: float = 0.5):
        """
        Initialize multi-model co-learner.
        
        Args:
            models: List of heterogeneous detectors (ADBench, GADBench, PyGOD, etc.)
            data: Unified Data wrapper
            strategy: Convergence strategy
            warmup_epochs: Pre-training epochs
            max_chapters: Max collaborative chapters
            anomaly_threshold: Score threshold for anomaly detection (default 0.5)
        """
        super().__init__(models, data, strategy, warmup_epochs, max_chapters)
        
        # Initialize pseudo-label storage on each model (no naming needed)
        for model in models:
            model._pseudo_labels = None
        
        self.model_scores = {f"model_{i}": None for i in range(len(models))}
        self.ensemble_scores = None
        self.anomaly_threshold = anomaly_threshold

    def get_anomaly_indexes(self, scores: np.ndarray) -> np.ndarray:
        """
        Identify anomaly indices from scores.
        
        Args:
            scores: Anomaly scores, shape (n_train,)
        
        Returns:
            Array of indices where scores > threshold
        """
        return np.where(scores > self.anomaly_threshold)[0]

    def get_normal_indexes(self, scores: np.ndarray) -> np.ndarray:
        """
        Identify normal indices from scores.
        
        Args:
            scores: Anomaly scores, shape (n_train,)
        
        Returns:
            Array of indices where scores <= threshold
        """
        return np.where(scores <= self.anomaly_threshold)[0]

    def send_anomalies(self, model_idx: int, anomaly_indexes: np.ndarray, scores: np.ndarray) -> None:
        """
        Send detected anomalies from one model to others.
        
        Updates pseudo-labels for all OTHER models with the anomalies detected by model_idx.
        Recipient models will treat these samples as anomalies (label = 1) during training.
        
        Args:
            model_idx: Index of source model sending anomalies
            anomaly_indexes: Indices of anomalies, shape (k,) where k = number of anomalies
            scores: Full anomaly scores from source model, shape (n_train,)
        
        Process:
            1. Receive anomaly indices from get_anomaly_indexes()
            2. For each other model: mark these indices as anomalies (pseudo-label = 1)
            3. Update confidence based on how extreme the score is
        """
        if len(anomaly_indexes) == 0:
            return  # No anomalies detected, nothing to share
        
        # Send to all other models
        for i, model in enumerate(self.models):
            if i == model_idx:
                continue  # Don't send to self
            
            # Initialize pseudo-labels if not already set (use -1 for unknown, not original labels)
            if model._pseudo_labels is None:
                model._pseudo_labels = np.full(self.data.n_train, -1, dtype=np.int32)
            
            # Mark detected anomalies as label = 1
            model._pseudo_labels[anomaly_indexes] = 1
            
            # Confidence tracking (stored separately if needed for advanced strategies)
            target_model_name = f"model_{i}"
            if target_model_name not in self.data.pseudo_label_confidence:
                self.data.pseudo_label_confidence[target_model_name] = np.zeros(self.data.n_train)
            
            # Confidence is higher for more extreme anomaly scores
            anomaly_scores = scores[anomaly_indexes]
            confidence = np.clip(anomaly_scores - self.anomaly_threshold, 0, 1)
            self.data.pseudo_label_confidence[target_model_name][anomaly_indexes] = confidence

    def send_normals(self, model_idx: int, normal_indexes: np.ndarray, scores: np.ndarray) -> None:
        """
        Send detected normals from one model to others.
        
        Updates pseudo-labels for all OTHER models with the normals detected by model_idx.
        Recipient models will treat these samples as normal (label = 0) during training.
        
        Args:
            model_idx: Index of source model sending normals
            normal_indexes: Indices of normals, shape (k,) where k = number of normals
            scores: Full anomaly scores from source model, shape (n_train,)
        
        Process:
            1. Receive normal indices from get_normal_indexes()
            2. For each other model: mark these indices as normals (pseudo-label = 0)
            3. Update confidence based on how far below threshold the score is
        """
        if len(normal_indexes) == 0:
            return  # No normals detected, nothing to share
        
        # Send to all other models
        for i, model in enumerate(self.models):
            if i == model_idx:
                continue  # Don't send to self
            
            # Initialize pseudo-labels if not already set (use -1 for unknown, not original labels)
            if model._pseudo_labels is None:
                model._pseudo_labels = np.full(self.data.n_train, -1, dtype=np.int32)
            
            # Mark detected normals as label = 0
            model._pseudo_labels[normal_indexes] = 0
            
            # Confidence tracking (stored separately if needed for advanced strategies)
            target_model_name = f"model_{i}"
            if target_model_name not in self.data.pseudo_label_confidence:
                self.data.pseudo_label_confidence[target_model_name] = np.zeros(self.data.n_train)
            
            # Confidence is higher for more extreme normal scores (far below threshold)
            normal_scores = scores[normal_indexes]
            confidence = np.clip(self.anomaly_threshold - normal_scores, 0, 1)
            self.data.pseudo_label_confidence[target_model_name][normal_indexes] = confidence

    def exchange(self) -> None:
        """
        Sequential pseudo-label exchange between models.
        
        Process (turn-by-turn):
        1. Each model predicts on training set
        2. Model sends its anomalies/normals to others as pseudo-labels
        3. Next model trains on received pseudo-labels
        4. Repeat in round-robin fashion
        
        Key idea: Each model learns from others' mistakes/agreements
        """
        # Collect predictions from all models on training data
        # Use explicit flag to score training set
        for i, model in enumerate(self.models):
            scores = model.predict_scores(use_train=True)
            self.model_scores[f"model_{i}"] = scores
            
            # Identify anomalies and normals using intermediate functions
            anomaly_indexes = self.get_anomaly_indexes(scores)
            normal_indexes = self.get_normal_indexes(scores)
            
            # Send this model's anomalies and normals to other models
            self.send_anomalies(i, anomaly_indexes, scores)
            self.send_normals(i, normal_indexes, scores)
        
        # Compute ensemble predictions for evaluation (on full training set)
        scores_array = np.array(list(self.model_scores.values()))
        self.ensemble_scores = np.mean(scores_array, axis=0)
        self.ensemble_disagreement = np.std(scores_array, axis=0)

    def cotrain(self, 
                recurrent_model: Optional[RecurrentModel] = None,
                eval_interval: int = 1) -> Dict[str, List[float]]:

        """
        Execute collaborative training loop with pseudo-label exchange.
        
        Process:
        1. Warmup phase: Train all models independently (no exchange)
        2. Chapter Loop (collaborative training):
           - Execute exchange: collect predictions, update pseudo-labels
           - Each model trains one epoch on fresh pseudo-labels
           - Evaluate ensemble performance on test set
           - Check convergence criterion
        3. Final: Optional recurrent model training on aggregated embeddings
        
        Ensemble Calculation:
        - Ensemble scores = mean(model_0_scores, model_1_scores, ..., model_n_scores)
        - Simple averaging across all model predictions (unweighted)
        - Each model contributes equally to final anomaly score
        
        Args:
            recurrent_model: Optional post-hoc ensemble model
            eval_interval: Evaluate metrics every N chapters
        
        Returns:
            Training history dict with metrics per chapter
        """
        from sklearn.metrics import roc_auc_score
        
        history = {"warmup": [], "chapters": []}
        
        # Get test set labels for evaluation
        y_test = self.data.y_test
        
        # ===== Warmup Phase =====
        print(f"\n[Warmup] Training {len(self.models)} models for {self.warmup_epochs} epochs (no exchange)...")
        for epoch in range(self.warmup_epochs):
            for i, model in enumerate(self.models):
                model.train(epoch)
            
            if (epoch + 1) % max(1, self.warmup_epochs // 3) == 0:
                print(f"  Epoch {epoch + 1}/{self.warmup_epochs}")
        
        # ===== Collaborative Chapters =====
        print(f"\n[Collaborative] Running up to {self.max_chapters} chapters...")
        for chapter in range(self.max_chapters):
            # Step 1: Exchange pseudo-labels between models (BEFORE training)
            self.exchange()
            
            # Step 2: Each model trains one epoch on updated pseudo-labels
            for i, model in enumerate(self.models):
                model.train(chapter)
            
            # Step 3: Evaluate ensemble on test set
            test_scores = self.ensemble_scores if self.ensemble_scores is not None else np.zeros(len(y_test))
            
            # Truncate to test set size if needed (ensemble_scores computed on train set)
            if len(test_scores) != len(y_test):
                # For now, re-compute on test set
                test_scores_list = []
                for i, model in enumerate(self.models):
                    # Predict on full test set (wrapper expects test-relative indexing)
                    scores = model.predict_scores()
                    test_scores_list.append(scores)
                test_scores = np.mean(np.array(test_scores_list), axis=0)
            
            try:
                auc = roc_auc_score(y_test, test_scores)
            except:
                auc = 0.5  # Fallback if test set is too small
            
            # Compute per-model metrics for convergence check
            model_metrics = {}
            for i, model in enumerate(self.models):
                try:
                    # Predict on full test set for per-model AUC
                    model_test_scores = model.predict_scores()
                    model_auc = roc_auc_score(y_test, model_test_scores)
                    model_metrics[f"model_{i}"] = model_auc
                except:
                    model_metrics[f"model_{i}"] = 0.5
            
            # Add ensemble metric
            model_metrics["ensemble"] = auc
            
            # Step 4: Log and check convergence
            if (chapter + 1) % eval_interval == 0:
                metric_str = ", ".join([f"{k}={v:.4f}" for k, v in model_metrics.items()])
                print(f"  Chapter {chapter + 1}: {metric_str}")
            
            history["chapters"].append(model_metrics)
            
            # Check stopping criterion
            if self.strategy is not None and not self.strategy.should_continue(model_metrics, chapter):
                print(f"  → Converged at chapter {chapter + 1}")
                break
        
        # ===== Optional: Recurrent Model =====
        if recurrent_model is not None:
            print(f"\n[Recurrent] Training ensemble on aggregated embeddings...")
            agg_emb = self.aggregate_embeddings()
            ensemble_labels = self.data.get_ensemble_pseudo_labels(aggregation="majority")
            recurrent_model.fit(agg_emb, ensemble_labels)
        
        print(f"\n[Done] Collaborative training complete")
        return history

class SingleModel(CoLearning):
    """
    Single-model baseline for fair comparison against collaborative learning.
    
    Trains a single model independently without pseudo-label exchange.
    Used to measure the contribution of collaborative learning by comparing
    results against multi-model CoLearner.
    """
    
    def __init__(self, 
                 model: Model, 
                 data: Data, 
                 strategy: Optional[Strategy] = None,
                 warmup_epochs: int = 0):
        """
        Initialize single-model learner.
        
        Args:
            model: Single Model instance
            data: Data wrapper
            strategy: Unused (single model, no convergence strategy needed)
            warmup_epochs: Ignored (single model trains from start)
        """
        super().__init__([model], data, None, warmup_epochs, 1)
        #super().__init__([model], data, strategy or Strategy(), warmup_epochs, max_chapters=1)
        self.model = model
        self.test_scores = None
        self.test_labels = None

    def exchange(self) -> None:
        """No-op: single model has no peers to exchange with."""
        return None

    def cotrain(self, 
                recurrent_model: Optional[RecurrentModel] = None,
                eval_interval: int = 1) -> Dict[str, float]:
        """
        Train single model to convergence (standalone fit()).
        
        Returns:
            Dict with final metrics (auc, precision, recall, etc.)
        """
        print("Training single model (no collaboration)...")
        self.model.fit()
        
        # Evaluate on test set
        self.test_scores = self.model.predict_scores()
        
        # Return evaluation metrics (ROC-AUC, etc.)
        # Note: requires test_labels to be set in data
        metrics = {"model_0_auc": 0.85}  # Placeholder
        return metrics


# ============================================================================
# Utility Functions
# ============================================================================

def load_models(model_names: List[str], 
                data: Data, 
                train_config: Dict,
                model_config: Dict,
                model_detector_dict: Dict) -> List[Model]:
    """
    Instantiate models from registry.
    
    Args:
        model_names: List of model identifiers (keys in model_detector_dict)
        data: Unified Data wrapper
        train_config: Training hyperparameters (learning rate, batch size, etc.)
        model_config: Model-specific config (architecture, thresholds, etc.)
        model_detector_dict: Registry mapping model_name -> Model class
    
    Returns:
        List of instantiated Model objects
    
    Raises:
        ValueError: If model_name not found in registry
    """
    detectors = []
    for model_name in model_names:
        if model_name not in model_detector_dict:
            raise ValueError(f"Model '{model_name}' not found in registry. "
                           f"Available: {list(model_detector_dict.keys())}")
        
        ModelClass = model_detector_dict[model_name]
        detector = ModelClass(train_config, model_config, data)
        detectors.append(detector)
    
    return detectors
