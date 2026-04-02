"""ADBench data loader"""

from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from base import Data


class ClassicalADBenchData(Data):
    """Loads ADBench .npz datasets with train/val/test splitting and normalization"""
    
    def __init__(self,
                 dataset_path: Path,
                 train_test_split_ratio: float = 0.8,
                 val_test_split_ratio: float = 0.0,
                 random_state: int = 42,
                 normalize: bool = True,
                 preserve_labeled: bool = False,
                 data_type: str = "tabular",
                 labeled_ratio=None,
                 stratified=True,
                 max_anomalies=None,
                 anomaly_ratio=None,
                 unlabeled_policy: str = "unlabeled_as_normal"):
        self.dataset_path = Path(dataset_path)
        self.normalize = normalize
        self._scaler = None
        
        # Cached data arrays
        self._X_train_cached = None
        self._X_test_cached = None
        self._X_val_cached = None
        self._y_test_cached = None
        self._y_val_cached = None
        
        # Load raw data
        self._raw_data = np.load(self.dataset_path, allow_pickle=True)
        
        # Parent __init__ calls _load() then _label_split()
        super().__init__(
            dataset=self._raw_data,
            train_test_split=train_test_split_ratio,
            random_state=random_state,
            preserve_labeled=preserve_labeled,
            data_type=data_type,
            val_test_split=val_test_split_ratio,
            labeled_ratio=labeled_ratio,
            stratified=stratified,
            max_anomalies=max_anomalies,
            anomaly_ratio=anomaly_ratio,
            unlabeled_policy=unlabeled_policy
        )
    
    def assign_global_indexes(self) -> None:
        data = self._raw_data
        files = set(data.files)
        
        # Determine total sample count
        if {"X_train", "y_train", "X_test", "y_test"}.issubset(files):
            n_total = len(data["X_train"]) + len(data["X_test"])
        elif {"X", "y"}.issubset(files):
            n_total = len(data["X"])
        else:
            raise ValueError(f"Unexpected keys in dataset: {files}")
        
        self.n_samples = n_total
        self.global_indexes = np.arange(self.n_samples)
    
    def _load(self) -> None:
        from sklearn.model_selection import train_test_split as sklearn_train_test_split
        
        data = self._raw_data
        files = set(data.files)
        
        # Load raw arrays from NPZ
        if {"X_train", "y_train", "X_test", "y_test"}.issubset(files):
            # Combine pre-split data so we can re-split with our own ratio
            X_train_orig = np.asarray(data["X_train"], dtype=np.float32)
            y_train_orig = np.asarray(data["y_train"], dtype=np.int32)
            X_test_orig = np.asarray(data["X_test"], dtype=np.float32)
            y_test_orig = np.asarray(data["y_test"], dtype=np.int32)
            
            X_all = np.vstack([X_train_orig, X_test_orig])
            y_all = np.hstack([y_train_orig, y_test_orig])
        elif {"X", "y"}.issubset(files):
            # Already combined
            X_all = np.asarray(data["X"], dtype=np.float32)
            y_all = np.asarray(data["y"], dtype=np.int32)
        else:
            raise ValueError(f"Unexpected keys in dataset: {files}")
        
        # Train vs (test+val)
        X_train, X_test_val, y_train, y_test_val = sklearn_train_test_split(
            X_all, y_all,
            train_size=self.train_test_split,
            shuffle=True,
            stratify=y_all,
            random_state=self.random_state
        )
        
        # Split off validation set (if requested)
        if self.val_test_split > 0:
            X_test, X_val, y_test, y_val = sklearn_train_test_split(
                X_test_val, y_test_val,
                train_size=self.val_test_split,
                shuffle=True,
                stratify=y_test_val,
                random_state=self.random_state
            )
        else:
            # No validation set
            X_test, X_val = X_test_val, np.empty((0, X_test_val.shape[1]), dtype=np.float32)
            y_test, y_val = y_test_val, np.empty(0, dtype=np.int32)
        
        # Normalize (scaler fit on train only)
        if self.normalize:
            self._scaler = MinMaxScaler()
            X_train = self._scaler.fit_transform(X_train)
            X_test = self._scaler.transform(X_test)
            if len(X_val) > 0:
                X_val = self._scaler.transform(X_val)
        
        # Store splits
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_test = y_test
        self.y_val = y_val
        
        # Index bookkeeping
        self.n_train = len(X_train)
        self.n_test = len(X_test)
        self.n_val = len(X_val)
        self.train_indexes = np.arange(self.n_train)
        self.test_indexes = np.arange(self.n_train, self.n_train + self.n_test)
        if self.n_val > 0:
            self.val_indexes = np.arange(self.n_train + self.n_test, self.n_train + self.n_test + self.n_val)
        else:
            self.val_indexes = np.empty(0, dtype=int)
        
        # Keep originals for evaluation
        self.y_train_original = y_train.copy()
    
    def __repr__(self) -> str:
        return (
            f"ClassicalADBenchData("
            f"dataset={self.dataset_path.name}, "
            f"n_train={self.n_train}, "
            f"n_test={self.n_test}, "
            f"n_val={self.n_val}, "
            f"n_features={self.X_train.shape[1]})"
        )



def load_data(dataset_path: Path,
              preserve_labeled: bool = False,
              data_type: str = "tabular") -> ClassicalADBenchData:
    """Shortcut to load an ADBench .npz dataset"""
    return ClassicalADBenchData(dataset_path, preserve_labeled=preserve_labeled, data_type=data_type)
