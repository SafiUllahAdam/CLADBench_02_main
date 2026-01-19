"""Data loaders and wrappers for ADBench Classical datasets."""

from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from base import Data


class ClassicalADBenchData(Data):
    """
    Data wrapper for ADBench Classical datasets (numpy arrays).
    
    Handles both formats:
    1. Pre-split: X_train, y_train, X_test, y_test
    2. Unsplit: X, y (automatically split 80/20)
    
    Features:
    - Automatic MinMaxScaler normalization
    - Index-based pseudo-label management
    - Backward-compatible dict-like access
    """
    
    def __init__(self, 
                 dataset_path: Path,
                 train_test_split_ratio: float = 0.8,
                 random_state: int = 42,
                 normalize: bool = True,
                 preserve_labeled: bool = False,
                 data_type: str = "tabular"):
        """
        Initialize Classical ADBench dataset.
        
        Args:
            dataset_path: Path to .npz file
            train_test_split_ratio: Fraction for training if X, y provided
            random_state: Seed for reproducibility
            normalize: Whether to apply MinMaxScaler
            preserve_labeled: If True, preserve ground-truth labels on labeled_indexes
            data_type: Type of data (default "tabular")
        """
        self.dataset_path = Path(dataset_path)
        self.normalize = normalize
        self._scaler = None
        self._X_train_cached = None
        self._X_test_cached = None
        self._y_test_cached = None
        
        # Load raw data
        npz_data = np.load(self.dataset_path, allow_pickle=True)
        self._raw_data = npz_data
        
        # Initialize parent with raw data
        super().__init__(
            dataset=npz_data,
            train_test_split=train_test_split_ratio,
            random_state=random_state,
            preserve_labeled=preserve_labeled,
            data_type=data_type
        )
    
    def assign_global_indexes(self) -> None:
        """Assign global indices to all samples."""
        # Load data to determine n_samples
        data = self._raw_data
        files = set(data.files)
        
        if {"X_train", "y_train", "X_test", "y_test"}.issubset(files):
            X_train = data["X_train"]
            X_test = data["X_test"]
            self.n_samples = len(X_train) + len(X_test)
        elif {"X", "y"}.issubset(files):
            X = data["X"]
            self.n_samples = len(X)
        else:
            raise ValueError(f"Unexpected keys in dataset: {files}")
        
        # Global indices are just 0, 1, 2, ..., n_samples-1
        self.global_indexes = np.arange(self.n_samples)
    
    def generate_splits(self) -> None:
        """Load data, handle both formats, and create train/test split."""
        data = self._raw_data
        files = set(data.files)
        
        if {"X_train", "y_train", "X_test", "y_test"}.issubset(files):
            # Pre-split format
            X_train = np.asarray(data["X_train"], dtype=np.float32)
            y_train = np.asarray(data["y_train"], dtype=np.int32)
            X_test = np.asarray(data["X_test"], dtype=np.float32)
            y_test = np.asarray(data["y_test"], dtype=np.int32)
        elif {"X", "y"}.issubset(files):
            # Unsplit format
            X = np.asarray(data["X"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.int32)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=1.0 - self.train_test_split,
                shuffle=True,
                stratify=y,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unexpected keys in dataset: {files}")
        
        # Normalize
        if self.normalize:
            self._scaler = MinMaxScaler()
            X_train = self._scaler.fit_transform(X_train)
            X_test = self._scaler.transform(X_test)
        
        # Cache data
        self._X_train_cached = X_train
        self._X_test_cached = X_test
        self._y_test_cached = y_test
        
        # Set split info
        self.n_train = len(X_train)
        self.n_test = len(X_test)
        self.train_indexes = np.arange(self.n_train)
        self.test_indexes = np.arange(self.n_train, self.n_train + self.n_test)
        
        # Store original training labels
        self.y_train_original = y_train.copy()
    
    def _get_X_train(self) -> np.ndarray:
        """Return cached training features."""
        return self._X_train_cached
    
    def _get_X_test(self) -> np.ndarray:
        """Return cached test features."""
        return self._X_test_cached
    
    def _get_y_test(self) -> np.ndarray:
        """Return cached test labels."""
        return self._y_test_cached
    
    def apply_threshold_pseudo_label(self, 
                                     scores: np.ndarray, 
                                     threshold: float = 0.5) -> np.ndarray:
        """Apply threshold to convert scores to binary labels."""
        return (scores > threshold).astype(np.int32)
    
    def __repr__(self) -> str:
        return (
            f"ClassicalADBenchData("
            f"dataset={self.dataset_path.name}, "
            f"n_train={self.n_train}, "
            f"n_test={self.n_test}, "
            f"n_features={self.X_train.shape[1]})"
        )


def load_adbench_classical(dataset_name: str, 
                          root: Optional[Path] = None,
                          train_test_split_ratio: float = 0.8,
                          random_state: int = 42) -> ClassicalADBenchData:
    """
    Load an ADBench Classical dataset by name.
    
    Args:
        dataset_name: e.g., "annthyroid", "arrhythmia", "cardio", etc.
        root: Root directory containing ADBench datasets. 
              If None, searches SubModules/ADBench/adbench/datasets/Classical/
        train_test_split_ratio: Fraction for training
        random_state: Seed for reproducibility
    
    Returns:
        ClassicalADBenchData instance
    
    Example:
        data = load_adbench_classical("annthyroid")
        print(data.X_train.shape)  # (n_train, n_features)
        print(data.X_test.shape)   # (n_test, n_features)
    """
    if root is None:
        # Infer root from project structure
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent
        root = project_root / "SubModules" / "ADBench" / "adbench" / "datasets" / "Classical"
    else:
        root = Path(root)
    
    # Handle dataset name format (with or without number prefix)
    if not dataset_name[0].isdigit():
        dataset_name = f"2_{dataset_name}.npz"
    if not dataset_name.endswith(".npz"):
        dataset_name = f"{dataset_name}.npz"
    
    dataset_path = root / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Available datasets in {root}:\n"
            f"{list(root.glob('*.npz'))}"
        )
    
    return ClassicalADBenchData(
        dataset_path,
        train_test_split_ratio=train_test_split_ratio,
        random_state=random_state
    )


def load_data(dataset_path: Path, 
              preserve_labeled: bool = False,
              data_type: str = "tabular") -> ClassicalADBenchData:
    """
    Load dataset from ADBench format (backward-compatible function).
    
    This function maintains compatibility with the mvp.ipynb load_data pattern
    while returning a Data object.
    
    Args:
        dataset_path: Path to .npz file
        preserve_labeled: If True, preserve ground-truth labels on labeled_indexes
        data_type: Type of data (default "tabular")
    
    Returns:
        ClassicalADBenchData instance
    
    Example:
        data = load_data(Path("SubModules/ADBench/adbench/datasets/Classical/2_annthyroid.npz"))
        print(data.X_train.shape)
        print(data["X_train"].shape)  # Dict-like access still works
    """
    return ClassicalADBenchData(dataset_path, preserve_labeled=preserve_labeled, data_type=data_type)
