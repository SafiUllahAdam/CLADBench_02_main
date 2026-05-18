"""GADBench graph data loader."""

from pathlib import Path
from typing import Optional, Union

import numpy as np

from base import GraphData
from benchmark_config import PROJECT_ROOT


GADBENCH_DATASETS = PROJECT_ROOT / "SubModules" / "GADBench" / "datasets"


class GADBenchGraphData(GraphData):
    """Loads official GADBench DGL node graphs under the CoBench Data API."""

    def __init__(self,
                 dataset: Union[str, Path],
                 dataset_root: Optional[Path] = None,
                 train_test_split_ratio: float = 0.8,
                 val_test_split_ratio: float = 0.0,
                 random_state: int = 42,
                 preserve_labeled: bool = False,
                 labeled_ratio=None,
                 stratified: bool = True,
                 max_anomalies=None,
                 anomaly_ratio=None,
                 unlabeled_policy: str = "unlabeled_as_normal"):
        self.dataset_name = str(dataset)
        self.dataset_root = Path(dataset_root) if dataset_root is not None else GADBENCH_DATASETS
        self.dataset_path = self._resolve_path(dataset)
        self.stratified_splits = stratified

        super().__init__(
            dataset=self.dataset_path,
            train_test_split=train_test_split_ratio,
            random_state=random_state,
            preserve_labeled=preserve_labeled,
            data_type="graph",
            val_test_split=val_test_split_ratio,
            labeled_ratio=labeled_ratio,
            stratified=stratified,
            max_anomalies=max_anomalies,
            anomaly_ratio=anomaly_ratio,
            unlabeled_policy=unlabeled_policy,
        )

    def _resolve_path(self, dataset: Union[str, Path]) -> Path:
        path = Path(dataset)
        if path.exists():
            return path
        return self.dataset_root / path

    def _load_graph(self) -> None:
        try:
            from dgl.data.utils import load_graphs
        except ImportError as e:
            raise ImportError("dgl is required to load GADBench datasets") from e

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"GADBench dataset not found: {self.dataset_path}")

        self.graph = load_graphs(str(self.dataset_path))[0][0]

        feature = self._node_data("feature")
        label = self._node_data("label")
        X = self._to_numpy(feature).astype(np.float32, copy=False)
        y = self._to_numpy(label).astype(np.int32, copy=False).reshape(-1)
        if X.shape[0] != len(y):
            raise ValueError("GADBench feature/label node counts do not match")
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("GADBenchGraphData expects binary node labels {0, 1}")

        self.n_samples = int(len(y))
        self.train_indexes, self.test_indexes, self.val_indexes = self._split_indexes(y)
        self.n_train = int(len(self.train_indexes))
        self.n_test = int(len(self.test_indexes))
        self.n_val = int(len(self.val_indexes))
        self._write_split_masks()

        self.X_train = X[self.train_indexes]
        self.X_val = X[self.val_indexes]
        self.X_test = X[self.test_indexes]
        self.y_train_original = y[self.train_indexes].copy()
        self.y_val = y[self.val_indexes].copy()
        self.y_test = y[self.test_indexes].copy()

    def _split_indexes(self, y: np.ndarray) -> tuple:
        from sklearn.model_selection import train_test_split as sklearn_train_test_split

        idx = np.arange(len(y))
        stratify = y if self.stratified_splits else None
        train_idx, test_val_idx = sklearn_train_test_split(
            idx, train_size=self.train_test_split, shuffle=True,
            stratify=stratify, random_state=self.random_state,
        )
        if self.val_test_split > 0:
            split_stratify = y[test_val_idx] if self.stratified_splits else None
            test_idx, val_idx = sklearn_train_test_split(
                test_val_idx, train_size=self.val_test_split, shuffle=True,
                stratify=split_stratify, random_state=self.random_state,
            )
        else:
            test_idx = test_val_idx
            val_idx = np.empty(0, dtype=int)
        return np.sort(train_idx), np.sort(test_idx), np.sort(val_idx)

    def _write_split_masks(self) -> None:
        self.graph.ndata["train_mask"] = self._make_mask(self.train_indexes)
        self.graph.ndata["val_mask"] = self._make_mask(self.val_indexes)
        self.graph.ndata["test_mask"] = self._make_mask(self.test_indexes)

    def _make_mask(self, indexes: np.ndarray):
        label = self._node_data("label")
        if hasattr(label, "device"):
            import torch
            mask = torch.zeros(self.n_samples, dtype=torch.bool, device=label.device)
            mask[torch.as_tensor(indexes, dtype=torch.long, device=label.device)] = True
            return mask
        mask = np.zeros(self.n_samples, dtype=bool)
        mask[indexes] = True
        return mask

    def __repr__(self) -> str:
        return (
            f"GADBenchGraphData(dataset={self.dataset_path.name}, "
            f"n_train={self.n_train}, n_val={self.n_val}, n_test={self.n_test}, "
            f"n_features={self.X_train.shape[1]})"
        )


def load_data(dataset: Union[str, Path],
              preserve_labeled: bool = False,
              labeled_ratio=None,
              train_test_split_ratio: float = 0.8,
              val_test_split_ratio: float = 0.0,
              **kwargs) -> GADBenchGraphData:
    """Shortcut to load an official GADBench graph dataset."""
    return GADBenchGraphData(
        dataset,
        train_test_split_ratio=train_test_split_ratio,
        val_test_split_ratio=val_test_split_ratio,
        preserve_labeled=preserve_labeled,
        labeled_ratio=labeled_ratio,
        **kwargs,
    )
