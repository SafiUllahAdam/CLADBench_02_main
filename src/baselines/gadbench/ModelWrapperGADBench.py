"""GADBench graph model wrappers for the CoBench Model interface."""

from typing import Dict, Optional
import importlib.util
import random

import numpy as np
from sklearn.metrics import log_loss

from base import Model
from benchmark_config import PROJECT_ROOT

try:
    import torch
    import torch.nn.functional as F
    import dgl
    import dgl.function as fn
    GRAPH_DEPS_ERROR = None
except ImportError as e:
    torch = None
    F = None
    dgl = None
    fn = None
    GRAPH_DEPS_ERROR = e

try:
    import xgboost as xgb
    XGBOOST_ERROR = None
except ImportError as e:
    xgb = None
    XGBOOST_ERROR = e

try:
    gnn_path = PROJECT_ROOT / "SubModules" / "GADBench" / "models" / "gnn.py"
    spec = importlib.util.spec_from_file_location("_gadbench_gnn", gnn_path)
    gadbench_gnn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gadbench_gnn)
    BWGNN = gadbench_gnn.BWGNN
    GraphSAGE = gadbench_gnn.GraphSAGE
    BWGNN_ERROR = None
except Exception as e:
    BWGNN = None
    GraphSAGE = None
    BWGNN_ERROR = e


def get_model_detector_dict() -> Dict[str, type]:
    return {
        "bwgnn": BWGNNWrapper,
        "BWGNN": BWGNNWrapper,
        "XGBGraph": XGBGraphWrapper,
        "xgbgraph": XGBGraphWrapper,
        "graphsage": GraphSAGEWrapper,
        "GraphSAGE": GraphSAGEWrapper,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_edge_index(edge_index) -> tuple:
    edge_index = np.asarray(edge_index)
    if edge_index.ndim != 2:
        raise ValueError("edge_index must be a 2D array with shape (2, E) or (E, 2)")
    if edge_index.shape[0] == 2:
        return edge_index[0], edge_index[1]
    if edge_index.shape[1] == 2:
        return edge_index[:, 0], edge_index[:, 1]
    raise ValueError("edge_index must have shape (2, E) or (E, 2)")


class _GraphWrapperBase(Model):
    """Shared graph/split plumbing for GADBench-style node detectors."""

    def __init__(self, train_config: dict, model_config: dict, data):
        if GRAPH_DEPS_ERROR is not None:
            raise ImportError("dgl and torch are required for GADBench wrappers") from GRAPH_DEPS_ERROR
        super().__init__(train_config=train_config, model_config=model_config, data=None)
        self.data = data
        self.device = self._resolve_device(self.train_config.get("device", "auto"))
        self.inductive = bool(self.train_config.get("inductive", False))
        self.graph = self._build_graph(data).to(self.device)
        self.features = self.graph.ndata["feature"]
        self.labels_full = self.graph.ndata["label"].long()
        self.train_node_idx, self.val_node_idx, self.test_node_idx = self._split_nodes(data)
        self._init_base_arrays()
        self._train_loss_history = []
        self._val_loss_history = []
        self._last_train_loss = None
        self._last_val_loss = None

    @staticmethod
    def _resolve_device(device: str):
        if device in (None, "auto"):
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _build_graph(self, data):
        graph = getattr(data, "graph", None)
        if graph is None and isinstance(data, dict):
            graph = data.get("graph")
        if graph is not None:
            graph = graph.clone()
            if "feature" not in graph.ndata or "label" not in graph.ndata:
                raise ValueError("GADBench graph needs ndata['feature'] and ndata['label']")
            graph.ndata["feature"] = graph.ndata["feature"].float()
            graph.ndata["label"] = graph.ndata["label"].long()
            return graph

        X = self._get_any(data, "X", "features", required=False)
        y = self._get_any(data, "y", "labels", required=False)
        edge_index = self._get_any(data, "edge_index", required=True)
        if X is None or y is None:
            X_split, y_split = self._stack_split_arrays(data)
            X = X_split if X is None else X
            y = y_split if y is None else y
        src, dst = _as_edge_index(edge_index)
        graph = dgl.graph((src, dst), num_nodes=len(X))
        graph.ndata["feature"] = torch.as_tensor(X, dtype=torch.float32)
        graph.ndata["label"] = torch.as_tensor(y, dtype=torch.long)
        if self.train_config.get("add_self_loop", False):
            graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
        return graph

    @staticmethod
    def _get_any(obj, *keys, required=True):
        for key in keys:
            val = getattr(obj, key, None)
            if val is not None:
                return val
            if isinstance(obj, dict) and obj.get(key) is not None:
                return obj[key]
        if required:
            raise KeyError(f"Missing: {keys}")
        return None

    def _stack_split_arrays(self, data):
        X_train = self._get_any(data, "X_train")
        X_test = self._get_any(data, "X_test")
        X_val = self._get_any(data, "X_val", required=False)
        y_train = self._get_any(data, "y_train_original", "y_train")
        y_test = self._get_any(data, "y_test")
        y_val = self._get_any(data, "y_val", required=False)
        X_parts = [X_train] + ([X_val] if X_val is not None else []) + [X_test]
        y_parts = [y_train] + ([y_val] if y_val is not None else []) + [y_test]
        return np.vstack(X_parts), np.concatenate(y_parts)

    def _split_nodes(self, data):
        masks = [self._mask_to_idx(name) for name in ("train_mask", "val_mask", "test_mask")]
        if masks[0] is not None and masks[2] is not None:
            return masks

        split_idx = [self._get_any(data, name, required=False)
                     for name in ("train_indexes", "val_indexes", "test_indexes")]
        if split_idx[0] is not None and split_idx[2] is not None:
            train = torch.as_tensor(split_idx[0], dtype=torch.long, device=self.device)
            val = None if split_idx[1] is None else torch.as_tensor(split_idx[1], dtype=torch.long, device=self.device)
            test = torch.as_tensor(split_idx[2], dtype=torch.long, device=self.device)
            return train, val, test

        X_val = self._get_any(data, "X_val", required=False)
        n_train = int(getattr(data, "n_train", 0) or len(self._get_any(data, "X_train")))
        n_val = int(getattr(data, "n_val", 0) or (0 if X_val is None else len(X_val)))
        n_test = int(getattr(data, "n_test", 0) or len(self._get_any(data, "X_test")))
        train = torch.arange(n_train, device=self.device)
        val = torch.arange(n_train, n_train + n_val, device=self.device)
        test = torch.arange(n_train + n_val, n_train + n_val + n_test, device=self.device)
        return train, val, test

    def _mask_to_idx(self, name: str):
        mask = self.graph.ndata.get(name)
        if mask is None:
            return None
        return torch.nonzero(mask.bool(), as_tuple=False).reshape(-1).to(self.device)

    def _init_base_arrays(self) -> None:
        full_X = self.features.detach().cpu().numpy()
        full_y = self.labels_full.detach().cpu().numpy()
        train = self.train_node_idx.detach().cpu().numpy()
        test = self.test_node_idx.detach().cpu().numpy()
        val = None if self.val_node_idx is None else self.val_node_idx.detach().cpu().numpy()
        self.X_train = full_X[train]
        self.X_test = full_X[test]
        self.X_val = None if val is None or len(val) == 0 else full_X[val]
        self.y_train_original = full_y[train]
        self.y_test = full_y[test]
        self.y_val = None if val is None or len(val) == 0 else full_y[val]

    def _known_train_rows(self):
        """Train rows with known {0,1} labels (graph-view local when inductive) and labels."""
        y = torch.as_tensor(np.array(self.y_train), dtype=torch.long, device=self.device)
        mask = torch.isin(y, torch.tensor([0, 1], device=self.device))
        if mask.sum() == 0 or torch.unique(y[mask]).numel() < 2:
            raise ValueError("GADBench wrappers need both normal and anomaly training labels")
        base = torch.arange(len(y), device=self.device) if self.inductive else self.train_node_idx
        return base[mask], y[mask]

    def _node_indexes(self, indexes=None, use_train: bool = False, use_val: bool = False):
        nodes = self.train_node_idx if use_train else (self.val_node_idx if use_val else self.test_node_idx)
        if nodes is None:
            raise ValueError(f"No graph split for use_train={use_train}, use_val={use_val}")
        return nodes if indexes is None else nodes[torch.as_tensor(indexes, dtype=torch.long, device=self.device)]

    @staticmethod
    def _class_weight(labels):
        positives = torch.sum(labels == 1).float()
        negatives = torch.sum(labels == 0).float()
        return torch.tensor([1.0, (negatives / positives).item()], device=labels.device)

    def fit(self) -> None:
        remaining = self.train_config["total_epochs"] - self._current_epoch
        if remaining > 0:
            self.train(remaining)
        self._fitted = True

    def get_loss(self, use_val: bool = False) -> Optional[float]:
        if not self._fitted:
            return None
        return self._last_val_loss if use_val else self._last_train_loss


class BWGNNWrapper(_GraphWrapperBase):
    """BWGNN node anomaly detector (transductive: trained on the full graph)."""
    _gnn = BWGNN
    _inductive_default = False
    _gnn_cfg = {"h_feats": 32, "num_classes": 2, "num_layers": 2,
                "mlp_layers": 2, "dropout_rate": 0.0, "activation": "ReLU"}

    def __init__(self, train_config: dict, model_config: dict, data):
        if BWGNN_ERROR is not None:
            raise ImportError("Could not import GADBench GNN models") from BWGNN_ERROR
        defaults = {"seed": 42, "total_epochs": 100, "lr": 0.01, "weight_decay": 0.0,
                    "device": "auto", "inductive": self._inductive_default}
        config = {**defaults, **(train_config or {})}
        super().__init__(train_config=config, model_config=model_config or {}, data=data)
        _set_seed(self.train_config["seed"])
        self.train_graph, self.eval_graph, self._val_rows = self._graph_views()
        self.model = self._gnn(**self._model_defaults()).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.train_config["lr"],
            weight_decay=self.train_config["weight_decay"],
        )

    def _graph_views(self):
        """Train/eval graph views; inductive runs hide held-out nodes via node subgraphs."""
        if not self.inductive:
            return self.graph, self.graph, self.val_node_idx
        train_graph = dgl.node_subgraph(self.graph, self.train_node_idx)
        if self.val_node_idx is None or self.val_node_idx.numel() == 0:
            return train_graph, train_graph, self.val_node_idx
        eval_graph = dgl.node_subgraph(self.graph, torch.cat([self.train_node_idx, self.val_node_idx]))
        n_train, n_val = self.train_node_idx.numel(), self.val_node_idx.numel()
        return train_graph, eval_graph, torch.arange(n_train, n_train + n_val, device=self.device)

    def _model_defaults(self) -> dict:
        cfg = {"in_feats": self.features.shape[1], **self._gnn_cfg}
        cfg.update({**self.train_config, **self.model_config})
        if "drop_rate" in cfg and "dropout_rate" not in self.model_config:
            cfg["dropout_rate"] = cfg.pop("drop_rate")
        for key in ("model", "lr", "seed", "total_epochs", "weight_decay",
                    "device", "add_self_loop", "inductive"):
            cfg.pop(key, None)
        return cfg

    def train(self, epochs: int = 1) -> None:
        for _ in range(epochs):
            rows, labels = self._known_train_rows()
            self.model.train()
            logits = self.model(self.train_graph)[rows]
            loss = F.cross_entropy(logits, labels, weight=self._class_weight(labels))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self._current_epoch += 1
            self._fitted = True
            self._record_losses(float(loss.detach().cpu()))

    def _record_losses(self, train_loss: float) -> None:
        self._last_train_loss = train_loss
        self._train_loss_history.append(train_loss)
        if self.val_node_idx is not None and self.val_node_idx.numel() > 0:
            val_loss = self._compute_loss()
            self._last_val_loss = val_loss
            self._val_loss_history.append(val_loss)

    def _compute_loss(self) -> float:
        """Validation cross-entropy on the eval graph view (train+val nodes when inductive)."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.eval_graph)[self._val_rows]
            return float(F.cross_entropy(logits, self.labels_full[self.val_node_idx]).detach().cpu())

    def _logits(self, nodes):
        self.model.eval()
        with torch.no_grad():
            return self.model(self.graph)[nodes]

    def predict_scores(self, indexes: Optional[np.ndarray] = None,
                       use_train: bool = False, use_val: bool = False) -> np.ndarray:
        if not self._fitted:
            self.fit()
        nodes = self._node_indexes(indexes, use_train=use_train, use_val=use_val)
        return torch.softmax(self._logits(nodes), dim=1)[:, 1].detach().cpu().numpy()

    def get_embeddings(self, indexes: Optional[np.ndarray] = None,
                       use_train: bool = True, use_val: bool = False) -> np.ndarray:
        if not self._fitted:
            self.fit()
        nodes = self._node_indexes(indexes, use_train=use_train, use_val=use_val)
        return self._logits(nodes).detach().cpu().numpy()


class GraphSAGEWrapper(BWGNNWrapper):
    """GraphSAGE detector: inductive minibatch training with neighbor sampling (Hamilton et al., 2017)."""
    _gnn = GraphSAGE
    _inductive_default = True
    _gnn_cfg = {"h_feats": 32, "num_classes": 2, "num_layers": 2,
                "agg": "pool", "dropout_rate": 0.0, "activation": "ReLU"}

    def __init__(self, train_config: dict, model_config: dict, data):
        super().__init__(train_config, model_config, data)
        self._sampler, self._batch_size = self._build_sampler()

    def _build_sampler(self):
        """Neighbor sampler with one fanout per GNN layer (paper default S1=25, S2=10)."""
        params = {**self.train_config, **self.model_config}
        n_layers = len(self.model.layers)
        fanout = params.get("fanout", [25, 10])
        fanout = [int(fanout)] * n_layers if isinstance(fanout, int) else [int(f) for f in fanout]
        if len(fanout) != n_layers:
            raise ValueError(f"fanout needs one value per GNN layer ({n_layers})")
        return dgl.dataloading.NeighborSampler(fanout), int(params.get("batch_size", 512))

    def train(self, epochs: int = 1) -> None:
        rows, labels = self._known_train_rows()
        weight = self._class_weight(labels)
        label_vec = torch.full((self.train_graph.num_nodes(),), -1, dtype=torch.long, device=self.device)
        label_vec[rows] = labels
        loader = dgl.dataloading.DataLoader(
            self.train_graph, rows, self._sampler, batch_size=self._batch_size,
            shuffle=True, drop_last=False, device=self.device)
        for _ in range(epochs):
            self.model.train()
            epoch_loss, n_seeds = 0.0, 0
            for input_nodes, output_nodes, blocks in loader:
                logits = self._forward_blocks(input_nodes, blocks)
                loss = F.cross_entropy(logits, label_vec[output_nodes], weight=weight)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss.detach().cpu()) * len(output_nodes)
                n_seeds += len(output_nodes)
            self._current_epoch += 1
            self._fitted = True
            self._record_losses(epoch_loss / max(n_seeds, 1))

    def _forward_blocks(self, input_nodes, blocks):
        """Forward SAGEConv layers along the sampled message-flow blocks."""
        h = self.train_graph.ndata["feature"][input_nodes]
        for layer, block in zip(self.model.layers, blocks):
            h = layer(block, self.model.dropout(h))
        return self.model.output_linear(h)


class XGBGraphWrapper(_GraphWrapperBase):
    """XGBoost on graph-aggregated node features."""

    def __init__(self, train_config: dict, model_config: dict, data):
        if XGBOOST_ERROR is not None:
            raise ImportError("xgboost is required for XGBGraphWrapper") from XGBOOST_ERROR
        defaults = {"seed": 42, "total_epochs": 1, "device": "auto"}
        config = {**defaults, **(train_config or {})}
        super().__init__(train_config=config, model_config=model_config or {}, data=data)
        self.X_graph = self._graph_features().detach().cpu().numpy()
        self.model = self._build_model()

    def _build_model(self):
        cfg = {
            "n_estimators": 100, "learning_rate": 0.05, "max_depth": 6,
            "subsample": 1.0, "tree_method": "hist", "eval_metric": "logloss",
            "random_state": self.train_config["seed"], "n_jobs": -1,
        }
        cfg.update({**self.train_config, **self.model_config})
        for key in (
            "model", "num_layers", "agg", "lr", "drop_rate", "h_feats",
            "seed", "total_epochs", "device", "add_self_loop",
        ):
            cfg.pop(key, None)
        return xgb.XGBClassifier(**cfg)

    def _graph_features(self):
        h = self.features
        out = [h.detach()]
        params = {**self.train_config, **self.model_config}
        agg = params.get("agg", "mean")
        for _ in range(int(params.get("num_layers", 2))):
            h = self._aggregate(h, agg)
            out.append(h.detach())
        return torch.cat(out, dim=1)

    def _aggregate(self, h, agg: str):
        with self.graph.local_scope():
            self.graph.ndata["h"] = h
            if agg == "sum":
                self.graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            elif agg == "max":
                self.graph.update_all(fn.copy_u("h", "m"), fn.max("m", "h"))
                self.graph.ndata["h"] = torch.nan_to_num(self.graph.ndata["h"], neginf=0.0)
            elif agg == "mean":
                self.graph.update_all(fn.copy_u("h", "m"), fn.mean("m", "h"))
            else:
                raise ValueError("agg must be one of: mean, sum, max")
            return self.graph.ndata["h"]

    def train(self, epochs: int = 1) -> None:
        for _ in range(epochs):
            nodes, labels = self._known_train_rows()
            idx = nodes.detach().cpu().numpy()
            y = labels.detach().cpu().numpy()
            weights = np.where(y == 0, 1.0, max(1.0, np.sum(y == 0) / max(np.sum(y == 1), 1)))
            self.model = self._build_model()
            self.model.fit(self.X_graph[idx], y, sample_weight=weights, verbose=False)
            self._current_epoch += 1
            self._fitted = True
            self._record_losses()

    def _record_losses(self) -> None:
        train_loss = self._compute_loss(self.train_node_idx.detach().cpu().numpy(), self.y_train)
        self._last_train_loss = train_loss
        self._train_loss_history.append(train_loss)
        if self.val_node_idx is not None and self.y_val is not None and len(self.y_val) > 0:
            val_loss = self._compute_loss(self.val_node_idx.detach().cpu().numpy(), self.y_val)
            self._last_val_loss = val_loss
            self._val_loss_history.append(val_loss)

    def _compute_loss(self, nodes: np.ndarray, y: np.ndarray) -> Optional[float]:
        mask = np.isin(y, [0, 1])
        if np.unique(y[mask]).size < 2:
            return None
        scores = np.clip(self._proba(nodes[mask]), 1e-7, 1.0 - 1e-7)
        return float(log_loss(y[mask], scores, labels=[0, 1]))

    def _proba(self, nodes: np.ndarray) -> np.ndarray:
        proba = np.asarray(self.model.predict_proba(self.X_graph[nodes]))
        classes = np.asarray(self.model.classes_)
        pos = np.where(classes == 1)[0]
        if len(pos) == 0:
            return np.zeros(len(nodes), dtype=np.float32)
        return proba[:, pos[0]].astype(np.float32)

    def predict_scores(self, indexes: Optional[np.ndarray] = None,
                       use_train: bool = False, use_val: bool = False) -> np.ndarray:
        if not self._fitted:
            self.fit()
        nodes = self._node_indexes(indexes, use_train=use_train, use_val=use_val)
        return np.clip(self._proba(nodes.detach().cpu().numpy()), 0.0, 1.0)

    def get_embeddings(self, indexes: Optional[np.ndarray] = None,
                       use_train: bool = True, use_val: bool = False) -> np.ndarray:
        if not self._fitted:
            self.fit()
        nodes = self._node_indexes(indexes, use_train=use_train, use_val=use_val)
        scores = self._proba(nodes.detach().cpu().numpy())
        return scores.reshape(-1, 1)
