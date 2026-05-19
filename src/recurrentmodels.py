import copy
from typing import Optional, List

import numpy as np
import torch
from torch import nn

from base import RecurrentModel


class GRURecurrentModel(RecurrentModel):
    """GRU that scores anomalies from stacked detector embeddings"""

    def __init__(self, hidden_size: int = 128, num_layers: int = 1, dropout: float = 0.0,
                 lr: float = 1e-3, batch_size: int = 256, num_epochs: int = 50,
                 seed: Optional[int] = None, max_grad_norm: float = 1.0,
                 n_detectors: Optional[int] = None, device: Optional[str] = None):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.n_detectors = n_detectors
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self._gru: Optional[nn.GRU] = None
        self._classifier: Optional[nn.Linear] = None
        self._optimizer = None
        self._criterion = nn.BCEWithLogitsLoss()
        self._input_size: Optional[int] = None
        self._fitted = False
        self._last_loss: Optional[float] = None
        self._train_loss_history: List[float] = []
        self._val_loss_history: List[float] = []
        # NEW 
        self._best_val_auc = -float("inf")
        self._best_state = None
        self._invert = False
        # NEW
        
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    # def _build(self, input_size: int) -> None:
    #     self._gru = nn.GRU(input_size=input_size, hidden_size=self.hidden_size,
    #                        num_layers=self.num_layers,
    #                        dropout=self.dropout if self.num_layers > 1 else 0.0,
    #                        batch_first=True, bidirectional=True).to(self.device)
    #     self._classifier = nn.Linear(self.hidden_size * 2, 1).to(self.device)
    #     self._optimizer = torch.optim.Adam(
    #         list(self._gru.parameters()) + list(self._classifier.parameters()), lr=self.lr)
    #     self._input_size = input_size

    def _build(self, input_size: int) -> None:
        self._gru = nn.GRU(input_size=input_size, hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           dropout=self.dropout if self.num_layers > 1 else 0.0,
                           batch_first=True, bidirectional=True).to(self.device)
        self._classifier = nn.Linear(self.hidden_size * 2, 1).to(self.device)
        self._optimizer = torch.optim.Adam(
            list(self._gru.parameters()) + list(self._classifier.parameters()), lr=self.lr)
        self._input_size = input_size
        self._best_val_auc = -float("inf")
        self._best_state = None
        self._invert = False
        
    def _prepare_batch(self, embeddings: np.ndarray, labels: Optional[np.ndarray] = None):
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"embeddings must be 3D [n_samples, n_detectors, feature_dim], got shape {x.shape}")
        if self.n_detectors is not None:
            assert x.shape[1] == self.n_detectors, \
                f"expected n_detectors={self.n_detectors}, got {x.shape[1]}"

        if self._gru is None or self._input_size != x.shape[-1]:
            self._build(x.shape[-1])
        xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        if labels is None:
            return xt, None, None
        y = np.asarray(labels, dtype=np.float32).ravel()
        if y.shape[0] != xt.shape[0]:
            raise ValueError("labels length != n_samples")
        mask = y != -1
        if not np.any(mask):
            return xt, None, None
        return (xt,
                torch.as_tensor(y, dtype=torch.float32, device=self.device),
                torch.as_tensor(mask, dtype=torch.bool, device=self.device))

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self._gru(x)
        h = h_n.view(self.num_layers, 2, -1, self.hidden_size)[-1]
        return self._classifier(torch.cat([h[0], h[1]], dim=-1)).squeeze(-1)


    def _train_epochs(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                    epochs: int, x_val: Optional[torch.Tensor] = None,
                    y_val: Optional[torch.Tensor] = None,
                    mask_val: Optional[torch.Tensor] = None) -> None:
        xl, yl = x[mask], y[mask]
        n = xl.size(0)
        if n == 0:
            return
        if len(torch.unique(yl)) < 2:
            return
        params = list(self._gru.parameters()) + list(self._classifier.parameters())

        for _ in range(max(1, epochs)):
            self._gru.train()
            self._classifier.train()
            perm = torch.randperm(n, device=self.device)
            losses = []
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                logits = self._forward(xl[idx])
                loss = self._criterion(logits, yl[idx])
                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=self.max_grad_norm)
                self._optimizer.step()
                losses.append(loss.item())
            
            if losses:
                epoch_loss = float(np.mean(losses))
                self._last_loss = epoch_loss
                self._train_loss_history.append(epoch_loss)
                self._fitted = True
                if x_val is not None and y_val is not None and mask_val is not None and mask_val.any():
                    self._gru.eval()
                    self._classifier.eval()
                    with torch.no_grad():
                        v_loss = self._criterion(self._forward(x_val[mask_val]), y_val[mask_val]).item()
                    self._val_loss_history.append(v_loss)
    def update_best(self, val_auc: float) -> bool:
        if self._gru is None or self._classifier is None:
            return False
        if val_auc is None or not np.isfinite(val_auc):
            return False

        effective_auc = max(float(val_auc), 1.0 - float(val_auc))
        if effective_auc <= self._best_val_auc:
            return False

        self._best_val_auc = effective_auc
        self._best_state = {
            "gru": copy.deepcopy(self._gru.state_dict()),
            "classifier": copy.deepcopy(self._classifier.state_dict()),
            "invert": float(val_auc) < 0.5,
        }
        return True

    def restore_best(self) -> bool:
        if self._best_state is None or self._gru is None or self._classifier is None:
            return False

        self._gru.load_state_dict(self._best_state["gru"])
        self._classifier.load_state_dict(self._best_state["classifier"])
        self._invert = bool(self._best_state["invert"])
        return True
    
    def train(self, aggregated_embeddings: np.ndarray, labels: np.ndarray,
              epochs: int = 1, val_embeddings: Optional[np.ndarray] = None,
              val_labels: Optional[np.ndarray] = None) -> None:
        x, y, mask = self._prepare_batch(aggregated_embeddings, labels)
        if y is None or mask is None:
            return
        xv, yv, mv = (None, None, None)
        if val_embeddings is not None and val_labels is not None:
            xv, yv, mv = self._prepare_batch(val_embeddings, val_labels)
        self._train_epochs(x, y, mask, epochs, xv, yv, mv)

    def fit(self, aggregated_embeddings: np.ndarray, labels: np.ndarray) -> None:
        self.train(aggregated_embeddings, labels, epochs=self.num_epochs)

    def get_loss(self, aggregated_embeddings: np.ndarray, labels: np.ndarray) -> Optional[float]:
        x, y, mask = self._prepare_batch(aggregated_embeddings, labels)
        if y is None or mask is None:
            return None
        self._gru.eval()
        self._classifier.eval()
        with torch.no_grad():
            return self._criterion(self._forward(x[mask]), y[mask]).item()

    # def predict_scores(self, aggregated_embeddings: np.ndarray) -> np.ndarray:
    #     if not self._fitted:
    #         raise RuntimeError("GRURecurrentModel must be fit before predict_scores()")
    #     x, _, _ = self._prepare_batch(aggregated_embeddings)
    #     self._gru.eval()
    #     self._classifier.eval()
    #     with torch.no_grad():
    #         return torch.sigmoid(self._forward(x)).cpu().numpy().astype(np.float32).ravel()

    def predict_scores(self, aggregated_embeddings: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("GRURecurrentModel must be fit before predict_scores()")
        x, _, _ = self._prepare_batch(aggregated_embeddings)
        self._gru.eval()
        self._classifier.eval()
        with torch.no_grad():
            scores = torch.sigmoid(self._forward(x)).cpu().numpy().astype(np.float32).ravel()
        return (1.0 - scores).astype(np.float32) if self._invert else scores