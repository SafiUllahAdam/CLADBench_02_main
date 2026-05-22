import torch
import torch.nn as nn
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

        effective_auc = float(val_auc)
        if effective_auc <= self._best_val_auc:
            return False

        self._best_val_auc = effective_auc
        self._best_state = {
            "gru": copy.deepcopy(self._gru.state_dict()),
            "classifier": copy.deepcopy(self._classifier.state_dict()),
        }
        return True

    def restore_best(self) -> bool:
        if self._best_state is None or self._gru is None or self._classifier is None:
            return False

        self._gru.load_state_dict(self._best_state["gru"])
        self._classifier.load_state_dict(self._best_state["classifier"])
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
            return torch.sigmoid(self._forward(x)).cpu().numpy().astype(np.float32).ravel()
    
    
    
class LSTMRecurrentModel(GRURecurrentModel):
    """LSTM recurrent judge with the same API as GRURecurrentModel"""

    def _build(self, input_size: int) -> None:
        self._input_size = input_size
        self._gru = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=False,
        ).to(self.device)

        self._classifier = nn.Linear(self.hidden_size, 1).to(self.device)

        params = list(self._gru.parameters()) + list(self._classifier.parameters())
        self._optimizer = torch.optim.Adam(
            params,
            lr=self.lr,
            weight_decay=getattr(self, "weight_decay", 0.0),
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self._gru(x)
        h = h_n[-1]
        return self._classifier(h).squeeze(-1)
    
    
    
class SlidingWindowLSTMRecurrentModel(LSTMRecurrentModel):
    """LSTM judge: one continuous pass, sliding-window read, max-pool over windows"""

    def __init__(self, *args, window_size: int = 6, stride: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size
        self.stride = stride

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self._gru(x)                                  # one continuous pass, hidden flows whole sequence
        seq_len = out.size(1)
        if seq_len < self.window_size:
            return self._classifier(out[:, -1]).squeeze(-1)    # short sequence: score final continuous state
        starts = range(0, seq_len - self.window_size + 1, self.stride)
        logits = [self._classifier(out[:, s + self.window_size - 1]) for s in starts]  # window-end state, full prior context
        return torch.stack(logits, dim=1).squeeze(-1).max(dim=1).values                # strongest window, no dilution


class _DeepSetsPool(nn.Module):
    """Per-detector MLP then permutation-invariant pooling (mean or attention)"""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0, pool: str = "attention"):
        super().__init__()
        self.pool = pool
        self.phi = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_size, hidden_size), nn.ReLU())
        self.attn = nn.Linear(hidden_size, 1) if pool == "attention" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.phi(x)                                    # [B, n_detectors, H] per-detector encoding
        if self.pool == "mean":
            return h.mean(dim=1)                           # permutation-invariant mean
        w = torch.softmax(self.attn(h), dim=1)             # [B, n_detectors, 1] attention over detectors
        return (w * h).sum(dim=1)                          # weighted permutation-invariant sum


class PermInvariantModel(GRURecurrentModel):
    """Permutation-invariant judge: per-detector MLP + mean/attention pooling (DeepSets)"""

    def __init__(self, *args, pool: str = "attention", **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = pool

    def _build(self, input_size: int) -> None:
        self._gru = _DeepSetsPool(input_size, self.hidden_size, self.dropout, self.pool).to(self.device)
        self._classifier = nn.Linear(self.hidden_size, 1).to(self.device)
        self._optimizer = torch.optim.Adam(
            list(self._gru.parameters()) + list(self._classifier.parameters()), lr=self.lr)
        self._input_size = input_size
        self._best_val_auc = -float("inf")
        self._best_state = None

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._classifier(self._gru(x)).squeeze(-1)  # pool over detectors then score