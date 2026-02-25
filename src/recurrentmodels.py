from typing import Optional

import numpy as np
import torch
from torch import nn

from base import RecurrentModel


class GRURecurrentModel(RecurrentModel):
	"""GRU-based recurrent ensemble model operating over model embeddings."""

	def __init__(
		self,
		hidden_size: int = 128,
		num_layers: int = 1,
		dropout: float = 0.0,
		lr: float = 1e-3,
		batch_size: int = 256,
		num_epochs: int = 50,
		seed: Optional[int] = None,
		max_grad_norm: float = 1.0,
		n_detectors: Optional[int] = None,
		device: Optional[str] = None,
	) -> None:
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.dropout = dropout
		self.lr = lr
		self.batch_size = batch_size
		self.num_epochs = num_epochs
		self.max_grad_norm = max_grad_norm
		self.n_detectors = n_detectors
		self.device = torch.device(
			device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
		)

		self._gru: Optional[nn.GRU] = None
		self._classifier: Optional[nn.Linear] = None
		self._optimizer: Optional[torch.optim.Optimizer] = None
		self._criterion = nn.BCEWithLogitsLoss() #might add options 
		self._input_size: Optional[int] = None
		self._fitted = False
		self._last_loss: Optional[float] = None
		self._train_loss_history: list[float] = []
		self._val_loss_history: list[float] = []

		if seed is not None:
			self._set_seed(seed)

	def _set_seed(self, seed: int) -> None:
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)

	def _build(self, input_size: int) -> None:
		self._gru = nn.GRU(
			input_size=input_size,
			hidden_size=self.hidden_size,
			num_layers=self.num_layers,
			dropout=self.dropout if self.num_layers > 1 else 0.0,
			batch_first=True,
		)
		self._classifier = nn.Linear(self.hidden_size, 1)
		self._gru.to(self.device)
		self._classifier.to(self.device)
		self._optimizer = torch.optim.Adam(
			list(self._gru.parameters()) + list(self._classifier.parameters()),
			lr=self.lr,
		)
		self._input_size = input_size

	def _prepare_batch(
		self,
		aggregated_embeddings: np.ndarray,
		labels: Optional[np.ndarray] = None,
	):
		x_np = np.asarray(aggregated_embeddings, dtype=np.float32)
		if x_np.ndim == 2:
			if self.n_detectors is not None and self.n_detectors > 1:
				if x_np.shape[1] % self.n_detectors != 0:
					raise ValueError("feature_dim must be divisible by n_detectors")
				x_np = x_np.reshape(x_np.shape[0], self.n_detectors, x_np.shape[1] // self.n_detectors)
			else:
				x_np = x_np[:, None, :]
		if x_np.ndim != 3:
			raise ValueError("aggregated_embeddings must have shape [n_samples, seq_len, feature_dim]")

		if self._gru is None or self._classifier is None or self._input_size != x_np.shape[-1]:
			self._build(x_np.shape[-1])
		x = torch.as_tensor(x_np, dtype=torch.float32, device=self.device)

		if labels is None:
			return x, None, None

		y_np = np.asarray(labels, dtype=np.float32).reshape(-1)
		if y_np.shape[0] != x.shape[0]:
			raise ValueError("labels length does not match number of samples")
		mask_np = y_np != -1
		if not np.any(mask_np):
			return x, None, None
		y = torch.as_tensor(y_np, dtype=torch.float32, device=self.device)
		mask = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)
		return x, y, mask

	def _forward_logits(self, x: torch.Tensor) -> torch.Tensor:
		if self._gru is None or self._classifier is None:
			raise RuntimeError("Model is not initialized.")
		_, h_n = self._gru(x)
		return self._classifier(h_n[-1]).squeeze(-1)

	def _train_epochs(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, epochs: int) -> None:
		if self._optimizer is None:
			raise RuntimeError("Optimizer not initialized.")

		x_labeled = x[mask]
		y_labeled = y[mask]
		n_samples = x_labeled.size(0)
		if n_samples == 0:
			return

		for _ in range(max(1, epochs)):
			self._gru.train()
			self._classifier.train()
			perm = torch.randperm(n_samples, device=self.device)
			epoch_losses = []
			for start in range(0, n_samples, self.batch_size):
				idx = perm[start : start + self.batch_size]
				xb = x_labeled[idx]
				yb = y_labeled[idx]

				logits = self._forward_logits(xb)
				loss = self._criterion(logits, yb)

				self._optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(
					list(self._gru.parameters()) + list(self._classifier.parameters()),
					max_norm=self.max_grad_norm,
				)
				self._optimizer.step()
				batch_loss = float(loss.detach().cpu().item())
				epoch_losses.append(batch_loss)

			if epoch_losses:
				epoch_loss = float(np.mean(epoch_losses))
				self._last_loss = epoch_loss
				self._train_loss_history.append(epoch_loss)

		self._fitted = True

	def train(self, aggregated_embeddings: np.ndarray, labels: np.ndarray, epochs: int = 1) -> None:
		x, y, mask = self._prepare_batch(aggregated_embeddings, labels)
		if y is None or mask is None:
			return
		self._train_epochs(x, y, mask, epochs=epochs)

	def fit(self, aggregated_embeddings: np.ndarray, labels: np.ndarray) -> None:
		self.train(aggregated_embeddings, labels, epochs=self.num_epochs)

	def get_loss(self, aggregated_embeddings: np.ndarray, labels: np.ndarray) -> Optional[float]:
		x, y, mask = self._prepare_batch(aggregated_embeddings, labels)
		if y is None or mask is None:
			return None
		self._gru.eval()
		self._classifier.eval()
		with torch.no_grad():
			logits = self._forward_logits(x[mask])
			loss = self._criterion(logits, y[mask])
		return float(loss.detach().cpu().item())

	def predict_scores(self, aggregated_embeddings: np.ndarray) -> np.ndarray:
		if not self._fitted:
			raise RuntimeError("GRURecurrentModel must be fit before predict_scores().")

		x, _, _ = self._prepare_batch(aggregated_embeddings)
		self._gru.eval()
		self._classifier.eval()
		with torch.no_grad():
			return torch.sigmoid(self._forward_logits(x)).detach().cpu().numpy().astype(np.float32).reshape(-1)

