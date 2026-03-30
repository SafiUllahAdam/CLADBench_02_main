"""ADBench model wrappers for the unified Model API."""

from typing import Dict, Optional
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

from base import Model
from adbench.baseline.PReNet.model import prenet
from adbench.baseline.PReNet.fit import fit
from adbench.myutils import Utils
from adbench.baseline.DeepSAD.src.datasets.main import load_dataset
from adbench.baseline.DeepSAD.src.deepsad import deepsad
from adbench.baseline.DeepSAD.src.optim.DeepSAD_trainer import DeepSADTrainer

# DevNet requires TensorFlow; this is checked lazily in DevNetWrapper.__init__
TF_AVAILABLE = False
try:
    from adbench.baseline.DevNet.run import DevNet
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    DevNet = None
    tf = None

try:
    from adbench.baseline.PyOD import PYOD
    PYOD_AVAILABLE = True
    PYOD_IMPORT_ERROR = None
except (ImportError, ValueError) as e:
    PYOD_AVAILABLE = False
    PYOD_IMPORT_ERROR = e


def get_model_detector_dict() -> Dict[str, type]:
    registry: Dict[str, type] = {
        "prenet": PReNetWrapper,
        "deepsad": DeepSADWrapper,
    }
    if TF_AVAILABLE:
        registry["devnet"] = DevNetWrapper
    if PYOD_AVAILABLE:
        registry["xgbod"] = XGBODWrapper
    return registry


class PReNetWrapper(Model):
    """ADBench PReNet with epoch-level training."""

    def __init__(self, train_config: dict, model_config: dict, data: dict):
        defaults = {
            "seed": 42, "total_epochs": 100, "batch_num": 10,
            "batch_size": 256, "lr": 1e-3, "weight_decay": 1e-2,
            "s_a_a": 8, "s_a_u": 4, "s_u_u": 0,
        }
        config = {**defaults, **(train_config or {})}
        super().__init__(train_config=config, model_config=model_config, data=data)

        self.utils = Utils()
        self.device = self.utils.get_device(gpu_specific=True)
        cfg = self.train_config

        self.X_train_tensor = torch.from_numpy(self.X_train).float()

        self.utils.set_seed(cfg["seed"])
        input_size = self.X_train.shape[1]
        self.model = prenet(input_size=input_size, act_fun=torch.nn.ReLU()).to(self.device)
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        )
        self.fit_fn = fit

        self._train_loss_history = []
        self._val_loss_history = []
        self._last_train_loss = None
        self._last_val_loss = None

    def fit(self) -> None:
        remaining = self.train_config["total_epochs"] - self._current_epoch
        if remaining > 0:
            self.train(remaining)
        self._fitted = True

    def train(self, epochs: int = 1) -> None:
        y_train = self.y_train
        cfg = self.train_config

        for _ in range(epochs):
            self.fit_fn(
                X_train_tensor=self.X_train_tensor, y_train=y_train,
                model=self.model, optimizer=self.optimizer, epochs=1,
                batch_num=cfg["batch_num"], batch_size=cfg["batch_size"],
                s_a_a=cfg["s_a_a"], s_a_u=cfg["s_a_u"], s_u_u=cfg["s_u_u"],
                device=self.device,
            )
            self._current_epoch += 1
            self._fitted = True

            train_loss = self._compute_prenet_loss(self.X_train, y_train)
            self._last_train_loss = train_loss
            self._train_loss_history.append(train_loss)

            if self.X_val is not None and self.y_val is not None and len(self.X_val) > 0:
                val_loss = self._compute_prenet_loss(self.X_val, self.y_val)
                self._last_val_loss = val_loss
                self._val_loss_history.append(val_loss)

    def _compute_prenet_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Pairwise ranking loss: anomaly-normal pairs should outscore normal-normal."""
        self.model.eval()
        anomaly_idx = np.where(y == 1)[0]
        normal_idx = np.where(y == 0)[0]
        if len(anomaly_idx) == 0 or len(normal_idx) == 0:
            return 0.0

        X_tensor = torch.from_numpy(X).float()
        num_pairs = min(100, len(anomaly_idx) * len(normal_idx))
        a_idxs = np.random.choice(anomaly_idx, num_pairs, replace=True)
        n_idxs = np.random.choice(normal_idx, num_pairs, replace=True)

        x_a = X_tensor[a_idxs].to(self.device)
        x_n = X_tensor[n_idxs].to(self.device)
        with torch.no_grad():
            score_an = self.model(x_a, x_n)
            score_nn = self.model(x_n, x_n)
            loss = torch.clamp(1.0 - (score_an - score_nn), min=0).mean().item()
        return loss

    def _get_labeled_indexes(self) -> tuple:
        labels = self.y_train
        known = labels != -1
        return np.where((labels == 1) & known)[0], np.where((labels == 0) & known)[0]

    def predict_scores(self, indexes: Optional[np.ndarray] = None, use_train: bool = False, use_val: bool = False) -> np.ndarray:
        if not self._fitted:
            self.fit()

        if use_train:
            X_data = self.X_train
        elif use_val:
            X_data = self.X_val
        else:
            X_data = self.X_test
        X_batch = X_data if indexes is None else X_data[indexes]

        self.model.eval()
        num = 30
        anomaly_idx, normal_idx = self._get_labeled_indexes()
        if len(anomaly_idx) == 0 or len(normal_idx) == 0:
            anomaly_idx = np.where(self.y_train_original == 1)[0]
            normal_idx = np.where(self.y_train_original == 0)[0]

        n = len(X_batch)
        index_a = np.random.choice(anomaly_idx, (n, num), replace=True)
        index_u = np.random.choice(normal_idx, (n, num), replace=True)
        X_batch_tensor = torch.from_numpy(X_batch).float().to(self.device)

        with torch.no_grad():
            scores = np.empty(n, dtype=np.float32)
            chunk_size = 256
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                chunk_len = end - start

                X_i = X_batch_tensor[start:end]
                X_i_rep = X_i.unsqueeze(1).expand(-1, num, -1).reshape(chunk_len * num, -1)
                X_a = self.X_train_tensor[index_a[start:end].ravel()].to(self.device)
                X_u = self.X_train_tensor[index_u[start:end].ravel()].to(self.device)

                score_a_x = self.model(X_a, X_i_rep).reshape(chunk_len, num)
                score_x_u = self.model(X_i_rep, X_u).reshape(chunk_len, num)
                scores[start:end] = (score_a_x + score_x_u).mean(dim=1).cpu().numpy()

        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-8:
            scores = (scores - s_min) / (s_max - s_min)
        else:
            scores = np.full_like(scores, 0.5)
        return scores

    def get_embeddings(self, indexes: Optional[np.ndarray] = None, use_train: bool = True) -> np.ndarray:
        if not self._fitted:
            self.fit()

        X_src = self.X_train if use_train else (self.X_test or self.X_val)
        if X_src is None:
            raise ValueError(f"No data for embeddings (use_train={use_train})")
        X_batch = X_src if indexes is None else X_src[indexes]
        tensor = torch.from_numpy(X_batch).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            return self.model.feature(tensor).cpu().numpy()

    def get_loss(self, use_val: bool = False) -> Optional[float]:
        if not self._fitted:
            return None
        if use_val:
            if self._last_val_loss is not None:
                return self._last_val_loss
            if self.X_val is not None and self.y_val is not None and len(self.X_val) > 0:
                return self._compute_prenet_loss(self.X_val, self.y_val)
            return None
        return self._last_train_loss


class XGBODWrapper(Model):
    """PyOD XGBOD wrapper."""

    def __init__(self, train_config: dict, model_config: dict, data: dict):
        if not PYOD_AVAILABLE:
            raise ImportError(
                "PyOD not available. Install scikit-learn==1.0.2 and pyod==1.0.9 with --no-cache-dir"
            ) from PYOD_IMPORT_ERROR

        config = {**{"seed": 42, "total_epochs": 10}, **(train_config or {})}
        super().__init__(train_config=config, model_config=model_config, data=data)
        self.detector: Optional[PYOD] = None

    def _build_detector(self) -> None:
        self.detector = PYOD(seed=self.train_config["seed"], model_name="XGBOD", tune=False)

    def fit(self) -> None:
        remaining = self.train_config["total_epochs"] - self._current_epoch
        if remaining > 0:
            self.train(remaining)
        self._fitted = True

    def train(self, epochs: int = 1) -> None:
        if self.detector is None:
            self._build_detector()
        y_train = self.y_train
        for _ in range(epochs):
            self.detector.fit(self.X_train, y_train)
            self._current_epoch += 1
        self._fitted = True

    def predict_scores(self, indexes: Optional[np.ndarray] = None, use_train: bool = False, use_val: bool = False) -> np.ndarray:
        if not self._fitted:
            self.fit()
        if use_train:
            X_data = self.X_train
        elif use_val:
            X_data = self.X_val
        else:
            X_data = self.X_test
        X_batch = X_data if indexes is None else X_data[indexes]
        scores = self.detector.model.decision_function(X_batch).astype(float)
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-8:
            scores = (scores - s_min) / (s_max - s_min)
        else:
            scores = np.full_like(scores, 0.5)
        return scores

    def get_embeddings(self, indexes: Optional[np.ndarray] = None, use_train: bool = True) -> np.ndarray:
        if not self._fitted:
            self.fit()
        X_src = self.X_train if use_train else (self.X_test or self.X_val)
        if X_src is None:
            raise ValueError(f"No data for embeddings (use_train={use_train})")
        X_batch = X_src if indexes is None else X_src[indexes]
        # XGBOD only exposes decision scores, no internal embeddings
        return self.detector.model.decision_function(X_batch).astype(float).reshape(-1, 1)


class DeepSADWrapper(Model):
    """ADBench DeepSAD with epoch-level training."""

    def __init__(self, train_config: dict, model_config: dict, data: dict):
        defaults = {
            "seed": 42, "total_epochs": 50, "pretrain": True,
            "ae_epochs": 100, "batch_size": 128, "lr": 0.001,
            "weight_decay": 1e-6, "eta": 1.0, "net_name": "dense",
        }
        config = {**defaults, **(train_config or {})}
        super().__init__(train_config=config, model_config=model_config, data=data)

        self._pretrained = False
        self.utils = Utils()
        self.device = self.utils.get_device(gpu_specific=True)

        cfg = self.train_config
        self.utils.set_seed(cfg["seed"])

        if self.data is not None and hasattr(self.data, 'resolve_labels'):
            y_init = self.data.resolve_labels()
        else:
            y_init = self.y_train_original.copy()
            y_init[y_init == -1] = 0
        self.dataset = load_dataset(data={"X_train": self.X_train, "y_train": y_init}, train=True)
        input_size = self.dataset.train_set.data.size(1)

        self.deepsad = deepsad(cfg["eta"])
        self.deepsad.set_network(cfg["net_name"], input_size)

        self.optimizer_name = "adam"
        self.lr = cfg["lr"]
        self.batch_size = cfg["batch_size"]
        self.weight_decay = cfg["weight_decay"]
        self.lr_milestones = []

        # AE pretrain is label-free; test step needs both classes for AUC
        if cfg["pretrain"] and not self._pretrained:
            self.deepsad.pretrain(
                self.dataset, input_size, optimizer_name="adam",
                lr=cfg["lr"], n_epochs=cfg["ae_epochs"], lr_milestones=(),
                batch_size=cfg["batch_size"], weight_decay=cfg["weight_decay"],
                device=self.device, n_jobs_dataloader=0,
            )
            self._pretrained = True

        self.trainer = None
        self._train_loss_history = []
        self._val_loss_history = []
        self._last_train_loss = None
        self._last_val_loss = None

    def fit(self) -> None:
        remaining = self.train_config["total_epochs"] - self._current_epoch
        if remaining > 0:
            self.train(remaining)
        self._fitted = True

    def train(self, epochs: int = 1) -> None:
        y_train = self.y_train

        # Rebuild dataset only when labels change
        y_hash = hash(y_train.tobytes())
        if not hasattr(self, '_last_y_hash') or self._last_y_hash != y_hash:
            self.dataset = load_dataset(
                data={"X_train": self.X_train, "y_train": y_train}, train=True,
            )
            self._last_y_hash = y_hash

        if self.trainer is None:
            self.trainer = DeepSADTrainer(
                c=self.deepsad.c, eta=self.train_config["eta"],
                optimizer_name=self.optimizer_name, lr=self.lr, n_epochs=1,
                lr_milestones=(), batch_size=self.batch_size,
                weight_decay=self.weight_decay, device=self.device, n_jobs_dataloader=0,
            )

        for _ in range(epochs):
            self.deepsad.net = self.trainer.train(self.dataset, self.deepsad.net)
            self.deepsad.c = self.trainer.c.cpu().data.numpy().tolist()
            self._current_epoch += 1
            self._fitted = True

            train_loss = self._compute_deepsad_loss(self.X_train, self.y_train)
            if train_loss is not None:
                self._last_train_loss = train_loss
                self._train_loss_history.append(train_loss)

            if self.X_val is not None and self.y_val is not None and len(self.X_val) > 0:
                val_loss = self._compute_deepsad_loss(self.X_val, self.y_val)
                if val_loss is not None:
                    self._last_val_loss = val_loss
                    self._val_loss_history.append(val_loss)

    def predict_scores(self, indexes: Optional[np.ndarray] = None, use_train: bool = False, use_val: bool = False) -> np.ndarray:
        if not self._fitted:
            self.fit()

        if use_train:
            X_data = self.X_train
        elif use_val:
            X_data = self.X_val
        else:
            X_data = self.X_test
        X_batch = X_data if indexes is None else X_data[indexes]

        # Dummy labels for dataset loader; caching via object id to avoid rebuild on data swap
        x_id = id(X_batch)
        if not hasattr(self, '_test_ds_cache_id') or self._test_ds_cache_id != x_id:
            self._test_ds_cache = load_dataset(
                data={"X_test": X_batch, "y_test": np.zeros(len(X_batch))}, train=False,
            )
            self._test_ds_cache_id = x_id

        scores = self.deepsad.test(self._test_ds_cache, device=self.device, n_jobs_dataloader=0)
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-8:
            scores = (scores - s_min) / (s_max - s_min)
        else:
            scores = np.full_like(scores, 0.5)
        return scores

    def get_embeddings(self, indexes: Optional[np.ndarray] = None, use_train: bool = True) -> np.ndarray:
        if not self._fitted:
            self.fit()
        X_src = self.X_train if use_train else (self.X_test or self.X_val)
        if X_src is None:
            raise ValueError(f"No data for embeddings (use_train={use_train})")
        X_batch = X_src if indexes is None else X_src[indexes]

        tensor = torch.from_numpy(X_batch).float().to(self.device)
        self.deepsad.net.eval()
        with torch.no_grad():
            return self.deepsad.net(tensor).cpu().numpy()

    def _compute_deepsad_loss(self, X: np.ndarray, y: np.ndarray) -> Optional[float]:
        """Hypersphere loss: normals minimize dist to center, anomalies maximize."""
        if self.trainer is None:
            return None
        self.deepsad.net.eval()
        c = torch.tensor(self.deepsad.c, device=self.device).float()
        with torch.no_grad():
            tensor = torch.from_numpy(X).float().to(self.device)
            outputs = self.deepsad.net(tensor)
            dist = torch.sum((outputs - c) ** 2, dim=1)
            labels = torch.tensor(y, device=self.device).float()
            losses = torch.where(
                labels == 0, dist,
                torch.clamp(self.train_config["eta"] - dist, min=0),
            )
            return torch.mean(losses).item()

    def get_loss(self, use_val: bool = False) -> Optional[float]:
        if not self._fitted or self.trainer is None:
            return None
        if use_val:
            if self._last_val_loss is not None:
                return self._last_val_loss
            if self.X_val is not None and self.y_val is not None and len(self.X_val) > 0:
                return self._compute_deepsad_loss(self.X_val, self.y_val)
            return None
        return self._last_train_loss


class DevNetWrapper(Model):
    """ADBench DevNet with epoch-level training."""

    def __init__(self, train_config: dict, model_config: dict, data: dict):
        defaults = {
            "seed": 42, "total_epochs": 50, "batch_size": 512,
            "nb_batch": 20, "network_depth": 2,
        }
        config = {**defaults, **(train_config or {})}
        super().__init__(train_config=config, model_config=model_config, data=data)

        self.devnet = DevNet(seed=self.train_config["seed"], save_suffix=self.model_config.get("save_suffix", "wrapper"))
        self.devnet.args.batch_size = self.train_config["batch_size"]
        self.devnet.args.nb_batch = self.train_config["nb_batch"]
        self.devnet.args.epochs = 1
        self.devnet.network_depth = int(self.train_config["network_depth"])
        # Precreate ref to avoid K.variable path (newer Keras compat)
        self.devnet.ref = tf.Variable(
            np.random.normal(loc=0.0, scale=1.0, size=5000).astype(np.float32),
            trainable=False,
        )
        self._loss_fn = self._build_devnet_loss(self.devnet.ref)

        self.model = None
        self.input_shape = None
        self.outlier_indices = None
        self.inlier_indices = None
        self.rng = np.random.RandomState(self.train_config["seed"])

        self._train_loss_history = []
        self._val_loss_history = []
        self._last_train_loss = None
        self._last_val_loss = None

    @staticmethod
    def _build_devnet_loss(ref_var: tf.Variable):
        def loss(y_true, y_pred):
            mean = tf.reduce_mean(ref_var)
            std = tf.math.reduce_std(ref_var)
            dev = (y_pred - mean) / (std + 1e-8)
            confidence_margin = 5.0
            inlier_loss = tf.abs(dev)
            outlier_loss = tf.abs(tf.nn.relu(confidence_margin - dev))
            return tf.reduce_mean((1.0 - y_true) * inlier_loss + y_true * outlier_loss)
        return loss

    def _ensure_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        if self.model is not None:
            return
        self.input_shape = X_train.shape[1:]
        self.outlier_indices = np.where(y_train == 1)[0]
        self.inlier_indices = np.where(y_train == 0)[0]
        self.model = self.devnet.deviation_network(self.input_shape, self.devnet.network_depth)
        opt = tf.keras.optimizers.RMSprop(clipnorm=1.0)
        self.model.compile(loss=self._loss_fn, optimizer=opt)
        self._model_path = self.devnet.modelpath + "/devnet_wrapper.h5"

    def fit(self) -> None:
        remaining = self.train_config["total_epochs"] - self._current_epoch
        if remaining > 0:
            self.train(remaining)
        self._fitted = True

    def train(self, epochs: int = 1) -> None:
        y_train = self.y_train
        self.outlier_indices = np.where(y_train == 1)[0]
        self.inlier_indices = np.where(y_train == 0)[0]
        self._ensure_model(self.X_train, y_train)

        batch_size = self.train_config["batch_size"]
        nb_batch = self.train_config["nb_batch"]

        for _ in range(epochs):
            generator = self.devnet.batch_generator_sup(
                self.X_train, self.outlier_indices, self.inlier_indices,
                batch_size, nb_batch, self.rng,
            )
            self.model.fit(generator, steps_per_epoch=nb_batch, epochs=1, verbose=0)
            self._current_epoch += 1
            self._fitted = True

            train_loss = self._compute_devnet_loss(self.X_train, self.y_train)
            if train_loss is not None:
                self._last_train_loss = train_loss
                self._train_loss_history.append(train_loss)

            if self.X_val is not None and self.y_val is not None and len(self.X_val) > 0:
                val_loss = self._compute_devnet_loss(self.X_val, self.y_val)
                if val_loss is not None:
                    self._last_val_loss = val_loss
                    self._val_loss_history.append(val_loss)

    def predict_scores(self, indexes: Optional[np.ndarray] = None, use_train: bool = False, use_val: bool = False) -> np.ndarray:
        if not self._fitted:
            self.fit()
        if use_train:
            X_data = self.X_train
        elif use_val:
            X_data = self.X_val
        else:
            X_data = self.X_test
        X_batch = X_data if indexes is None else X_data[indexes]
        scores = np.asarray(self.model.predict(X_batch)).reshape(-1)
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-8:
            scores = (scores - s_min) / (s_max - s_min)
        else:
            scores = np.full_like(scores, 0.5)
        return scores

    def get_embeddings(self, indexes: Optional[np.ndarray] = None, use_train: bool = True) -> np.ndarray:
        # DevNet has no internal embeddings; use scores
        return self.predict_scores(indexes, use_train=use_train).reshape(-1, 1)

    def _compute_devnet_loss(self, X: np.ndarray, y: np.ndarray) -> Optional[float]:
        if self.model is None:
            return None
        scores = np.asarray(self.model.predict(X, verbose=0)).reshape(-1)
        ref_mean = np.mean(self.devnet.ref.numpy())
        ref_std = np.std(self.devnet.ref.numpy())
        dev = (scores - ref_mean) / (ref_std + 1e-8)
        confidence_margin = 5.0
        inlier_loss = np.abs(dev)
        outlier_loss = np.abs(np.maximum(0, confidence_margin - dev))
        losses = (1.0 - y) * inlier_loss + y * outlier_loss
        return float(np.mean(losses))

    def get_loss(self, use_val: bool = False) -> Optional[float]:
        if not self._fitted or self.model is None:
            return None
        if use_val:
            if self._last_val_loss is not None:
                return self._last_val_loss
            if self.X_val is not None and self.y_val is not None and len(self.X_val) > 0:
                return self._compute_devnet_loss(self.X_val, self.y_val)
            return None
        return self._last_train_loss