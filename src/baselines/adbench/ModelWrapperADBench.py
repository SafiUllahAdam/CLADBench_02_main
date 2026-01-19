"""ADBench model wrappers aligned with the unified Model API."""

from typing import Dict, Optional

import numpy as np
import torch
import tensorflow as tf

from base import Model
from adbench.baseline.PReNet.model import prenet
from adbench.baseline.PReNet.fit import fit
from adbench.myutils import Utils
from adbench.baseline.DeepSAD.src.datasets.main import load_dataset
from adbench.baseline.DeepSAD.src.deepsad import deepsad
from adbench.baseline.DeepSAD.src.optim.DeepSAD_trainer import DeepSADTrainer
from adbench.baseline.DevNet.run import DevNet

try:
    from adbench.baseline.PyOD import PYOD
    PYOD_AVAILABLE = True
    PYOD_IMPORT_ERROR = None
except (ImportError, ValueError) as e:
    PYOD_AVAILABLE = False
    PYOD_IMPORT_ERROR = e


def get_model_detector_dict() -> Dict[str, type]:
    """Registry of available ADBench wrappers keyed by short name."""

    registry: Dict[str, type] = {
        "prenet": PReNetWrapper,
        "deepsad": DeepSADWrapper,
        "devnet": DevNetWrapper,
    }


    if PYOD_AVAILABLE:
        registry["xgbod"] = XGBODWrapper

    return registry


class PReNetWrapper(Model):
    """Wrapper adapting ADBench PReNet to the unified Model API with epoch-level training."""

    def __init__(self, train_config: dict, model_config: dict, data: dict):
        defaults = {
            "seed": 42,
            "total_epochs": 100,
            "batch_num": 10,
            "batch_size": 256,
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "s_a_a": 8,
            "s_a_u": 4,
            "s_u_u": 0,
        }
        self.train_config = {**defaults, **(train_config or {})}
        self.model_config = model_config or {}
        self.data = data
        self._current_epoch = 0
        self._fitted = False
        self._pseudo_labels = None  # Will be set by CoLearner during collaborative learning

        self.utils = Utils()
        self.device = self.utils.get_device()
        cfg = self.train_config

        X_train = self.data["X_train"]
        y_train = self.data["y_train"]
        self.X_train_tensor = torch.from_numpy(X_train).float()
        self.y_train = y_train

        self.utils.set_seed(cfg["seed"])
        input_size = X_train.shape[1]
        self.model = prenet(input_size=input_size, act_fun=torch.nn.ReLU())
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
        self.fit_fn = fit

    def fit(self) -> None:
        cfg = self.train_config
        total_epochs = cfg["total_epochs"]

        for epoch in range(self._current_epoch, total_epochs):
            self.train(epoch)
        self._fitted = True

    def train(self, epoch: int) -> None:
        """Train for a single epoch with current pseudo-labels."""
        # Use pseudo-labels from CoLearner if available, otherwise use original labels
        if self._pseudo_labels is not None:
            y_pseudo = self._pseudo_labels
            # Use only samples with valid pseudo-labels (not -1 = unknown)
            valid_mask = y_pseudo != -1
            if np.any(valid_mask):
                y_train = y_pseudo.copy()
                # For invalid samples, use original labels
                y_train[~valid_mask] = self.y_train[~valid_mask]
            else:
                # No valid pseudo-labels yet, use original
                y_train = self.y_train
        else:
            # Fall back to original labels
            y_train = self.y_train

        cfg = self.train_config

        self.fit_fn(
            X_train_tensor=self.X_train_tensor,
            y_train=y_train,
            model=self.model,
            optimizer=self.optimizer,
            epochs=1,
            batch_num=cfg["batch_num"],
            batch_size=cfg["batch_size"],
            s_a_a=cfg["s_a_a"],
            s_a_u=cfg["s_a_u"],
            s_u_u=cfg["s_u_u"],
            device=self.device,
        )

        self._current_epoch = epoch + 1
        self._fitted = True

    def predict_scores(self, indexes: Optional[np.ndarray] = None, use_train: bool = False) -> np.ndarray:
        if not self._fitted:
            self.fit()

        data_split = "X_train" if use_train else "X_test"
        X_data = self.data.get(data_split, self.data["X_train"])
        X_batch = X_data if indexes is None else X_data[indexes]

        self.model.eval()
        scores = []
        num = 30

        for i in range(len(X_batch)):
            index_a = np.random.choice(np.where(self.y_train == 1)[0], num, replace=True)
            index_u = np.random.choice(np.where(self.y_train == 0)[0], num, replace=True)

            X_train_a = self.X_train_tensor[index_a]
            X_train_u = self.X_train_tensor[index_u]

            X_i = torch.from_numpy(X_batch[i:i + 1]).float().to(self.device)

            with torch.no_grad():
                score_a_x = self.model(X_train_a.to(self.device), X_i.repeat(num, 1))
                score_x_u = self.model(X_i.repeat(num, 1), X_train_u.to(self.device))

            score_sub = torch.mean(score_a_x + score_x_u).cpu().numpy()
            scores.append(score_sub)

        scores = np.array(scores)

        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-8:
            scores = (scores - s_min) / (s_max - s_min)
        return scores

    def get_embeddings(self, indexes: Optional[np.ndarray] = None) -> np.ndarray:
        if not self._fitted:
            self.fit()

        X_train = self.data["X_train"]
        X_batch = X_train if indexes is None else X_train[indexes]
        tensor = torch.from_numpy(X_batch).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.feature(tensor).cpu().numpy()
        return embeddings


class XGBODWrapper(Model):
    """Wrapper adapting PyOD XGBOD to the unified Model API."""

    def __init__(self, train_config: dict, model_config: dict, data: dict):
        super().__init__()  # Initialize parent Model class
        if not PYOD_AVAILABLE:
            raise ImportError(
                "PyOD not available. Install scikit-learn==1.0.2 and pyod==1.0.9 with --no-cache-dir"
            ) from PYOD_IMPORT_ERROR

        defaults = {
            "seed": 42,
            "total_epochs": 10,
        }
        self.train_config = {**defaults, **(train_config or {})}
        self.model_config = model_config or {}
        self.data = data
        self.detector: Optional[PYOD] = None
        self._fitted = False
        self._current_epoch = 0
        self._pseudo_labels = None  # Will be set by CoLearner during collaborative learning

    def _build_detector(self) -> None:
        cfg = self.train_config
        self.detector = PYOD(seed=cfg["seed"], model_name="XGBOD", tune=False)

    def fit(self) -> None:
        cfg = self.train_config
        total_epochs = cfg["total_epochs"]

        for epoch in range(self._current_epoch, total_epochs):
            self.train(epoch)
        self._fitted = True

    def train(self, epoch: int) -> None:
        if self.detector is None:
            self._build_detector()

        X_train = self.data["X_train"]
        
        # Use pseudo-labels from CoLearner if available, otherwise use original labels
        if self._pseudo_labels is not None:
            y_pseudo = self._pseudo_labels
            # Use only samples with valid pseudo-labels (not -1 = unknown)
            valid_mask = y_pseudo != -1
            if np.any(valid_mask):
                y_train = y_pseudo.copy()
                # For invalid samples, use original labels (if available)
                y_train[~valid_mask] = self.data.get("y_train", None)[~valid_mask]
            else:
                # No valid pseudo-labels yet, use original
                y_train = self.data.get("y_train", None)
        else:
            # Fall back to original labels (or None for unsupervised)
            y_train = self.data.get("y_train", None)

        self.detector.fit(X_train, y_train)
        self._current_epoch = epoch + 1
        self._fitted = True

    def predict_scores(self, indexes: Optional[np.ndarray] = None, use_train: bool = False) -> np.ndarray:
        if not self._fitted:
            self.fit()

        data_split = "X_train" if use_train else "X_test"
        X_data = self.data.get(data_split, self.data["X_train"])
        X_batch = X_data if indexes is None else X_data[indexes]

        scores = self.detector.model.decision_function(X_batch).astype(float)

        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-8:
            scores = (scores - s_min) / (s_max - s_min)
        return scores

    def get_embeddings(self, indexes: Optional[np.ndarray] = None) -> np.ndarray:
        if not self._fitted:
            self.fit()

        X_train = self.data["X_train"]
        X_batch = X_train if indexes is None else X_train[indexes]

        embeddings = self.detector.model.decision_function(X_batch).astype(float)
        return embeddings.reshape(-1, 1)


class DeepSADWrapper(Model):
    """Wrapper adapting ADBench DeepSAD to the unified Model API."""

    def __init__(self, train_config: dict, model_config: dict, data: dict):
        super().__init__()  # Initialize parent Model class
        defaults = {
            "seed": 42,
            "total_epochs": 50,
            "pretrain": True,
            "ae_epochs": 100,
            "batch_size": 128,
            "lr": 0.001,
            "weight_decay": 1e-6,
            "eta": 1.0,
            "net_name": "dense",
        }
        self.train_config = {**defaults, **(train_config or {})}
        self.model_config = model_config or {}
        self.data = data
        self._current_epoch = 0
        self._fitted = False
        self._pretrained = False
        self._pseudo_labels = None  # Will be set by CoLearner during collaborative learning

        self.utils = Utils()
        self.device = self.utils.get_device()

        cfg = self.train_config
        self.utils.set_seed(cfg["seed"])

        X_train = self.data["X_train"]
        y_train = self.data["y_train"]
        self.dataset = load_dataset(data={"X_train": X_train, "y_train": y_train}, train=True)
        input_size = self.dataset.train_set.data.size(1)

        self.deepsad = deepsad(cfg["eta"])
        self.deepsad.set_network(cfg["net_name"], input_size)

        self.optimizer_name = "adam"
        self.lr = cfg["lr"]
        self.batch_size = cfg["batch_size"]
        self.weight_decay = cfg["weight_decay"]
        self.lr_milestones = []

        if cfg["pretrain"] and not self._pretrained:
            self.deepsad.pretrain(
                self.dataset,
                input_size,
                optimizer_name="adam",
                lr=cfg["lr"],
                n_epochs=cfg["ae_epochs"],
                lr_milestones=(),
                batch_size=cfg["batch_size"],
                weight_decay=cfg["weight_decay"],
                device=self.device,
                n_jobs_dataloader=0,
            )
            self._pretrained = True

        self.trainer = None

    def fit(self) -> None:
        cfg = self.train_config
        total_epochs = cfg["total_epochs"]

        for epoch in range(self._current_epoch, total_epochs):
            self.train(epoch)
        self._fitted = True

    def train(self, epoch: int) -> None:
        # Use pseudo-labels from CoLearner if available, otherwise use original labels
        if self._pseudo_labels is not None:
            y_pseudo = self._pseudo_labels
            # Use only samples with valid pseudo-labels (not -1 = unknown)
            valid_mask = y_pseudo != -1
            if np.any(valid_mask):
                y_train = y_pseudo.copy()
                # For invalid samples, use original labels (if available)
                y_train[~valid_mask] = self.data["y_train"][~valid_mask]
            else:
                # No valid pseudo-labels yet, use original
                y_train = self.data["y_train"]
        else:
            # Fall back to original labels
            y_train = self.data["y_train"]

        self.dataset = load_dataset(
            data={"X_train": self.data["X_train"], "y_train": y_train},
            train=True,
        )

        if self.trainer is None:
            self.trainer = DeepSADTrainer(
                c=self.deepsad.c,
                eta=self.train_config["eta"],
                optimizer_name=self.optimizer_name,
                lr=self.lr,
                n_epochs=1,
                lr_milestones=(),
                batch_size=self.batch_size,
                weight_decay=self.weight_decay,
                device=self.device,
                n_jobs_dataloader=0,
            )

        self.deepsad.net = self.trainer.train(self.dataset, self.deepsad.net)
        self.deepsad.c = self.trainer.c.cpu().data.numpy().tolist()

        self._current_epoch = epoch + 1
        self._fitted = True

    def predict_scores(self, indexes: Optional[np.ndarray] = None, use_train: bool = False) -> np.ndarray:
        if not self._fitted:
            self.fit()

        data_split = "X_train" if use_train else "X_test"
        X_data = self.data.get(data_split, self.data["X_train"])
        X_batch = X_data if indexes is None else X_data[indexes]

        test_dataset = load_dataset(
            data={"X_test": X_batch, "y_test": np.zeros(len(X_batch))},
            train=False,
        )

        scores = self.deepsad.test(test_dataset, device=self.device, n_jobs_dataloader=0)

        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-8:
            scores = (scores - s_min) / (s_max - s_min)
        return scores

    def get_embeddings(self, indexes: Optional[np.ndarray] = None) -> np.ndarray:
        if not self._fitted:
            self.fit()

        X_train = self.data["X_train"]
        X_batch = X_train if indexes is None else X_train[indexes]

        tensor = torch.from_numpy(X_batch).float().to(self.device)

        self.deepsad.net.eval()
        with torch.no_grad():
            embeddings = self.deepsad.net(tensor).cpu().numpy()

        return embeddings


class DevNetWrapper(Model):
    """Wrapper adapting ADBench DevNet to the unified Model API."""

    def __init__(self, train_config: dict, model_config: dict, data: dict):
        super().__init__()  # Initialize parent Model class
        defaults = {
            "seed": 42,
            "total_epochs": 50,
            "batch_size": 512,
            "nb_batch": 20,
            "network_depth": 2,  # 1, 2, or 4
        }
        self.train_config = {**defaults, **(train_config or {})}
        self.model_config = model_config or {}
        self.data = data
        self._current_epoch = 0
        self._fitted = False
        self._pseudo_labels = None  # Will be set by CoLearner during collaborative learning

        self.devnet = DevNet(seed=self.train_config["seed"], save_suffix=self.model_config.get("save_suffix", "wrapper"))
        # Override parsed args to align with train_config
        self.devnet.args.batch_size = self.train_config["batch_size"]
        self.devnet.args.nb_batch = self.train_config["nb_batch"]
        self.devnet.args.epochs = 1  # we run one epoch per train() call
        self.devnet.network_depth = int(self.train_config["network_depth"])
        # Precreate ref to avoid K.variable path (compat with newer Keras)
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

    @staticmethod
    def _build_devnet_loss(ref_var: tf.Variable):
        """TF-native deviation loss to replace legacy keras.backend ops."""

        def loss(y_true, y_pred):
            # y_true expected in {0,1}; broadcast-safe
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

        # Initialize the Keras model once; reuse across epochs
        self.model = self.devnet.deviation_network(self.input_shape, self.devnet.network_depth)
        # Recompile with TF-native loss to avoid legacy Keras backend calls
        opt = tf.keras.optimizers.RMSprop(clipnorm=1.0)
        self.model.compile(loss=self._loss_fn, optimizer=opt)
        self._model_path = self.devnet.modelpath + "/devnet_wrapper.h5"

    def fit(self) -> None:
        cfg = self.train_config
        total_epochs = cfg["total_epochs"]

        for epoch in range(self._current_epoch, total_epochs):
            self.train(epoch)
        self._fitted = True

    def train(self, epoch: int) -> None:
        X_train = self.data["X_train"]
        
        # Use pseudo-labels from CoLearner if available, otherwise use original labels
        if self._pseudo_labels is not None:
            y_pseudo = self._pseudo_labels
            # Use only samples with valid pseudo-labels (not -1 = unknown)
            valid_mask = y_pseudo != -1
            if np.any(valid_mask):
                y_train = y_pseudo.copy()
                # For invalid samples, use original labels (if available)
                y_train[~valid_mask] = self.data["y_train"][~valid_mask]
            else:
                # No valid pseudo-labels yet, use original
                y_train = self.data["y_train"]
        else:
            # Fall back to original labels
            y_train = self.data["y_train"]

        # Refresh indices to reflect potential pseudo-label updates
        self.outlier_indices = np.where(y_train == 1)[0]
        self.inlier_indices = np.where(y_train == 0)[0]

        self._ensure_model(X_train, y_train)

        batch_size = self.train_config["batch_size"]
        nb_batch = self.train_config["nb_batch"]
        generator = self.devnet.batch_generator_sup(
            X_train,
            self.outlier_indices,
            self.inlier_indices,
            batch_size,
            nb_batch,
            self.rng,
        )

        # Single-epoch training step (fit_generator deprecated in modern TF/Keras)
        self.model.fit(generator, steps_per_epoch=nb_batch, epochs=1, verbose=0)

        self._current_epoch = epoch + 1
        self._fitted = True

    def predict_scores(self, indexes: Optional[np.ndarray] = None, use_train: bool = False) -> np.ndarray:
        if not self._fitted:
            self.fit()

        data_split = "X_train" if use_train else "X_test"
        X_data = self.data.get(data_split, self.data["X_train"])
        X_batch = X_data if indexes is None else X_data[indexes]

        scores = self.model.predict(X_batch)
        scores = np.asarray(scores).reshape(-1)

        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-8:
            scores = (scores - s_min) / (s_max - s_min)
        return scores

    def get_embeddings(self, indexes: Optional[np.ndarray] = None) -> np.ndarray:
        # DevNet exposes only the score head; use scores as 1D embeddings
        return self.predict_scores(indexes).reshape(-1, 1)