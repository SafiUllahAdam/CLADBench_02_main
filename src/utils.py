import logging
import os
import random
from typing import Dict, List, Optional, Iterable
import numpy as np
import torch

from base import Model, Data
from benchmark_config import get_model_config

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_models(registry: Dict[str, type], model_names: List[str], data: Data) -> List[Model]:
    models = []
    for name in model_names:
        if name not in registry:
            raise KeyError(f"Model '{name}' not in registry. Available: {list(registry.keys())}")
        try:
            m = registry[name](train_config=get_model_config(name), model_config={}, data=data)
            models.append(m)
            logger.info(f"Created model: {name}")
        except Exception as e:
            raise ValueError(f"Failed to create '{name}': {e}") from e
    logger.info(f"Created {len(models)} models: {model_names}")
    return models


def _iter_batches(array: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    if batch_size <= 0:
        yield array
        return
    for i in range(0, len(array), batch_size):
        yield array[i:i + batch_size]


def _resolve_input(model: Model, indexes: Optional[np.ndarray], use_train: bool) -> np.ndarray:
    X = model.X_train if use_train else model.X_test
    if X is None:
        X = model.X_val
    if X is None:
        raise ValueError("No input data on model (X_train/X_test/X_val all None)")
    return X if indexes is None else X[indexes]


def extract_embeddings_auto(model: Model, indexes: Optional[np.ndarray] = None,
                            use_train: bool = False, use_val: bool = False,
                            layer: Optional[str] = None,
                            batch_size: int = 1024, device: Optional[str] = None) -> np.ndarray:
    """Try get_embeddings() -> torch hook -> keras sub-model -> fall back to scores."""
    name = model.__class__.__name__

    try:
        emb = model.get_embeddings(indexes, use_train=use_train)
        if emb is not None:
            logger.info(f"[{name}] Embeddings via get_embeddings()")
            return emb
    except Exception as e:
        logger.warning(f"[{name}] get_embeddings() failed: {e}")

    X = _resolve_input(model, indexes, use_train=use_train or not use_val)

    try:
        import torch
        torch_model = getattr(model, "model", None) or getattr(getattr(model, "detector", None), "model", None)
        if isinstance(torch_model, torch.nn.Module):
            return _extract_torch(torch_model, X, name, layer, batch_size, device)
    except Exception as e:
        logger.debug(f"[{name}] Torch path failed: {e}")

    try:
        import tensorflow as tf
        keras_model = getattr(model, "model", None) or getattr(getattr(model, "detector", None), "model", None)
        if isinstance(keras_model, tf.keras.Model):
            return _extract_keras(keras_model, X, name, layer, batch_size)
    except Exception as e:
        logger.debug(f"[{name}] Keras path failed: {e}")

    logger.warning(f"[{name}] Falling back to anomaly scores as 1D embeddings")
    return np.asarray(model.predict_scores(indexes=indexes, use_train=use_train, use_val=use_val)).reshape(-1, 1)


def _extract_torch(torch_model, X, name, layer, batch_size, device):
    import torch
    torch_model.eval()
    params = list(torch_model.parameters())
    dev = torch.device(device) if device else (params[0].device if params else torch.device("cpu"))
    if device:
        torch_model.to(dev)

    named = dict(torch_model.named_modules())
    if layer and layer in named:
        target = named[layer]
    else:
        # Pick last trainable leaf, or just last leaf
        leaves = [m for m in named.values() if not list(m.children())]
        target = next((m for m in reversed(leaves) if any(p.requires_grad for p in m.parameters(recurse=False))), None)
        if target is None and leaves:
            target = leaves[-1]

    if target is None:
        raise RuntimeError("No target layer found")

    activations: List = []

    def hook_fn(_, __, output):
        t = output if torch.is_tensor(output) else (output[0] if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]) else None)
        if t is not None:
            activations.append(t.detach().cpu())

    handle = target.register_forward_hook(hook_fn)
    with torch.no_grad():
        for batch in _iter_batches(X, batch_size):
            torch_model(torch.as_tensor(batch, dtype=torch.float32, device=dev))
    handle.remove()

    if activations:
        logger.info(f"[{name}] Embeddings via torch hook from {target}")
        return torch.cat(activations, dim=0).numpy()
    raise RuntimeError("Hook captured no activations")


def _extract_keras(keras_model, X, name, layer, batch_size):
    import tensorflow as tf
    try:
        out = keras_model.get_layer(layer).output if layer else keras_model.layers[-2].output
    except Exception:
        out = keras_model.layers[-2].output
    sub = tf.keras.Model(keras_model.input, out)
    logger.info(f"[{name}] Embeddings via Keras sub-model")
    return sub.predict(X, batch_size=batch_size, verbose=0)


# --- Data validation ---

def validate_splits(data: Data) -> None:
    assert data.X_train.shape[0] == data.n_train, "X_train/n_train mismatch"
    assert data.X_test.shape[0] == data.n_test, "X_test/n_test mismatch"
    if data.n_val and data.n_val > 0:
        assert data.X_val.shape[0] == data.n_val, "X_val/n_val mismatch"

    if data.labeled_indexes is not None:
        all_train = np.union1d(data.labeled_indexes, data.unlabeled_indexes)
        assert np.array_equal(all_train, np.arange(data.n_train)), "labeled + unlabeled != train"
        assert len(np.intersect1d(data.labeled_indexes, data.unlabeled_indexes)) == 0, "labeled/unlabeled overlap"
        assert np.any(data.y_train_original[data.labeled_indexes] == 1), "no labeled anomaly"
        assert np.any(data.y_train_original[data.labeled_indexes] == 0), "no labeled normal"

        # 30% relative tolerance on stratification
        if len(data.labeled_indexes) > 5:
            full_rate = np.mean(data.y_train_original == 1)
            labeled_rate = np.mean(data.y_train_original[data.labeled_indexes] == 1)
            if full_rate > 0:
                assert abs(labeled_rate - full_rate) / full_rate < 0.30, \
                    f"Stratification drift: full={full_rate:.3f} labeled={labeled_rate:.3f}"

    if data.semisupervised_labels is not None and data.labeled_indexes is not None:
        ssl = data.semisupervised_labels
        assert np.all(ssl[data.labeled_indexes] != -1), "labeled idx has -1"
        assert np.all(ssl[data.unlabeled_indexes] == -1), "unlabeled idx not -1"
        assert np.array_equal(ssl[data.labeled_indexes], data.y_train_original[data.labeled_indexes])

    for name, y in [("train", data.y_train_original), ("test", data.y_test), ("val", data.y_val)]:
        if y is not None and len(y) > 0:
            logger.info(f"  {name}: n={len(y)}, anomaly_rate={np.mean(y == 1):.4f}")
    logger.info("[validate_splits] Passed")


def check_no_test_leakage(data: Data) -> None:
    if hasattr(data, '_scaler') and data._scaler is not None:
        assert data._scaler.n_samples_seen_ == data.n_train, \
            f"Scaler saw {data._scaler.n_samples_seen_} samples, n_train={data.n_train}"
    logger.info("[check_no_test_leakage] Passed")


def check_unlabeled_hidden(data: Data) -> None:
    if data.unlabeled_indexes is None or not hasattr(data, 'resolve_labels'):
        return
    resolved = data.resolve_labels()
    policy = getattr(data, 'unlabeled_policy', 'unlabeled_as_normal')
    if policy == "unlabeled_as_normal":
        assert np.all(resolved[data.unlabeled_indexes] == 0), "unlabeled not mapped to 0"
    elif policy == "unlabeled_as_is":
        assert np.all(resolved[data.unlabeled_indexes] == -1), "unlabeled not kept as -1"
    logger.info(f"[check_unlabeled_hidden] Passed (policy={policy})")


def validate_data(data: Data) -> None:
    validate_splits(data)
    check_no_test_leakage(data)
    check_unlabeled_hidden(data)

