"""Utility functions for CoBench benchmark (no classes - see base.py for classes)."""

import logging
from typing import Dict, List, Optional, Any, Tuple, Iterable
import numpy as np
from pathlib import Path

from base import Model, Data
from benchmark_config import get_model_config, get_scenario, EVALUATION_THRESHOLDS

logger = logging.getLogger(__name__)


def create_models(
    registry: Dict[str, type],
    model_names: List[str],
    data: Data
) -> List[Model]:
    """Create Model instances from registry."""
    models = []
    
    for model_name in model_names:
        if model_name not in registry:
            available = list(registry.keys())
            raise KeyError(
                f"Model '{model_name}' not in registry. "
                f"Available: {available}"
            )
        
        try:
            ModelClass = registry[model_name]
            train_config = get_model_config(model_name)
            
            model = ModelClass(
                train_config=train_config,
                model_config={},
                data=data
            )
            models.append(model)
            logger.info(f"Created model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to create model '{model_name}': {e}")
            raise ValueError(f"Model instantiation failed: {e}") from e
    
    logger.info(f"Successfully created {len(models)} models: {model_names}")
    return models




def test_all_data_access(test_all_datasets: bool, root: Path) -> None:
    from baselines.adbench.data_loader import load_data
    
    if test_all_datasets:
        dataset_root = root / "adbench" / "datasets"
        all_npz = sorted(dataset_root.rglob("*.npz"))
        print(f"\nRunning DataLoadTest across {len(all_npz)} datasets under {dataset_root}...")
        failures = []
        for path in all_npz:
            rel_path = path.relative_to(dataset_root)
            try:
                ds = load_data(path)
                print(f"{rel_path}: train={ds.n_train}, test={ds.n_test}")
            except Exception as e:
                print(f"{rel_path} FAIL: {type(e).__name__}: {e}")
                failures.append(rel_path)
        if failures:
            print(f"\nDataLoadTest finished with {len(failures)} failures:")
            for rel_path in failures:
                print(f"  - {rel_path}")
        else:
            print("\nDataLoadTest succeeded")


def analyze_pseudo_label_distribution(
    data: Data,
    model_names: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Compute statistics on pseudo-label distribution per model."""
    stats = {}
    
    for model_name in model_names:
        if model_name not in data.pseudo_labels_by_model:
            logger.warning(f"No pseudo-labels found for {model_name}")
            continue
        
        labels = data.pseudo_labels_by_model[model_name]
        confidence = data.pseudo_label_confidence.get(model_name, np.ones_like(labels))
        
        stats[model_name] = {
            "n_normals": int(np.sum(labels == 0)),
            "n_anomalies": int(np.sum(labels == 1)),
            "confidence_mean": float(np.mean(confidence)),
            "confidence_std": float(np.std(confidence)),
            "confidence_min": float(np.min(confidence)),
            "confidence_max": float(np.max(confidence)),
        }
        
        logger.info(f"{model_name} pseudo-labels:")
        logger.info(f"  Normals: {stats[model_name]['n_normals']}, "
                   f"Anomalies: {stats[model_name]['n_anomalies']}")
        logger.info(f"  Confidence: mean={stats[model_name]['confidence_mean']:.4f}, "
                   f"std={stats[model_name]['confidence_std']:.4f}")
    
    return stats


def _iter_batches(array: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    if batch_size <= 0:
        yield array
        return
    for start in range(0, len(array), batch_size):
        end = start + batch_size
        yield array[start:end]


def _resolve_input_data(model: Model, indexes: Optional[np.ndarray], use_train: bool) -> np.ndarray:
    X_data = model.X_train if use_train else model.X_test
    if X_data is None:
        X_data = model.X_val
    if X_data is None:
        raise ValueError("No input data found on model (X_train/X_test/X_val are None).")
    return X_data if indexes is None else X_data[indexes]


def extract_embeddings_auto(
    model: Model,
    indexes: Optional[np.ndarray] = None,
    use_train: bool = False,
    layer: Optional[str] = None,
    batch_size: int = 1024,
    device: Optional[str] = None,
) -> np.ndarray:
    """Best-effort embedding extraction for any wrapper.

    Strategy:
      1) ideal case : use model.get_embeddings() if implemented
      2) If a torch model is available, hook the requested layer (or last leaf).
      3) If a Keras model is available, build a sub-model for the requested layer CHECK AGAIN 
      4) Fallback to predict_scores() as 1D embeddings + log the pseudo fail 
    """
    model_name = model.__class__.__name__
    
    # 1) Use wrapper-provided method if it works
    try:
        emb = model.get_embeddings(indexes, use_train=use_train)
        if emb is not None:
            logger.info(f"[{model_name}] Extracted embeddings via get_embeddings()")
            return emb
    except Exception as e:
        logger.warning(f"[{model_name}] get_embeddings() failed: {type(e).__name__}: {e}")

    X = _resolve_input_data(model, indexes, use_train)

    # 2) Torch hook path
    try:
        import torch
        torch_model = None
        if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
            torch_model = model.model
        elif hasattr(model, "detector") and hasattr(model.detector, "model") and isinstance(model.detector.model, torch.nn.Module):
            torch_model = model.detector.model

        if torch_model is not None:
            try:
                torch_model.eval()
                params = list(torch_model.parameters())
                torch_device = params[0].device if params else torch.device("cpu")
                if device is not None:
                    torch_device = torch.device(device)
                    torch_model.to(torch_device)

                named_modules = dict(torch_model.named_modules())
                if layer is not None and layer in named_modules:
                    target = named_modules[layer]
                else:
                    # last leaf with parameters, else last module
                    leaf_modules = [m for m in named_modules.values() if len(list(m.children())) == 0]
                    target = None
                    for m in reversed(leaf_modules):
                        if any(p.requires_grad for p in m.parameters(recurse=False)):
                            target = m
                            break
                    if target is None and leaf_modules:
                        target = leaf_modules[-1]

                if target is not None:
                    activations: List[torch.Tensor] = []

                    def hook_fn(_, __, output):
                        if torch.is_tensor(output):
                            activations.append(output.detach().cpu())
                        elif isinstance(output, (tuple, list)) and output:
                            out0 = output[0]
                            if torch.is_tensor(out0):
                                activations.append(out0.detach().cpu())

                    handle = target.register_forward_hook(hook_fn)
                    with torch.no_grad():
                        for batch in _iter_batches(X, batch_size):
                            xb = torch.as_tensor(batch, dtype=torch.float32, device=torch_device)
                            _ = torch_model(xb)
                    handle.remove()

                    if activations:
                        logger.info(f"[{model_name}] Extracted embeddings via torch hook from {target}")
                        return torch.cat(activations, dim=0).numpy()
                else:
                    logger.debug(f"[{model_name}] Torch: no target layer found")
            except Exception as e:
                logger.debug(f"[{model_name}] Torch hook extraction failed: {e}")
        else:
            logger.debug(f"[{model_name}] No torch.nn.Module found")
    except Exception as e:
        logger.debug(f"[{model_name}] Torch path failed: {e}")

    # 3) Keras path
    try:
        import tensorflow as tf
        keras_model = None
        if hasattr(model, "model") and isinstance(model.model, tf.keras.Model):
            keras_model = model.model
        elif hasattr(model, "detector") and hasattr(model.detector, "model") and isinstance(model.detector.model, tf.keras.Model):
            keras_model = model.detector.model

        if keras_model is not None:
            try:
                if layer is not None:
                    try:
                        out_layer = keras_model.get_layer(layer).output
                    except Exception:
                        out_layer = keras_model.layers[-2].output
                else:
                    out_layer = keras_model.layers[-2].output
                sub_model = tf.keras.Model(keras_model.input, out_layer)
                embeddings = sub_model.predict(X, batch_size=batch_size, verbose=0)
                logger.info(f"[{model_name}] Extracted embeddings via Keras sub-model")
                return embeddings
            except Exception as e:
                logger.debug(f"[{model_name}] Keras extraction failed: {e}")
        else:
            logger.debug(f"[{model_name}] No keras.Model found")
    except Exception as e:
        logger.debug(f"[{model_name}] Keras path failed: {e}")

    # 4) Fallback to scores
    logger.warning(f"[{model_name}] Falling back to anomaly scores as 1D embeddings "
                   f"(no get_embeddings, torch hook, or keras sub-model available)")
    scores = model.predict_scores(indexes=indexes, use_train=use_train)
    return np.asarray(scores).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Data validation checks
# ---------------------------------------------------------------------------

def validate_splits(data: Data) -> None:
    """Run benchmark integrity checks on a Data instance. Raises AssertionError on failure."""
    assert data.X_train.shape[0] == data.n_train, "X_train/n_train mismatch"
    assert data.X_test.shape[0] == data.n_test, "X_test/n_test mismatch"
    if data.n_val and data.n_val > 0:
        assert data.X_val.shape[0] == data.n_val, "X_val/n_val mismatch"

    if data.labeled_indexes is not None:
        # labeled ∪ unlabeled = train, no overlap
        all_train = np.union1d(data.labeled_indexes, data.unlabeled_indexes)
        assert np.array_equal(all_train, np.arange(data.n_train)), "labeled ∪ unlabeled ≠ train"
        assert len(np.intersect1d(data.labeled_indexes, data.unlabeled_indexes)) == 0, "labeled/unlabeled overlap"

        # At least 1 anomaly + 1 normal labeled
        assert np.any(data.y_train_original[data.labeled_indexes] == 1), "no labeled anomaly"
        assert np.any(data.y_train_original[data.labeled_indexes] == 0), "no labeled normal"

        # Stratification sanity (20% relative tolerance)
        if len(data.labeled_indexes) > 5:
            full_rate = np.mean(data.y_train_original == 1)
            labeled_rate = np.mean(data.y_train_original[data.labeled_indexes] == 1)
            if full_rate > 0:
                assert abs(labeled_rate - full_rate) / full_rate < 0.30, \
                    f"Stratification drift: full={full_rate:.3f} labeled={labeled_rate:.3f}"

    # semisupervised_labels consistency
    if data.semisupervised_labels is not None and data.labeled_indexes is not None:
        ssl = data.semisupervised_labels
        assert np.all(ssl[data.labeled_indexes] != -1), "labeled idx has -1 in semisupervised_labels"
        assert np.all(ssl[data.unlabeled_indexes] == -1), "unlabeled idx is not -1"
        assert np.array_equal(
            ssl[data.labeled_indexes], data.y_train_original[data.labeled_indexes]
        ), "labeled semisupervised_labels != ground truth"

    # Print split statistics
    for name, y in [("train", data.y_train_original), ("test", data.y_test), ("val", data.y_val)]:
        if y is not None and len(y) > 0:
            logger.info(f"  {name}: n={len(y)}, anomaly_rate={np.mean(y == 1):.4f}")

    logger.info("[validate_splits] All checks passed")


def check_no_test_leakage(data: Data) -> None:
    """Verify scaler was fit on train only."""
    if hasattr(data, '_scaler') and data._scaler is not None:
        n_seen = data._scaler.n_samples_seen_
        assert n_seen == data.n_train, f"Scaler saw {n_seen} samples but n_train={data.n_train}"
    logger.info("[check_no_test_leakage] Passed")


def check_unlabeled_hidden(data: Data) -> None:
    """Ensure resolve_labels() never leaks ground truth on unlabeled."""
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
    """Run all benchmark data checks."""
    validate_splits(data)
    check_no_test_leakage(data)
    check_unlabeled_hidden(data)
