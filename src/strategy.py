from typing import Dict

import numpy as np

from base import Strategy


class SimpleStrategy(Strategy):
    """Stop when ensemble AUC stops improving"""

    def __init__(self, max_chapters: int = 50, patience: int = 5,
                 patience_threshold: float = 0.001):
        self.max_chapters = max_chapters
        self.patience = patience
        self.patience_threshold = patience_threshold
        self.best_ensemble_auc = 0.0
        self.patience_counter = 0

    def should_continue(self, model_metrics: Dict[str, float], chapter: int) -> bool:
        auc = model_metrics.get("ensemble", 0.5)
        if auc > self.best_ensemble_auc + self.patience_threshold:
            self.best_ensemble_auc = auc
            self.patience_counter = 0
            return True
        self.patience_counter += 1
        return self.patience_counter < self.patience

    def reset(self):
        self.best_ensemble_auc = 0
        self.patience_counter = 0


class PlateauStrategy(Strategy):
    """Stop when a metric plateaus for `patience` chapters"""

    def __init__(self, metric_key: str = "ensemble", mode: str = "max",
                 patience: int = 5, min_delta: float = 0.001):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode}")
        self.metric_key = metric_key
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = -np.inf if mode == "max" else np.inf
        self.patience_counter = 0

    def _improved(self, val: float) -> bool:
        if self.mode == "max":
            return val > self.best_value + self.min_delta
        return val < self.best_value - self.min_delta

    def should_continue(self, model_metrics: Dict[str, float], chapter: int) -> bool:
        if self.metric_key not in model_metrics:
            raise ValueError(f"Missing metric '{self.metric_key}'")
        val = model_metrics[self.metric_key]
        if not np.isfinite(val):
            raise ValueError(f"Non-finite metric '{self.metric_key}' at chapter {chapter}")
        if self._improved(val):
            self.best_value = val
            self.patience_counter = 0
            return True
        self.patience_counter += 1
        return self.patience_counter < self.patience

    def reset(self) -> None:
        self.best_value = -np.inf if self.mode == "max" else np.inf
        self.patience_counter = 0


class AdaptivePlateauStrategy(Strategy):
    """Plateau strategy where patience shrinks after each stop"""

    def __init__(self, metric_key: str = "ensemble", mode: str = "max",
                 patience: int = 5, min_delta: float = 0.001,
                 decay_rate: int = 1, min_patience: int = 1):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode}")
        self.metric_key = metric_key
        self.mode = mode
        self.min_delta = min_delta
        self.initial_patience = patience
        self.patience = patience
        self.decay_rate = decay_rate
        self.min_patience = min_patience
        self.best_value = -np.inf if mode == "max" else np.inf
        self.patience_counter = 0
        self.stop_count = 0

    def _improved(self, val: float) -> bool:
        if self.mode == "max":
            return val > self.best_value + self.min_delta
        return val < self.best_value - self.min_delta

    def should_continue(self, model_metrics: Dict[str, float], chapter: int) -> bool:
        if self.metric_key not in model_metrics:
            raise ValueError(f"Missing metric '{self.metric_key}'")
        val = model_metrics[self.metric_key]
        if not np.isfinite(val):
            raise ValueError(f"Non-finite metric '{self.metric_key}' at chapter {chapter}")

        if self._improved(val):
            self.best_value = val
            self.patience_counter = 0
            return True

        self.patience_counter += 1
        if self.patience_counter >= self.patience:
            self.stop_count += 1
            new_p = max(self.min_patience, self.patience - self.decay_rate)
            print(f"    [Strategy] Stop #{self.stop_count}, patience: {self.patience} -> {new_p}")
            self.patience = new_p
            return False
        return True

    def reset(self) -> None:
        self.best_value = -np.inf if self.mode == "max" else np.inf
        self.patience_counter = 0
        # Keep decayed patience across resets

    def full_reset(self) -> None:
        self.reset()
        self.patience = self.initial_patience
        self.stop_count = 0


class RecurrentPlateauStrategy(Strategy):
    """Monitors ensemble before GRU starts, then switches to GRU metric"""

    def __init__(self, recurrent_start_chapter: int = 3,
                 fallback_key: str = "ensemble", recurrent_key: str = "gru",
                 mode: str = "max", patience_fallback: int = 3,
                 patience_recurrent: int = 2, min_delta: float = 0.001):
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode}")
        self.recurrent_start_chapter = recurrent_start_chapter
        self.fallback_key = fallback_key
        self.recurrent_key = recurrent_key
        self.mode = mode
        self.patience_fallback = patience_fallback
        self.patience_recurrent = patience_recurrent
        self.min_delta = min_delta
        self._best_fallback = -np.inf if mode == "max" else np.inf
        self._best_recurrent = -np.inf if mode == "max" else np.inf
        self._counter_fallback = 0
        self._counter_recurrent = 0

    def _improved(self, current: float, best: float) -> bool:
        if self.mode == "max":
            return current > best + self.min_delta
        return current < best - self.min_delta

    def should_continue(self, model_metrics: Dict[str, float], chapter: int) -> bool:
        if chapter < self.recurrent_start_chapter or self.recurrent_key not in model_metrics:
            key = self.fallback_key
            if key not in model_metrics:
                raise ValueError(f"Missing fallback metric '{key}'")
            val = model_metrics[key]
            if self._improved(val, self._best_fallback):
                self._best_fallback = val
                self._counter_fallback = 0
            else:
                self._counter_fallback += 1
            return self._counter_fallback < self.patience_fallback

        val = model_metrics[self.recurrent_key]
        if not np.isfinite(val):
            raise ValueError(f"Non-finite '{self.recurrent_key}' at chapter {chapter}")
        if self._improved(val, self._best_recurrent):
            self._best_recurrent = val
            self._counter_recurrent = 0
        else:
            self._counter_recurrent += 1
        return self._counter_recurrent < self.patience_recurrent

    def reset(self) -> None:
        self._best_fallback = -np.inf if self.mode == "max" else np.inf
        self._best_recurrent = -np.inf if self.mode == "max" else np.inf
        self._counter_fallback = 0
        self._counter_recurrent = 0
