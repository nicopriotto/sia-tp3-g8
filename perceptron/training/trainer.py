"""Training loop."""
from __future__ import annotations

import time
import numpy as np

from .history import TrainingHistory
from ..config import ExperimentConfig
from ..metrics import accuracy
from ..models.base import BasePerceptron


class Trainer:
    def __init__(self, model: BasePerceptron, config: ExperimentConfig):
        self.model = model
        self.config = config
        self.history = TrainingHistory()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        stop_on_perfect: bool = True,
        loss_threshold: float | None = None,
    ) -> TrainingHistory:
        rng = np.random.default_rng(self.config.seed)
        start = time.time()
        n_epochs = self.config.epochs

        for epoch in range(1, n_epochs + 1):
            loss = self.model.train_epoch(X, y, self.config.learning_rate, rng)

            train_acc = self._safe_accuracy(y, self.model.predict(X))
            elapsed = round(time.time() - start, 3)

            record = {
                "epoch": epoch,
                "loss": loss,
                "accuracy": train_acc,
                "elapsed_sec": elapsed,
            }
            if X_val is not None and y_val is not None:
                record["val_accuracy"] = self._safe_accuracy(y_val, self.model.predict(X_val))

            self.history.record(**record)

            log_every = max(1, self.config.log_every)
            if epoch == 1 or epoch % log_every == 0 or epoch == n_epochs:
                msg = (
                    f"epoca {epoch:4d}/{n_epochs} | "
                    f"loss={loss:.4f} | acc={train_acc:.4f} | "
                    f"elapsed={elapsed:.2f}s"
                )
                if "val_accuracy" in record:
                    msg += f" | val_acc={record['val_accuracy']:.4f}"
                print(msg)

            if stop_on_perfect and train_acc == 1.0 and loss == 0.0:
                print(f"Convergio en la epoca {epoch}")
                break
            if loss_threshold is not None and loss <= loss_threshold:
                print(f"Convergio en la epoca {epoch} (loss={loss:.6e} <= {loss_threshold})")
                break

        return self.history

    @staticmethod
    def _safe_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)

        if y_true_arr.shape != y_pred_arr.shape:
            return accuracy(y_true_arr, y_pred_arr)

        unique_true = set(np.unique(y_true_arr).tolist())
        if unique_true.issubset({-1.0, 1.0}):
            y_pred_disc = np.where(y_pred_arr >= 0.0, 1.0, -1.0)
            return accuracy(y_true_arr, y_pred_disc)

        if unique_true.issubset({0.0, 1.0}):
            y_pred_disc = np.where(y_pred_arr >= 0.5, 1.0, 0.0)
            return accuracy(y_true_arr, y_pred_disc)

        return accuracy(y_true_arr, y_pred_arr)
