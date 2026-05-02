"""Training loop."""
from __future__ import annotations

import copy
import time
from typing import Optional
import numpy as np

from .history import TrainingHistory
from .lr_schedules import build_lr_schedule
from .optimizers import build_optimizer
from ..config import ExperimentConfig
from ..metrics import accuracy, evaluate_predictions, mse, multiclass_accuracy
from ..models.base import BasePerceptron


class Trainer:
    def __init__(self, model: BasePerceptron, config: ExperimentConfig):
        self.model = model
        self.config = config
        self.history = TrainingHistory()
        self.optimizer = build_optimizer(config.optimizer, **config.optimizer_params)
        self.lr_schedule = build_lr_schedule(
            config.lr_schedule,
            config.learning_rate,
            config.epochs,
            config.lr_schedule_params,
        )
        if hasattr(self.model, "loss"):
            self.model.loss = config.loss

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        stop_on_perfect: bool = True,
        loss_threshold: float | None = None,
        restore_best_weights: bool = False,
    ) -> TrainingHistory:
        rng = np.random.default_rng(self.config.seed)
        start = time.time()
        n_epochs = self.config.epochs
        task_type = self._resolve_task_type(y)
        best_monitor = float("inf")
        stale_epochs = 0
        best_weights = None

        for epoch in range(1, n_epochs + 1):
            current_lr = self.lr_schedule(epoch)
            loss = self.model.train_epoch(X, y, current_lr, rng, optimizer=self.optimizer)

            train_pred = self.model.predict(X)
            train_acc = self._safe_accuracy(y, train_pred)
            train_metrics = evaluate_predictions(y, train_pred, task_type, threshold=self.config.threshold)
            elapsed = round(time.time() - start, 3)

            record = {
                "epoch": epoch,
                "loss": loss,
                "train_loss": loss,
                "accuracy": train_acc,
                "lr": current_lr,
                "elapsed_sec": elapsed,
            }
            record.update(train_metrics)

            if X_val is not None and y_val is not None:
                val_pred = self.model.predict(X_val)
                val_loss = self._compute_loss(y_val, val_pred)
                record["val_loss"] = val_loss
                record["val_accuracy"] = self._safe_accuracy(y_val, val_pred)
                for key, value in evaluate_predictions(
                    y_val, val_pred, task_type, threshold=self.config.threshold
                ).items():
                    record[f"val_{key}"] = value

            self.history.record(**record)

            log_every = max(1, self.config.log_every)
            if epoch == 1 or epoch % log_every == 0 or epoch == n_epochs:
                acc_txt = f"{train_acc:.4f}" if train_acc is not None else "n/a"
                msg = (
                    f"epoca {epoch:4d}/{n_epochs} | "
                    f"loss={loss:.4f} | acc={acc_txt} | "
                    f"elapsed={elapsed:.2f}s"
                )
                if "val_accuracy" in record:
                    val_acc = record["val_accuracy"]
                    val_txt = f"{val_acc:.4f}" if val_acc is not None else "n/a"
                    msg += f" | val_acc={val_txt}"
                if "val_loss" in record:
                    msg += f" | val_loss={record['val_loss']:.4f}"
                print(msg)

            if stop_on_perfect and train_acc is not None and train_acc == 1.0 and loss == 0.0:
                print(f"Convergio en la epoca {epoch}")
                break
            if loss_threshold is not None and loss <= loss_threshold:
                print(f"Convergio en la epoca {epoch} (loss={loss:.6e} <= {loss_threshold})")
                break
            if self.config.early_stopping:
                monitor = record.get("val_loss", loss)
                if monitor < best_monitor - self.config.min_delta:
                    best_monitor = monitor
                    stale_epochs = 0
                    if restore_best_weights:
                        best_weights = copy.deepcopy(self.model.get_weights())
                else:
                    stale_epochs += 1
                    if stale_epochs >= self.config.patience:
                        print(f"Early stopping en la epoca {epoch}")
                        break

        if restore_best_weights and best_weights is not None:
            self.model.set_weights(best_weights)

        return self.history

    def _resolve_task_type(self, y: np.ndarray) -> str:
        if self.config.task_type != "auto":
            return self.config.task_type

        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim > 1 and y_arr.shape[1] > 1:
            return "multiclass"

        unique_true = set(np.unique(y_arr).tolist())
        if unique_true.issubset({-1.0, 1.0}):
            return "bipolar_binary"
        if unique_true.issubset({0.0, 1.0}):
            return "binary"
        return "regression"

    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)
        if y_true_arr.ndim == 1 and y_pred_arr.ndim == 2 and y_pred_arr.shape[1] == 1:
            y_true_arr = y_true_arr.reshape(-1, 1)
        if self.config.loss == "categorical_cross_entropy":
            clipped = np.clip(y_pred_arr, 1e-12, 1.0)
            return float(np.mean(-np.sum(y_true_arr * np.log(clipped), axis=1)))
        if self.config.loss == "binary_cross_entropy":
            clipped = np.clip(y_pred_arr, 1e-12, 1.0 - 1e-12)
            per_output = -(
                y_true_arr * np.log(clipped) + (1.0 - y_true_arr) * np.log(1.0 - clipped)
            )
            return float(np.mean(per_output))
        return mse(y_true_arr, y_pred_arr)

    @staticmethod
    def _safe_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)

        if y_true_arr.ndim > 1 and y_true_arr.shape[1] > 1:
            return multiclass_accuracy(y_true_arr, y_pred_arr)

        if y_true_arr.shape != y_pred_arr.shape:
            return None

        unique_true = set(np.unique(y_true_arr).tolist())
        if unique_true.issubset({-1.0, 1.0}):
            y_pred_disc = np.where(y_pred_arr >= 0.0, 1.0, -1.0)
            return accuracy(y_true_arr, y_pred_disc)

        if unique_true.issubset({0.0, 1.0}):
            y_pred_disc = np.where(y_pred_arr >= 0.5, 1.0, 0.0)
            return accuracy(y_true_arr, y_pred_disc)

        return None
