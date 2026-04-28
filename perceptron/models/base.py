"""Abstract base class for perceptron variants."""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BasePerceptron(ABC):
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def train_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        rng: np.random.Generator,
    ) -> float:
        """One pass over the data. Returns epoch loss."""
        ...

    @abstractmethod
    def get_weights(self) -> dict[str, np.ndarray]:
        """Weights as a dict for np.savez."""
        ...

    @abstractmethod
    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        ...
