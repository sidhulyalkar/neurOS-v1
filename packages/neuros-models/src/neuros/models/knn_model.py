"""
K‑nearest neighbours classifier for neurOS.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from .base_model import BaseModel


class KNNModel(BaseModel):
    """K‑nearest neighbours classifier wrapper."""

    def __init__(self, n_neighbors: int = 5) -> None:
        super().__init__()
        self.n_neighbors = n_neighbors
        self.clf: KNeighborsClassifier | None = None
        self._model: object | None = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        clf = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        clf.fit(X, y)
        self.clf = clf
        self._model = clf
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Model has not been trained")
        return self.clf.predict(X)