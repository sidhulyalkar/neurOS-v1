"""
Gradient boosting classifier for neurOS.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .base_model import BaseModel


class GBDTModel(BaseModel):
    """Gradient boosting decision tree classifier."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.clf: GradientBoostingClassifier | None = None
        self._model: object | None = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        clf = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42,
        )
        clf.fit(X, y)
        self.clf = clf
        self._model = clf
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Model has not been trained")
        return self.clf.predict(X)