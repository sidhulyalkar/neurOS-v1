"""
Random forest classifier for neurOS.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseModel


class RandomForestModel(BaseModel):
    """Random forest classifier wrapper."""

    def __init__(self, n_estimators: int = 100, max_depth: int | None = None) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.clf: RandomForestClassifier | None = None
        self._model: object | None = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        clf = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=42)
        clf.fit(X, y)
        self.clf = clf
        self._model = clf
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Model has not been trained")
        return self.clf.predict(X)