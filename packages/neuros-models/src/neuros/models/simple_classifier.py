"""A simple logistic regression classifier for demonstration purposes."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from neuros.models.base_model import BaseModel


class SimpleClassifier(BaseModel):
    """Baseline classifier based on logistic regression."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._model = LogisticRegression(**kwargs)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")
        return self._model.predict_proba(X)
