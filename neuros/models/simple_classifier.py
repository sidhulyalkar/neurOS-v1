"""
A simple logistic regression classifier for demonstration purposes.

This model uses scikit‑learn's :class:`LogisticRegression` to classify
feature vectors.  It serves as a baseline that can be replaced by more
sophisticated models like EEGNet or transformers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from neuros.models.base_model import BaseModel


class SimpleClassifier(BaseModel):
    """Baseline classifier based on logistic regression."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        # allow hyperparameters to be passed through kwargs
        self._model = LogisticRegression(**kwargs)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model has not been trained.  Call train() first.")
        return self._model.predict(X)
