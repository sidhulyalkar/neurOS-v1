"""
Support vector machine classifier for neurOS.
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC

from neuros.models.base_model import BaseModel


class SVMModel(BaseModel):
    """SVM classifier with RBF kernel."""

    def __init__(self, C: float = 1.0, gamma: str | float = "scale") -> None:
        super().__init__()
        self.C = C
        self.gamma = gamma
        self.clf: SVC | None = None
        self._model: object | None = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        clf = SVC(C=self.C, gamma=self.gamma, probability=True, random_state=42)
        clf.fit(X, y)
        self.clf = clf
        self._model = clf
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Model has not been trained")
        return self.clf.predict(X)
