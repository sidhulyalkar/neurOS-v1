"""
Base model classes for neurOS.

A model encapsulates both training and inference.  Concrete subclasses must
implement :meth:`train` and :meth:`predict`.  They may also define
stateful adaptation logic for online learning.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

import numpy as np


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, **kwargs) -> None:
        self.is_trained = False

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on labelled feature vectors.

        Parameters
        ----------
        X: np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y: np.ndarray
            Target labels of shape (n_samples,).
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for a batch of feature vectors.

        Parameters
        ----------
        X: np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted labels of shape (n_samples,).
        """

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Incrementally update the model with new samples.

        This method is optional; subclasses may override it if incremental
        learning is supported.  By default it delegates to :meth:`train`.
        """
        self.train(X, y)

    def adapt(self, *args, **kwargs) -> None:
        """Hook for online adaptation.

        Models can override this method to adjust internal parameters based on
        runtime conditions such as signal quality or classification confidence.
        The base implementation does nothing.
        """
        return