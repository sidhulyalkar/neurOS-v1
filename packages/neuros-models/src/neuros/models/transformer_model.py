"""
Placeholder transformer model for neurOS.

This module defines a simple fully connected neural network to emulate
transformer‑like behaviour when deep learning frameworks such as
PyTorch or TensorFlow are unavailable.  It uses scikit‑learn's
``MLPClassifier`` with multiple hidden layers to approximate the
hierarchical representation learning characteristic of transformer
architectures.  When genuine transformer implementations become
available, this class can be replaced with a proper model using
libraries like PyTorch or TensorFlow.

The :class:`TransformerModel` conforms to the :class:`BaseModel`
interface, providing :meth:`train` and :meth:`predict` methods.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from neuros.models.base_model import BaseModel


class TransformerModel(BaseModel):
    """Approximate transformer classifier using a multi‑layer perceptron.

    Parameters
    ----------
    hidden_layers : tuple[int, ...], optional
        Sizes of the hidden layers.  A deeper network increases
        representational capacity.  Defaults to a four‑layer network.
    max_iter : int, optional
        Maximum number of training iterations for the MLP.  Defaults
        to 200.
    """

    def __init__(self, hidden_layers: Tuple[int, ...] = (256, 128, 64, 32), max_iter: int = 200) -> None:
        self.scaler = StandardScaler()
        self.clf = MLPClassifier(hidden_layer_sizes=hidden_layers, max_iter=max_iter)
        self._trained = False
        # expose underlying classifier for ModelAgent
        self._model: object = self.clf

    def train(self, X: Iterable[np.ndarray], y: Iterable[int]) -> None:
        """Train the MLP classifier on the provided features and labels.

        Data is first standardized using :class:`StandardScaler`.

        Parameters
        ----------
        X : iterable of ndarray
            Feature vectors for each training sample.
        y : iterable of int
            Corresponding integer labels.
        """
        X_arr = np.asarray(list(X), dtype=np.float32)
        y_arr = np.asarray(list(y), dtype=int)
        # fit scaler and transform features
        X_scaled = self.scaler.fit_transform(X_arr)
        self.clf.fit(X_scaled, y_arr)
        self._trained = True

    def predict(self, X: Iterable[np.ndarray]) -> Iterable[Tuple[int, float]]:
        """Predict class labels and confidence scores for the given data.

        Parameters
        ----------
        X : iterable of ndarray
            Feature vectors.

        Yields
        ------
        tuple[int, float]
            The predicted label and confidence for each sample.
        """
        if not self._trained:
            raise RuntimeError("Model must be trained before prediction")
        X_arr = np.asarray(list(X), dtype=np.float32)
        X_scaled = self.scaler.transform(X_arr)
        probs = self.clf.predict_proba(X_scaled)
        preds = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        for label, conf in zip(preds, confidences):
            yield int(label), float(conf)