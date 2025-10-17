"""
Pseudo‑EEGNet model implementation for neurOS.

This module provides a lightweight approximation of the EEGNet deep learning
architecture using scikit‑learn's `MLPClassifier`.  While it does not
implement convolutional layers, it serves as a drop‑in replacement within
neurOS to demonstrate integration of advanced models.  When full deep
learning frameworks such as TensorFlow or PyTorch are available, this class
can be replaced with a true EEGNet implementation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.neural_network import MLPClassifier

from neuros.models.base_model import BaseModel

try:
    # Attempt to import TensorFlow for a real deep learning implementation.
    import tensorflow as _tf  # pragma: no cover
    _TENSORFLOW_AVAILABLE = True
except Exception:  # ImportError or other errors
    _tf = None  # type: ignore
    _TENSORFLOW_AVAILABLE = False


class EEGNetModel(BaseModel):
    """Approximate EEGNet classifier using a multilayer perceptron.

    Parameters
    ----------
    hidden_layer_sizes : tuple[int, ...], optional
        Sizes of hidden layers.  Defaults to (64, 32).
    max_iter : int, optional
        Maximum number of iterations for training.  Defaults to 200.
    """

    def __init__(self, hidden_layer_sizes: tuple[int, ...] = (64, 32), max_iter: int = 200) -> None:
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.clf: Optional[MLPClassifier] = None
        # underlying model object for compatibility with ModelAgent
        self._model: Optional[object] = None
        # flag indicating whether a deep learning backend is used
        self._use_deep: bool = False

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        # Use a real deep learning model if TensorFlow is available.  EEGNet
        # normally requires 2D convolutional layers with depthwise and
        # separable filters.  Here we implement a simplified 1D CNN to
        # approximate EEGNet when TensorFlow is installed.  Otherwise fall
        # back to an MLP.
        if _TENSORFLOW_AVAILABLE:
            # reshape features to (samples, channels, 1) where channels = features
            import numpy as np  # local import to satisfy Pyright
            n_samples, n_features = X.shape
            X_r = X.reshape((n_samples, n_features, 1)).astype('float32')
            # one‑hot encode labels
            y_cat = _tf.keras.utils.to_categorical(y)
            # build a simple 1D CNN
            model = _tf.keras.models.Sequential([
                _tf.keras.layers.Conv1D(16, 3, activation='relu', input_shape=(n_features, 1)),
                _tf.keras.layers.BatchNormalization(),
                _tf.keras.layers.MaxPooling1D(2),
                _tf.keras.layers.Conv1D(32, 3, activation='relu'),
                _tf.keras.layers.BatchNormalization(),
                _tf.keras.layers.GlobalAveragePooling1D(),
                _tf.keras.layers.Dense(y_cat.shape[1], activation='softmax'),
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            # train for a few epochs silently
            model.fit(X_r, y_cat, epochs=5, batch_size=32, verbose=0)
            self._model = model
            self._use_deep = True
            self.is_trained = True
            return
        # fallback: normalise features to zero mean and unit variance for MLP stability
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
        clf = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter, random_state=42)
        clf.fit(X_norm, y)
        self.clf = clf
        # assign to _model for compatibility with ModelAgent
        self._model = clf
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model has not been trained")
        # if using deep model, perform inference via TensorFlow
        if self._use_deep and _TENSORFLOW_AVAILABLE and isinstance(self._model, _tf.keras.Model):
            # reshape and predict
            import numpy as np  # ensure numpy available locally
            X_r = X.reshape((X.shape[0], X.shape[1], 1)).astype('float32')
            probs = self._model.predict(X_r, verbose=0)
            return np.argmax(probs, axis=1)
        # otherwise, fallback to MLP
        if self.clf is None:
            raise RuntimeError("Model has not been trained")
        # normalise features as during training
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
        return self.clf.predict(X_norm)
