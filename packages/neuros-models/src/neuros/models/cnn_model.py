"""
Pseudo‑CNN model for neurOS.

This model uses a multilayer perceptron to approximate the behaviour of
a convolutional neural network for EEG classification tasks.  It is
intended as a placeholder until true convolutional models are integrated.
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier

from .base_model import BaseModel

try:
    import tensorflow as _tf  # pragma: no cover
    _TF_AVAILABLE = True
except Exception:
    _tf = None  # type: ignore
    _TF_AVAILABLE = False


class CNNModel(BaseModel):
    """Approximate convolutional neural network using a deep MLP."""

    def __init__(self, hidden_layer_sizes: tuple[int, ...] = (128, 64, 32), max_iter: int = 300) -> None:
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.clf: MLPClassifier | None = None
        self._model: object | None = None
        self._use_deep: bool = False

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        # attempt to use a simple deep CNN if TensorFlow is available
        if _TF_AVAILABLE:
            import numpy as np
            n_samples, n_features = X.shape
            # reshape features to (samples, features, 1)
            X_r = X.reshape((n_samples, n_features, 1)).astype('float32')
            y_cat = _tf.keras.utils.to_categorical(y)
            model = _tf.keras.models.Sequential([
                _tf.keras.layers.Conv1D(32, 5, activation='relu', input_shape=(n_features, 1)),
                _tf.keras.layers.BatchNormalization(),
                _tf.keras.layers.MaxPooling1D(2),
                _tf.keras.layers.Conv1D(64, 5, activation='relu'),
                _tf.keras.layers.BatchNormalization(),
                _tf.keras.layers.GlobalAveragePooling1D(),
                _tf.keras.layers.Dense(y_cat.shape[1], activation='softmax'),
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_r, y_cat, epochs=5, batch_size=32, verbose=0)
            self._model = model
            self._use_deep = True
            self.is_trained = True
            return
        # fallback to MLP
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
        clf = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, max_iter=self.max_iter, random_state=42)
        clf.fit(X_norm, y)
        self.clf = clf
        self._model = clf
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model has not been trained")
        # if using deep model with TensorFlow
        if self._use_deep and _TF_AVAILABLE and isinstance(self._model, _tf.keras.Model):
            import numpy as np
            X_r = X.reshape((X.shape[0], X.shape[1], 1)).astype('float32')
            probs = self._model.predict(X_r, verbose=0)
            return np.argmax(probs, axis=1)
        # fallback: MLP
        if self.clf is None:
            raise RuntimeError("Model has not been trained")
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
        return self.clf.predict(X_norm)