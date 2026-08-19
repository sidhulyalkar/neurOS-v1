"""Small bridges from foundation-model embeddings into the neurOS runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from neuros.models.base_model import BaseModel


class FoundationEmbeddingDecoder(BaseModel):
    """Attach a transparent linear readout to any foundation-model encoder.

    This makes an upstream ``encoder(X) -> embeddings`` callable usable as a
    neurOS ``BaseModel`` and therefore as a decoder node in ``neuros.Pipeline``
    or ``RuntimeGraph``. The upstream model remains responsible for modality-
    specific preprocessing, channel geometry, and checkpoint loading.
    """

    def __init__(
        self,
        encoder: Callable[[Any], Any],
        *,
        task: Literal["classification", "regression"] = "classification",
        alpha: float = 1e-3,
        model_id: str = "foundation-encoder",
    ) -> None:
        super().__init__()
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.encoder = encoder
        self.task = task
        self.alpha = float(alpha)
        self.model_id = model_id
        self._weights: np.ndarray | None = None
        self._classes: np.ndarray | None = None
        self._embedding_dim: int | None = None

    @staticmethod
    def _as_embeddings(values: Any) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if matrix.ndim != 2:
            raise ValueError(f"encoder must return a 2D matrix, got {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError("encoder returned NaN or infinite values")
        return matrix

    def _encode(self, X: Any) -> np.ndarray:
        embeddings = self._as_embeddings(self.encoder(X))
        if self._embedding_dim is not None and embeddings.shape[1] != self._embedding_dim:
            raise ValueError(
                f"embedding dimension changed from {self._embedding_dim} to {embeddings.shape[1]}"
            )
        return embeddings

    def _fit_ridge(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        design = np.concatenate([x, np.ones((len(x), 1), dtype=x.dtype)], axis=1)
        regularizer = np.eye(design.shape[1], dtype=x.dtype) * self.alpha
        regularizer[-1, -1] = 0.0
        return np.linalg.pinv(design.T @ design + regularizer) @ design.T @ y

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        embeddings = self._encode(X)
        targets = np.asarray(y)
        if len(embeddings) != len(targets):
            raise ValueError("encoder output and target lengths must match")
        self._embedding_dim = int(embeddings.shape[1])

        if self.task == "classification":
            labels = targets.reshape(-1)
            classes = np.unique(labels)
            if len(classes) < 2:
                raise ValueError("classification requires at least two classes")
            self._classes = classes
            mapping = {value: index for index, value in enumerate(classes.tolist())}
            encoded = np.zeros((len(labels), len(classes)), dtype=np.float64)
            for row, value in enumerate(labels.tolist()):
                encoded[row, mapping[value]] = 1.0
            self._weights = self._fit_ridge(embeddings, encoded)
        else:
            values = np.asarray(targets, dtype=np.float64)
            if values.ndim == 1:
                values = values[:, None]
            self._weights = self._fit_ridge(embeddings, values)

        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self._weights is None:
            raise RuntimeError("Model has not been trained. Call train() first.")
        embeddings = self._encode(X)
        design = np.concatenate(
            [embeddings, np.ones((len(embeddings), 1), dtype=embeddings.dtype)], axis=1
        )
        values = design @ self._weights
        if self.task == "classification":
            if self._classes is None:
                raise RuntimeError("classification classes are missing")
            return self._classes[np.argmax(values, axis=1)]
        return values[:, 0] if values.shape[1] == 1 else values

    def __repr__(self) -> str:
        return (
            f"FoundationEmbeddingDecoder(model_id={self.model_id!r}, task={self.task!r}, "
            f"alpha={self.alpha}, trained={self.is_trained})"
        )
