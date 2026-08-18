"""Base model classes for neurOS.

``BaseModel`` retains the familiar train/predict API while exposing the new
structured ``infer`` contract used by the neurOS kernel.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from neuros.contracts import DecoderCapabilities, DecoderOutput


class BaseModel(ABC):
    """Abstract base class for trainable neurOS models."""

    def __init__(self, **kwargs: Any) -> None:
        self.is_trained = False

    @property
    def capabilities(self) -> DecoderCapabilities:
        return DecoderCapabilities(
            probabilities=type(self).predict_proba is not BaseModel.predict_proba,
            online_fit=type(self).partial_fit is not BaseModel.partial_fit,
        )

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on labelled feature vectors."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for a batch of feature vectors."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """Return class probabilities when supported."""
        return None

    def infer(self, X: np.ndarray) -> DecoderOutput:
        """Run structured inference without fabricating confidence."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        started_ns = time.perf_counter_ns()
        predictions = np.asarray(self.predict(X))
        probabilities = self.predict_proba(X)
        elapsed_ns = time.perf_counter_ns() - started_ns

        prediction: Any
        if predictions.size == 1:
            prediction = predictions.reshape(-1)[0].item()
        else:
            prediction = predictions

        confidence = None
        probability_row = None
        if probabilities is not None:
            probs = np.asarray(probabilities, dtype=float)
            probability_row = probs[0] if probs.ndim > 1 and len(probs) else probs
            if probability_row.size:
                confidence = float(np.max(probability_row))

        return DecoderOutput(
            prediction=prediction,
            confidence=confidence,
            probabilities=probability_row,
            model_id=self.__class__.__name__,
            inference_time_ns=elapsed_ns,
        )

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.train(X, y)

    def adapt(self, *args: Any, **kwargs: Any) -> None:
        return None
