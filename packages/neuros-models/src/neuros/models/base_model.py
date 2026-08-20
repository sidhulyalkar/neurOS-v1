"""Base decoder contract for neurOS models."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from neuros.contracts import DecoderCapabilities, DecoderOutput
from neuros.models.analysis import InterpretabilityManifest


class BaseModel(ABC):
    """Abstract base class for trainable neurOS decoders.

    The runtime contract remains deliberately small.  Optional methods expose
    logits, embeddings, and a model-analysis surface without forcing every
    classical estimator to pretend it has mechanistic internals.
    """

    model_version: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        del kwargs
        self.is_trained = False

    @property
    def capabilities(self) -> DecoderCapabilities:
        return DecoderCapabilities(
            probabilities=type(self).predict_proba is not BaseModel.predict_proba,
            online_fit=type(self).partial_fit is not BaseModel.partial_fit,
            embeddings=type(self).encode is not BaseModel.encode,
        )

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on labelled samples."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels or values for a batch."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        return None

    def predict_logits(self, X: np.ndarray) -> np.ndarray | None:
        return None

    def encode(self, X: np.ndarray) -> np.ndarray | None:
        """Return a stable pooled representation when the model exposes one."""

        return None

    def analysis_manifest(self) -> InterpretabilityManifest:
        """Describe inspectable components without importing ``neuros-mechint``."""

        return InterpretabilityManifest.opaque(type(self).__name__)

    def analysis_model(self) -> Any | None:
        """Return the underlying framework model used for mechanistic experiments."""

        return None

    def mechint_adapter(self) -> Any:
        """Create a ``neuros-mechint`` adapter lazily.

        This keeps mechanistic interpretability optional for normal deployment
        while making the research path one method call when installed.
        """

        try:
            from neuros_mechint.adapters import NeurOSModelAdapter
        except ImportError as exc:  # pragma: no cover - exercised in minimal installs
            raise ImportError(
                "Install neuros-models[mechint] (or neuros-mechint) to create a mechanistic adapter."
            ) from exc
        return NeurOSModelAdapter(self)

    def infer(self, X: np.ndarray) -> DecoderOutput:
        """Run structured inference without fabricating confidence."""

        if not self.is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")

        started_ns = time.perf_counter_ns()
        predictions = np.asarray(self.predict(X))
        probabilities = self.predict_proba(X)
        logits = self.predict_logits(X)
        embedding = self.encode(X)
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

        logits_row = None
        if logits is not None:
            logits_arr = np.asarray(logits, dtype=float)
            logits_row = logits_arr[0] if logits_arr.ndim > 1 and len(logits_arr) else logits_arr

        embedding_row = None
        if embedding is not None:
            embedding_arr = np.asarray(embedding, dtype=float)
            embedding_row = (
                embedding_arr[0] if embedding_arr.ndim > 1 and len(embedding_arr) else embedding_arr
            )

        manifest = self.analysis_manifest()
        return DecoderOutput(
            prediction=prediction,
            confidence=confidence,
            probabilities=probability_row,
            logits=logits_row,
            embedding=embedding_row,
            model_id=self.__class__.__name__,
            model_version=self.model_version,
            inference_time_ns=elapsed_ns,
            metadata={
                "architecture": manifest.architecture,
                "backend": manifest.backend,
                "analysis_manifest": manifest.fingerprint(),
                "mechint_ready": manifest.mechint_ready,
            },
        )

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.train(X, y)

    def adapt(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None
