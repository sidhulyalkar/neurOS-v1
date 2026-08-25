"""Faithful optional adapters for selected Braindecode raw-window decoders.

neurOS owns the runtime/evidence boundary; Braindecode owns these published model
implementations and their Skorch-based training loop.  This module therefore
wraps upstream objects rather than copying architectures or training code.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import sys
import time
from typing import Any, Mapping

import numpy as np

from neuros.contracts import DecoderCapabilities, DecoderOutput
from neuros.models.analysis import InterpretabilityManifest
from neuros.models.base_model import BaseModel


_SUPPORTED_MODELS = {
    "eegnet": "EEGNet",
    "eegconformer": "EEGConformer",
    "shallowfbcspnet": "ShallowFBCSPNet",
    "deep4net": "Deep4Net",
}

_RESERVED_MODEL_OPTIONS = {
    "n_chans",
    "n_outputs",
    "n_times",
    "sfreq",
    "input_window_seconds",
}


class BraindecodeDecoder(BaseModel):
    """Train/infer through upstream Braindecode without hidden preprocessing.

    Inputs are exactly ``(batch, channels, time)``. The adapter never resamples,
    pads, crops, changes channel order, or constructs sensor geometry. Windowing
    belongs to the neurOS ``NeuralWindow`` runtime contract and preprocessing
    belongs to explicit transforms/evidence protocols.
    """

    def __init__(
        self,
        model_name: str,
        n_channels: int,
        n_times: int,
        n_classes: int,
        *,
        sample_rate_hz: float | None = None,
        model_options: Mapping[str, Any] | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        n_epochs: int = 10,
        batch_size: int = 32,
        device: str = "cpu",
        random_state: int = 0,
    ) -> None:
        super().__init__()
        key = model_name.replace("_", "").replace("-", "").lower()
        if key not in _SUPPORTED_MODELS:
            supported = ", ".join(sorted(_SUPPORTED_MODELS.values()))
            raise ValueError(
                f"Unsupported Braindecode model {model_name!r}. "
                f"Qualified adapter models: {supported}"
            )
        if n_channels <= 0 or n_times <= 0:
            raise ValueError("n_channels and n_times must be positive")
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if sample_rate_hz is not None and sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive when provided")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if n_epochs <= 0 or batch_size <= 0:
            raise ValueError("n_epochs and batch_size must be positive")

        options = dict(model_options or {})
        conflicts = sorted(_RESERVED_MODEL_OPTIONS.intersection(options))
        if conflicts:
            raise ValueError(
                "model_options cannot override neurOS geometry parameters: "
                + ", ".join(conflicts)
            )

        self.model_name = _SUPPORTED_MODELS[key]
        self.n_channels = int(n_channels)
        self.n_times = int(n_times)
        self.n_classes = int(n_classes)
        self.sample_rate_hz = float(sample_rate_hz) if sample_rate_hz is not None else None
        self.model_options = options
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.device = str(device)
        self.random_state = int(random_state)

        self._module: Any | None = None
        self._classifier: Any | None = None
        self._braindecode_version: str | None = None

    @property
    def capabilities(self) -> DecoderCapabilities:
        return DecoderCapabilities(probabilities=True)

    @staticmethod
    def _require_upstream() -> tuple[Any, Any, Any]:
        if sys.version_info < (3, 11):
            raise ImportError(
                "Braindecode 1.7 requires Python 3.11+. "
                "The neurOS kernel remains compatible with Python 3.10."
            )
        try:
            import torch
            from braindecode import EEGClassifier
            import braindecode.models as models
        except ImportError as exc:  # pragma: no cover - exercised in minimal installs
            raise ImportError(
                "Braindecode support is optional. Install with "
                "`pip install 'neuros-models[braindecode]'` on Python 3.11+."
            ) from exc
        return torch, EEGClassifier, models

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        expected = (self.n_channels, self.n_times)
        if arr.ndim != 3:
            raise ValueError(
                "BraindecodeDecoder expects X with shape "
                f"(batch, channels, time); received {tuple(arr.shape)}"
            )
        if tuple(arr.shape[1:]) != expected:
            raise ValueError(
                f"Expected neural windows with geometry {expected}, "
                f"received {tuple(arr.shape[1:])}"
            )
        if arr.shape[0] == 0:
            raise ValueError("X must contain at least one window")
        if not np.isfinite(arr).all():
            raise ValueError("X contains NaN or infinite values")
        return arr

    def _build_module(self) -> Any:
        _, _, models = self._require_upstream()
        model_type = getattr(models, self.model_name, None)
        if model_type is None:
            raise RuntimeError(
                f"Installed Braindecode does not expose qualified model {self.model_name}"
            )

        kwargs: dict[str, Any] = {
            "n_chans": self.n_channels,
            "n_outputs": self.n_classes,
            "n_times": self.n_times,
            **self.model_options,
        }
        if self.sample_rate_hz is not None:
            signature = inspect.signature(model_type)
            if "sfreq" in signature.parameters:
                kwargs["sfreq"] = self.sample_rate_hz
        return model_type(**kwargs)

    def _ensure_classifier(self) -> Any:
        if self._classifier is not None:
            return self._classifier

        torch, classifier_type, _ = self._require_upstream()
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self._module = self._build_module()
        self._classifier = classifier_type(
            self._module,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            optimizer__lr=self.learning_rate,
            optimizer__weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            max_epochs=self.n_epochs,
            train_split=None,
            device=self.device,
            classes=np.arange(self.n_classes),
            verbose=0,
        )
        try:
            self._braindecode_version = importlib.metadata.version("braindecode")
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover
            self._braindecode_version = None
        self.model_version = self._braindecode_version
        return self._classifier

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        X_arr = self._validate_X(X)
        y_arr = np.asarray(y, dtype=np.int64)
        if y_arr.ndim != 1 or len(y_arr) != len(X_arr):
            raise ValueError("y must be one-dimensional and aligned with X")
        if y_arr.size and (y_arr.min() < 0 or y_arr.max() >= self.n_classes):
            raise ValueError(f"labels must be in [0, {self.n_classes - 1}]")
        self._ensure_classifier().fit(X_arr, y_arr)
        self.is_trained = True

    def _require_trained(self) -> Any:
        if not self.is_trained or self._classifier is None:
            raise RuntimeError("Model has not been trained. Call train() first.")
        return self._classifier

    def predict(self, X: np.ndarray) -> np.ndarray:
        classifier = self._require_trained()
        return np.asarray(classifier.predict(self._validate_X(X)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        classifier = self._require_trained()
        probabilities = np.asarray(
            classifier.predict_proba(self._validate_X(X)), dtype=np.float64
        )
        if probabilities.ndim != 2 or probabilities.shape[1] != self.n_classes:
            raise RuntimeError(
                "Braindecode classifier returned unexpected probability geometry: "
                f"{tuple(probabilities.shape)}"
            )
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Braindecode classifier returned non-finite probabilities")
        return probabilities

    def infer(self, X: np.ndarray) -> DecoderOutput:
        X_arr = self._validate_X(X)
        classifier = self._require_trained()
        started_ns = time.perf_counter_ns()
        probabilities = np.asarray(classifier.predict_proba(X_arr), dtype=np.float64)
        predictions = np.asarray(classifier.predict(X_arr))
        elapsed_ns = time.perf_counter_ns() - started_ns

        if probabilities.ndim != 2 or probabilities.shape != (
            len(X_arr),
            self.n_classes,
        ):
            raise RuntimeError(
                "Braindecode classifier returned unexpected probability geometry: "
                f"{tuple(probabilities.shape)}"
            )
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Braindecode classifier returned non-finite probabilities")

        prediction: Any = (
            predictions.reshape(-1)[0].item() if predictions.size == 1 else predictions
        )
        probability_row = probabilities[0] if len(probabilities) else probabilities
        return DecoderOutput(
            prediction=prediction,
            confidence=float(np.max(probability_row)) if probability_row.size else None,
            probabilities=probability_row,
            model_id=f"Braindecode:{self.model_name}",
            model_version=self._braindecode_version,
            inference_time_ns=elapsed_ns,
            metadata={
                "architecture": self.model_name,
                "backend": "braindecode/torch",
                "braindecode_version": self._braindecode_version,
                "input_contract": "batch,channel,time",
                "n_channels": self.n_channels,
                "n_times": self.n_times,
                "sample_rate_hz": self.sample_rate_hz,
                "upstream_training": "EEGClassifier",
                "hidden_preprocessing": False,
            },
        )

    def analysis_model(self) -> Any | None:
        if self._classifier is not None and hasattr(self._classifier, "module_"):
            return self._classifier.module_
        return self._module

    def analysis_manifest(self) -> InterpretabilityManifest:
        return InterpretabilityManifest(
            model_type="BraindecodeDecoder",
            architecture=self.model_name,
            backend="braindecode/torch",
            input_axes=("batch", "channel", "time"),
            output_semantics="classification logits/probabilities",
            limitations=(
                "No stable neurOS activation paths are claimed for upstream Braindecode internals yet.",
                "Qualify architecture-specific hook paths before mechanistic intervention evidence.",
            ),
        )
