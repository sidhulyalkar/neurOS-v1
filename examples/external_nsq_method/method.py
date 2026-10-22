"""A normal external sklearn decoder participating in neurOS NSQ."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neuros.foundation_models.qualification import (
    ExternalDecoderMethodSpec,
    ExternalLearnedState,
)


class ExternalDecoder:
    """External model implementation; neurOS does not own this training code."""

    def __init__(self) -> None:
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                random_state=17,
                solver="lbfgs",
            ),
        )

    @staticmethod
    def _features(X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError("expected X=(sample, channel, time)")
        return values.reshape(len(values), -1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(self._features(X), np.asarray(y).astype(str))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(self._features(X))).astype(str)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.model.predict_proba(self._features(X)),
            dtype=np.float64,
        )

    def probability_class_labels(self) -> tuple[str, ...]:
        classifier = self.model[-1]
        return tuple(str(value) for value in classifier.classes_)

    def learned_state(self) -> ExternalLearnedState:
        """Hash fitted arrays explicitly instead of serializing with pickle."""

        classifier = self.model[-1]
        if not hasattr(classifier, "coef_"):
            raise RuntimeError("learned_state() requires fit() to complete")
        digest = hashlib.sha256()
        for name, raw in (
            ("scaler_mean", self.model[0].mean_),
            ("scaler_scale", self.model[0].scale_),
            ("coef", classifier.coef_),
            ("intercept", classifier.intercept_),
        ):
            array = np.ascontiguousarray(raw)
            digest.update(
                json.dumps(
                    {
                        "name": name,
                        "dtype": str(array.dtype),
                        "shape": list(array.shape),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(array.tobytes(order="C"))
        digest.update(
            json.dumps(
                [str(value) for value in classifier.classes_],
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return ExternalLearnedState(
            state_identity_kind="tensor_sha256",
            state_sha256=digest.hexdigest(),
            metadata={
                "serializer": "explicit-numpy-scaler-logistic-v1",
                "unsafe_pickle_used": False,
            },
        )


@dataclass(frozen=True)
class ExternalFactory:
    """NSQ calls create once per authorized calibration budget."""

    @property
    def method_spec(self) -> ExternalDecoderMethodSpec:
        return ExternalDecoderMethodSpec(
            method_id="example-sklearn-standard-scaler-logistic-regression",
            implementation=(
                "sklearn.pipeline.Pipeline(StandardScaler, "
                "LogisticRegression)"
            ),
            implementation_version=(
                f"scikit-learn={importlib.metadata.version('scikit-learn')}"
            ),
            input_axes=("sample", "channel", "time"),
            probability_semantics="uncalibrated_probability",
            target_adaptation_mode="none",
            uncertainty_semantics=(
                "predict_proba scores; not post-hoc calibrated"
            ),
            source_reference="examples/external_nsq_method",
            metadata={
                "feature_transform": "flatten-channel-time",
                "scaler": "StandardScaler",
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": 17,
            },
        )

    def create(self) -> ExternalDecoder:
        return ExternalDecoder()
