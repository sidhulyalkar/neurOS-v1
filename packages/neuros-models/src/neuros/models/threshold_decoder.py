"""A deterministic, training-free decoder for runtime smoke tests and tutorials."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from neuros.contracts import DecoderCapabilities, DecoderOutput


class ThresholdDecoder:
    """Classify by the mean feature value relative to a fixed threshold.

    This decoder is intentionally simple and interpretable. It is useful for
    validating an end-to-end runtime/configuration path without hiding model
    training inside the CLI. It is not intended as a scientific BCI baseline.
    """

    def __init__(self, threshold: float = 0.0, positive_label: Any = 1, negative_label: Any = 0) -> None:
        self.threshold = float(threshold)
        self.positive_label = positive_label
        self.negative_label = negative_label

    @property
    def capabilities(self) -> DecoderCapabilities:
        return DecoderCapabilities(probabilities=False, uncertainty=False)

    def infer(self, X: np.ndarray) -> DecoderOutput:
        started = time.perf_counter_ns()
        value = float(np.asarray(X, dtype=float).mean())
        prediction = self.positive_label if value >= self.threshold else self.negative_label
        return DecoderOutput(
            prediction=prediction,
            confidence=None,
            model_id="threshold",
            model_version="1",
            inference_time_ns=time.perf_counter_ns() - started,
            metadata={"mean_feature": value, "threshold": self.threshold},
        )
