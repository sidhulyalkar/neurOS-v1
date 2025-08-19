"""
Adaptation utilities.

These classes implement simple adaptive behaviours that can be used by the
orchestrator to adjust the processing pipeline at runtime.  They monitor
metrics such as classification confidence and update parameters accordingly.
"""

from __future__ import annotations

import numpy as np


class AdaptiveThreshold:
    """Adapt the decision threshold based on recent classifier confidence.

    A sliding window of confidence scores is maintained.  If the mean
    confidence drops below a target value, the threshold is lowered to make
    the classifier more permissive.  Conversely, if the mean confidence is
    high, the threshold is raised to reduce false positives.
    """

    def __init__(self, window_size: int = 50, base_threshold: float = 0.5) -> None:
        self.window_size = window_size
        self.base_threshold = base_threshold
        self.scores: list[float] = []
        self.threshold = base_threshold

    def update(self, score: float) -> None:
        self.scores.append(score)
        if len(self.scores) > self.window_size:
            self.scores.pop(0)
        if self.scores:
            mean_conf = float(np.mean(self.scores))
            # adjust threshold around base threshold
            delta = (0.5 - mean_conf) * 0.5  # scale adjustment
            self.threshold = np.clip(self.base_threshold + delta, 0.1, 0.9)

    def should_trigger(self, score: float) -> bool:
        return score >= self.threshold