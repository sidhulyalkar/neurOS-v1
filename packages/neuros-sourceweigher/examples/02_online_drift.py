"""Smooth source reliability as the target domain drifts over time."""
from __future__ import annotations

import numpy as np

from neuros_sourceweigher import DistanceWeigher, OnlineSourceWeigher

sources = np.array([[0.0], [2.0]])
online = OnlineSourceWeigher(
    DistanceWeigher(temperature=0.15, standardize=False),
    adaptation_rate=0.35,
    max_l1_step=0.30,
)

for target_value in np.linspace(0.0, 2.0, 9):
    result = online.update(sources, np.array([target_value]))
    print(
        f"target={target_value:0.2f}",
        "weights=",
        np.round(result.weights, 3),
        "drift=",
        round(float(result.diagnostics.metadata["l1_weight_drift"]), 3),
    )
