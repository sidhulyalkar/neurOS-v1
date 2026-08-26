"""Reference out-of-tree neurOS transform plugin."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from neuros.contracts import SignalFrame


class GainTransform:
    """Multiply data by a fixed gain while preserving ``SignalFrame`` identity."""

    def __init__(self, gain: float = 1.0) -> None:
        value = float(gain)
        if not np.isfinite(value):
            raise ValueError("gain must be finite")
        self.gain = value

    def transform(self, item: Any) -> Any:
        if not isinstance(item, SignalFrame):
            return np.asarray(item) * self.gain
        return replace(
            item,
            data=np.asarray(item.data) * self.gain,
            metadata={
                **dict(item.metadata),
                "example_gain": self.gain,
                "transform_plugin": "neuros-example-plugin",
            },
        )
