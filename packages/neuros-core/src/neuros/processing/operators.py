"""Composable processing operators for the typed neurOS runtime."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

import numpy as np

from neuros.contracts import SignalFrame


class FeatureTransform:
    """Apply a sequence of legacy filters followed by a feature extractor.

    This adapter is deliberately small: it lets the existing processing library
    participate in RuntimeGraph execution without teaching the kernel about any
    concrete filter or extractor implementation.
    """

    def __init__(self, *, filters: Iterable[Any] = (), extractor: Any) -> None:
        if extractor is None or not hasattr(extractor, "extract"):
            raise TypeError("extractor must provide extract(data)")
        self.filters = tuple(filters)
        self.extractor = extractor

    def transform(self, item: Any) -> Any:
        frame = item if isinstance(item, SignalFrame) else None
        data = np.asarray(frame.data if frame is not None else item)
        for filter_obj in self.filters:
            if not hasattr(filter_obj, "apply"):
                raise TypeError(f"Filter {type(filter_obj).__name__} lacks apply(data)")
            data = np.asarray(filter_obj.apply(data))
        features = np.asarray(self.extractor.extract(data))
        if frame is None:
            return features
        return replace(
            frame,
            data=features,
            metadata={
                **dict(frame.metadata),
                "representation": "features",
                "extractor": type(self.extractor).__name__,
            },
        )
