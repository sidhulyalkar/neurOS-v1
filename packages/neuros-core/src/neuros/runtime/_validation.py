"""Internal validation helpers for runtime authority-bearing numeric fields."""
from __future__ import annotations

import math
from numbers import Real
from typing import Any


def positive_finite_real(value: Any, *, field_name: str) -> float:
    """Return a canonical finite positive float without accepting coercions.

    Authority-bearing durations must not silently accept strings, booleans,
    NaN, or infinities. ``numbers.Real`` intentionally accepts ordinary Python
    numeric scalars and NumPy real scalars while excluding text-like values.
    """

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real numeric scalar")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return resolved
