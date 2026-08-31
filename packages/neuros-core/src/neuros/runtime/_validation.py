"""Internal validators for runtime authority-bearing fields."""
from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any


def nonblank_string(value: Any, *, field_name: str) -> str:
    """Require an explicit nonblank string without rewriting its identity."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be nonblank")
    return value


def positive_integral(value: Any, *, field_name: str) -> int:
    """Return a canonical positive Python int without bool/text coercion."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive")
    return resolved


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
