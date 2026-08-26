"""Reference external plugin package for neurOS."""

from .source import SineSource
from .transforms import GainTransform

__all__ = ["GainTransform", "SineSource"]
