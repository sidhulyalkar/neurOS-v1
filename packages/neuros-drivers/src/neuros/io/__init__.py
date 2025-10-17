"""
Input/Output utilities for NeurOS.

This module provides loaders and writers for various neural data formats.
"""

from __future__ import annotations

# NWB support
try:
    from .nwb_loader import NWBLoader, NWBWriter, NWB_AVAILABLE
except ImportError:
    NWB_AVAILABLE = False
    NWBLoader = None
    NWBWriter = None

__all__ = [
    "NWBLoader",
    "NWBWriter",
    "NWB_AVAILABLE",
]
