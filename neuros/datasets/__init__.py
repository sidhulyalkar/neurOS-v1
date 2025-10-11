"""
Dataset loaders for public neuroscience and BCI datasets.

This module provides unified interfaces for loading and preprocessing datasets
from various sources including Allen Institute, BNCI Horizon, PhysioNet, and more.
"""

from __future__ import annotations

# Public BCI datasets
from .bci_datasets import load_bnci_horizon, load_physionet_mi

# Allen Institute datasets (for foundation models)
from .allen_datasets import (
    load_allen_visual_coding,
    load_allen_neuropixels,
    load_allen_mock_data,
    AllenDatasetConfig,
)

__all__ = [
    # BCI datasets
    "load_bnci_horizon",
    "load_physionet_mi",
    # Allen Institute datasets
    "load_allen_visual_coding",
    "load_allen_neuropixels",
    "load_allen_mock_data",
    "AllenDatasetConfig",
]
