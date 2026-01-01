"""
Dataset loaders for public neuroscience and BCI datasets.

This module provides unified interfaces for loading and preprocessing datasets
from various sources including Allen Institute, BNCI Horizon, PhysioNet, and more.

Validation Framework
-------------------
The module also includes validation-focused dataset loaders that implement the
BaseNeuralDataset interface for SAE feature validation across modalities:
- AllenVisualCodingValidator: Orientation selectivity validation with spike data
- BCIMotorImageryValidator: Motor laterality validation with EEG data
"""

from __future__ import annotations

# Base dataset interface
from neuros.datasets.base_dataset import BaseNeuralDataset, NeuralWindow

# Public BCI datasets
from neuros.datasets.bci_datasets import load_bnci_horizon, load_physionet_mi

# Allen Institute datasets (for foundation models)
from neuros.datasets.allen_datasets import (
    load_allen_visual_coding,
    load_allen_neuropixels,
    load_allen_mock_data,
    AllenDatasetConfig,
)

# Validation dataset loaders
from neuros.datasets.allen_datasets import AllenVisualCodingValidator
from neuros.datasets.bci_datasets import BCIMotorImageryValidator

__all__ = [
    # Base interfaces
    "BaseNeuralDataset",
    "NeuralWindow",
    # BCI datasets
    "load_bnci_horizon",
    "load_physionet_mi",
    # Allen Institute datasets
    "load_allen_visual_coding",
    "load_allen_neuropixels",
    "load_allen_mock_data",
    "AllenDatasetConfig",
    # Validation datasets
    "AllenVisualCodingValidator",
    "BCIMotorImageryValidator",
]
