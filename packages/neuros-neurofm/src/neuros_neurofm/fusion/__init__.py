"""
Multi-modal fusion modules for NeuroFM-X.

This module contains the Perceiver-IO architecture for fusing
features from multiple modalities (spikes, LFP, behavior, etc.).
"""

from neuros_neurofm.fusion.perceiver import PerceiverIO

__all__ = [
    "PerceiverIO",
]
