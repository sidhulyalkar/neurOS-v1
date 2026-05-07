"""
Models for NeuroFM-X.

This module contains the core NeuroFM-X model and its components.
"""

from neuros_neurofm.models.mamba_backbone import MambaBackbone
from neuros_neurofm.models.popt import PopT, PopTWithLatents

# ENGRAM-FMx backbone
from neuros_neurofm.backbones.engram_fmx import (
    ENGRAMFMxConfig,
    ENGRAMBlock,
    ENGRAMBackbone,
    ENGRAMBackboneOutput,
)
from neuros_neurofm.models.heads import (
    DecoderHead,
    EncoderHead,
    ContrastiveHead,
    ForecastHead,
    MultiTaskHeads,
)
from neuros_neurofm.models.neurofmx import NeuroFMX

__all__ = [
    # Backbones
    "MambaBackbone",
    "ENGRAMFMxConfig",
    "ENGRAMBlock",
    "ENGRAMBackbone",
    "ENGRAMBackboneOutput",
    # Aggregation
    "PopT",
    "PopTWithLatents",
    # Heads
    "DecoderHead",
    "EncoderHead",
    "ContrastiveHead",
    "ForecastHead",
    "MultiTaskHeads",
    # Complete models
    "NeuroFMX",
    "NeuroFMXComplete",
    "NeuroFMXMultiTask",
]
