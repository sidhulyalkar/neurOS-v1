"""
neuros - a modular operating system for brain–computer interfaces.

This package exposes the core classes required to build, run and benchmark
neural processing pipelines.  It defines drivers for acquiring signals,
processing modules for cleaning and feature extraction, model wrappers for
classification and an agent‑based orchestrator to coordinate everything.

The public API re-exports the most commonly used classes so they can be
imported directly from the `neuros` package.
"""

from .drivers.base_driver import BaseDriver  # noqa: F401
from .drivers.mock_driver import MockDriver  # noqa: F401
from .models.base_model import BaseModel  # noqa: F401
from .models.simple_classifier import SimpleClassifier  # noqa: F401
from .processing.filters import BandpassFilter, SmoothingFilter  # noqa: F401
from .processing.feature_extraction import BandPowerExtractor  # noqa: F401
from .pipeline import Pipeline, MultiModalPipeline  # noqa: F401
from .agents.orchestrator_agent import Orchestrator  # noqa: F401
from .agents.multimodal_orchestrator import MultiModalOrchestrator  # noqa: F401
from .agents.fusion_agent import FusionAgent  # noqa: F401
from .models.composite_model import CompositeModel  # noqa: F401

__all__ = [
    "BaseDriver",
    "MockDriver",
    "BaseModel",
    "SimpleClassifier",
    "BandpassFilter",
    "SmoothingFilter",
    "BandPowerExtractor",
    "Pipeline",
    "MultiModalPipeline",
    "Orchestrator",
    "MultiModalOrchestrator",
    "FusionAgent",
    "CompositeModel",
]