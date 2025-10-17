"""
neuros - a modular operating system for brain–computer interfaces.

This package exposes the core classes required to build, run and benchmark
neural processing pipelines.  It defines drivers for acquiring signals,
processing modules for cleaning and feature extraction, model wrappers for
classification and an agent‑based orchestrator to coordinate everything.

The public API re-exports the most commonly used classes so they can be
imported directly from the `neuros` package.
"""

from neuros.drivers.base_driver import BaseDriver  # noqa: F401
from neuros.drivers.mock_driver import MockDriver  # noqa: F401
from neuros.models.base_model import BaseModel  # noqa: F401
from neuros.models.simple_classifier import SimpleClassifier  # noqa: F401
from neuros.processing.filters import BandpassFilter, SmoothingFilter  # noqa: F401
from neuros.processing.feature_extraction import BandPowerExtractor  # noqa: F401
from neuros.pipeline import Pipeline, MultiModalPipeline  # noqa: F401
from neuros.agents.orchestrator_agent import Orchestrator  # noqa: F401
from neuros.agents.multimodal_orchestrator import MultiModalOrchestrator  # noqa: F401
from neuros.agents.fusion_agent import FusionAgent  # noqa: F401
from neuros.models.composite_model import CompositeModel  # noqa: F401

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
