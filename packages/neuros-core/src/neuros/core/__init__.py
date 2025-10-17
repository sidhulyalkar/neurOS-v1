"""
neurOS Core Package
===================

Core functionality for neurOS including pipelines, agents, and orchestration.
"""

from neuros.pipeline import Pipeline
from neuros.agents import (
    BaseAgent,
    DeviceAgent,
    ProcessingAgent,
    ModelAgent,
    FusionAgent,
    MultiModalOrchestrator,
)

__all__ = [
    "Pipeline",
    "BaseAgent",
    "DeviceAgent",
    "ProcessingAgent",
    "ModelAgent",
    "FusionAgent",
    "MultiModalOrchestrator",
]

__version__ = "2.0.0"
