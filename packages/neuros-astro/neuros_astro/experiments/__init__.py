"""
Experiment tracking and ablation framework for neuros-astro.

Provides tools for organizing ablation experiments, tracking results,
and comparing model performance with/without astrocyte modality.
"""

from neuros_astro.experiments.tracker import (
    ExperimentTracker,
    ExperimentConfig,
    ExperimentResult,
)

from neuros_astro.experiments.ablation import (
    AblationStudy,
    AblationCondition,
    compare_ablation_results,
)

__all__ = [
    "ExperimentTracker",
    "ExperimentConfig",
    "ExperimentResult",
    "AblationStudy",
    "AblationCondition",
    "compare_ablation_results",
]
