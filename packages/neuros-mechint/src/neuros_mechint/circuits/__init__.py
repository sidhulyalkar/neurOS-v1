"""Circuit localization and comparison methods."""

from .acdc import AutomatedCircuitDiscovery, Circuit, Edge, ModuleCircuitDiscovery
from .path_patching import (
    ActivationPatcher,
    ActivationPatchingResult,
    ModuleActivationPatcher,
    PatchEffect,
    PathEffect,
    PathPatcher,
    PathPatchingResult,
)

# Existing optional circuit-analysis classes remain available from their
# implementation modules. Import them lazily so the stable causal core does
# not pull visualization and scientific extras at package import time.
_LEGACY = {
    "CircuitComparator": (".circuit_comparator", "CircuitComparator"),
    "MotifDetector": (".motif_detection", "MotifDetector"),
    "LatentCircuitModel": (".latent_rnn", "LatentCircuitModel"),
    "CircuitFitter": (".latent_rnn", "CircuitFitter"),
    "RecurrentDynamicsAnalyzer": (".latent_rnn", "RecurrentDynamicsAnalyzer"),
    "DUNLModel": (".dunl", "DUNLModel"),
    "MixedSelectivityAnalyzer": (".dunl", "MixedSelectivityAnalyzer"),
    "FactorDecomposition": (".dunl", "FactorDecomposition"),
    "FeatureVisualizer": (".feature_viz", "FeatureVisualizer"),
    "OptimalStimulus": (".feature_viz", "OptimalStimulus"),
    "ActivationMaximization": (".feature_viz", "ActivationMaximization"),
}


def __getattr__(name):
    if name not in _LEGACY:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attr = _LEGACY[name]
    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value
    return value


# Wildcard imports remain focused on the causal circuit contract. Historical
# classes continue to work through explicit named imports.
__all__ = [
    "ActivationPatcher",
    "ActivationPatchingResult",
    "AutomatedCircuitDiscovery",
    "Circuit",
    "Edge",
    "ModuleActivationPatcher",
    "ModuleCircuitDiscovery",
    "PatchEffect",
    "PathEffect",
    "PathPatcher",
    "PathPatchingResult",
]
