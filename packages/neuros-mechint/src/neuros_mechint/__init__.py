"""neuros-mechint: causal experiments for understanding neural computation.

The stable surface is intentionally small. Broad exploratory analyses from the
0.1 research package remain available through their owning modules and lazy
compatibility imports.
"""

from __future__ import annotations

from importlib import import_module

from neuros_mechint.adapters import (
    CircuitTracerAdapter,
    ModelAdapter,
    ModelCall,
    NNsightAdapter,
    PyTorchAdapter,
    SAELensFeatureAdapter,
    TransformerLensAdapter,
    integration_status,
)
from neuros_mechint.core import (
    CURRENT_MANIFEST_SCHEMA,
    AblationIntervention,
    ComponentRef,
    CounterfactualPair,
    EvidenceTier,
    ExperimentManifest,
    ExperimentResult,
    InputCausalExperiment,
    InputExperimentResult,
    InputIntervention,
    InputInterventionEffect,
    InputMetric,
    InterventionEffect,
    MechanisticExperiment,
    MethodCard,
    MethodMaturity,
    OutputMetric,
    PatchIntervention,
    get_method_card,
    list_method_cards,
    logit_difference,
    migrate_artifact_envelope,
    migrate_manifest_payload,
    schema_catalog,
    stable_hash,
    stable_hash_or_none,
)
from neuros_mechint.release import (
    EvidenceClosureRequirement,
    EvidenceRequirementState,
    V1EvidenceStatus,
    default_v1_evidence_status,
)

__version__ = "1.0.0"

_LEGACY_EXPORTS = {
    "NeuronActivationAnalyzer": ("neuros_mechint.neuron_analysis", "NeuronActivationAnalyzer"),
    "CircuitDiscovery": ("neuros_mechint.circuit_discovery", "CircuitDiscovery"),
    "SparseAutoencoder": ("neuros_mechint.sparse_autoencoder", "SparseAutoencoder"),
    "HierarchicalSAE": ("neuros_mechint.concept_sae", "HierarchicalSAE"),
    "ConceptDictionary": ("neuros_mechint.concept_sae", "ConceptDictionary"),
    "ConceptLabel": ("neuros_mechint.concept_sae", "ConceptLabel"),
    "CausalSAEProbe": ("neuros_mechint.concept_sae", "CausalSAEProbe"),
    "ActivationCache": ("neuros_mechint.sae_training", "ActivationCache"),
    "MultiLayerSAETrainer": ("neuros_mechint.sae_training", "MultiLayerSAETrainer"),
    "SAETrainingPipeline": ("neuros_mechint.sae_training", "SAETrainingPipeline"),
    "SAEVisualizer": ("neuros_mechint.sae_visualization", "SAEVisualizer"),
    "MultiLayerSAEVisualizer": ("neuros_mechint.sae_visualization", "MultiLayerSAEVisualizer"),
    "FeatureAttributionAnalyzer": ("neuros_mechint.feature_analysis", "FeatureAttributionAnalyzer"),
    "TemporalDynamicsAnalyzer": ("neuros_mechint.feature_analysis", "TemporalDynamicsAnalyzer"),
    "CausalImportanceAnalyzer": ("neuros_mechint.feature_analysis", "CausalImportanceAnalyzer"),
    "FeatureClusteringAnalyzer": ("neuros_mechint.feature_analysis", "FeatureClusteringAnalyzer"),
    "FeatureSteeringAnalyzer": ("neuros_mechint.feature_analysis", "FeatureSteeringAnalyzer"),
    "AutomatedCircuitDiscovery": ("neuros_mechint.circuits", "AutomatedCircuitDiscovery"),
    "ModuleCircuitDiscovery": ("neuros_mechint.circuits", "ModuleCircuitDiscovery"),
    "ModuleActivationPatcher": ("neuros_mechint.circuits", "ModuleActivationPatcher"),
    "ActivationPatcher": ("neuros_mechint.circuits", "ActivationPatcher"),
    "PathPatcher": ("neuros_mechint.circuits", "PathPatcher"),
    "CircuitComparator": ("neuros_mechint.circuits", "CircuitComparator"),
    "MotifDetector": ("neuros_mechint.circuits", "MotifDetector"),
    "MechIntResult": ("neuros_mechint.results", "MechIntResult"),
    "CircuitResult": ("neuros_mechint.results", "CircuitResult"),
    "MechIntDatabase": ("neuros_mechint.database", "MechIntDatabase"),
    "MechIntPipeline": ("neuros_mechint.pipeline", "MechIntPipeline"),
    "PipelineConfig": ("neuros_mechint.pipeline", "PipelineConfig"),
    "IntegratedGradients": ("neuros_mechint.attribution", "IntegratedGradients"),
    "DeepLIFT": ("neuros_mechint.attribution", "DeepLIFT"),
    "GradientSHAP": ("neuros_mechint.attribution", "GradientSHAP"),
    "GenerativePathAttribution": ("neuros_mechint.attribution", "GenerativePathAttribution"),
    "visualize_attributions": ("neuros_mechint.attribution", "visualize_attributions"),
    "CCA": ("neuros_mechint.alignment.cca", "CCA"),
    "RSA": ("neuros_mechint.alignment.rsa", "RSA"),
    "PLS": ("neuros_mechint.alignment.pls", "PLS"),
    "DynamicsAnalyzer": ("neuros_mechint.dynamics.analyzer", "DynamicsAnalyzer"),
    "KoopmanOperator": ("neuros_mechint.dynamics.koopman", "KoopmanOperator"),
    "LyapunovAnalyzer": ("neuros_mechint.dynamics.lyapunov", "LyapunovAnalyzer"),
    "FixedPointFinder": ("neuros_mechint.dynamics.fixed_points", "FixedPointFinder"),
    "ManifoldAnalyzer": ("neuros_mechint.dynamics.manifold", "ManifoldAnalyzer"),
    "LatentSurgery": ("neuros_mechint.counterfactuals", "LatentSurgery"),
    "DoCalculusInterventions": ("neuros_mechint.counterfactuals", "DoCalculusInterventions"),
    "SyntheticLesions": ("neuros_mechint.counterfactuals", "SyntheticLesions"),
    "TrainingPhase": ("neuros_mechint.meta_dynamics", "TrainingPhase"),
    "RepresentationalTrajectory": ("neuros_mechint.meta_dynamics", "RepresentationalTrajectory"),
    "TrainingPhaseDetection": ("neuros_mechint.meta_dynamics", "TrainingPhaseDetection"),
    "GradientAttribution": ("neuros_mechint.meta_dynamics", "GradientAttribution"),
    "ManifoldGeometry": ("neuros_mechint.geometry_topology", "ManifoldGeometry"),
    "TopologicalAnalysis": ("neuros_mechint.geometry_topology", "TopologicalAnalysis"),
    "ManifoldVisualization": ("neuros_mechint.geometry_topology", "ManifoldVisualization"),
    "UnifiedMechIntReporter": ("neuros_mechint.reporting", "UnifiedMechIntReporter"),
    "MechIntReport": ("neuros_mechint.reporting", "MechIntReport"),
    "ReportSection": ("neuros_mechint.reporting", "ReportSection"),
    "ReportMetric": ("neuros_mechint.reporting", "ReportMetric"),
    "ReportTemplate": ("neuros_mechint.reporting", "ReportTemplate"),
}


def __getattr__(name: str):
    if name not in _LEGACY_EXPORTS:
        raise AttributeError(f"module 'neuros_mechint' has no attribute {name!r}")
    module_name, attribute = _LEGACY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = [
    "__version__",
    "CURRENT_MANIFEST_SCHEMA",
    "AblationIntervention",
    "CircuitTracerAdapter",
    "ComponentRef",
    "CounterfactualPair",
    "EvidenceClosureRequirement",
    "EvidenceRequirementState",
    "EvidenceTier",
    "ExperimentManifest",
    "ExperimentResult",
    "InputCausalExperiment",
    "InputExperimentResult",
    "InputIntervention",
    "InputInterventionEffect",
    "InputMetric",
    "InterventionEffect",
    "MechanisticExperiment",
    "MethodCard",
    "MethodMaturity",
    "ModelAdapter",
    "ModelCall",
    "NNsightAdapter",
    "OutputMetric",
    "PatchIntervention",
    "PyTorchAdapter",
    "SAELensFeatureAdapter",
    "TransformerLensAdapter",
    "V1EvidenceStatus",
    "default_v1_evidence_status",
    "get_method_card",
    "integration_status",
    "list_method_cards",
    "logit_difference",
    "migrate_artifact_envelope",
    "migrate_manifest_payload",
    "schema_catalog",
    "stable_hash",
    "stable_hash_or_none",
]
