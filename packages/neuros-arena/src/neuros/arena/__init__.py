"""Deterministic closed-loop synthetic BCI systems arena."""

from .application import (
    ApplicationEvent,
    ApplicationTrace,
    evaluate_application_trace,
    load_application_trace,
    save_application_trace,
)
from .baselines import export_mne_raw_baseline, save_eeg_baseline
from .benchmark import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkPack,
    BenchmarkPackResult,
    MetricRule,
    load_benchmark_pack,
    run_benchmark_pack,
    save_benchmark_pack,
)
from .conformance import (
    AdversarialSearchResult,
    MetamorphicResult,
    check_display_drop_monotonicity,
    check_fail_closed_degradation,
    check_transport_drop_monotonicity,
    search_counterexamples,
)
from .device_presets import unicorn_hybrid_black_eeg_profile
from .evaluation import ArenaDecision, evaluate_decisions
from .evidence import WorldModelEvidenceCard, evidence_card_for_model
from .leadfield import LeadFieldDrivenWorldModel, export_mne_forward_bundle, save_leadfield_bundle
from .manifest import ArenaManifest, load_manifest, save_manifest
from .participant import (
    PARTICIPANT_RESPONSE_MODEL,
    ParticipantStateTrace,
    compile_participant_state_trace,
)
from .population import ParameterDistribution, PopulationRun, PopulationSpec, run_population
from .presets import get_preset, list_presets
from .reality import RealityAnchorResult, anchor_worlds_by_covariance, anchor_worlds_by_embeddings
from .recording import (
    ElectrodeCoordinate,
    RecordingMetadata,
    load_recording_metadata,
    recording_sidecar_path,
    save_recording_metadata,
)
from .reference import compare_feature_signatures, feature_signature
from .runner import ArenaRun, run_scenario
from .semi_synthetic import SemiSyntheticReplayWorldModel
from .specs import (
    ArenaScenario,
    ArtifactEvent,
    DeviceProfile,
    DisplayProfile,
    ParticipantProfile,
    StageSpec,
    TransportProfile,
    WorldModelProfile,
)
from .studies import CohortAnchorFold, CohortAnchorStudy, run_leave_one_domain_out_covariance_study
from .validation import HeldOutRealityValidation, split_contiguous_recording, validate_covariance_anchor_held_out
from .world_input import WorldInputBlock
from .world_models import DrivenStateSpaceWorldModel, LegacySyntheticWorldModel, NeuralWorldModel, WorldModelEmission

__all__ = [
    "AdversarialSearchResult",
    "ApplicationEvent",
    "ApplicationTrace",
    "ArenaDecision",
    "ArenaManifest",
    "ArenaRun",
    "ArenaScenario",
    "ArtifactEvent",
    "BenchmarkCase",
    "BenchmarkCaseResult",
    "BenchmarkPack",
    "BenchmarkPackResult",
    "CohortAnchorFold",
    "CohortAnchorStudy",
    "DeviceProfile",
    "DisplayProfile",
    "DrivenStateSpaceWorldModel",
    "ElectrodeCoordinate",
    "HeldOutRealityValidation",
    "LeadFieldDrivenWorldModel",
    "LegacySyntheticWorldModel",
    "MetamorphicResult",
    "MetricRule",
    "NeuralWorldModel",
    "PARTICIPANT_RESPONSE_MODEL",
    "ParameterDistribution",
    "ParticipantProfile",
    "ParticipantStateTrace",
    "PopulationRun",
    "PopulationSpec",
    "RealityAnchorResult",
    "RecordingMetadata",
    "SemiSyntheticReplayWorldModel",
    "StageSpec",
    "TransportProfile",
    "WorldInputBlock",
    "WorldModelEmission",
    "WorldModelEvidenceCard",
    "WorldModelProfile",
    "anchor_worlds_by_covariance",
    "anchor_worlds_by_embeddings",
    "check_display_drop_monotonicity",
    "check_fail_closed_degradation",
    "check_transport_drop_monotonicity",
    "compare_feature_signatures",
    "compile_participant_state_trace",
    "evaluate_application_trace",
    "evaluate_decisions",
    "evidence_card_for_model",
    "export_mne_forward_bundle",
    "export_mne_raw_baseline",
    "feature_signature",
    "get_preset",
    "list_presets",
    "load_application_trace",
    "load_benchmark_pack",
    "load_manifest",
    "load_recording_metadata",
    "recording_sidecar_path",
    "run_benchmark_pack",
    "run_leave_one_domain_out_covariance_study",
    "run_population",
    "run_scenario",
    "save_application_trace",
    "save_benchmark_pack",
    "save_eeg_baseline",
    "save_leadfield_bundle",
    "save_manifest",
    "save_recording_metadata",
    "search_counterexamples",
    "split_contiguous_recording",
    "unicorn_hybrid_black_eeg_profile",
    "validate_covariance_anchor_held_out",
]
