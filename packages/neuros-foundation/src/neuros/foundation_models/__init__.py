"""neurOS foundation-model interoperability, discovery, probes, and benchmarks.

The modern API is registry-first. Historical model classes remain importable
for compatibility, but many of those wrappers predate the upstream-adapter
architecture and must not be used as evidence of upstream model performance.
"""

from __future__ import annotations

from neuros.foundation_models.base_foundation_model import BaseFoundationModel
from neuros.foundation_models.benchmark import (
    BenchmarkReport,
    EvaluationProtocol,
    benchmark_embeddings,
    sample_efficiency_curve,
)
from neuros.foundation_models.catalog import DEFAULT_MODEL_CARDS, catalog_by_id
from neuros.foundation_models.integration import FoundationEmbeddingDecoder
from neuros.foundation_models.longitudinal import (
    NestedCalibrationSplit,
    chronological_partition,
    make_nested_calibration_split,
    ordered_group_values,
)
from neuros.foundation_models.longitudinal_authority import (
    LongitudinalCaseAuthority,
    processed_data_sha256,
)
from neuros.foundation_models.longitudinal_baseline import CSPCaseResult, run_csp_case
from neuros.foundation_models.longitudinal_external import (
    ExternalTaskDecoderCaseResult,
    ExternalTaskDecoderMethodSpec,
    PairedTaskPerformanceResult,
    pair_task_performance,
    run_external_task_decoder_case,
)
from neuros.foundation_models.longitudinal_ladder import (
    LADDER_METHODS,
    SOURCEWEIGHER_METHODS,
    LadderRuntimeConfig,
    frontier_auc,
    paired_case_set_audit,
    render_ladder_report,
    run_ladder_method,
    seed_averaged_case_rows,
    summarize_ladder_rows,
)
from neuros.foundation_models.longitudinal_methods import (
    TaskDecoderCaseResult,
    TaskDecoderMethodSpec,
    run_task_decoder_case,
)
from neuros.foundation_models.longitudinal_transfer import (
    FrozenTransferCaseResult,
    FrozenTransferMethodSpec,
    PreparedFrozenEncoderCase,
    prepare_frozen_encoder_case,
    run_frozen_transfer_case,
)
from neuros.foundation_models.moabb_longitudinal import (
    MOABB_LONGITUDINAL_DATASETS,
    MOABBLongitudinalDatasetSpec,
    build_moabb_longitudinal_dataset,
    get_moabb_longitudinal_spec,
    validate_observed_sessions,
)
from neuros.foundation_models.probes import (
    domain_leakage_probe,
    effective_rank,
    invariance_score,
    linear_cka,
    linear_probe,
    mean_pairwise_cosine,
    pairwise_cka,
    representation_report,
)
from neuros.foundation_models.real_world import (
    REAL_WORLD_EVIDENCE_SOURCES,
    EvaluationPartition,
    EvidenceSource,
    GroupedEvaluationData,
    collect_moabb,
    find_evidence_sources,
    get_evidence_source,
    hold_out_groups,
)
from neuros.foundation_models.registry import (
    DEFAULT_REGISTRY,
    AdapterUnavailableError,
    CallableAdapter,
    FoundationAdapter,
    FoundationModelError,
    ModelRegistry,
    NeuroFMXAdapter,
    UnsupportedCapabilityError,
    ZunaAdapter,
    build_default_registry,
)
from neuros.foundation_models.schema import (
    AccessLevel,
    AdapterAvailability,
    FoundationModelCard,
    IntegrationLevel,
    ModelStatus,
    ModelTask,
    NeuralModality,
)

try:
    from neuros.foundation_models.poyo_model import POYOModel, POYOPlusModel
    POYO_AVAILABLE = True
except ImportError:
    POYO_AVAILABLE = False
    POYOModel = None
    POYOPlusModel = None

try:
    from neuros.foundation_models.ndt_model import NDT2Model, NDT3Model
    NDT_AVAILABLE = True
except ImportError:
    NDT_AVAILABLE = False
    NDT2Model = None
    NDT3Model = None

try:
    from neuros.foundation_models.cebra_model import CEBRAModel
    CEBRA_AVAILABLE = True
except ImportError:
    CEBRAModel = None
    CEBRA_AVAILABLE = False

try:
    from neuros.foundation_models.neuroformer_model import NeuroformerModel
    NEUROFORMER_AVAILABLE = True
except ImportError:
    NEUROFORMER_AVAILABLE = False
    NeuroformerModel = None

LEGACY_WRAPPER_NOTICE = (
    "POYOModel/NDT2Model/NDT3Model/CEBRAModel/NeuroformerModel are retained for "
    "backward compatibility. Prefer DEFAULT_REGISTRY and verified upstream adapters "
    "for scientific comparisons; legacy placeholder code paths are not benchmark evidence."
)

__all__ = [
    "AccessLevel",
    "AdapterAvailability",
    "FoundationModelCard",
    "IntegrationLevel",
    "ModelStatus",
    "ModelTask",
    "NeuralModality",
    "DEFAULT_MODEL_CARDS",
    "catalog_by_id",
    "FoundationAdapter",
    "CallableAdapter",
    "ZunaAdapter",
    "NeuroFMXAdapter",
    "ModelRegistry",
    "DEFAULT_REGISTRY",
    "build_default_registry",
    "FoundationModelError",
    "AdapterUnavailableError",
    "UnsupportedCapabilityError",
    "effective_rank",
    "mean_pairwise_cosine",
    "linear_cka",
    "invariance_score",
    "representation_report",
    "linear_probe",
    "domain_leakage_probe",
    "pairwise_cka",
    "EvaluationProtocol",
    "BenchmarkReport",
    "benchmark_embeddings",
    "sample_efficiency_curve",
    "EvidenceSource",
    "REAL_WORLD_EVIDENCE_SOURCES",
    "get_evidence_source",
    "find_evidence_sources",
    "GroupedEvaluationData",
    "EvaluationPartition",
    "collect_moabb",
    "hold_out_groups",
    "ordered_group_values",
    "chronological_partition",
    "NestedCalibrationSplit",
    "make_nested_calibration_split",
    "LongitudinalCaseAuthority",
    "processed_data_sha256",
    "CSPCaseResult",
    "run_csp_case",
    "TaskDecoderMethodSpec",
    "TaskDecoderCaseResult",
    "run_task_decoder_case",
    "ExternalTaskDecoderMethodSpec",
    "ExternalTaskDecoderCaseResult",
    "PairedTaskPerformanceResult",
    "run_external_task_decoder_case",
    "pair_task_performance",
    "FrozenTransferMethodSpec",
    "FrozenTransferCaseResult",
    "PreparedFrozenEncoderCase",
    "prepare_frozen_encoder_case",
    "run_frozen_transfer_case",
    "LADDER_METHODS",
    "SOURCEWEIGHER_METHODS",
    "LadderRuntimeConfig",
    "run_ladder_method",
    "seed_averaged_case_rows",
    "frontier_auc",
    "paired_case_set_audit",
    "summarize_ladder_rows",
    "render_ladder_report",
    "MOABBLongitudinalDatasetSpec",
    "MOABB_LONGITUDINAL_DATASETS",
    "get_moabb_longitudinal_spec",
    "build_moabb_longitudinal_dataset",
    "validate_observed_sessions",
    "FoundationEmbeddingDecoder",
    "BaseFoundationModel",
    "POYOModel",
    "POYOPlusModel",
    "POYO_AVAILABLE",
    "NDT2Model",
    "NDT3Model",
    "NDT_AVAILABLE",
    "CEBRAModel",
    "CEBRA_AVAILABLE",
    "NeuroformerModel",
    "NEUROFORMER_AVAILABLE",
    "LEGACY_WRAPPER_NOTICE",
]
