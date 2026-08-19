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
    CEBRA_AVAILABLE = False
    CEBRAModel = None

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
