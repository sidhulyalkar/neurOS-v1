"""Optional integrations with neighboring neurOS research layers."""

from .correspondence import (
    AdapterCausalSubstitutionEvaluator,
    AdapterFeatureSpaceView,
    AdapterPairedExampleSpec,
    TensorFeatureProjector,
    build_adapter_feature_pair_examples,
    factorial_origin_from_report,
    run_adapter_feature_correspondence_study,
)
from .factorial_study import FactorialEvidenceCellInput, run_factorial_evidence_study

__all__ = [
    "AdapterCausalSubstitutionEvaluator",
    "AdapterFeatureSpaceView",
    "AdapterPairedExampleSpec",
    "FactorialEvidenceCellInput",
    "TensorFeatureProjector",
    "build_adapter_feature_pair_examples",
    "factorial_origin_from_report",
    "run_adapter_feature_correspondence_study",
    "run_factorial_evidence_study",
]
