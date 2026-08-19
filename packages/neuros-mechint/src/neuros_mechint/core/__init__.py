"""Stable experiment primitives for :mod:`neuros_mechint`."""

from .components import ComponentRef
from .evidence import EvidenceTier
from .experiment import CounterfactualPair, MechanisticExperiment
from .input_experiment import InputCausalExperiment, InputIntervention, InputMetric
from .interventions import AblationIntervention, Intervention, PatchIntervention
from .manifest import ExperimentManifest, stable_hash, stable_hash_or_none
from .maturity import (
    MethodCard,
    MethodMaturity,
    get_method_card,
    list_method_cards,
    register_method_card,
)
from .metrics import OutputMetric, ScalarMetric, logit_difference
from .results import (
    ExperimentResult,
    InputExperimentResult,
    InputInterventionEffect,
    InterventionEffect,
)
from .schema import (
    ARTIFACT_ENVELOPE_SCHEMA,
    CURRENT_MANIFEST_SCHEMA,
    ArtifactSchemaSpec,
    get_artifact_schema,
    make_artifact_envelope,
    migrate_artifact_envelope,
    migrate_manifest_payload,
    schema_catalog,
    validate_artifact_envelope,
    validate_manifest_payload,
)
from .v04_maturity import register_v04_method_cards
from .v05_maturity import register_v05_method_cards
from .v06_maturity import register_v06_method_cards
from .v07_maturity import register_v07_method_cards
from .v08_maturity import register_v08_method_cards
from .v09_maturity import register_v09_method_cards
from .v10_maturity import register_v10_method_cards

register_v04_method_cards()
register_v05_method_cards()
register_v06_method_cards()
register_v07_method_cards()
register_v08_method_cards()
register_v09_method_cards()
register_v10_method_cards()

__all__ = [
    "ARTIFACT_ENVELOPE_SCHEMA",
    "CURRENT_MANIFEST_SCHEMA",
    "AblationIntervention",
    "ArtifactSchemaSpec",
    "ComponentRef",
    "CounterfactualPair",
    "EvidenceTier",
    "ExperimentManifest",
    "ExperimentResult",
    "InputCausalExperiment",
    "InputExperimentResult",
    "InputIntervention",
    "InputInterventionEffect",
    "InputMetric",
    "Intervention",
    "InterventionEffect",
    "MechanisticExperiment",
    "MethodCard",
    "MethodMaturity",
    "OutputMetric",
    "PatchIntervention",
    "ScalarMetric",
    "get_artifact_schema",
    "get_method_card",
    "list_method_cards",
    "logit_difference",
    "make_artifact_envelope",
    "migrate_artifact_envelope",
    "migrate_manifest_payload",
    "register_method_card",
    "schema_catalog",
    "stable_hash",
    "stable_hash_or_none",
    "validate_artifact_envelope",
    "validate_manifest_payload",
]
