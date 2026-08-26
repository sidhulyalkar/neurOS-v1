"""Scientific evidence cards for Arena neural world models.

The purpose of this module is claim governance. Different synthetic generators
can satisfy the same software protocol while supporting very different
scientific statements. Evidence cards keep those differences machine-readable
in every report and benchmark artifact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class WorldModelEvidenceCard:
    """Declared scientific scope of a neural world model.

    ``evidence_level`` is an Arena engineering level, not a ranking of biological
    truth. Higher levels require additional external evidence but never convert a
    simulator into a substitute for human closed-loop validation.
    """

    model_name: str
    evidence_level: str
    model_family: str
    stimulus_causal: bool
    spatial_model: str
    recorded_human_background: bool
    known_intervention_ground_truth: bool
    artifact_ground_truth: bool
    uncertainty_representation: str
    validated_against: tuple[str, ...]
    intended_uses: tuple[str, ...]
    unsupported_claims: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_BUILTIN_CARDS: dict[str, WorldModelEvidenceCard] = {
    "legacy_synthetic": WorldModelEvidenceCard(
        model_name="legacy_synthetic",
        evidence_level="W0-regression-fixture",
        model_family="analytic frequency fixture",
        stimulus_causal=False,
        spatial_model="hand-authored channel weights",
        recorded_human_background=False,
        known_intervention_ground_truth=True,
        artifact_ground_truth=True,
        uncertainty_representation="seeded stochastic nuisance only",
        validated_against=("internal deterministic contracts",),
        intended_uses=(
            "driver smoke tests",
            "decoder regression tests",
            "known-frequency fixtures",
        ),
        unsupported_claims=(
            "display-to-cortex causality",
            "human response distribution",
            "anatomical source realism",
        ),
    ),
    "driven_state_space": WorldModelEvidenceCard(
        model_name="driven_state_space",
        evidence_level="W1-causal-phenomenological",
        model_family="stochastic driven state-space",
        stimulus_causal=True,
        spatial_model="hand-authored posterior/central/frontal mixing",
        recorded_human_background=False,
        known_intervention_ground_truth=True,
        artifact_ground_truth=True,
        uncertainty_representation="seeded stochastic latent/background dynamics",
        validated_against=(
            "internal deterministic contracts",
            "display-causality metamorphic tests",
        ),
        intended_uses=(
            "closed-loop systems qualification",
            "display/device/network stress testing",
            "population and counterexample search",
        ),
        unsupported_claims=(
            "biophysical cortical mechanism",
            "human prevalence estimates",
            "subject-specific physiological prediction",
        ),
        notes=(
            "Phenomenological oscillator/background dynamics are chosen for causal controllability rather than cortical-mechanism fidelity.",
        ),
    ),
    "semi_synthetic_replay": WorldModelEvidenceCard(
        model_name="semi_synthetic_replay",
        evidence_level="W2-recorded-background-semi-synthetic",
        model_family="recorded EEG background plus controlled intervention",
        stimulus_causal=True,
        spatial_model="recorded sensor covariance plus declared injected topography",
        recorded_human_background=True,
        known_intervention_ground_truth=True,
        artifact_ground_truth=False,
        uncertainty_representation="recorded nuisance plus seeded intervention",
        validated_against=(
            "portable recorded-background contract",
            "internal intervention-causality contracts",
        ),
        intended_uses=(
            "real-background decoder stress tests",
            "public-dataset anchored development",
            "held-out subject/session studies",
        ),
        unsupported_claims=(
            "the injected response occurred in the recorded participant",
            "full physiological interaction between intervention and background",
            "human closed-loop performance",
        ),
    ),
    "leadfield_driven": WorldModelEvidenceCard(
        model_name="leadfield_driven",
        evidence_level="W3-source-projected-causal",
        model_family="display-driven source response projected by frozen lead field",
        stimulus_causal=True,
        spatial_model="explicit forward/lead-field projection",
        recorded_human_background=False,
        known_intervention_ground_truth=True,
        artifact_ground_truth=True,
        uncertainty_representation="seeded source/background nuisance with fixed spatial projection",
        validated_against=(
            "portable lead-field contract",
            "explicit source-selection provenance",
            "internal display-causality contracts",
        ),
        intended_uses=(
            "montage/topography sensitivity studies",
            "source-to-sensor systems qualification",
            "forward-model based population studies",
        ),
        unsupported_claims=(
            "subject-specific source localization unless the bundle is subject-specific and independently validated",
            "human response amplitude distribution",
            "human closed-loop performance",
        ),
    ),
}


def evidence_card_for_model(model: Any, model_name: str) -> WorldModelEvidenceCard:
    """Resolve a model's evidence card, failing scientifically closed for plugins.

    Third-party plugins may expose ``evidence_card()`` returning either a
    ``WorldModelEvidenceCard`` or a compatible dictionary. Plugins without a
    card remain runnable, but reports mark them as scientifically unqualified
    rather than inheriting the claims of a built-in model.
    """

    provider = getattr(model, "evidence_card", None)
    if callable(provider):
        raw = provider()
        if isinstance(raw, WorldModelEvidenceCard):
            return raw
        if isinstance(raw, dict):
            return WorldModelEvidenceCard(**raw)
        raise TypeError("world-model evidence_card() must return a WorldModelEvidenceCard or dict")
    if model_name in _BUILTIN_CARDS:
        return _BUILTIN_CARDS[model_name]
    return WorldModelEvidenceCard(
        model_name=model_name,
        evidence_level="W?-external-unqualified",
        model_family="external plugin",
        stimulus_causal=False,
        spatial_model="undeclared",
        recorded_human_background=False,
        known_intervention_ground_truth=False,
        artifact_ground_truth=False,
        uncertainty_representation="undeclared",
        validated_against=(),
        intended_uses=("software experimentation",),
        unsupported_claims=(
            "physiological validity",
            "human performance prediction",
            "causal neural interpretation",
        ),
        notes=("External plugin did not provide a machine-readable Arena evidence card.",),
    )
