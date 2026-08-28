from __future__ import annotations

import numpy as np
import pytest

from neuros.foundation_models.qualification import (
    ExternalDecoderMethodSpec,
    ExternalLearnedState,
    ExternalQualificationDecoder,
    ExternalQualificationFactory,
    QualificationProtocolSpec,
    QualificationRunContract,
    bind_learned_state,
    validate_probability_output,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _protocol(**overrides):
    values = {
        "protocol_id": "nsq-kumar2024-mi-v1",
        "dataset_id": "MOABB:Kumar2024",
        "dataset_lineage_sha256": SHA_D,
        "task_id": "left-vs-right-motor-imagery",
        "independent_unit": "participant",
        "grouping_hierarchy": ("participant", "session", "trial"),
        "calibration_budgets_per_class": (0, 1, 2, 5, 10),
        "metadata": {"cohorts": ["GR", "PAR"]},
    }
    values.update(overrides)
    return QualificationProtocolSpec(**values)


def _method(**overrides):
    values = {
        "method_id": "external-eegnet",
        "implementation": "braindecode.models.EEGNet",
        "implementation_version": "1.7.0",
        "input_axes": ("sample", "channel", "time"),
        "probability_semantics": "uncalibrated_softmax",
        "source_reference": "doi:10.1000/example",
        "metadata": {"optimizer": "adamw"},
    }
    values.update(overrides)
    return ExternalDecoderMethodSpec(**values)


def _run(method=None, **overrides):
    method = _method() if method is None else method
    values = {
        "protocol_sha256": _protocol().sha256,
        "method_spec_sha256": method.sha256,
        "case_authority_sha256": SHA_A,
    }
    values.update(overrides)
    return QualificationRunContract(**values)


def test_protocol_identity_is_full_deterministic_and_metadata_order_independent():
    first = _protocol(metadata={"cohorts": ["GR", "PAR"], "site": "public"})
    second = _protocol(metadata={"site": "public", "cohorts": ["GR", "PAR"]})
    assert len(first.sha256) == 64
    assert first.sha256 == second.sha256
    assert first.display_fingerprint == first.sha256[:16]
    with pytest.raises(TypeError):
        first.metadata["site"] = "mutated"  # type: ignore[index]


def test_protocol_binds_dataset_lineage_metric_authority_and_lifecycle():
    first = _protocol(dataset_lineage_sha256=SHA_D)
    second = _protocol(dataset_lineage_sha256=SHA_C)
    assert first.sha256 != second.sha256
    assert first.protocol_status == "draft"
    assert first.metric_scorecard_sha256 is None

    with pytest.raises(ValueError, match="frozen qualification protocol requires"):
        _protocol(protocol_status="frozen")
    frozen = _protocol(protocol_status="frozen", metric_scorecard_sha256=SHA_C)
    assert frozen.sha256 != first.sha256
    assert frozen.metric_scorecard_sha256 == SHA_C

    with pytest.raises(ValueError, match="64-character"):
        _protocol(dataset_lineage_sha256="short")
    with pytest.raises(ValueError, match="64-character"):
        _protocol(metric_scorecard_sha256="short")
    with pytest.raises(ValueError, match="draft, frozen, or retired"):
        _protocol(protocol_status="published")


def test_metric_scorecard_identity_changes_protocol_even_when_display_names_do_not():
    first = _protocol(metric_scorecard_sha256=SHA_A)
    second = _protocol(metric_scorecard_sha256=SHA_B)
    assert first.primary_metric == second.primary_metric == "balanced_accuracy"
    assert first.secondary_metrics == second.secondary_metrics
    assert first.sha256 != second.sha256


def test_protocol_requires_participant_first_hierarchy_and_zero_budget():
    with pytest.raises(ValueError, match="start with the declared independent_unit"):
        _protocol(grouping_hierarchy=("session", "participant", "trial"))
    with pytest.raises(ValueError, match="start at zero"):
        _protocol(calibration_budgets_per_class=(1, 2, 5))
    with pytest.raises(ValueError, match="strictly increasing"):
        _protocol(calibration_budgets_per_class=(0, 2, 1, 5))
    with pytest.raises(ValueError, match="strictly increasing"):
        _protocol(calibration_budgets_per_class=(0, 1, 1, 2))
    with pytest.raises(ValueError, match="integer without coercion"):
        _protocol(calibration_budgets_per_class=(0, 1.0, 2))


def test_final_assessment_role_cannot_be_redefined_by_a_submission():
    with pytest.raises(ValueError, match="untouched_final_assessment"):
        _protocol(final_assessment_role="test")


def test_method_identity_does_not_contain_learned_checkpoint_state():
    method = _method()
    assert len(method.sha256) == 64
    assert "state_sha256" not in method.to_dict()
    assert "calibration_state_sha256" not in method.to_dict()


def test_method_model_lineage_is_explicitly_known_or_unknown():
    unknown = _method(model_lineage_sha256=None)
    known = _method(model_lineage_sha256=SHA_C)
    assert unknown.lineage_known is False
    assert known.lineage_known is True
    assert known.model_lineage_sha256 == SHA_C
    assert unknown.sha256 != known.sha256
    with pytest.raises(ValueError, match="64-character"):
        _method(model_lineage_sha256="unknown")


def test_learned_state_separates_scientific_comparison_from_content_addressability():
    opaque = ExternalLearnedState()
    assert opaque.state_addressable is False

    verified = ExternalLearnedState(
        state_identity_kind="tensor_sha256",
        state_sha256=SHA_B,
    )
    assert verified.state_addressable is True

    with pytest.raises(ValueError, match="verified state identity requires"):
        ExternalLearnedState(state_identity_kind="checkpoint_sha256")
    with pytest.raises(ValueError, match="opaque_unverified state cannot claim"):
        ExternalLearnedState(state_sha256=SHA_B)


def test_calibration_state_is_bound_to_each_fitted_model_state_not_method_spec():
    method = _method(probability_semantics="calibrated_probability")
    run = _run(method)
    with pytest.raises(ValueError, match="requires calibration_state_sha256"):
        bind_learned_state(
            method,
            run,
            ExternalLearnedState(
                state_identity_kind="tensor_sha256",
                state_sha256=SHA_B,
            ),
        )

    bound = bind_learned_state(
        method,
        run,
        ExternalLearnedState(
            state_identity_kind="tensor_sha256",
            state_sha256=SHA_B,
            calibration_state_sha256=SHA_C,
        ),
    )
    assert bound.state_addressable is True
    assert len(bound.sha256) == 64
    assert bound.learned_state.calibration_state_sha256 == SHA_C


def test_uncalibrated_method_cannot_smuggle_a_calibration_state():
    method = _method()
    run = _run(method)
    with pytest.raises(ValueError, match="may only accompany"):
        bind_learned_state(
            method,
            run,
            ExternalLearnedState(
                state_identity_kind="tensor_sha256",
                state_sha256=SHA_B,
                calibration_state_sha256=SHA_C,
            ),
        )


def test_bound_state_rejects_a_run_authorized_for_a_different_method():
    method = _method(method_id="method-a")
    other = _method(method_id="method-b")
    run = _run(other)
    with pytest.raises(ValueError, match="does not authorize"):
        bind_learned_state(
            method,
            run,
            ExternalLearnedState(
                state_identity_kind="tensor_sha256",
                state_sha256=SHA_B,
            ),
        )


def test_zero_shot_means_zero_labeled_and_zero_unlabeled_target_information():
    zero = _run()
    assert zero.zero_shot is True
    assert _run(labeled_target_examples=1).zero_shot is False
    assert _run(unlabeled_target_examples=1).zero_shot is False
    assert _run(unlabeled_target_seconds=0.25).zero_shot is False


def test_run_contract_rejects_short_hashes_and_lossy_budget_coercion():
    method = _method()
    with pytest.raises(ValueError, match="64-character"):
        QualificationRunContract(
            protocol_sha256=_protocol().sha256[:16],
            method_spec_sha256=method.sha256,
            case_authority_sha256=SHA_A,
        )
    with pytest.raises(ValueError, match="integer without coercion"):
        _run(labeled_target_examples=1.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        _run(unlabeled_target_seconds=float("nan"))


def test_probability_outputs_are_validated_not_repaired():
    method = _method()
    good = np.asarray([[0.7, 0.3], [0.2, 0.8]], dtype=np.float32)
    returned = validate_probability_output(
        method,
        good,
        expected_samples=2,
        expected_classes=2,
    )
    assert returned is good

    with pytest.raises(ValueError, match="exact shape"):
        validate_probability_output(
            method,
            good[:, :1],
            expected_samples=2,
            expected_classes=2,
        )
    with pytest.raises(ValueError, match="floating dtype"):
        validate_probability_output(
            method,
            np.asarray([[1, 0], [0, 1]], dtype=np.int64),
            expected_samples=2,
            expected_classes=2,
        )
    with pytest.raises(ValueError, match="finite"):
        validate_probability_output(
            method,
            np.asarray([[np.nan, np.nan], [0.2, 0.8]], dtype=np.float32),
            expected_samples=2,
            expected_classes=2,
        )
    with pytest.raises(ValueError, match="within \[0, 1\]"):
        validate_probability_output(
            method,
            np.asarray([[1.1, -0.1], [0.2, 0.8]], dtype=np.float32),
            expected_samples=2,
            expected_classes=2,
        )
    with pytest.raises(ValueError, match="will not renormalize"):
        validate_probability_output(
            method,
            np.asarray([[0.6, 0.6], [0.2, 0.8]], dtype=np.float32),
            expected_samples=2,
            expected_classes=2,
        )


def test_probability_unavailable_is_not_silently_treated_as_scores():
    method = _method(probability_semantics="unavailable")
    with pytest.raises(ValueError, match="unavailable"):
        validate_probability_output(
            method,
            np.asarray([[0.5, 0.5]], dtype=np.float32),
            expected_samples=1,
            expected_classes=2,
        )


def test_external_factory_is_structural_not_an_import_mechanism_and_can_create_fresh_models():
    class ToyDecoder:
        def fit(self, X: np.ndarray, y: np.ndarray) -> None:
            del X, y

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.full((len(X), 2), 0.5, dtype=np.float32)

        def learned_state(self) -> ExternalLearnedState:
            return ExternalLearnedState(
                state_identity_kind="tensor_sha256",
                state_sha256=SHA_B,
            )

    class ToyFactory:
        @property
        def method_spec(self) -> ExternalDecoderMethodSpec:
            return _method(method_id="toy")

        def create(self) -> ExternalQualificationDecoder:
            return ToyDecoder()

    factory = ToyFactory()
    assert isinstance(factory, ExternalQualificationFactory)
    first = factory.create()
    second = factory.create()
    assert isinstance(first, ExternalQualificationDecoder)
    assert isinstance(second, ExternalQualificationDecoder)
    assert first is not second
    assert factory.method_spec.method_id == "toy"
