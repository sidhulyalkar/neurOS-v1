from __future__ import annotations

import numpy as np
import pytest

from neuros.foundation_models.qualification import (
    ExternalDecoderMethodSpec,
    ExternalLearnedState,
    ExternalProbabilityDecoder,
    ExternalQualificationDecoder,
    QualificationRunContract,
    validate_prediction_output,
    validate_run_capabilities,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def _method(*, probability_semantics: str) -> ExternalDecoderMethodSpec:
    return ExternalDecoderMethodSpec(
        method_id=f"method-{probability_semantics}",
        implementation="external.fixture.Decoder",
        implementation_version="1.0.0",
        input_axes=("sample", "feature"),
        probability_semantics=probability_semantics,  # type: ignore[arg-type]
    )


def _run(method: ExternalDecoderMethodSpec) -> QualificationRunContract:
    return QualificationRunContract(
        protocol_sha256=SHA_A,
        method_spec_sha256=method.sha256,
        case_authority_sha256=SHA_B,
    )


class LabelOnlyDecoder:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        del X, y

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=np.int64)

    def learned_state(self) -> ExternalLearnedState:
        return ExternalLearnedState()


class ProbabilityDecoder(LabelOnlyDecoder):
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full((len(X), 2), 0.5, dtype=np.float32)


def test_label_only_method_can_participate_when_probability_is_unavailable():
    method = _method(probability_semantics="unavailable")
    decoder = LabelOnlyDecoder()
    assert isinstance(decoder, ExternalQualificationDecoder)
    assert not isinstance(decoder, ExternalProbabilityDecoder)
    validate_run_capabilities(method, _run(method), decoder)
    prediction = validate_prediction_output(
        decoder.predict(np.zeros((3, 2), dtype=np.float32)),
        expected_samples=3,
        allowed_labels=(0, 1),
    )
    assert prediction.tolist() == [0, 0, 0]


def test_declared_probability_requires_probability_surface():
    method = _method(probability_semantics="uncalibrated_probability")
    with pytest.raises(TypeError, match="lacks predict_proba"):
        validate_run_capabilities(method, _run(method), LabelOnlyDecoder())

    decoder = ProbabilityDecoder()
    assert isinstance(decoder, ExternalProbabilityDecoder)
    validate_run_capabilities(method, _run(method), decoder)


def test_generic_probability_semantics_are_not_softmax_specific():
    logistic = _method(probability_semantics="uncalibrated_probability")
    softmax = _method(probability_semantics="uncalibrated_softmax")
    assert logistic.sha256 != softmax.sha256


def test_prediction_output_rejects_shape_object_nan_and_unknown_classes():
    with pytest.raises(ValueError, match="exact shape"):
        validate_prediction_output(
            np.asarray([[0], [1]]), expected_samples=2, allowed_labels=(0, 1)
        )
    with pytest.raises(ValueError, match="object dtype"):
        validate_prediction_output(
            np.asarray([object(), object()], dtype=object),
            expected_samples=2,
            allowed_labels=(0, 1),
        )
    with pytest.raises(ValueError, match="must be finite"):
        validate_prediction_output(
            np.asarray([0.0, np.nan]), expected_samples=2, allowed_labels=(0, 1)
        )
    with pytest.raises(ValueError, match="outside the declared task classes"):
        validate_prediction_output(
            np.asarray([0, 2]), expected_samples=2, allowed_labels=(0, 1)
        )
