from __future__ import annotations

import numpy as np
import pytest

from neuros.foundation_models.qualification import (
    ExternalDecoderMethodSpec,
    ExternalLearnedState,
    ExternalQualificationDecoder,
    ExternalUnlabeledTargetAdapter,
    QualificationRunContract,
    validate_run_capabilities,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def _method(*, mode: str = "none") -> ExternalDecoderMethodSpec:
    return ExternalDecoderMethodSpec(
        method_id=f"fixture-{mode}",
        implementation="external.fixture.Decoder",
        implementation_version="1.0.0",
        input_axes=("sample", "channel", "time"),
        probability_semantics="uncalibrated_softmax",
        target_adaptation_mode=mode,  # type: ignore[arg-type]
    )


def _run(method: ExternalDecoderMethodSpec, **overrides) -> QualificationRunContract:
    values = {
        "protocol_sha256": SHA_A,
        "method_spec_sha256": method.sha256,
        "case_authority_sha256": SHA_B,
    }
    values.update(overrides)
    return QualificationRunContract(**values)


class PlainDecoder:
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        del X, y

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full((len(X), 2), 0.5, dtype=np.float32)

    def learned_state(self) -> ExternalLearnedState:
        return ExternalLearnedState()


class UnlabeledDecoder(PlainDecoder):
    def __init__(self) -> None:
        self.observed = 0

    def adapt_unlabeled(self, X: np.ndarray) -> None:
        self.observed += len(X)


def test_zero_target_run_does_not_require_unlabeled_adapter():
    method = _method(mode="none")
    decoder = PlainDecoder()
    validate_run_capabilities(method, _run(method), decoder)
    assert isinstance(decoder, ExternalQualificationDecoder)
    assert not isinstance(decoder, ExternalUnlabeledTargetAdapter)


def test_unlabeled_target_budget_requires_method_declaration():
    method = _method(mode="none")
    with pytest.raises(ValueError, match="does not declare unlabeled adaptation"):
        validate_run_capabilities(
            method,
            _run(method, unlabeled_target_examples=4),
            PlainDecoder(),
        )


def test_unlabeled_target_declaration_requires_separate_adapter_surface():
    method = _method(mode="unlabeled")
    with pytest.raises(TypeError, match="lacks adapt_unlabeled"):
        validate_run_capabilities(
            method,
            _run(method, unlabeled_target_seconds=1.5),
            PlainDecoder(),
        )


def test_unlabeled_target_run_is_authorized_only_when_both_ledger_and_surface_agree():
    method = _method(mode="unlabeled")
    decoder = UnlabeledDecoder()
    run = _run(method, unlabeled_target_examples=3)
    validate_run_capabilities(method, run, decoder)
    assert isinstance(decoder, ExternalUnlabeledTargetAdapter)
    decoder.adapt_unlabeled(np.zeros((3, 2, 16), dtype=np.float32))
    assert decoder.observed == 3
    assert run.zero_shot is False
    assert run.consumes_unlabeled_target is True


def test_run_for_different_method_cannot_borrow_an_unlabeled_adapter():
    authorized = _method(mode="unlabeled")
    different = ExternalDecoderMethodSpec(
        method_id="different",
        implementation="external.fixture.Decoder",
        implementation_version="1.0.0",
        input_axes=("sample", "channel", "time"),
        probability_semantics="uncalibrated_softmax",
        target_adaptation_mode="unlabeled",
    )
    with pytest.raises(ValueError, match="does not authorize"):
        validate_run_capabilities(
            different,
            _run(authorized, unlabeled_target_examples=2),
            UnlabeledDecoder(),
        )


def test_invalid_adaptation_mode_is_rejected_as_method_identity():
    with pytest.raises(ValueError, match="target_adaptation_mode"):
        _method(mode="sometimes")
