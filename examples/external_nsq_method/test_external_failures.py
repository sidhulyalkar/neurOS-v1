"""Issue #197: black-box failure challenge against the public NSQ referee.

Uses only public ``neuros.foundation_models`` imports plus the example's
``method.py`` / ``demo.py``, so it also runs from a standalone copy outside
the neurOS checkout. Synthetic software-contract evidence only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from neuros.foundation_models.qualification import QualificationProtocolSpec
from neuros.foundation_models.qualification_runner import (
    DEFAULT_CLASSIFICATION_SCORECARD,
    QualificationExecutionContext,
    run_external_qualification_case,
)

if __package__:  # inside the neurOS repo
    from .demo import _sha, build_authority, synthetic_data
    from .method import ExternalDecoder, ExternalFactory
else:  # standalone copy next to demo.py / method.py
    from demo import _sha, build_authority, synthetic_data  # type: ignore[no-redef]
    from method import ExternalDecoder, ExternalFactory  # type: ignore[no-redef]

LINEAGE = _sha("synthetic-external-nsq-method-v1:dataset-lineage")


# ---------------------------------------------------------------- faulty decoders
class NaNProbaDecoder(ExternalDecoder):
    def predict_proba(self, X):
        proba = super().predict_proba(X)
        proba[0, 0] = np.nan
        return proba


class WrongShapeProbaDecoder(ExternalDecoder):
    def predict_proba(self, X):
        return super().predict_proba(X)[:, :1]  # drops a class column


class ReversedClassOrderDecoder(ExternalDecoder):
    def probability_class_labels(self):
        return tuple(reversed(super().probability_class_labels()))


class FitExplodesDecoder(ExternalDecoder):
    def fit(self, X, y):
        raise RuntimeError("intentional external fit failure (#197)")


@dataclass(frozen=True)
class FaultyFactory(ExternalFactory):
    decoder_cls: type = ExternalDecoder
    created: list = field(default_factory=list, compare=False)

    def create(self):
        self.created.append(1)
        return self.decoder_cls()


# ---------------------------------------------------------------- harness
def _run(factory, *, observed_lineage: str = LINEAGE):
    data = synthetic_data()
    authority = build_authority(data)
    split = authority.restore(data)
    protocol = QualificationProtocolSpec(
        protocol_id="synthetic-external-method-referee-v1",
        dataset_id=data.dataset_id,
        dataset_lineage_sha256=LINEAGE,
        task_id="synthetic-left-vs-right",
        independent_unit="participant",
        grouping_hierarchy=("participant", "session", "trial"),
        calibration_budgets_per_class=tuple(range(split.max_budget_per_class + 1)),
        metric_scorecard_sha256=DEFAULT_CLASSIFICATION_SCORECARD.sha256,
        protocol_status="frozen",
        metadata={"synthetic_fixture": True, "numerical_result_interpretable": False},
    )
    snapshot = (data.X.copy(), np.asarray(data.y).copy())
    result = run_external_qualification_case(
        data,
        authority,
        protocol,
        factory,
        execution_context=QualificationExecutionContext(
            observed_dataset_lineage_sha256=observed_lineage,
            preprocessing_authority_sha256s=(
                _sha("synthetic-external-nsq-method-v1:preprocessing"),
            ),
            calibration_authority_sha256s=(
                _sha("synthetic-external-nsq-method-v1:calibration-policy"),
            ),
        ),
    )
    return data, snapshot, protocol, result


@pytest.fixture(scope="module")
def baseline():
    result = _run(FaultyFactory())[3]
    assert all(row.status == "success" for row in result.rows)
    return result


@pytest.mark.parametrize(
    ("decoder_cls", "failure_type", "reason_fragment"),
    [
        (NaNProbaDecoder, "ValueError", "finite"),
        (WrongShapeProbaDecoder, "ValueError", "shape"),
        (ReversedClassOrderDecoder, "ValueError", "class order"),
        (FitExplodesDecoder, "RuntimeError", "intentional external fit failure"),
    ],
)
def test_faulty_decoder_rows_are_failure_preserving(
    baseline, decoder_cls, failure_type, reason_fragment
):
    factory = FaultyFactory(decoder_cls=decoder_cls)
    data, (X0, y0), protocol, result = _run(factory)

    # Complete frontier is preserved: one row per frozen budget, none scored.
    assert [r.calibration_per_class for r in result.rows] == list(
        protocol.calibration_budgets_per_class
    )
    assert len(factory.created) == len(protocol.calibration_budgets_per_class)
    for row, ref in zip(result.rows, baseline.rows):
        assert row.status == "failed"
        assert row.score is None
        assert row.failure_type == failure_type
        assert reason_fragment in row.failure_reason.lower()
        # Untouched final-assessment partition: same identity as the valid run.
        assert row.evaluation_indices_sha256 == ref.evaluation_indices_sha256
        assert row.fit_indices_sha256 == ref.fit_indices_sha256
        assert row.run_contract_sha256 == ref.run_contract_sha256
    if decoder_cls is FitExplodesDecoder:
        assert all(r.external_learned_state_sha256 is None for r in result.rows)
    np.testing.assert_array_equal(data.X, X0)
    np.testing.assert_array_equal(np.asarray(data.y), y0)


def test_lineage_mismatch_aborts_before_decoder_creation():
    factory = FaultyFactory()
    with pytest.raises(ValueError, match="lineage"):
        _run(factory, observed_lineage=_sha("tampered-lineage"))
    assert factory.created == []
