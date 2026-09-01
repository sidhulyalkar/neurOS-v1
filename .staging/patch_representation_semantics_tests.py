from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one replacement, found {count}")
    target.write_text(text.replace(old, new, 1))


example = "packages/neuros-mechint/examples/12_representation_noise_sweep.py"
replace_once(
    example,
    '''                "fit_regime": summary.fit_regime.value,
                "noise_std": summary.noise_std,
''',
    '''                "fit_regime": summary.fit_regime.value,
                "evaluation_scope": summary.evaluation_scope.value,
                "noise_std": summary.noise_std,
''',
)
replace_once(
    example,
    '''                "nonconverged_cases": summary.nonconverged_cases,
                "failure_rate": summary.failure_rate,
                "metric_mean": dict(summary.metric_mean),
''',
    '''                "nonconverged_cases": summary.nonconverged_cases,
                "non_ok_rate": summary.non_ok_rate,
                "failed_rate": summary.failed_rate,
                "unavailable_rate": summary.unavailable_rate,
                "nonconverged_rate": summary.nonconverged_rate,
                "metric_mean": dict(summary.metric_mean),
''',
)
replace_once(
    example,
    '''                "metric_sem": dict(summary.metric_sem),
''',
    '''                "metric_sem": dict(summary.metric_sem),
                "metric_n": dict(summary.metric_n),
''',
)
replace_once(
    example,
    '''                "fit_regime": record.fit_regime.value,
                "status": record.status.value,
''',
    '''                "fit_regime": record.fit_regime.value,
                "evaluation_scope": record.evaluation_scope.value,
                "status": record.status.value,
''',
)

case_tests = "packages/neuros-mechint/tests/test_mechint_representation_cases.py"
replace_once(
    case_tests,
    '''    FitRegime,
    PCARepresentation,
''',
    '''    EvaluationScope,
    FitRegime,
    PCARepresentation,
''',
)

target = ROOT / case_tests
text = target.read_text()
for marker in (
    '        fit_regime = FitRegime.TRANSDUCTIVE_TARGET_OBSERVED\n',
    '        fit_regime = FitRegime.EXTERNAL_PRETRAINED\n',
):
    if marker not in text:
        raise RuntimeError(f"{case_tests}: missing marker {marker!r}")
    text = text.replace(
        marker,
        marker + '        evaluation_scope = EvaluationScope.SEQUENCE_LOCAL\n',
    )
text = text.replace(
    '        fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,\n        status=CaseStatus.OK,\n',
    '        fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,\n'
    '        evaluation_scope=EvaluationScope.BATCH_TRANSFORM,\n'
    '        status=CaseStatus.OK,\n',
)
text = text.replace(
    '            fit_regime=FitRegime.TRANSDUCTIVE_TARGET_OBSERVED,\n'
    '            status=CaseStatus.FAILED,\n',
    '            fit_regime=FitRegime.TRANSDUCTIVE_TARGET_OBSERVED,\n'
    '            evaluation_scope=EvaluationScope.SEQUENCE_LOCAL,\n'
    '            status=CaseStatus.FAILED,\n',
)
target.write_text(text)

replace_once(
    case_tests,
    '    assert summary.failure_rate == 0.0\n',
    '    assert summary.non_ok_rate == 0.0\n    assert summary.failed_rate == 0.0\n',
)
replace_once(
    case_tests,
    '    assert summary.failure_rate == pytest.approx(1 / 3)\n',
    '    assert summary.non_ok_rate == pytest.approx(1 / 3)\n'
    '    assert summary.failed_rate == pytest.approx(1 / 3)\n'
    '    assert summary.unavailable_rate == 0.0\n',
)
replace_once(
    case_tests,
    '    assert summary.failure_rate == pytest.approx(1 / 3)\n',
    '    assert summary.non_ok_rate == pytest.approx(1 / 3)\n'
    '    assert summary.failed_rate == 0.0\n'
    '    assert summary.unavailable_rate == pytest.approx(1 / 3)\n',
)

target = ROOT / case_tests
target.write_text(
    target.read_text()
    + r'''


def test_metric_values_reject_bool_and_text_coercion() -> None:
    for value in (True, "0.5"):
        with pytest.raises(TypeError, match="finite real"):
            RepresentationCaseOutcome(
                method_id="pca",
                sequence_id="a",
                fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,
                evaluation_scope=EvaluationScope.BATCH_TRANSFORM,
                status=CaseStatus.OK,
                embedding=np.ones((4, 2)),
                metrics={"score": value},
            )


def test_case_result_validates_train_sequence_identity() -> None:
    case = RepresentationCaseOutcome(
        method_id="pca",
        sequence_id="a",
        fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,
        evaluation_scope=EvaluationScope.BATCH_TRANSFORM,
        status=CaseStatus.OK,
        embedding=np.ones((4, 2)),
        metrics={"score": 1.0},
    )
    with pytest.raises(ValueError, match="train sequence IDs must be unique"):
        CasePreservingRepresentationResult(
            train_sequence_ids=("train", "train"),
            evaluation_sequence_ids=("a",),
            method_ids=("pca",),
            cases=(case,),
        )


def test_external_pretrained_scope_can_batch_transform_all_sequences_once() -> None:
    train, evaluation, _ = _data()

    class BatchExternal:
        method_id = "batch_external"
        fit_regime = FitRegime.EXTERNAL_PRETRAINED
        evaluation_scope = EvaluationScope.BATCH_TRANSFORM

        def __init__(self) -> None:
            self.calls = 0

        def embed(self, train, evaluation):
            self.calls += 1
            from neuros_mechint.representations import RepresentationEmbedding

            return RepresentationEmbedding(
                method_id=self.method_id,
                sequences=tuple(sequence[:, :2] for sequence in evaluation.sequences),
                sequence_ids=evaluation.sequence_ids,
                fit_regime=self.fit_regime,
            )

    method = BatchExternal()
    result = CasePreservingRepresentationBenchmark([method]).run(train, evaluation)
    assert method.calls == 1
    assert len(result.cases) == 3
    assert all(
        case.evaluation_scope is EvaluationScope.BATCH_TRANSFORM
        for case in result.cases
    )


def test_method_without_explicit_evaluation_scope_is_rejected() -> None:
    class AmbiguousMethod:
        method_id = "ambiguous"
        fit_regime = FitRegime.EXTERNAL_PRETRAINED

        def embed(self, train, evaluation):  # pragma: no cover
            raise AssertionError

    with pytest.raises(ValueError, match="evaluation_scope"):
        CasePreservingRepresentationBenchmark([AmbiguousMethod()])


def test_metric_schema_drift_is_not_silently_averaged() -> None:
    cases = (
        RepresentationCaseOutcome(
            method_id="x",
            sequence_id="a",
            fit_regime=FitRegime.EXTERNAL_PRETRAINED,
            evaluation_scope=EvaluationScope.SEQUENCE_LOCAL,
            status=CaseStatus.OK,
            embedding=np.ones((4, 2)),
            metrics={"a": 1.0},
        ),
        RepresentationCaseOutcome(
            method_id="x",
            sequence_id="b",
            fit_regime=FitRegime.EXTERNAL_PRETRAINED,
            evaluation_scope=EvaluationScope.SEQUENCE_LOCAL,
            status=CaseStatus.OK,
            embedding=np.ones((4, 2)),
            metrics={"b": 1.0},
        ),
    )
    result = CasePreservingRepresentationResult(
        train_sequence_ids=("train",),
        evaluation_sequence_ids=("a", "b"),
        method_ids=("x",),
        cases=cases,
    )
    with pytest.raises(ValueError, match="identical metric schema"):
        result.summary_for_method("x")


def test_metric_bug_aborts_benchmark_instead_of_becoming_method_failure(monkeypatch) -> None:
    train, evaluation, _ = _data()
    benchmark = CasePreservingRepresentationBenchmark(
        [PCARepresentation(2)],
        neighborhood_k=3,
    )

    def broken_metrics(*args, **kwargs):
        raise RuntimeError("metric implementation bug")

    monkeypatch.setattr(benchmark, "_metrics", broken_metrics)
    with pytest.raises(RuntimeError, match="metric implementation bug"):
        benchmark.run(train, evaluation)


def test_method_summary_reports_per_metric_denominator() -> None:
    cases = (
        RepresentationCaseOutcome(
            method_id="x",
            sequence_id="a",
            fit_regime=FitRegime.EXTERNAL_PRETRAINED,
            evaluation_scope=EvaluationScope.SEQUENCE_LOCAL,
            status=CaseStatus.OK,
            embedding=np.ones((4, 2)),
            metrics={"score": 1.0, "optional": None},
        ),
        RepresentationCaseOutcome(
            method_id="x",
            sequence_id="b",
            fit_regime=FitRegime.EXTERNAL_PRETRAINED,
            evaluation_scope=EvaluationScope.SEQUENCE_LOCAL,
            status=CaseStatus.OK,
            embedding=np.ones((4, 2)),
            metrics={"score": 3.0, "optional": 2.0},
        ),
    )
    result = CasePreservingRepresentationResult(
        train_sequence_ids=("train",),
        evaluation_sequence_ids=("a", "b"),
        method_ids=("x",),
        cases=cases,
    )
    summary = result.summary_for_method("x")
    assert summary.metrics["score"] == pytest.approx(2.0)
    assert summary.metric_n == {"optional": 1, "score": 2}
'''
)

sweep_tests = "packages/neuros-mechint/tests/test_mechint_representation_sweep.py"
replace_once(
    sweep_tests,
    '''    FitRegime,
    PCARepresentation,
''',
    '''    EvaluationScope,
    FitRegime,
    PCARepresentation,
''',
)

target = ROOT / sweep_tests
text = target.read_text()
text = text.replace(
    '        fit_regime = FitRegime.TRANSDUCTIVE_TARGET_OBSERVED\n',
    '        fit_regime = FitRegime.TRANSDUCTIVE_TARGET_OBSERVED\n'
    '        evaluation_scope = EvaluationScope.SEQUENCE_LOCAL\n',
)
text = text.replace(
    '        fit_regime = FitRegime.EXTERNAL_PRETRAINED\n',
    '        fit_regime = FitRegime.EXTERNAL_PRETRAINED\n'
    '        evaluation_scope = EvaluationScope.SEQUENCE_LOCAL\n',
)
text = text.replace(
    '        fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,\n'
    '        status=CaseStatus.OK,\n',
    '        fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,\n'
    '        evaluation_scope=EvaluationScope.BATCH_TRANSFORM,\n'
    '        status=CaseStatus.OK,\n',
)
target.write_text(text)

replace_once(
    sweep_tests,
    '    assert summary.failure_rate == 0.0\n',
    '    assert summary.non_ok_rate == 0.0\n'
    '    assert summary.failed_rate == 0.0\n'
    '    assert summary.metric_n["reference_pairwise_distance_rank"] == 4\n',
)
replace_once(
    sweep_tests,
    '        assert summary.failure_rate == 1.0\n',
    '        assert summary.non_ok_rate == 1.0\n'
    '        assert summary.failed_rate == 0.0\n'
    '        assert summary.unavailable_rate == 1.0\n',
)

target = ROOT / sweep_tests
target.write_text(
    target.read_text()
    + r'''


def test_sweep_metric_values_reject_bool_and_text_coercion() -> None:
    for value in (True, "0.5"):
        with pytest.raises(TypeError, match="finite real"):
            SweepCaseRecord(
                noise_std=0.0,
                seed=1,
                method_id="pca",
                sequence_id="eval",
                fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,
                evaluation_scope=EvaluationScope.BATCH_TRANSFORM,
                status=CaseStatus.OK,
                metrics={"score": value},
            )


def test_method_factory_evaluation_scope_cannot_drift_across_points() -> None:
    calls = 0

    class ScopedIdentity:
        method_id = "scoped"
        fit_regime = FitRegime.EXTERNAL_PRETRAINED

        def __init__(self, scope):
            self.evaluation_scope = scope

        def embed(self, train, evaluation):
            source = evaluation.sequences[0]
            return RepresentationEmbedding(
                method_id=self.method_id,
                sequences=(source[:, :2],),
                sequence_ids=evaluation.sequence_ids,
                fit_regime=self.fit_regime,
            )

    def factory():
        nonlocal calls
        calls += 1
        scope = (
            EvaluationScope.SEQUENCE_LOCAL
            if calls == 1
            else EvaluationScope.BATCH_TRANSFORM
        )
        return (ScopedIdentity(scope),)

    with pytest.raises(ValueError, match="evaluation scopes"):
        run_controlled_noise_sweep(
            factory,
            noise_levels=(0.0, 0.1),
            seeds=(1,),
        )


def test_sweep_summary_rejects_metric_schema_drift() -> None:
    records = (
        SweepCaseRecord(
            noise_std=0.0,
            seed=1,
            method_id="x",
            sequence_id="eval",
            fit_regime=FitRegime.EXTERNAL_PRETRAINED,
            evaluation_scope=EvaluationScope.SEQUENCE_LOCAL,
            status=CaseStatus.OK,
            metrics={"a": 1.0},
        ),
        SweepCaseRecord(
            noise_std=0.0,
            seed=2,
            method_id="x",
            sequence_id="eval",
            fit_regime=FitRegime.EXTERNAL_PRETRAINED,
            evaluation_scope=EvaluationScope.SEQUENCE_LOCAL,
            status=CaseStatus.OK,
            metrics={"b": 1.0},
        ),
    )
    result = ControlledNoiseSweepResult(
        noise_levels=(0.0,),
        seeds=(1, 2),
        method_ids=("x",),
        evaluation_sequence_ids=("eval",),
        records=records,
    )
    with pytest.raises(ValueError, match="identical metric schema"):
        result.summary("x", 0.0)
'''
)

print("example/test semantics patch applied")
