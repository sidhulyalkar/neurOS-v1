from pathlib import Path

# Applicator revision 2: rerun after fixing the negative-seed adversary harness.
SOURCE = Path("packages/neuros-mechint/src/neuros_mechint/representations/temporal_sweep.py")
TEST = Path("packages/neuros-mechint/tests/test_mechint_temporal_sweep.py")


def replace_exact(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one marker, found {count}")
    return text.replace(old, new, 1)


source = SOURCE.read_text()
source = replace_exact(
    source,
    '''def _metric_mapping(\n    values: Mapping[str, float | None],\n) -> Mapping[str, float | None]:\n    output: dict[str, float | None] = {}\n    for key, value in dict(values).items():\n''',
    '''def _metric_mapping(\n    values: Mapping[str, float | None],\n) -> Mapping[str, float | None]:\n    if not isinstance(values, Mapping):\n        raise TypeError("scientific metrics must be a mapping")\n    output: dict[str, float | None] = {}\n    for key, value in values.items():\n''',
    label="metric mapping type authority",
)
source = replace_exact(
    source,
    '''        metrics = _metric_mapping(self.metrics or {})\n''',
    '''        metrics = _metric_mapping({} if self.metrics is None else self.metrics)\n''',
    label="record metrics None authority",
)
source = replace_exact(
    source,
    '''        metric_n: dict[str, int] = {}\n        for key, value in dict(self.metric_n).items():\n''',
    '''        if not isinstance(self.metric_n, Mapping):\n            raise TypeError("metric_n must be a mapping")\n        metric_n: dict[str, int] = {}\n        for key, value in self.metric_n.items():\n''',
    label="metric_n mapping authority",
)
SOURCE.write_text(source)

test = TEST.read_text()
test = replace_exact(
    test,
    '''from neuros_mechint.representations.temporal_sweep import (\n    ControlledTemporalAblationResult,\n    run_controlled_temporal_ablation,\n)\n''',
    '''from neuros_mechint.representations.temporal_sweep import (\n    ControlledTemporalAblationResult,\n    TemporalAblationRecord,\n    TemporalAblationSummary,\n    run_controlled_temporal_ablation,\n)\n''',
    label="temporal sweep test imports",
)
test += '''\n\ndef test_temporal_record_rejects_dict_coercible_non_mapping_metrics():\n    with pytest.raises(TypeError, match="mapping"):\n        TemporalAblationRecord(\n            corruption=TemporalCorruption.IID_GAUSSIAN,\n            corruption_scale=0.0,\n            seed=0,\n            method_id="fake",\n            sequence_id="eval",\n            fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,\n            evaluation_scope=EvaluationScope.SEQUENCE_LOCAL,\n            status=CaseStatus.OK,\n            metrics=[("score", 1.0)],\n        )\n    with pytest.raises(TypeError, match="mapping"):\n        TemporalAblationRecord(\n            corruption=TemporalCorruption.IID_GAUSSIAN,\n            corruption_scale=0.0,\n            seed=0,\n            method_id="fake",\n            sequence_id="eval",\n            fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,\n            evaluation_scope=EvaluationScope.SEQUENCE_LOCAL,\n            status=CaseStatus.OK,\n            metrics=[],\n        )\n\n\ndef test_temporal_summary_rejects_dict_coercible_non_mapping_containers():\n    common = dict(\n        method_id="fake",\n        corruption=TemporalCorruption.IID_GAUSSIAN,\n        corruption_scale=0.0,\n        fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,\n        evaluation_scope=EvaluationScope.SEQUENCE_LOCAL,\n        total_cases=1,\n        ok_cases=1,\n        failed_cases=0,\n        unavailable_cases=0,\n        nonconverged_cases=0,\n        metric_std={"score": None},\n        metric_sem={"score": None},\n        metric_n={"score": 1},\n    )\n    with pytest.raises(TypeError, match="mapping"):\n        TemporalAblationSummary(\n            **common,\n            metric_mean=[("score", 1.0)],\n        )\n    with pytest.raises(TypeError, match="metric_n must be a mapping"):\n        TemporalAblationSummary(\n            **{**common, "metric_n": [("score", 1)]},\n            metric_mean={"score": 1.0},\n        )\n'''
TEST.write_text(test)
