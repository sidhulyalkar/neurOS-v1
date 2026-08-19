from dataclasses import dataclass

from neuros_mechint import EvidenceTier, InputCausalExperiment, InputMetric


@dataclass(frozen=True)
class Scale:
    factor: float
    name: str = "scale_input"
    target: str = "value"

    def apply(self, reference):
        return reference * self.factor

    def metadata(self):
        return {"factor": self.factor}


def test_input_causal_audit_runs_each_edit_from_same_reference():
    result = InputCausalExperiment(
        reference=4.0,
        metric=InputMetric(lambda value: value, name="identity"),
        experiment_name="input-audit",
        model_id="scorer",
        evidence_tier=EvidenceTier.UNIT,
    ).run(
        [Scale(0.5), Scale(0.25)],
        controls=[Scale(1.0, name="identity_control")],
    )

    assert result.baseline_metric == 4.0
    assert [effect.intervened_metric for effect in result.effects] == [2.0, 1.0]
    assert result.controls[0].effect == 0.0
    assert result.specificity_gap == 3.0
