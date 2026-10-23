"""Zero-download Circuit Auditor walkthrough: positive AND negative evidence packs.

Run from the neurOS workspace after installing neuros-mechint:
  python packages/neuros-mechint/examples/23_circuit_auditor_quickstart.py --out mechint-demo

Scientific boundary: both cases are synthetic known-answer tests. These do not
demonstrate real neural/LLM mechanisms or any biological homology.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from neuros_mechint.adapters import PyTorchAdapter
from neuros_mechint.benchmarks import (
    CircuitCandidate,
    EvidenceExample,
    EvidencePackPolicy,
    EvidencePackSpec,
    EvidenceSplit,
    run_adapter_evidence_pack,
    write_evidence_pack_artifact,
)
from neuros_mechint.core import EvidenceTier, OutputMetric


class KnownRouteModel(nn.Module):
    """A planted causal route with two functionally irrelevant branches."""

    def __init__(self) -> None:
        super().__init__()
        self.route = nn.Linear(2, 1, bias=False)
        self.nuisance_a = nn.Linear(2, 1, bias=False)
        self.nuisance_b = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.route.weight.copy_(torch.tensor([[1.0, 0.0]]))
            self.nuisance_a.weight.zero_()
            self.nuisance_b.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.route(x) + self.nuisance_a(x) + self.nuisance_b(x)


def main(output_dir: Path) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = KnownRouteModel().eval()
    adapter = PyTorchAdapter(model)
    examples = tuple(
        EvidenceExample(
            f"{split.value}-{value}",
            torch.tensor([[float(value), 0.0]]),
            split,
        )
        for split, values in (
            (EvidenceSplit.DISCOVERY, (1, 2)),
            (EvidenceSplit.VALIDATION, (3, 4)),
        )
        for value in values
    )
    metric = OutputMetric(lambda output: output.mean(), name="mean_output")
    policy = EvidencePackPolicy(
        require_multiple_intervention_baselines=False,
        bootstrap_samples=100,
    )
    outcomes: dict[str, object] = {}
    for label, target in (("known-route", "route"), ("nuisance-route", "nuisance_a")):
        spec = EvidencePackSpec(
            pack_id=f"circuit-auditor-quickstart-{label}",
            model_id="KnownRouteModel",
            model_revision="synthetic-v1",
            dataset_id="synthetic:known-route",
            dataset_revision="synthetic-v1",
            metric_name=metric.name,
            target_universe=("route", "nuisance_a", "nuisance_b"),
            discovery_method="planted-known-answer",
            intervention_baselines=("zero",),
            random_trials=100,
            evidence_tier=EvidenceTier.SCIENTIFIC_SYNTHETIC,
        )
        result = run_adapter_evidence_pack(
            spec=spec,
            adapter=adapter,
            metric=metric,
            examples=examples,
            candidate=CircuitCandidate(
                name=label, targets=(target,), source="planted-synthetic-fixture"
            ),
            pack_policy=policy,
        )
        artifact = write_evidence_pack_artifact(result, output_dir / f"{label}.json")
        outcomes[label] = {
            "passed": result.promotion.passed,
            "reasons": list(result.promotion.reasons),
            "validation_pass_rate": result.validation_aggregate.pass_rate,
            "artifact": str(artifact),
            "study_fingerprint": result.study_fingerprint,
        }
    if not outcomes["known-route"]["passed"] or outcomes["nuisance-route"]["passed"]:
        raise AssertionError("Circuit Auditor toy positive/negative falsification failed")
    return outcomes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("mechint-demo"))
    args = parser.parse_args()
    print(json.dumps(main(args.out), indent=2, sort_keys=True))
