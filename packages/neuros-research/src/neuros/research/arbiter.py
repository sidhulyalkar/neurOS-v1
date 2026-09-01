"""Deterministic vector-gated promotion for research evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ._canonical import canonical_sha256, require_nonempty
from .contracts import ExperimentPacket
from .evidence import ExperimentEvidence

Comparator = Literal["ge", "gt", "le", "lt"]
DecisionState = Literal["promoted", "rejected", "non_evaluable"]


@dataclass(frozen=True, slots=True)
class MetricGate:
    """One independent metric requirement; gates are never combined into a winner scalar."""

    name: str
    domain: str
    comparator: Comparator
    threshold: float | None = None
    min_delta_from_baseline: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_nonempty(self.name, name="gate.name"))
        object.__setattr__(self, "domain", require_nonempty(self.domain, name="gate.domain"))
        if self.threshold is None and self.min_delta_from_baseline is None:
            raise ValueError("a MetricGate requires threshold and/or min_delta_from_baseline")

    @property
    def key(self) -> tuple[str, str]:
        return self.domain, self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "comparator": self.comparator,
            "threshold": self.threshold,
            "min_delta_from_baseline": self.min_delta_from_baseline,
        }


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    """Predeclared evidence vector required for promotion."""

    policy_id: str
    metric_gates: tuple[MetricGate, ...]
    required_checks: tuple[str, ...] = ()
    require_all_checks_pass: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", require_nonempty(self.policy_id, name="policy_id"))
        if not self.metric_gates and not self.required_checks:
            raise ValueError("promotion policy must require at least one metric or check")
        gate_keys = [gate.key for gate in self.metric_gates]
        if len(set(gate_keys)) != len(gate_keys):
            raise ValueError("promotion policy cannot define duplicate metric gates")
        checks = tuple(require_nonempty(check, name="required_check") for check in self.required_checks)
        if len(set(checks)) != len(checks):
            raise ValueError("required_checks must be unique")
        object.__setattr__(self, "required_checks", checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "metric_gates": [gate.to_dict() for gate in self.metric_gates],
            "required_checks": list(self.required_checks),
            "require_all_checks_pass": self.require_all_checks_pass,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    experiment_id: str
    packet_fingerprint: str
    evidence_fingerprint: str
    policy_fingerprint: str
    state: DecisionState
    reasons: tuple[str, ...]

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "experiment_id": self.experiment_id,
            "packet_fingerprint": self.packet_fingerprint,
            "evidence_fingerprint": self.evidence_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
            "state": self.state,
            "reasons": list(self.reasons),
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict(include_fingerprint=False))


def _compare(value: float, comparator: Comparator, threshold: float) -> bool:
    if comparator == "ge":
        return value >= threshold
    if comparator == "gt":
        return value > threshold
    if comparator == "le":
        return value <= threshold
    if comparator == "lt":
        return value < threshold
    raise AssertionError(f"unsupported comparator {comparator!r}")


def _authority_errors(packet: ExperimentPacket, evidence: ExperimentEvidence) -> list[str]:
    errors: list[str] = []
    if evidence.experiment_id != packet.experiment_id:
        errors.append("evidence experiment_id does not match packet")
    if evidence.packet_fingerprint != packet.fingerprint:
        errors.append("evidence packet fingerprint does not match packet")
    if evidence.evaluation_fingerprint != packet.evaluation.fingerprint:
        errors.append("evidence evaluation fingerprint does not match packet authority")
    if evidence.status != "completed":
        errors.append(f"evidence status is {evidence.status!r}, not completed")

    allowed_metrics = set(packet.evaluation.metric_names)
    allowed_domains = set(packet.evaluation.evaluation_domains)
    for metric in evidence.metrics:
        if metric.name not in allowed_metrics:
            errors.append(f"evidence metric {metric.name!r} is not declared by evaluation authority")
        if metric.domain not in allowed_domains:
            errors.append(f"evidence domain {metric.domain!r} is not declared by evaluation authority")
    return errors


class EvidenceArbiter:
    """Pure evaluator over frozen packet, evidence, policy, and optional baseline."""

    def evaluate(
        self,
        packet: ExperimentPacket,
        evidence: ExperimentEvidence,
        policy: PromotionPolicy,
        *,
        baseline_packet: ExperimentPacket | None = None,
        baseline: ExperimentEvidence | None = None,
    ) -> PromotionDecision:
        reasons = _authority_errors(packet, evidence)

        allowed_metrics = set(packet.evaluation.metric_names)
        allowed_domains = set(packet.evaluation.evaluation_domains)
        for gate in policy.metric_gates:
            if gate.name not in allowed_metrics:
                reasons.append(f"policy metric {gate.name!r} is outside evaluation authority")
            if gate.domain not in allowed_domains:
                reasons.append(f"policy domain {gate.domain!r} is outside evaluation authority")

        if (baseline_packet is None) != (baseline is None):
            reasons.append("baseline_packet and baseline evidence must be supplied together")
        elif baseline_packet is not None and baseline is not None:
            reasons.extend(f"baseline: {reason}" for reason in _authority_errors(baseline_packet, baseline))
            if baseline_packet.dataset.fingerprint != packet.dataset.fingerprint:
                reasons.append("baseline uses a different dataset authority")
            if baseline_packet.evaluation.fingerprint != packet.evaluation.fingerprint:
                reasons.append("baseline uses a different evaluation authority")

        if reasons:
            return PromotionDecision(
                experiment_id=packet.experiment_id,
                packet_fingerprint=packet.fingerprint,
                evidence_fingerprint=evidence.fingerprint,
                policy_fingerprint=policy.fingerprint,
                state="non_evaluable",
                reasons=tuple(reasons),
            )

        metric_map = evidence.metric_map()
        baseline_map = baseline.metric_map() if baseline is not None else {}
        failures: list[str] = []

        for gate in policy.metric_gates:
            if gate.key not in metric_map:
                failures.append(f"missing required metric {gate.domain}/{gate.name}")
                continue
            value = metric_map[gate.key]
            if gate.threshold is not None and not _compare(value, gate.comparator, gate.threshold):
                failures.append(
                    f"metric {gate.domain}/{gate.name}={value:g} failed "
                    f"{gate.comparator} {gate.threshold:g}"
                )
            if gate.min_delta_from_baseline is not None:
                if baseline is None:
                    failures.append(
                        f"metric {gate.domain}/{gate.name} requires baseline delta but no baseline was supplied"
                    )
                elif gate.key not in baseline_map:
                    failures.append(
                        f"baseline missing metric {gate.domain}/{gate.name} required for delta"
                    )
                else:
                    delta = value - baseline_map[gate.key]
                    if delta < gate.min_delta_from_baseline:
                        failures.append(
                            f"metric {gate.domain}/{gate.name} delta={delta:g} "
                            f"is below {gate.min_delta_from_baseline:g}"
                        )

        check_map = evidence.check_map()
        for check_id in policy.required_checks:
            check = check_map.get(check_id)
            if check is None:
                failures.append(f"missing required check {check_id}")
            elif policy.require_all_checks_pass and check.status != "pass":
                failures.append(f"required check {check_id} has status {check.status!r}")

        state: DecisionState = "rejected" if failures else "promoted"
        return PromotionDecision(
            experiment_id=packet.experiment_id,
            packet_fingerprint=packet.fingerprint,
            evidence_fingerprint=evidence.fingerprint,
            policy_fingerprint=policy.fingerprint,
            state=state,
            reasons=tuple(failures) if failures else ("all predeclared evidence gates passed",),
        )
