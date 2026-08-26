"""Portable benchmark packs for reproducible BCI systems claims."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from .manifest import ArenaManifest, manifest_from_dict
from .runner import ArenaRun, run_scenario

BENCHMARK_SCHEMA = "neuros.synthetic_bci_arena.benchmark_pack.v1"
RuleOperator = Literal["<", "<=", ">", ">=", "==", "!="]
Scalar = str | int | float | bool | None


@dataclass(frozen=True)
class MetricRule:
    path: str
    operator: RuleOperator
    expected: Scalar
    description: str = ""

    def validate(self) -> None:
        if not self.path:
            raise ValueError("benchmark rule path is required")
        if self.operator not in {"<", "<=", ">", ">=", "==", "!="}:
            raise ValueError(f"unsupported benchmark operator: {self.operator}")
        if self.operator in {"<", "<=", ">", ">="} and not isinstance(self.expected, (int, float)):
            raise ValueError("ordered benchmark comparisons require a numeric expected value")


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    manifest: ArenaManifest
    rules: tuple[MetricRule, ...]
    description: str = ""

    def validate(self) -> None:
        if not self.name or not self.rules:
            raise ValueError("benchmark case requires a name and at least one rule")
        self.manifest.validate()
        for rule in self.rules:
            rule.validate()


@dataclass(frozen=True)
class BenchmarkPack:
    name: str
    version: str
    cases: tuple[BenchmarkCase, ...]
    description: str = ""
    claim_scope: str = "synthetic systems conformance"

    def validate(self) -> None:
        if not self.name or not self.version or not self.cases:
            raise ValueError("benchmark pack requires name, version, and cases")
        names = [case.name for case in self.cases]
        if len(names) != len(set(names)):
            raise ValueError("benchmark case names must be unique")
        for case in self.cases:
            case.validate()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BENCHMARK_SCHEMA,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "claim_scope": self.claim_scope,
            "cases": [
                {
                    "name": case.name,
                    "description": case.description,
                    "manifest": case.manifest.to_dict(),
                    "rules": [asdict(rule) for rule in case.rules],
                }
                for case in self.cases
            ],
        }


@dataclass(frozen=True)
class BenchmarkCaseResult:
    name: str
    passed: bool
    observations: dict[str, Scalar]
    failures: tuple[str, ...]
    report: dict[str, Any]


@dataclass(frozen=True)
class BenchmarkPackResult:
    pack_name: str
    pack_version: str
    passed: bool
    cases: tuple[BenchmarkCaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "neuros.synthetic_bci_arena.benchmark_result.v1",
            "pack_name": self.pack_name,
            "pack_version": self.pack_version,
            "passed": self.passed,
            "cases": [asdict(case) for case in self.cases],
            "evidence_boundary": (
                "Passing this pack establishes only the declared synthetic systems-conformance claims. "
                "It does not establish human physiological performance."
            ),
        }


def _lookup(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"benchmark metric path not found: {path!r}")
        value = value[part]
    return value


def _compare(observed: Any, operator: RuleOperator, expected: Scalar) -> bool:
    if operator == "==":
        return observed == expected
    if operator == "!=":
        return observed != expected
    if not isinstance(observed, (int, float)) or not isinstance(expected, (int, float)):
        return False
    if operator == "<":
        return observed < expected
    if operator == "<=":
        return observed <= expected
    if operator == ">":
        return observed > expected
    return observed >= expected


def run_benchmark_pack(
    pack: BenchmarkPack,
    evaluator: Callable[[ArenaRun], dict[str, Scalar]] | None = None,
) -> BenchmarkPackResult:
    """Run a versioned conformance pack against Arena and optional application metrics."""
    pack.validate()
    case_results: list[BenchmarkCaseResult] = []
    for case in pack.cases:
        manifest = case.manifest
        run = run_scenario(
            manifest.scenario,
            manifest.participant,
            manifest.device,
            manifest.display,
            manifest.transport,
            manifest.world_model,
        )
        payload = dict(run.report)
        if evaluator is not None:
            payload["application"] = dict(evaluator(run))
        observations: dict[str, Scalar] = {}
        failures: list[str] = []
        for rule in case.rules:
            try:
                observed = _lookup(payload, rule.path)
            except KeyError as exc:
                failures.append(str(exc))
                continue
            observations[rule.path] = observed
            if not _compare(observed, rule.operator, rule.expected):
                message = f"{rule.path}: observed {observed!r} {rule.operator} expected {rule.expected!r} failed"
                if rule.description:
                    message += f" ({rule.description})"
                failures.append(message)
        case_results.append(BenchmarkCaseResult(
            name=case.name,
            passed=not failures,
            observations=observations,
            failures=tuple(failures),
            report=payload,
        ))
    return BenchmarkPackResult(
        pack_name=pack.name,
        pack_version=pack.version,
        passed=all(case.passed for case in case_results),
        cases=tuple(case_results),
    )


def benchmark_pack_from_dict(raw: dict[str, Any]) -> BenchmarkPack:
    if raw.get("schema") != BENCHMARK_SCHEMA:
        raise ValueError(f"expected benchmark schema {BENCHMARK_SCHEMA!r}")
    cases = []
    for case_raw in raw.get("cases", []):
        cases.append(BenchmarkCase(
            name=str(case_raw["name"]),
            description=str(case_raw.get("description", "")),
            manifest=manifest_from_dict(dict(case_raw["manifest"])),
            rules=tuple(MetricRule(**dict(rule)) for rule in case_raw.get("rules", [])),
        ))
    pack = BenchmarkPack(
        name=str(raw["name"]),
        version=str(raw["version"]),
        description=str(raw.get("description", "")),
        claim_scope=str(raw.get("claim_scope", "synthetic systems conformance")),
        cases=tuple(cases),
    )
    pack.validate()
    return pack


def load_benchmark_pack(path: str | Path) -> BenchmarkPack:
    return benchmark_pack_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_benchmark_pack(pack: BenchmarkPack, path: str | Path) -> None:
    pack.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pack.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
