"""BCI-relevant runtime quality metrics and version-controlled gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    """Generic runtime acceptance thresholds.

    These defaults are intentionally conservative enough for shared CI. Device-
    specific and application-specific latency requirements belong in hardware
    qualification profiles rather than being smuggled into generic unit tests.
    """

    min_decoder_samples: int = 1
    max_drop_fraction: float = 0.0
    max_decoder_p99_ms: float = 250.0
    max_transform_p99_ms: float = 250.0
    max_failed_nodes: int = 0

    def __post_init__(self) -> None:
        if self.min_decoder_samples < 0:
            raise ValueError("min_decoder_samples must be >= 0")
        if not 0.0 <= self.max_drop_fraction <= 1.0:
            raise ValueError("max_drop_fraction must be in [0, 1]")
        if self.max_decoder_p99_ms <= 0 or self.max_transform_p99_ms <= 0:
            raise ValueError("latency limits must be positive")
        if self.max_failed_nodes < 0:
            raise ValueError("max_failed_nodes must be >= 0")


@dataclass(frozen=True, slots=True)
class QualityGateResult:
    passed: bool
    checks: Mapping[str, bool]
    metrics: Mapping[str, float | int]
    failures: tuple[str, ...]


def _edge_counts(snapshot: Mapping[str, Any]) -> tuple[int, int]:
    accepted = 0
    dropped = 0
    for edge in snapshot.get("edges", {}).values():
        accepted += int(edge.get("accepted", 0))
        dropped += int(edge.get("dropped", 0))
    return accepted, dropped


def evaluate_runtime_snapshot(
    snapshot: Mapping[str, Any],
    thresholds: QualityThresholds = QualityThresholds(),
) -> QualityGateResult:
    """Evaluate a RuntimeExecutor snapshot against explicit acceptance gates."""

    nodes = snapshot.get("nodes", {})
    decoder = nodes.get("decoder:primary", {})
    decoder_samples = int(decoder.get("processed", 0))
    decoder_p99 = float(decoder.get("p99_latency_ms", 0.0))
    transform_p99 = max(
        (
            float(metrics.get("p99_latency_ms", 0.0))
            for node_id, metrics in nodes.items()
            if str(node_id).startswith("transform:")
        ),
        default=0.0,
    )
    failed_nodes = sum(int(metrics.get("failed", 0)) > 0 for metrics in nodes.values())
    accepted, dropped = _edge_counts(snapshot)
    attempted = accepted + dropped
    drop_fraction = dropped / attempted if attempted else 0.0

    checks = {
        "runtime_stopped_cleanly": snapshot.get("state") == "stopped" and snapshot.get("failure") is None,
        "decoder_activity": decoder_samples >= thresholds.min_decoder_samples,
        "edge_loss": drop_fraction <= thresholds.max_drop_fraction,
        "decoder_p99_latency": decoder_p99 <= thresholds.max_decoder_p99_ms,
        "transform_p99_latency": transform_p99 <= thresholds.max_transform_p99_ms,
        "node_failures": failed_nodes <= thresholds.max_failed_nodes,
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    return QualityGateResult(
        passed=not failures,
        checks=checks,
        metrics={
            "decoder_samples": decoder_samples,
            "decoder_p99_ms": decoder_p99,
            "transform_p99_ms": transform_p99,
            "failed_nodes": failed_nodes,
            "accepted_edge_items": accepted,
            "dropped_edge_items": dropped,
            "drop_fraction": drop_fraction,
            "runtime_seconds": float(snapshot.get("runtime_seconds", 0.0)),
        },
        failures=failures,
    )
