"""Score decoder/application decisions against the Arena's known causal timeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .runner import ArenaRun


@dataclass(frozen=True)
class ArenaDecision:
    timestamp_s: float
    target_frequency_hz: float | None
    accepted: bool
    confidence: float = 0.0
    reason: str | None = None


def _truth_at(run: "ArenaRun", timestamp_s: float) -> float | None:
    times = run.device_output.timestamps_s
    if times.size == 0:
        return None
    index = int(np.clip(np.searchsorted(times, timestamp_s), 0, times.size - 1))
    value = run.ground_truth_target_hz[index]
    return None if np.isnan(value) else float(value)


def evaluate_decisions(run: "ArenaRun", decisions: list[ArenaDecision]) -> dict[str, float]:
    if not decisions:
        return {
            "decisions": 0.0,
            "accepted_fraction": 0.0,
            "accepted_precision": 0.0,
            "false_activation_fraction": 0.0,
            "median_switch_latency_s": 0.0,
        }
    accepted = [decision for decision in decisions if decision.accepted]
    correct = 0
    false_baseline = 0
    for decision in accepted:
        truth = _truth_at(run, decision.timestamp_s)
        if truth is None:
            false_baseline += 1
        elif decision.target_frequency_hz is not None and abs(decision.target_frequency_hz - truth) < 1e-6:
            correct += 1

    switch_latencies: list[float] = []
    prior_target: float | None = None
    for stage in run.stages:
        target = stage.target_frequency_hz
        if target is None or target == prior_target:
            prior_target = target
            continue
        match = next((decision for decision in decisions
                      if decision.accepted
                      and decision.timestamp_s >= stage.start_s
                      and decision.target_frequency_hz is not None
                      and abs(decision.target_frequency_hz - target) < 1e-6), None)
        if match is not None:
            switch_latencies.append(max(0.0, match.timestamp_s - stage.start_s))
        prior_target = target

    return {
        "decisions": float(len(decisions)),
        "accepted_fraction": float(len(accepted) / len(decisions)),
        "accepted_precision": float(correct / max(len(accepted), 1)),
        "false_activation_fraction": float(false_baseline / max(len(accepted), 1)),
        "median_switch_latency_s": float(np.median(switch_latencies)) if switch_latencies else 0.0,
    }
