"""Monte Carlo population and adversarial world exploration for neurOS Arena."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Callable, Literal

import numpy as np

from .manifest import ArenaManifest
from .runner import ArenaRun, run_scenario

DistributionKind = Literal["uniform", "log_uniform", "normal", "choice"]


@dataclass(frozen=True)
class ParameterDistribution:
    """A shareable distribution over one manifest parameter."""

    path: str
    kind: DistributionKind = "uniform"
    low: float | None = None
    high: float | None = None
    mean: float | None = None
    std: float | None = None
    choices: tuple[float, ...] = ()

    def validate(self) -> None:
        root = self.path.split(".", 1)[0]
        if root not in {"participant", "device", "display", "transport", "world_model"}:
            raise ValueError(f"unsupported population path root: {root!r}")
        if self.kind in {"uniform", "log_uniform"}:
            if self.low is None or self.high is None or self.high < self.low:
                raise ValueError(f"{self.kind} requires low <= high")
            if self.kind == "log_uniform" and self.low <= 0:
                raise ValueError("log_uniform requires low > 0")
        elif self.kind == "normal":
            if self.mean is None or self.std is None or self.std < 0:
                raise ValueError("normal requires mean and std >= 0")
        elif self.kind == "choice":
            if not self.choices:
                raise ValueError("choice requires at least one value")
        else:
            raise ValueError(f"unknown distribution: {self.kind}")

    def sample(self, rng: np.random.Generator) -> float:
        self.validate()
        if self.kind == "uniform":
            return float(rng.uniform(float(self.low), float(self.high)))
        if self.kind == "log_uniform":
            return float(np.exp(rng.uniform(np.log(float(self.low)), np.log(float(self.high)))))
        if self.kind == "normal":
            return float(rng.normal(float(self.mean), float(self.std)))
        return float(rng.choice(np.asarray(self.choices, dtype=float)))


@dataclass(frozen=True)
class PopulationSpec:
    size: int = 100
    seed: int = 7
    parameters: tuple[ParameterDistribution, ...] = ()

    def validate(self) -> None:
        if self.size <= 0:
            raise ValueError("population size must be positive")
        for parameter in self.parameters:
            parameter.validate()


@dataclass(frozen=True)
class PopulationTrial:
    index: int
    sampled: dict[str, float]
    metrics: dict[str, float]


@dataclass(frozen=True)
class PopulationRun:
    spec: PopulationSpec
    trials: tuple[PopulationTrial, ...]
    summary: dict[str, dict[str, float]]

    def to_dict(self) -> dict:
        return {
            "schema": "neuros.synthetic_bci_arena.population.v1",
            "spec": {
                "size": self.spec.size,
                "seed": self.spec.seed,
                "parameters": [asdict(value) for value in self.spec.parameters],
            },
            "trials": [asdict(trial) for trial in self.trials],
            "summary": self.summary,
            "evidence_boundary": "Synthetic population coverage only; not a human prevalence estimate.",
        }


def _replace_manifest_value(manifest: ArenaManifest, path: str, value: float) -> ArenaManifest:
    parts = path.split(".")
    if len(parts) < 2:
        raise ValueError(f"population path must include a field: {path!r}")
    root, tail = parts[0], parts[1:]
    if root == "world_model" and tail[0] == "parameters":
        if len(tail) != 2:
            raise ValueError("world_model population paths must be world_model.parameters.KEY")
        params = dict(manifest.world_model.parameters)
        params[tail[1]] = value
        return replace(manifest, world_model=replace(manifest.world_model, parameters=params))
    if len(tail) != 1:
        raise ValueError(f"nested population path is unsupported: {path!r}")
    obj = getattr(manifest, root)
    field = tail[0]
    if not hasattr(obj, field):
        raise ValueError(f"unknown population parameter: {path!r}")
    current = getattr(obj, field)
    cast_value: float | int
    if isinstance(current, int) and not isinstance(current, bool):
        cast_value = int(round(value))
    else:
        cast_value = float(value)
    return replace(manifest, **{root: replace(obj, **{field: cast_value})})


def default_run_metrics(run: ArenaRun) -> dict[str, float]:
    snr_values = list(run.report["metrics"]["target_snr_db"].values())
    display = run.report["metrics"]["display"]
    transport = run.report["metrics"]["transport"]
    return {
        "target_snr_db_mean": float(np.mean(snr_values)) if snr_values else 0.0,
        "target_snr_db_min": float(np.min(snr_values)) if snr_values else 0.0,
        "display_frequency_error_hz_max": float(max((row["frequency_error_hz"] for row in display), default=0.0)),
        "display_frame_drop_fraction_max": float(max((row["frame_drop_fraction"] for row in display), default=0.0)),
        "transport_packet_drop_fraction": float(transport["packet_drop_fraction"]),
        "transport_delivery_delay_p95_ms": float(transport["delivery_delay_p95_ms"]),
        "device_clipped_fraction": float(run.report["metrics"]["device_clipped_fraction"]),
    }


def _summary(trials: list[PopulationTrial]) -> dict[str, dict[str, float]]:
    keys = sorted({key for trial in trials for key in trial.metrics})
    result: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([trial.metrics[key] for trial in trials if key in trial.metrics], dtype=float)
        if values.size:
            result[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "p05": float(np.percentile(values, 5)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return result


def run_population(
    manifest: ArenaManifest,
    spec: PopulationSpec,
    evaluator: Callable[[ArenaRun], dict[str, float]] | None = None,
) -> PopulationRun:
    """Run a deterministic population over a shareable parameter envelope.

    ``evaluator`` can add decoder/application metrics to the generic Arena
    physics metrics, allowing the same population engine to qualify Mindforge or
    any other closed-loop consumer.
    """
    manifest.validate()
    spec.validate()
    rng = np.random.default_rng(spec.seed)
    trials: list[PopulationTrial] = []
    for index in range(spec.size):
        world = manifest
        sampled: dict[str, float] = {}
        for distribution in spec.parameters:
            value = distribution.sample(rng)
            sampled[distribution.path] = value
            world = _replace_manifest_value(world, distribution.path, value)
        # Each population member gets a deterministic background realization in
        # addition to its sampled physiological/system parameters.
        world = replace(world, participant=replace(world.participant, seed=world.participant.seed + spec.seed + index * 7919))
        run = run_scenario(
            world.scenario,
            world.participant,
            world.device,
            world.display,
            world.transport,
            world.world_model,
        )
        metrics = default_run_metrics(run)
        if evaluator is not None:
            metrics.update({key: float(value) for key, value in evaluator(run).items()})
        trials.append(PopulationTrial(index=index, sampled=sampled, metrics=metrics))
    return PopulationRun(spec=spec, trials=tuple(trials), summary=_summary(trials))
