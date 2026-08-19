"""Protocol-first benchmarking for neural foundation-model representations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

import numpy as np

from .probes import domain_leakage_probe, linear_probe, pairwise_cka, representation_report

SplitUnit = Literal["subject", "session", "site", "device", "recording", "sample"]
TransferRegime = Literal[
    "zero_shot",
    "linear_probe",
    "few_shot",
    "parameter_efficient",
    "full_fine_tune",
    "reconstruction",
]


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    """Minimum context required to interpret a foundation-model score."""

    name: str
    split_unit: SplitUnit = "subject"
    transfer_regime: TransferRegime = "linear_probe"
    pooling: str = "token_preserving_or_model_recommended"
    preprocessing: str = "model_recommended; identical downstream split"
    seed: int = 0
    leakage_controls: tuple[str, ...] = (
        "no subject overlap between train/test",
        "fit normalization on train split only",
    )
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("protocol name must be non-empty")
        if self.split_unit == "sample" and self.transfer_regime != "reconstruction":
            if not any("sample split" in note.lower() for note in self.notes):
                raise ValueError(
                    "sample-level splits can leak subject/session structure. "
                    "Add an explicit note containing 'sample split' if this is intentional."
                )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fingerprint"] = self.fingerprint
        return data


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    protocol: EvaluationProtocol
    model_results: tuple[dict[str, Any], ...]
    pairwise_similarity: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol.to_dict(),
            "model_results": list(self.model_results),
            "pairwise_similarity": list(self.pairwise_similarity),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def benchmark_embeddings(
    train_embeddings: Mapping[str, Any],
    test_embeddings: Mapping[str, Any],
    train_targets: Any,
    test_targets: Any,
    *,
    protocol: EvaluationProtocol,
    task: Literal["auto", "classification", "regression"] = "auto",
    alpha: float = 1e-3,
    train_domains: Any | None = None,
    test_domains: Any | None = None,
) -> BenchmarkReport:
    """Benchmark multiple models under one identical downstream protocol."""
    train_names = set(train_embeddings)
    test_names = set(test_embeddings)
    if train_names != test_names:
        missing_test = sorted(train_names - test_names)
        missing_train = sorted(test_names - train_names)
        raise ValueError(
            f"train/test model sets must match; missing_test={missing_test}, missing_train={missing_train}"
        )
    if not train_names:
        raise ValueError("at least one model embedding set is required")
    if (train_domains is None) != (test_domains is None):
        raise ValueError("train_domains and test_domains must be provided together")

    rows: list[dict[str, Any]] = []
    for model_id in sorted(train_names):
        train = np.asarray(train_embeddings[model_id])
        test = np.asarray(test_embeddings[model_id])
        probe = linear_probe(
            train,
            train_targets,
            test,
            test_targets,
            task=task,
            alpha=alpha,
        )
        row: dict[str, Any] = {
            "model_id": model_id,
            "probe": probe,
            "train_representation": representation_report(train),
            "test_representation": representation_report(test),
        }
        if train_domains is not None and test_domains is not None:
            row["domain_leakage"] = domain_leakage_probe(
                train,
                train_domains,
                test,
                test_domains,
                alpha=alpha,
            )
        rows.append(row)

    similarities = pairwise_cka({name: test_embeddings[name] for name in sorted(test_names)})
    return BenchmarkReport(
        protocol=protocol,
        model_results=tuple(rows),
        pairwise_similarity=tuple(similarities),
    )


def sample_efficiency_curve(
    train_embeddings: Any,
    train_targets: Any,
    test_embeddings: Any,
    test_targets: Any,
    *,
    fractions: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
    task: Literal["auto", "classification", "regression"] = "auto",
    alpha: float = 1e-3,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Measure downstream sample efficiency with a deterministic nested subset."""
    x = np.asarray(train_embeddings)
    y = np.asarray(train_targets)
    if len(x) != len(y):
        raise ValueError("train_embeddings and train_targets lengths must match")
    if len(x) < 2:
        raise ValueError("sample-efficiency curves require at least two training samples")

    normalized = tuple(sorted(set(float(value) for value in fractions)))
    if not normalized or normalized[0] <= 0 or normalized[-1] > 1:
        raise ValueError("fractions must lie in (0, 1]")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    rows: list[dict[str, Any]] = []
    for fraction in normalized:
        n = min(len(x), max(2, int(np.ceil(len(x) * fraction))))
        indices = order[:n]
        try:
            probe = linear_probe(
                x[indices],
                y[indices],
                test_embeddings,
                test_targets,
                task=task,
                alpha=alpha,
            )
        except ValueError as exc:
            probe = {"error": str(exc), "score": None}
        rows.append({"fraction": fraction, "n_train": n, "probe": probe})
    return rows
