"""Failure-preserving comparison of neural representation methods."""
from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

from .contracts import (
    MethodOutcome,
    MethodStatus,
    RepresentationBenchmarkResult,
    RepresentationMethod,
    RepresentationUnavailableError,
    SequenceBatch,
)
from .metrics import aggregate_geometry_metrics, aggregate_reference_metrics
from .pca import _positive_int


def _runtime_metadata(started: float) -> dict[str, float | str]:
    return {
        "runtime_seconds": max(0.0, perf_counter() - started),
        "runtime_domain": "operational",
    }


class RepresentationBenchmark:
    """Run requested methods under their declared fit regimes.

    The result intentionally does not rank methods. T-PHATE's transductive
    target fit, train-only PCA/autoencoder fitting, and fixed pretrained
    encoders represent different information regimes and must remain visible.
    Runtime is recorded as an operational diagnostic and is not a scientific
    geometry metric.
    """

    def __init__(
        self,
        methods: Iterable[RepresentationMethod],
        *,
        neighborhood_k: int = 5,
    ) -> None:
        methods = tuple(methods)
        if not methods:
            raise ValueError("at least one representation method is required")
        ids = [method.method_id for method in methods]
        if any(not isinstance(method_id, str) or not method_id.strip() for method_id in ids):
            raise ValueError("every method must expose a nonblank method_id")
        if len(set(ids)) != len(ids):
            raise ValueError("representation method IDs must be unique")
        self.methods = methods
        self.neighborhood_k = _positive_int(neighborhood_k, name="neighborhood_k")

    def run(
        self,
        train: SequenceBatch,
        evaluation: SequenceBatch,
        *,
        reference: SequenceBatch | None = None,
    ) -> RepresentationBenchmarkResult:
        if train.feature_count != evaluation.feature_count:
            raise ValueError("train and evaluation feature dimensions must match")
        if reference is not None:
            if reference.sequence_ids != evaluation.sequence_ids:
                raise ValueError(
                    "reference sequence identity must exactly match evaluation identity"
                )
            for source, reference_sequence in zip(
                evaluation.sequences,
                reference.sequences,
                strict=True,
            ):
                if source.shape[0] != reference_sequence.shape[0]:
                    raise ValueError(
                        "reference and evaluation sequences must have matching timepoints"
                    )

        outcomes: list[MethodOutcome] = []
        for method in self.methods:
            started = perf_counter()
            try:
                embedding = method.embed(train, evaluation)
                if embedding.sequence_ids != evaluation.sequence_ids:
                    raise ValueError(
                        "representation output sequence identity does not match evaluation batch"
                    )
                for source, latent in zip(
                    evaluation.sequences,
                    embedding.sequences,
                    strict=True,
                ):
                    if source.shape[0] != latent.shape[0]:
                        raise ValueError(
                            "representation output changed the evaluation timepoint count"
                        )
                metrics = aggregate_geometry_metrics(
                    evaluation.sequences,
                    embedding.sequences,
                    k=self.neighborhood_k,
                )
                if reference is not None:
                    metrics.update(
                        aggregate_reference_metrics(
                            reference.sequences,
                            embedding.sequences,
                            k=self.neighborhood_k,
                        )
                    )
                outcomes.append(
                    MethodOutcome(
                        method_id=method.method_id,
                        fit_regime=method.fit_regime,
                        status=MethodStatus.OK,
                        embedding=embedding,
                        metrics=metrics,
                        metadata={
                            "metric_scope": "trajectory_local_rigid_transform_invariant",
                            **_runtime_metadata(started),
                        },
                    )
                )
            except RepresentationUnavailableError as exc:
                outcomes.append(
                    MethodOutcome(
                        method_id=method.method_id,
                        fit_regime=method.fit_regime,
                        status=MethodStatus.UNAVAILABLE,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        metadata=_runtime_metadata(started),
                    )
                )
            # A representation plugin is an experiment subject. Any plugin-local failure must
            # become explicit FAILED evidence rather than aborting the remaining comparison.
            except Exception as exc:  # noqa: BLE001
                outcomes.append(
                    MethodOutcome(
                        method_id=method.method_id,
                        fit_regime=method.fit_regime,
                        status=MethodStatus.FAILED,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        metadata=_runtime_metadata(started),
                    )
                )

        return RepresentationBenchmarkResult(
            train_sequence_ids=train.sequence_ids,
            evaluation_sequence_ids=evaluation.sequence_ids,
            outcomes=tuple(outcomes),
            metadata={
                "neighborhood_k": self.neighborhood_k,
                "ranking_policy": "none",
                "claim_scope": "representation_geometry",
                "reference_geometry": "provided" if reference is not None else "none",
                "runtime_domain": "operational",
            },
        )
