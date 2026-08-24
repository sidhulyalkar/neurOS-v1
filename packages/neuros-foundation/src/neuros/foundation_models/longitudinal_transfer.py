"""Frozen-representation transfer methods for longitudinal BCI evidence.

The encoder is trained once on source history and then frozen. Target-session
calibration can change only the low-capacity readout and, for SourceWeigher, the
source-domain weights. This keeps representation adaptation distinct from
end-to-end target fine-tuning.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from .longitudinal_authority import LongitudinalCaseAuthority
from .longitudinal_methods import (
    TaskDecoderMethodSpec,
    _model_for,
    _parameter_count,
    _resolved_config,
    _score,
)
from .probes import representation_report
from .real_world import GroupedEvaluationData

FrozenTransferId = Literal["frozen-logistic", "sourceweigher-mean"]


@dataclass(frozen=True, slots=True)
class FrozenTransferMethodSpec:
    """Identity for a source-trained frozen encoder plus matched linear readout."""

    method_id: FrozenTransferId
    encoder_id: Literal["eegnet", "eeg-conformer"]
    encoder_seed: int
    encoder_kwargs: Mapping[str, Any] = field(default_factory=dict)
    readout_c: float = 1.0
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.method_id not in {"frozen-logistic", "sourceweigher-mean"}:
            raise ValueError(f"unsupported frozen transfer method {self.method_id!r}")
        if self.encoder_id not in {"eegnet", "eeg-conformer"}:
            raise ValueError(f"unsupported encoder {self.encoder_id!r}")
        if self.readout_c <= 0:
            raise ValueError("readout_c must be positive")
        forbidden = {"n_channels", "n_classes", "random_state"}.intersection(
            self.encoder_kwargs
        )
        if forbidden:
            raise ValueError(
                "n_channels, n_classes and random_state are evidence-controlled; "
                f"remove overrides {sorted(forbidden)}"
            )
        object.__setattr__(self, "encoder_kwargs", MappingProxyType(dict(self.encoder_kwargs)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "encoder_id": self.encoder_id,
            "encoder_seed": int(self.encoder_seed),
            "encoder_kwargs": dict(self.encoder_kwargs),
            "readout_c": float(self.readout_c),
        }

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class FrozenTransferCaseResult:
    authority_fingerprint: str
    method_spec: FrozenTransferMethodSpec
    encoder_config: Mapping[str, Any]
    encoder_parameter_count: int
    analysis_manifest_fingerprint: str
    encoder_fit_s: float
    rows: tuple[Mapping[str, Any], ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "encoder_config", MappingProxyType(dict(self.encoder_config)))
        object.__setattr__(self, "rows", tuple(MappingProxyType(dict(row)) for row in self.rows))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "authority_fingerprint": self.authority_fingerprint,
            "method_spec": self.method_spec.to_dict(),
            "method_spec_fingerprint": self.method_spec.fingerprint,
            "encoder_config": dict(self.encoder_config),
            "encoder_parameter_count": int(self.encoder_parameter_count),
            "analysis_manifest_fingerprint": self.analysis_manifest_fingerprint,
            "encoder_fit_s": float(self.encoder_fit_s),
            "rows": [dict(row) for row in self.rows],
        }


def _encode_labels(y: np.ndarray, source_indices: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    labels = tuple(sorted(np.unique(np.asarray(y)[source_indices].astype(str)).tolist()))
    if len(labels) < 2:
        raise ValueError("source history must contain at least two task classes")
    mapping = {label: index for index, label in enumerate(labels)}
    values = np.asarray(y).astype(str)
    unknown = sorted(set(values.tolist()) - set(mapping))
    if unknown:
        raise ValueError(f"target contains class labels absent from source history: {unknown}")
    return np.asarray([mapping[value] for value in values], dtype=np.int64), labels


def _fit_logistic(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    *,
    c: float,
    sample_weight: np.ndarray | None = None,
) -> LogisticRegression:
    model = LogisticRegression(
        C=float(c),
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    )
    model.fit(train_embeddings, train_labels, sample_weight=sample_weight)
    return model


def _sourceweigher_result(
    *,
    source_embeddings: Mapping[str, np.ndarray],
    target_embeddings: np.ndarray,
):
    try:
        from neuros_sourceweigher import RepresentationSourceWeigher
    except ImportError as exc:  # pragma: no cover - optional package boundary
        raise ImportError(
            "sourceweigher-mean requires neuros-sourceweigher. Install the local "
            "package or the matching published distribution."
        ) from exc

    estimator = RepresentationSourceWeigher(statistics=("mean",))
    return estimator.estimate(source_embeddings, target_embeddings)


def _source_sample_weights(
    source_sessions: np.ndarray,
    by_source: Mapping[str, float],
) -> np.ndarray:
    """Redistribute fixed total source sample mass across source sessions.

    Unweighted fitting gives every source example weight 1, for total source mass
    ``N``. This helper preserves total mass ``N`` while making each source
    session's aggregate mass equal ``N * source_weight``. No free target/source
    mass hyperparameter is introduced.
    """
    sessions = np.asarray(source_sessions).astype(str)
    total = len(sessions)
    result = np.zeros(total, dtype=np.float64)
    for source_id, weight in by_source.items():
        mask = sessions == str(source_id)
        count = int(mask.sum())
        if count <= 0:
            raise ValueError(f"SourceWeigher returned unknown/empty source {source_id!r}")
        result[mask] = float(weight) * total / count
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("computed source sample weights are invalid")
    if not np.isclose(result.sum(), float(total), rtol=1e-6, atol=1e-6):
        raise ValueError("source sample weighting failed to preserve total source mass")
    return result


def run_frozen_transfer_case(
    data: GroupedEvaluationData,
    authority: LongitudinalCaseAuthority,
    *,
    spec: FrozenTransferMethodSpec,
    budgets_per_class: Sequence[int],
) -> FrozenTransferCaseResult:
    """Run one frozen encoder across all calibration budgets.

    ``frozen-logistic`` uses labels from source history plus the declared target
    calibration examples.

    ``sourceweigher-mean`` additionally estimates source-session weights from
    source embeddings and the declared target calibration embeddings. Final
    evaluation embeddings are never passed to SourceWeigher. At budget 0 this
    target-dependent method emits an explicit unavailable row rather than using
    the evaluation set as unlabeled target data.
    """
    split = authority.restore(data)
    if authority.split_unit != "session":
        raise ValueError("frozen longitudinal transfer currently requires session authority")
    budgets = tuple(sorted(set(int(value) for value in budgets_per_class)))
    if not budgets or budgets[0] < 0:
        raise ValueError("budgets_per_class must contain non-negative values")
    if budgets[-1] > split.max_budget_per_class:
        raise ValueError(
            f"requested budget {budgets[-1]} exceeds authority maximum "
            f"{split.max_budget_per_class}/class"
        )

    X = np.asarray(data.X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError("frozen EEG transfer expects X=(sample, channel, time)")
    y_encoded, class_labels = _encode_labels(np.asarray(data.y), split.source_train_indices)

    encoder_spec = TaskDecoderMethodSpec(
        method_id=spec.encoder_id,
        model_seed=spec.encoder_seed,
        model_kwargs=spec.encoder_kwargs,
    )
    encoder = _model_for(
        encoder_spec,
        n_channels=int(X.shape[1]),
        n_classes=len(class_labels),
    )
    encoder_config = _resolved_config(encoder)
    parameter_count = _parameter_count(encoder)
    manifest_fingerprint = encoder.analysis_manifest().fingerprint()

    started = time.perf_counter()
    encoder.train(X[split.source_train_indices], y_encoded[split.source_train_indices])
    encoder_fit_s = time.perf_counter() - started

    source_embedding = encoder.encode(X[split.source_train_indices])
    evaluation_embedding = encoder.encode(X[split.evaluation_indices])
    target_pool_indices = np.sort(
        np.concatenate(
            [np.asarray(values, dtype=np.int64) for values in split.calibration_order_by_class.values()]
        )
    )
    target_pool_embedding = encoder.encode(X[target_pool_indices])
    target_embedding_by_index = {
        int(index): target_pool_embedding[position]
        for position, index in enumerate(target_pool_indices.tolist())
    }

    source_session = np.asarray(data.groups["session"])[split.source_train_indices].astype(str)
    source_embeddings_by_session = {
        session: source_embedding[source_session == session]
        for session in authority.source_group_values
    }

    rows: list[dict[str, Any]] = []
    for budget in budgets:
        calibration_indices = split.calibration_indices(budget)
        calibration_embedding = (
            np.stack([target_embedding_by_index[int(index)] for index in calibration_indices])
            if len(calibration_indices)
            else np.empty((0, source_embedding.shape[1]), dtype=source_embedding.dtype)
        )

        if spec.method_id == "sourceweigher-mean" and budget == 0:
            rows.append(
                {
                    "case_id": authority.case_id,
                    "authority_fingerprint": authority.authority_fingerprint,
                    "processed_data_sha256": authority.processed_data_sha256,
                    "method_id": spec.method_id,
                    "method_spec_fingerprint": spec.fingerprint,
                    "encoder_id": spec.encoder_id,
                    "encoder_seed": int(spec.encoder_seed),
                    "calibration_per_class": 0,
                    "status": "unavailable_no_target_observations",
                    "failure_reason": (
                        "target-dependent SourceWeigher requires declared target calibration "
                        "embeddings; final evaluation examples are forbidden"
                    ),
                }
            )
            continue

        train_embedding = source_embedding
        train_labels = y_encoded[split.source_train_indices]
        sample_weight: np.ndarray | None = None
        weighting_payload: dict[str, Any] | None = None

        if spec.method_id == "sourceweigher-mean":
            weighting = _sourceweigher_result(
                source_embeddings=source_embeddings_by_session,
                target_embeddings=calibration_embedding,
            )
            source_weights = _source_sample_weights(source_session, weighting.by_source())
            weighting_payload = weighting.to_dict()
        else:
            source_weights = np.ones(len(source_embedding), dtype=np.float64)

        if len(calibration_indices):
            train_embedding = np.concatenate([source_embedding, calibration_embedding], axis=0)
            train_labels = np.concatenate(
                [train_labels, y_encoded[calibration_indices]], axis=0
            )
            # Calibration labels retain ordinary per-example mass. SourceWeigher
            # only redistributes the source-history mass across source sessions.
            sample_weight = np.concatenate(
                [source_weights, np.ones(len(calibration_indices), dtype=np.float64)]
            )
        elif spec.method_id == "sourceweigher-mean":  # guarded above; defensive
            raise RuntimeError("SourceWeigher reached zero budget unexpectedly")

        started = time.perf_counter()
        readout = _fit_logistic(
            train_embedding,
            train_labels,
            c=spec.readout_c,
            sample_weight=sample_weight,
        )
        readout_fit_s = time.perf_counter() - started
        started = time.perf_counter()
        probability = readout.predict_proba(evaluation_embedding)
        inference_s = time.perf_counter() - started
        metrics = _score(y_encoded[split.evaluation_indices], probability)

        rows.append(
            {
                "case_id": authority.case_id,
                "authority_fingerprint": authority.authority_fingerprint,
                "processed_data_sha256": authority.processed_data_sha256,
                "partition_fingerprint": authority.partition_fingerprint,
                "calibration_split_fingerprint": authority.calibration_split_fingerprint,
                "method_id": spec.method_id,
                "method_spec_fingerprint": spec.fingerprint,
                "encoder_id": spec.encoder_id,
                "encoder_seed": int(spec.encoder_seed),
                "calibration_per_class": int(budget),
                "status": "ok",
                "failure_reason": None,
                "source_train_samples": int(len(split.source_train_indices)),
                "calibration_samples": int(len(calibration_indices)),
                "evaluation_samples": int(len(split.evaluation_indices)),
                "class_labels": list(class_labels),
                **metrics,
                "readout_fit_s": float(readout_fit_s),
                "inference_s": float(inference_s),
                "inference_ms_per_trial": float(
                    1000.0 * inference_s / max(len(split.evaluation_indices), 1)
                ),
                "sourceweigher": weighting_payload,
            }
        )

    return FrozenTransferCaseResult(
        authority_fingerprint=authority.authority_fingerprint,
        method_spec=spec,
        encoder_config=encoder_config,
        encoder_parameter_count=parameter_count,
        analysis_manifest_fingerprint=manifest_fingerprint,
        encoder_fit_s=float(encoder_fit_s),
        rows=tuple(rows),
    )
