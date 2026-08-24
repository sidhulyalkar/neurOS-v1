"""Frozen-representation transfer methods for longitudinal BCI evidence.

A frozen encoder is trained once on source history and materialized as a
``PreparedFrozenEncoderCase``. Multiple transfer strategies can then consume the
exact same representation tensors, making SourceWeigher-vs-unweighted readout a
clean comparison rather than two nominally identical encoder trainings.
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
from .real_world import GroupedEvaluationData

FrozenTransferStrategy = Literal["frozen-logistic", "sourceweigher-mean"]


def _readonly_array(values: Any, *, dtype: Any | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _hash_array(digest: Any, name: str, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))


@dataclass(frozen=True, slots=True)
class FrozenTransferMethodSpec:
    """Identity for a frozen representation plus one transfer/readout strategy."""

    method_id: str
    strategy: FrozenTransferStrategy
    encoder_id: Literal["eegnet", "eeg-conformer"]
    encoder_seed: int
    encoder_kwargs: Mapping[str, Any] = field(default_factory=dict)
    readout_c: float = 1.0
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not str(self.method_id).strip():
            raise ValueError("method_id must be non-empty")
        if self.strategy not in {"frozen-logistic", "sourceweigher-mean"}:
            raise ValueError(f"unsupported frozen transfer strategy {self.strategy!r}")
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
        object.__setattr__(self, "method_id", str(self.method_id).strip())
        object.__setattr__(self, "encoder_kwargs", MappingProxyType(dict(self.encoder_kwargs)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "strategy": self.strategy,
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
class PreparedFrozenEncoderCase:
    """One source-trained encoder and immutable representation tensors for a case."""

    authority_fingerprint: str
    encoder_id: str
    encoder_seed: int
    encoder_spec_fingerprint: str
    encoder_config: Mapping[str, Any]
    encoder_parameter_count: int
    analysis_manifest_fingerprint: str
    encoder_fit_s: float
    class_labels: tuple[str, ...]
    y_encoded: np.ndarray
    source_indices: np.ndarray
    evaluation_indices: np.ndarray
    target_pool_indices: np.ndarray
    source_embedding: np.ndarray
    evaluation_embedding: np.ndarray
    target_pool_embedding: np.ndarray
    source_session: np.ndarray
    source_embeddings_by_session: Mapping[str, np.ndarray]
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "encoder_config", MappingProxyType(dict(self.encoder_config)))
        object.__setattr__(self, "y_encoded", _readonly_array(self.y_encoded, dtype=np.int64))
        object.__setattr__(self, "source_indices", _readonly_array(self.source_indices, dtype=np.int64))
        object.__setattr__(self, "evaluation_indices", _readonly_array(self.evaluation_indices, dtype=np.int64))
        object.__setattr__(self, "target_pool_indices", _readonly_array(self.target_pool_indices, dtype=np.int64))
        object.__setattr__(self, "source_embedding", _readonly_array(self.source_embedding))
        object.__setattr__(self, "evaluation_embedding", _readonly_array(self.evaluation_embedding))
        object.__setattr__(self, "target_pool_embedding", _readonly_array(self.target_pool_embedding))
        object.__setattr__(self, "source_session", _readonly_array(self.source_session))
        frozen_sources = {
            str(key): _readonly_array(value)
            for key, value in self.source_embeddings_by_session.items()
        }
        object.__setattr__(self, "source_embeddings_by_session", MappingProxyType(frozen_sources))

    @property
    def representation_sha256(self) -> str:
        digest = hashlib.sha256()
        _hash_array(digest, "source_indices", self.source_indices)
        _hash_array(digest, "evaluation_indices", self.evaluation_indices)
        _hash_array(digest, "target_pool_indices", self.target_pool_indices)
        _hash_array(digest, "source_embedding", self.source_embedding)
        _hash_array(digest, "evaluation_embedding", self.evaluation_embedding)
        _hash_array(digest, "target_pool_embedding", self.target_pool_embedding)
        return digest.hexdigest()

    @property
    def fingerprint(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "authority_fingerprint": self.authority_fingerprint,
            "encoder_id": self.encoder_id,
            "encoder_seed": int(self.encoder_seed),
            "encoder_spec_fingerprint": self.encoder_spec_fingerprint,
            "analysis_manifest_fingerprint": self.analysis_manifest_fingerprint,
            "representation_sha256": self.representation_sha256,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "authority_fingerprint": self.authority_fingerprint,
            "encoder_id": self.encoder_id,
            "encoder_seed": int(self.encoder_seed),
            "encoder_spec_fingerprint": self.encoder_spec_fingerprint,
            "encoder_config": dict(self.encoder_config),
            "encoder_parameter_count": int(self.encoder_parameter_count),
            "analysis_manifest_fingerprint": self.analysis_manifest_fingerprint,
            "encoder_fit_s": float(self.encoder_fit_s),
            "class_labels": list(self.class_labels),
            "source_samples": int(len(self.source_indices)),
            "target_pool_samples": int(len(self.target_pool_indices)),
            "evaluation_samples": int(len(self.evaluation_indices)),
            "representation_sha256": self.representation_sha256,
            "encoder_state_fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class FrozenTransferCaseResult:
    authority_fingerprint: str
    method_spec: FrozenTransferMethodSpec
    encoder_state_manifest: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    schema_version: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "encoder_state_manifest", MappingProxyType(dict(self.encoder_state_manifest))
        )
        object.__setattr__(self, "rows", tuple(MappingProxyType(dict(row)) for row in self.rows))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "authority_fingerprint": self.authority_fingerprint,
            "method_spec": self.method_spec.to_dict(),
            "method_spec_fingerprint": self.method_spec.fingerprint,
            "encoder_state": dict(self.encoder_state_manifest),
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


def prepare_frozen_encoder_case(
    data: GroupedEvaluationData,
    authority: LongitudinalCaseAuthority,
    *,
    encoder_id: Literal["eegnet", "eeg-conformer"],
    encoder_seed: int,
    encoder_kwargs: Mapping[str, Any] | None = None,
) -> PreparedFrozenEncoderCase:
    """Train one encoder on source history and materialize immutable embeddings."""
    split = authority.restore(data)
    if authority.split_unit != "session":
        raise ValueError("frozen longitudinal transfer currently requires session authority")
    X = np.asarray(data.X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError("frozen EEG transfer expects X=(sample, channel, time)")
    y_encoded, class_labels = _encode_labels(np.asarray(data.y), split.source_train_indices)

    encoder_spec = TaskDecoderMethodSpec(
        method_id=encoder_id,
        model_seed=int(encoder_seed),
        model_kwargs=dict(encoder_kwargs or {}),
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
    source_session = np.asarray(data.groups["session"])[split.source_train_indices].astype(str)
    source_embeddings_by_session = {
        session: source_embedding[source_session == session]
        for session in authority.source_group_values
    }

    return PreparedFrozenEncoderCase(
        authority_fingerprint=authority.authority_fingerprint,
        encoder_id=encoder_id,
        encoder_seed=int(encoder_seed),
        encoder_spec_fingerprint=encoder_spec.fingerprint,
        encoder_config=encoder_config,
        encoder_parameter_count=parameter_count,
        analysis_manifest_fingerprint=manifest_fingerprint,
        encoder_fit_s=float(encoder_fit_s),
        class_labels=class_labels,
        y_encoded=y_encoded,
        source_indices=split.source_train_indices,
        evaluation_indices=split.evaluation_indices,
        target_pool_indices=target_pool_indices,
        source_embedding=source_embedding,
        evaluation_embedding=evaluation_embedding,
        target_pool_embedding=target_pool_embedding,
        source_session=source_session,
        source_embeddings_by_session=source_embeddings_by_session,
    )


def _validate_prepared(
    prepared: PreparedFrozenEncoderCase,
    authority: LongitudinalCaseAuthority,
    spec: FrozenTransferMethodSpec,
) -> None:
    if prepared.authority_fingerprint != authority.authority_fingerprint:
        raise ValueError("prepared encoder authority does not match requested case authority")
    encoder_spec = TaskDecoderMethodSpec(
        method_id=spec.encoder_id,
        model_seed=spec.encoder_seed,
        model_kwargs=spec.encoder_kwargs,
    )
    if prepared.encoder_spec_fingerprint != encoder_spec.fingerprint:
        raise ValueError("prepared encoder spec does not match transfer method encoder spec")


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
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sourceweigher-mean requires neuros-sourceweigher. Install the local "
            "package or the matching published distribution."
        ) from exc
    return RepresentationSourceWeigher(statistics=("mean",)).estimate(
        source_embeddings,
        target_embeddings,
    )


def _source_sample_weights(
    source_sessions: np.ndarray,
    by_source: Mapping[str, float],
) -> np.ndarray:
    """Redistribute fixed total source mass across source sessions."""
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
    prepared: PreparedFrozenEncoderCase | None = None,
) -> FrozenTransferCaseResult:
    """Run one transfer strategy on an exact frozen encoder representation."""
    split = authority.restore(data)
    budgets = tuple(sorted(set(int(value) for value in budgets_per_class)))
    if not budgets or budgets[0] < 0:
        raise ValueError("budgets_per_class must contain non-negative values")
    if budgets[-1] > split.max_budget_per_class:
        raise ValueError(
            f"requested budget {budgets[-1]} exceeds authority maximum "
            f"{split.max_budget_per_class}/class"
        )

    state = prepared or prepare_frozen_encoder_case(
        data,
        authority,
        encoder_id=spec.encoder_id,
        encoder_seed=spec.encoder_seed,
        encoder_kwargs=spec.encoder_kwargs,
    )
    _validate_prepared(state, authority, spec)

    target_embedding_by_index = {
        int(index): state.target_pool_embedding[position]
        for position, index in enumerate(state.target_pool_indices.tolist())
    }
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        calibration_indices = split.calibration_indices(budget)
        calibration_embedding = (
            np.stack([target_embedding_by_index[int(index)] for index in calibration_indices])
            if len(calibration_indices)
            else np.empty((0, state.source_embedding.shape[1]), dtype=state.source_embedding.dtype)
        )

        common_identity = {
            "case_id": authority.case_id,
            "authority_fingerprint": authority.authority_fingerprint,
            "processed_data_sha256": authority.processed_data_sha256,
            "partition_fingerprint": authority.partition_fingerprint,
            "calibration_split_fingerprint": authority.calibration_split_fingerprint,
            "method_id": spec.method_id,
            "transfer_strategy": spec.strategy,
            "method_spec_fingerprint": spec.fingerprint,
            "encoder_id": spec.encoder_id,
            "encoder_seed": int(spec.encoder_seed),
            "encoder_state_fingerprint": state.fingerprint,
            "representation_sha256": state.representation_sha256,
        }

        if spec.strategy == "sourceweigher-mean" and budget == 0:
            rows.append(
                {
                    **common_identity,
                    "calibration_per_class": 0,
                    "status": "unavailable_no_target_observations",
                    "failure_reason": (
                        "target-dependent SourceWeigher requires declared target calibration "
                        "embeddings; final evaluation examples are forbidden"
                    ),
                }
            )
            continue

        train_embedding = state.source_embedding
        train_labels = state.y_encoded[state.source_indices]
        sample_weight: np.ndarray | None = None
        weighting_payload: dict[str, Any] | None = None

        if spec.strategy == "sourceweigher-mean":
            weighting = _sourceweigher_result(
                source_embeddings=state.source_embeddings_by_session,
                target_embeddings=calibration_embedding,
            )
            source_weights = _source_sample_weights(state.source_session, weighting.by_source())
            weighting_payload = weighting.to_dict()
        else:
            source_weights = np.ones(len(state.source_embedding), dtype=np.float64)

        if len(calibration_indices):
            train_embedding = np.concatenate(
                [state.source_embedding, calibration_embedding], axis=0
            )
            train_labels = np.concatenate(
                [train_labels, state.y_encoded[calibration_indices]], axis=0
            )
            sample_weight = np.concatenate(
                [source_weights, np.ones(len(calibration_indices), dtype=np.float64)]
            )
        elif spec.strategy == "sourceweigher-mean":
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
        probability = readout.predict_proba(state.evaluation_embedding)
        inference_s = time.perf_counter() - started
        metrics = _score(state.y_encoded[state.evaluation_indices], probability)

        rows.append(
            {
                **common_identity,
                "calibration_per_class": int(budget),
                "status": "ok",
                "failure_reason": None,
                "source_train_samples": int(len(state.source_indices)),
                "calibration_samples": int(len(calibration_indices)),
                "evaluation_samples": int(len(state.evaluation_indices)),
                "class_labels": list(state.class_labels),
                **metrics,
                "readout_fit_s": float(readout_fit_s),
                "inference_s": float(inference_s),
                "inference_ms_per_trial": float(
                    1000.0 * inference_s / max(len(state.evaluation_indices), 1)
                ),
                "sourceweigher": weighting_payload,
            }
        )

    return FrozenTransferCaseResult(
        authority_fingerprint=authority.authority_fingerprint,
        method_spec=spec,
        encoder_state_manifest=state.manifest(),
        rows=tuple(rows),
    )