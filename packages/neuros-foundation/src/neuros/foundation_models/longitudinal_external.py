"""Paired longitudinal evidence for optional external decoder ecosystems.

External model ecosystems may participate in neurOS evidence only after a
:class:`LongitudinalCaseAuthority` has frozen the source history, target
calibration pool, and final evaluation examples. This module deliberately
reports task performance, calibration, runtime, and learned-state identity only.
It does not invent representation or mechanistic surfaces for an upstream model
that has not qualified those capabilities.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from neuros.models import BraindecodeDecoder

from .longitudinal_authority import LongitudinalCaseAuthority
from .longitudinal_methods import (
    TaskDecoderCaseResult,
    _encode_labels,
    _model_state_sha256,
    _score,
)
from .real_world import GroupedEvaluationData

ExternalTaskDecoderId = Literal["braindecode-eegnet"]


@dataclass(frozen=True, slots=True)
class ExternalTaskDecoderMethodSpec:
    """Frozen identity/configuration for one optional external task decoder."""

    method_id: ExternalTaskDecoderId
    model_seed: int
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.method_id != "braindecode-eegnet":
            raise ValueError(f"unsupported external task decoder {self.method_id!r}")
        forbidden = {
            "model_name",
            "n_channels",
            "n_times",
            "n_classes",
            "sample_rate_hz",
            "random_state",
        }.intersection(self.model_kwargs)
        if forbidden:
            raise ValueError(
                "model identity, geometry, sample-rate provenance and random_state are "
                f"controlled by the evidence runner; remove overrides {sorted(forbidden)}"
            )
        options = dict(self.model_kwargs)
        try:
            json.dumps(options, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("model_kwargs must be finite JSON-serializable evidence metadata") from exc
        object.__setattr__(self, "model_kwargs", MappingProxyType(options))

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "model_seed": int(self.model_seed),
            "model_kwargs": dict(self.model_kwargs),
        }


@dataclass(frozen=True, slots=True)
class ExternalTaskDecoderCaseResult:
    """One external method/seed evaluated across one frozen authority frontier."""

    authority_fingerprint: str
    method_spec: ExternalTaskDecoderMethodSpec
    rows: tuple[Mapping[str, Any], ...]
    resolved_model_config: Mapping[str, Any]
    parameter_count: int
    analysis_manifest_fingerprint: str
    upstream_version: str | None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.parameter_count <= 0:
            raise ValueError("parameter_count must be positive")
        object.__setattr__(self, "rows", tuple(MappingProxyType(dict(row)) for row in self.rows))
        object.__setattr__(
            self,
            "resolved_model_config",
            MappingProxyType(dict(self.resolved_model_config)),
        )

    @property
    def method_run_fingerprint(self) -> str:
        raw = json.dumps(
            self.to_dict(include_fingerprint=False),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "authority_fingerprint": self.authority_fingerprint,
            "method_spec": self.method_spec.to_dict(),
            "method_spec_fingerprint": self.method_spec.fingerprint,
            "resolved_model_config": dict(self.resolved_model_config),
            "parameter_count": int(self.parameter_count),
            "analysis_manifest_fingerprint": self.analysis_manifest_fingerprint,
            "upstream_version": self.upstream_version,
            "rows": [dict(row) for row in self.rows],
        }
        if include_fingerprint:
            payload["method_run_fingerprint"] = self.method_run_fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class PairedTaskPerformanceResult:
    """Paired native/external task evidence under one immutable authority."""

    authority_fingerprint: str
    native_method_run_fingerprint: str
    external_method_run_fingerprint: str
    rows: tuple[Mapping[str, Any], ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("paired task evidence must contain at least one budget row")
        object.__setattr__(self, "rows", tuple(MappingProxyType(dict(row)) for row in self.rows))

    @property
    def pair_fingerprint(self) -> str:
        raw = json.dumps(
            self.to_dict(include_fingerprint=False),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "authority_fingerprint": self.authority_fingerprint,
            "native_method_run_fingerprint": self.native_method_run_fingerprint,
            "external_method_run_fingerprint": self.external_method_run_fingerprint,
            "rows": [dict(row) for row in self.rows],
        }
        if include_fingerprint:
            payload["pair_fingerprint"] = self.pair_fingerprint
        return payload


def _external_model_for(
    spec: ExternalTaskDecoderMethodSpec,
    *,
    n_channels: int,
    n_times: int,
    n_classes: int,
) -> BraindecodeDecoder:
    if spec.method_id != "braindecode-eegnet":
        raise ValueError(f"unsupported external task decoder {spec.method_id!r}")
    return BraindecodeDecoder(
        "EEGNet",
        n_channels=int(n_channels),
        n_times=int(n_times),
        n_classes=int(n_classes),
        random_state=int(spec.model_seed),
        **dict(spec.model_kwargs),
    )


def run_external_task_decoder_case(
    data: GroupedEvaluationData,
    authority: LongitudinalCaseAuthority,
    *,
    spec: ExternalTaskDecoderMethodSpec,
    budgets_per_class: Sequence[int],
) -> ExternalTaskDecoderCaseResult:
    """Evaluate one external decoder under a previously frozen sample authority.

    A fresh upstream model is trained at every declared calibration budget. The
    adapter receives exactly the processed ``X=(sample, channel, time)`` array
    governed by ``authority``. It cannot choose samples, preprocess the final
    evaluation set, or silently change window geometry.
    """

    split = authority.restore(data)
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
        raise ValueError("external task decoder evidence expects X=(sample, channel, time)")
    if not np.isfinite(X).all():
        raise ValueError("external task decoder evidence refuses non-finite processed inputs")

    y_encoded, class_labels = _encode_labels(np.asarray(data.y), split.source_train_indices)
    n_channels = int(X.shape[1])
    n_times = int(X.shape[2])
    n_classes = len(class_labels)

    rows: list[dict[str, Any]] = []
    resolved: dict[str, Any] | None = None
    config_fingerprint: str | None = None
    parameters: int | None = None
    manifest_fingerprint: str | None = None
    upstream_version: str | None = None

    for budget in budgets:
        train_indices = split.train_indices_for_budget(budget)
        calibration_indices = split.calibration_indices(budget)
        evaluation_indices = split.evaluation_indices

        model = _external_model_for(
            spec,
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
        )
        current_resolved = model.configuration()
        current_config_fingerprint = model.configuration_fingerprint
        current_manifest = model.analysis_manifest().fingerprint()
        if resolved is None:
            resolved = current_resolved
            config_fingerprint = current_config_fingerprint
            manifest_fingerprint = current_manifest
        elif (
            current_resolved != resolved
            or current_config_fingerprint != config_fingerprint
            or current_manifest != manifest_fingerprint
        ):
            raise RuntimeError("external method identity changed across calibration budgets")

        started = time.perf_counter()
        model.train(X[train_indices], y_encoded[train_indices])
        fit_s = time.perf_counter() - started

        current_parameters = int(
            sum(parameter.numel() for parameter in model.analysis_model().parameters())
        )
        if parameters is None:
            parameters = current_parameters
            upstream_version = model.model_version
        elif current_parameters != parameters or model.model_version != upstream_version:
            raise RuntimeError("external learned architecture/version changed across budgets")

        state_sha256 = _model_state_sha256(model)
        started = time.perf_counter()
        probability = model.predict_proba(X[evaluation_indices])
        inference_s = time.perf_counter() - started
        metrics = _score(y_encoded[evaluation_indices], probability)

        rows.append(
            {
                "case_id": authority.case_id,
                "authority_fingerprint": authority.authority_fingerprint,
                "processed_data_sha256": authority.processed_data_sha256,
                "partition_fingerprint": authority.partition_fingerprint,
                "calibration_split_fingerprint": authority.calibration_split_fingerprint,
                "method_id": spec.method_id,
                "method_spec_fingerprint": spec.fingerprint,
                "adapter_config_fingerprint": current_config_fingerprint,
                "model_seed": int(spec.model_seed),
                "model_state_sha256": state_sha256,
                "upstream_model_version": model.model_version,
                "calibration_per_class": int(budget),
                "source_train_samples": int(len(split.source_train_indices)),
                "calibration_samples": int(len(calibration_indices)),
                "evaluation_samples": int(len(evaluation_indices)),
                "fit_samples": int(len(train_indices)),
                "class_labels": list(class_labels),
                **metrics,
                "fit_s": float(fit_s),
                "inference_s": float(inference_s),
                "inference_ms_per_trial": float(
                    1000.0 * inference_s / max(len(evaluation_indices), 1)
                ),
                "representation_evidence_available": False,
                "mechanistic_evidence_available": False,
            }
        )

    assert resolved is not None
    assert config_fingerprint is not None
    assert parameters is not None
    assert manifest_fingerprint is not None
    return ExternalTaskDecoderCaseResult(
        authority_fingerprint=authority.authority_fingerprint,
        method_spec=spec,
        rows=tuple(rows),
        resolved_model_config=resolved,
        parameter_count=parameters,
        analysis_manifest_fingerprint=manifest_fingerprint,
        upstream_version=upstream_version,
    )


def pair_task_performance(
    native: TaskDecoderCaseResult,
    external: ExternalTaskDecoderCaseResult,
) -> PairedTaskPerformanceResult:
    """Pair native and external rows only when their evidence identity is exact."""

    if native.authority_fingerprint != external.authority_fingerprint:
        raise ValueError("native and external results do not share one authority fingerprint")

    native_by_budget = {int(row["calibration_per_class"]): row for row in native.rows}
    external_by_budget = {int(row["calibration_per_class"]): row for row in external.rows}
    if set(native_by_budget) != set(external_by_budget):
        raise ValueError("native and external results must cover identical calibration budgets")

    identity_keys = (
        "case_id",
        "authority_fingerprint",
        "processed_data_sha256",
        "partition_fingerprint",
        "calibration_split_fingerprint",
        "source_train_samples",
        "calibration_samples",
        "evaluation_samples",
        "fit_samples",
        "class_labels",
    )
    score_metrics = (
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "brier_score",
        "expected_calibration_error",
    )

    rows: list[dict[str, Any]] = []
    for budget in sorted(native_by_budget):
        native_row = native_by_budget[budget]
        external_row = external_by_budget[budget]
        for key in identity_keys:
            if native_row.get(key) != external_row.get(key):
                raise ValueError(
                    f"paired evidence identity mismatch at budget={budget}: {key}"
                )

        row: dict[str, Any] = {
            key: native_row.get(key) for key in identity_keys
        }
        row.update(
            {
                "calibration_per_class": budget,
                "native_method_id": native_row["method_id"],
                "external_method_id": external_row["method_id"],
                "native_method_spec_fingerprint": native_row["method_spec_fingerprint"],
                "external_method_spec_fingerprint": external_row["method_spec_fingerprint"],
                "native_model_state_sha256": native_row["model_state_sha256"],
                "external_model_state_sha256": external_row["model_state_sha256"],
                "external_upstream_model_version": external_row["upstream_model_version"],
                "external_representation_evidence_available": bool(
                    external_row["representation_evidence_available"]
                ),
                "external_mechanistic_evidence_available": bool(
                    external_row["mechanistic_evidence_available"]
                ),
            }
        )
        for metric in score_metrics:
            native_value = native_row.get(metric)
            external_value = external_row.get(metric)
            row[f"native_{metric}"] = native_value
            row[f"external_{metric}"] = external_value
            if native_value is None or external_value is None:
                row[f"delta_external_minus_native_{metric}"] = None
            else:
                row[f"delta_external_minus_native_{metric}"] = float(external_value) - float(
                    native_value
                )

        native_fit = float(native_row["fit_s"])
        external_fit = float(external_row["fit_s"])
        native_inference = float(native_row["inference_ms_per_trial"])
        external_inference = float(external_row["inference_ms_per_trial"])
        row.update(
            {
                "native_fit_s": native_fit,
                "external_fit_s": external_fit,
                "fit_ratio_external_over_native": (
                    None if native_fit <= 0 else external_fit / native_fit
                ),
                "native_inference_ms_per_trial": native_inference,
                "external_inference_ms_per_trial": external_inference,
                "inference_ratio_external_over_native": (
                    None if native_inference <= 0 else external_inference / native_inference
                ),
            }
        )
        rows.append(row)

    return PairedTaskPerformanceResult(
        authority_fingerprint=native.authority_fingerprint,
        native_method_run_fingerprint=native.method_run_fingerprint,
        external_method_run_fingerprint=external.method_run_fingerprint,
        rows=tuple(rows),
    )
