"""Task-decoder methods for replayable longitudinal evidence cases.

Every method in this module receives a :class:`LongitudinalCaseAuthority` and
must restore it before fitting. The method therefore has no authority to choose
source, calibration, or final-evaluation examples.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from neuros.models import EEGConformerModel, EEGNetModel

from .longitudinal_authority import LongitudinalCaseAuthority
from .probes import representation_report
from .real_world import GroupedEvaluationData

TaskDecoderId = Literal["eegnet", "eeg-conformer"]


@dataclass(frozen=True, slots=True)
class TaskDecoderMethodSpec:
    """Frozen identity/configuration for one supervised task-decoder method."""

    method_id: TaskDecoderId
    model_seed: int
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.method_id not in {"eegnet", "eeg-conformer"}:
            raise ValueError(f"unsupported task decoder {self.method_id!r}")
        forbidden = {"n_channels", "n_classes", "random_state"}.intersection(self.model_kwargs)
        if forbidden:
            raise ValueError(
                "n_channels, n_classes and random_state are controlled by the evidence runner; "
                f"remove overrides {sorted(forbidden)}"
            )
        object.__setattr__(self, "model_kwargs", MappingProxyType(dict(self.model_kwargs)))

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "model_seed": int(self.model_seed),
            "model_kwargs": dict(self.model_kwargs),
        }


@dataclass(frozen=True, slots=True)
class TaskDecoderCaseResult:
    """One method/seed evaluated across calibration budgets for one authority."""

    authority_fingerprint: str
    method_spec: TaskDecoderMethodSpec
    rows: tuple[Mapping[str, Any], ...]
    resolved_model_config: Mapping[str, Any]
    parameter_count: int
    analysis_manifest_fingerprint: str
    schema_version: int = 2

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
        payload = self.to_dict(include_fingerprint=False)
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "authority_fingerprint": self.authority_fingerprint,
            "method_spec": self.method_spec.to_dict(),
            "method_spec_fingerprint": self.method_spec.fingerprint,
            "resolved_model_config": dict(self.resolved_model_config),
            "parameter_count": int(self.parameter_count),
            "analysis_manifest_fingerprint": self.analysis_manifest_fingerprint,
            "rows": [dict(row) for row in self.rows],
        }
        if include_fingerprint:
            payload["method_run_fingerprint"] = self.method_run_fingerprint
        return payload


def _model_for(
    spec: TaskDecoderMethodSpec,
    *,
    n_channels: int,
    n_classes: int,
):
    kwargs = dict(spec.model_kwargs)
    kwargs.update(
        {
            "n_channels": int(n_channels),
            "n_classes": int(n_classes),
            "random_state": int(spec.model_seed),
        }
    )
    if spec.method_id == "eegnet":
        return EEGNetModel(**kwargs)
    if spec.method_id == "eeg-conformer":
        return EEGConformerModel(**kwargs)
    raise ValueError(f"unsupported task decoder {spec.method_id!r}")


def _resolved_config(model: Any) -> dict[str, Any]:
    common = (
        "n_channels",
        "n_classes",
        "learning_rate",
        "weight_decay",
        "n_epochs",
        "batch_size",
        "device_spec",
        "random_state",
    )
    architecture = (
        "temporal_filters",
        "depth_multiplier",
        "separable_filters",
        "temporal_kernel",
        "separable_kernel",
        "embedding_dim",
        "pool_length",
        "pool_stride",
        "n_heads",
        "n_layers",
        "feedforward_multiplier",
        "dropout",
    )
    result: dict[str, Any] = {}
    for name in common + architecture:
        if hasattr(model, name):
            value = getattr(model, name)
            if isinstance(value, np.generic):
                value = value.item()
            result[name] = value
    return result


def _parameter_count(model: Any) -> int:
    module = model.analysis_model()
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _model_state_sha256(model: Any) -> str:
    """Hash the actual learned PyTorch state, including registered buffers.

    Config/seed identity is not sufficient evidence of learned-state identity on
    every accelerator. The state hash includes every ``state_dict`` tensor name,
    dtype, shape and raw CPU bytes in sorted-key order, including BatchNorm
    running statistics and other registered buffers.
    """
    module = model.analysis_model()
    state = module.state_dict()
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _encode_labels(y: np.ndarray, source_indices: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    labels = tuple(sorted(np.unique(np.asarray(y)[source_indices].astype(str)).tolist()))
    if len(labels) < 2:
        raise ValueError("source history must contain at least two task classes")
    mapping = {label: index for index, label in enumerate(labels)}
    all_labels = np.asarray(y).astype(str)
    unknown = sorted(set(all_labels.tolist()) - set(mapping))
    if unknown:
        raise ValueError(
            "loaded data contains labels absent from source-history class vocabulary: "
            f"{unknown}"
        )
    encoded = np.asarray([mapping[value] for value in all_labels], dtype=np.int64)
    return encoded, labels


def _multiclass_brier(y_true: np.ndarray, probability: np.ndarray, n_classes: int) -> float:
    one_hot = np.eye(n_classes, dtype=np.float64)[np.asarray(y_true, dtype=np.int64)]
    return float(np.mean(np.sum((np.asarray(probability) - one_hot) ** 2, axis=1)))


def _expected_calibration_error(
    y_true: np.ndarray,
    probability: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    probs = np.asarray(probability, dtype=np.float64)
    prediction = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = (prediction == np.asarray(y_true)).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for index in range(n_bins):
        if index == n_bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if not np.any(mask):
            continue
        weight = float(np.mean(mask))
        error += weight * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(error)


def _score(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float | None]:
    probs = np.asarray(probability, dtype=np.float64)
    prediction = probs.argmax(axis=1)
    y = np.asarray(y_true, dtype=np.int64)
    n_classes = probs.shape[1]
    roc_auc: float | None = None
    if n_classes == 2 and len(np.unique(y)) == 2:
        roc_auc = float(roc_auc_score(y, probs[:, 1]))
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "roc_auc": roc_auc,
        "brier_score": _multiclass_brier(y, probs, n_classes),
        "expected_calibration_error": _expected_calibration_error(y, probs),
    }


def run_task_decoder_case(
    data: GroupedEvaluationData,
    authority: LongitudinalCaseAuthority,
    *,
    spec: TaskDecoderMethodSpec,
    budgets_per_class: Sequence[int],
) -> TaskDecoderCaseResult:
    """Evaluate one task decoder under a previously frozen sample authority.

    A fresh model is trained at every calibration budget. No final evaluation
    example is used for fitting, preprocessing, adaptation, or model selection.
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
        raise ValueError("task decoder ladder expects X=(sample, channel, time)")
    y_encoded, class_labels = _encode_labels(np.asarray(data.y), split.source_train_indices)
    n_channels = int(X.shape[1])
    n_classes = len(class_labels)

    rows: list[dict[str, Any]] = []
    resolved: dict[str, Any] | None = None
    parameters: int | None = None
    manifest_fingerprint: str | None = None

    for budget in budgets:
        train_indices = split.train_indices_for_budget(budget)
        calibration_indices = split.calibration_indices(budget)
        evaluation_indices = split.evaluation_indices

        model = _model_for(
            spec,
            n_channels=n_channels,
            n_classes=n_classes,
        )
        current_resolved = _resolved_config(model)
        current_parameters = _parameter_count(model)
        current_manifest = model.analysis_manifest().fingerprint()
        if resolved is None:
            resolved = current_resolved
            parameters = current_parameters
            manifest_fingerprint = current_manifest
        elif (
            current_resolved != resolved
            or current_parameters != parameters
            or current_manifest != manifest_fingerprint
        ):
            raise RuntimeError("method identity changed across calibration budgets")

        started = time.perf_counter()
        model.train(X[train_indices], y_encoded[train_indices])
        fit_s = time.perf_counter() - started
        model_state_sha256 = _model_state_sha256(model)

        started = time.perf_counter()
        probability = model.predict_proba(X[evaluation_indices])
        inference_s = time.perf_counter() - started
        metrics = _score(y_encoded[evaluation_indices], probability)

        eval_embedding = model.encode(X[evaluation_indices])
        source_embedding = model.encode(X[split.source_train_indices])
        eval_report = representation_report(eval_embedding)
        source_report = representation_report(source_embedding)
        final_training = model.training_history[-1] if model.training_history else {}

        rows.append(
            {
                "case_id": authority.case_id,
                "authority_fingerprint": authority.authority_fingerprint,
                "processed_data_sha256": authority.processed_data_sha256,
                "partition_fingerprint": authority.partition_fingerprint,
                "calibration_split_fingerprint": authority.calibration_split_fingerprint,
                "method_id": spec.method_id,
                "method_spec_fingerprint": spec.fingerprint,
                "model_seed": int(spec.model_seed),
                "model_state_sha256": model_state_sha256,
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
                "training_final_loss": (
                    None if "loss" not in final_training else float(final_training["loss"])
                ),
                "training_final_accuracy": (
                    None
                    if "accuracy" not in final_training
                    else float(final_training["accuracy"])
                ),
                "evaluation_representation": eval_report,
                "source_representation": source_report,
            }
        )

    assert resolved is not None and parameters is not None and manifest_fingerprint is not None
    return TaskDecoderCaseResult(
        authority_fingerprint=authority.authority_fingerprint,
        method_spec=spec,
        rows=tuple(rows),
        resolved_model_config=resolved,
        parameter_count=parameters,
        analysis_manifest_fingerprint=manifest_fingerprint,
    )