"""Transparent CSP+LDA baseline under a frozen longitudinal sample authority."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from .longitudinal_authority import LongitudinalCaseAuthority
from .real_world import GroupedEvaluationData


@dataclass(frozen=True, slots=True)
class CSPCaseResult:
    authority_fingerprint: str
    csp_components: int
    rows: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "authority_fingerprint": self.authority_fingerprint,
            "method_id": "csp-lda",
            "csp_components": int(self.csp_components),
            "rows": [dict(row) for row in self.rows],
        }


def _build_csp_lda(components: int):
    try:
        from mne.decoding import CSP
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:  # pragma: no cover
        raise ImportError("CSP+LDA requires MNE and scikit-learn") from exc
    return make_pipeline(CSP(n_components=int(components), reg=None), LDA())


def _encode_labels(y: np.ndarray, source_indices: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    values = np.asarray(y).astype(str)
    labels = tuple(sorted(np.unique(values[source_indices]).tolist()))
    if len(labels) < 2:
        raise ValueError("source history must contain at least two task classes")
    mapping = {label: index for index, label in enumerate(labels)}
    unknown = sorted(set(values.tolist()) - set(mapping))
    if unknown:
        raise ValueError(f"target contains labels absent from source history: {unknown}")
    return np.asarray([mapping[value] for value in values], dtype=np.int64), labels


def run_csp_case(
    data: GroupedEvaluationData,
    authority: LongitudinalCaseAuthority,
    *,
    budgets_per_class: Sequence[int],
    csp_components: int = 8,
) -> CSPCaseResult:
    if csp_components <= 0:
        raise ValueError("csp_components must be positive")
    split = authority.restore(data)
    budgets = tuple(sorted(set(int(value) for value in budgets_per_class)))
    if not budgets or budgets[0] < 0:
        raise ValueError("budgets_per_class must contain non-negative values")
    if budgets[-1] > split.max_budget_per_class:
        raise ValueError(
            f"requested budget {budgets[-1]} exceeds authority maximum "
            f"{split.max_budget_per_class}/class"
        )

    X = np.asarray(data.X)
    y, class_labels = _encode_labels(np.asarray(data.y), split.source_train_indices)
    rows = []
    for budget in budgets:
        train = split.train_indices_for_budget(budget)
        calibration = split.calibration_indices(budget)
        evaluation = split.evaluation_indices
        model = _build_csp_lda(csp_components)

        started = time.perf_counter()
        model.fit(X[train], y[train])
        fit_s = time.perf_counter() - started
        started = time.perf_counter()
        prediction = np.asarray(model.predict(X[evaluation]), dtype=np.int64)
        probability = np.asarray(model.predict_proba(X[evaluation]), dtype=np.float64)
        inference_s = time.perf_counter() - started
        roc_auc = None
        if probability.shape[1] == 2 and len(np.unique(y[evaluation])) == 2:
            roc_auc = float(roc_auc_score(y[evaluation], probability[:, 1]))

        rows.append(
            {
                "case_id": authority.case_id,
                "authority_fingerprint": authority.authority_fingerprint,
                "processed_data_sha256": authority.processed_data_sha256,
                "partition_fingerprint": authority.partition_fingerprint,
                "calibration_split_fingerprint": authority.calibration_split_fingerprint,
                "method_id": "csp-lda",
                "model_seed": None,
                "calibration_per_class": int(budget),
                "status": "ok",
                "failure_reason": None,
                "source_train_samples": int(len(split.source_train_indices)),
                "calibration_samples": int(len(calibration)),
                "evaluation_samples": int(len(evaluation)),
                "fit_samples": int(len(train)),
                "class_labels": list(class_labels),
                "accuracy": float(accuracy_score(y[evaluation], prediction)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y[evaluation], prediction)
                ),
                "roc_auc": roc_auc,
                "fit_s": float(fit_s),
                "inference_s": float(inference_s),
                "inference_ms_per_trial": float(
                    1000.0 * inference_s / max(len(evaluation), 1)
                ),
            }
        )

    return CSPCaseResult(
        authority_fingerprint=authority.authority_fingerprint,
        csp_components=int(csp_components),
        rows=tuple(rows),
    )
