"""Cross-validation and evaluation utilities for neurOS-compatible models.

scikit-learn is imported lazily so the kernel can be installed without the
evaluation extra when only real-time execution is required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EvaluatableModel(Protocol):
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


@dataclass
class CVResults:
    fold_scores: List[Dict[str, float]] = field(default_factory=list)
    mean_scores: Dict[str, float] = field(default_factory=dict)
    std_scores: Dict[str, float] = field(default_factory=dict)
    confusion_matrices: List[np.ndarray] = field(default_factory=list)
    predictions: List[np.ndarray] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["Cross-Validation Results", "=" * 50]
        for metric, mean_val in self.mean_scores.items():
            std_val = self.std_scores.get(metric, 0.0)
            lines.append(f"{metric:20s}: {mean_val:.4f} ± {std_val:.4f}")
        return "\n".join(lines)


def _sklearn_metrics():
    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Evaluation utilities require scikit-learn. Install neuros-core[evaluation]."
        ) from exc
    return {
        "accuracy_score": accuracy_score,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
    }


def cross_validate_model(
    model: EvaluatableModel,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int = 5,
    stratified: bool = True,
    shuffle: bool = True,
    random_state: Optional[int] = 42,
    metrics: Optional[List[str]] = None,
    return_predictions: bool = False,
) -> CVResults:
    try:
        from sklearn.model_selection import KFold, StratifiedKFold
    except ImportError as exc:
        raise RuntimeError(
            "Cross-validation requires scikit-learn. Install neuros-core[evaluation]."
        ) from exc
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1"]
    splitter = (
        StratifiedKFold(
            n_splits=n_folds,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
        if stratified
        else KFold(
            n_splits=n_folds,
            shuffle=shuffle,
            random_state=random_state if shuffle else None,
        )
    )
    impl = _sklearn_metrics()
    results = CVResults()
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        try:
            model.train(X_train, y_train)
            y_pred = model.predict(X_val)
        except Exception as exc:
            logger.warning("Fold %d failed: %s", fold_idx + 1, exc)
            continue
        fold_metrics = compute_metrics(y_val, y_pred, metrics=metrics)
        results.fold_scores.append(fold_metrics)
        results.confusion_matrices.append(impl["confusion_matrix"](y_val, y_pred))
        if return_predictions:
            results.predictions.append(y_pred)
    for metric in {key for fold in results.fold_scores for key in fold.keys()}:
        values = [fold[metric] for fold in results.fold_scores if metric in fold]
        if values:
            results.mean_scores[metric] = float(np.mean(values))
            results.std_scores[metric] = float(np.std(values))
    return results


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metrics: Optional[List[str]] = None,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    impl = _sklearn_metrics()
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1"]
    results: Dict[str, float] = {}
    n_classes = len(np.unique(y_true))
    average = "binary" if n_classes == 2 else "macro"
    for metric in metrics:
        try:
            if metric == "accuracy":
                results[metric] = float(impl["accuracy_score"](y_true, y_pred))
            elif metric == "precision":
                results[metric] = float(
                    impl["precision_score"](y_true, y_pred, average=average, zero_division=0)
                )
            elif metric == "recall":
                results[metric] = float(
                    impl["recall_score"](y_true, y_pred, average=average, zero_division=0)
                )
            elif metric == "f1":
                results[metric] = float(
                    impl["f1_score"](y_true, y_pred, average=average, zero_division=0)
                )
            elif metric == "roc_auc" and y_prob is not None:
                if n_classes == 2:
                    results[metric] = float(impl["roc_auc_score"](y_true, y_prob[:, 1]))
                else:
                    results[metric] = float(
                        impl["roc_auc_score"](y_true, y_prob, multi_class="ovr", average=average)
                    )
            else:
                logger.warning("Metric '%s' not recognized or requires y_prob", metric)
        except Exception as exc:
            logger.warning("Failed to compute %s: %s", metric, exc)
    return results


def stratified_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError(
            "Train/test splitting requires scikit-learn. Install neuros-core[evaluation]."
        ) from exc
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def evaluate_model(
    model: EvaluatableModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    metrics: Optional[List[str]] = None,
    return_report: bool = False,
) -> Dict[str, Any]:
    impl = _sklearn_metrics()
    y_pred = model.predict(X_test)
    y_prob = None
    predict_proba = getattr(model, "predict_proba", None)
    if callable(predict_proba):
        try:
            y_prob = predict_proba(X_test)
        except (AttributeError, NotImplementedError):
            y_prob = None
    results: Dict[str, Any] = compute_metrics(y_test, y_pred, metrics=metrics, y_prob=y_prob)
    results["confusion_matrix"] = impl["confusion_matrix"](y_test, y_pred).tolist()
    if return_report:
        results["classification_report"] = impl["classification_report"](
            y_test, y_pred, output_dict=True
        )
    return results


def nested_cross_validation(
    model: EvaluatableModel,
    X: np.ndarray,
    y: np.ndarray,
    *,
    outer_folds: int = 5,
    inner_folds: int = 3,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    stratified: bool = True,
    random_state: Optional[int] = 42,
) -> CVResults:
    logger.info(
        "Nested CV outer_folds=%d, inner_folds=%d; hyperparameter search is not yet enabled",
        outer_folds,
        inner_folds,
    )
    return cross_validate_model(
        model,
        X,
        y,
        n_folds=outer_folds,
        stratified=stratified,
        random_state=random_state,
    )
