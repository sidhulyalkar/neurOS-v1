"""
Cross-validation and evaluation utilities for neurOS models.

This module provides tools for rigorous model evaluation including k-fold
cross-validation, stratified splits, and comprehensive performance metrics
for BCI classification tasks.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from .models.base_model import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class CVResults:
    """Results from cross-validation evaluation.

    Parameters
    ----------
    fold_scores : list of dict
        Per-fold metrics for each evaluation fold.
    mean_scores : dict
        Mean values across all folds for each metric.
    std_scores : dict
        Standard deviation across folds for each metric.
    confusion_matrices : list of np.ndarray, optional
        Confusion matrix for each fold.
    predictions : list of np.ndarray, optional
        Predictions for each fold (useful for ensembling).
    """

    fold_scores: List[Dict[str, float]] = field(default_factory=list)
    mean_scores: Dict[str, float] = field(default_factory=dict)
    std_scores: Dict[str, float] = field(default_factory=dict)
    confusion_matrices: List[np.ndarray] = field(default_factory=list)
    predictions: List[np.ndarray] = field(default_factory=list)

    def summary(self) -> str:
        """Generate a human-readable summary of CV results."""
        lines = ["Cross-Validation Results", "=" * 50]
        for metric, mean_val in self.mean_scores.items():
            std_val = self.std_scores.get(metric, 0.0)
            lines.append(f"{metric:20s}: {mean_val:.4f} ± {std_val:.4f}")
        return "\n".join(lines)


def cross_validate_model(
    model: BaseModel,
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
    """Perform k-fold cross-validation on a neurOS model.

    Parameters
    ----------
    model : BaseModel
        The model to evaluate (must implement train/predict interface).
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features) or (n_samples, n_channels, n_timepoints).
    y : np.ndarray
        Target labels of shape (n_samples,).
    n_folds : int, default=5
        Number of cross-validation folds.
    stratified : bool, default=True
        Whether to preserve class distribution in each fold.
    shuffle : bool, default=True
        Whether to shuffle data before splitting.
    random_state : int, optional, default=42
        Random seed for reproducibility.
    metrics : list of str, optional
        Metrics to compute. Default: ['accuracy', 'precision', 'recall', 'f1'].
    return_predictions : bool, default=False
        Whether to return per-fold predictions.

    Returns
    -------
    CVResults
        Cross-validation results including per-fold and aggregate metrics.

    Examples
    --------
    >>> from neuros.models import EEGNetModel
    >>> from neuros.evaluation import cross_validate_model
    >>> model = EEGNetModel(n_channels=8, n_classes=2)
    >>> X = np.random.randn(100, 8, 200)  # 100 samples, 8 channels, 200 timepoints
    >>> y = np.random.randint(0, 2, 100)
    >>> results = cross_validate_model(model, X, y, n_folds=5)
    >>> print(results.summary())
    """
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1"]

    # Select cross-validation strategy
    if stratified:
        cv_splitter = StratifiedKFold(
            n_splits=n_folds, shuffle=shuffle, random_state=random_state if shuffle else None
        )
    else:
        cv_splitter = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state if shuffle else None)

    results = CVResults()

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model on this fold
        try:
            model.train(X_train, y_train)
        except Exception as e:
            logger.warning(f"Training failed on fold {fold_idx + 1}: {e}")
            continue

        # Predict on validation set
        try:
            y_pred = model.predict(X_val)
        except Exception as e:
            logger.warning(f"Prediction failed on fold {fold_idx + 1}: {e}")
            continue

        # Compute metrics for this fold
        fold_metrics = compute_metrics(y_val, y_pred, metrics=metrics)
        results.fold_scores.append(fold_metrics)

        # Store confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        results.confusion_matrices.append(cm)

        # Optionally store predictions
        if return_predictions:
            results.predictions.append(y_pred)

        logger.info(
            f"Fold {fold_idx + 1}/{n_folds}: accuracy={fold_metrics.get('accuracy', 0):.4f}"
        )

    # Aggregate results across folds
    if results.fold_scores:
        all_metrics = set()
        for fold_score in results.fold_scores:
            all_metrics.update(fold_score.keys())

        for metric in all_metrics:
            values = [
                fold[metric] for fold in results.fold_scores if metric in fold
            ]
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
    """Compute classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    metrics : list of str, optional
        Metrics to compute. Default: ['accuracy', 'precision', 'recall', 'f1'].
    y_prob : np.ndarray, optional
        Predicted probabilities (required for 'roc_auc').

    Returns
    -------
    dict
        Dictionary mapping metric names to values.

    Examples
    --------
    >>> y_true = np.array([0, 1, 0, 1, 1])
    >>> y_pred = np.array([0, 1, 0, 0, 1])
    >>> metrics = compute_metrics(y_true, y_pred)
    >>> print(f"Accuracy: {metrics['accuracy']:.2f}")
    """
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1"]

    results = {}
    n_classes = len(np.unique(y_true))
    average = "binary" if n_classes == 2 else "macro"

    for metric in metrics:
        try:
            if metric == "accuracy":
                results[metric] = accuracy_score(y_true, y_pred)
            elif metric == "precision":
                results[metric] = precision_score(
                    y_true, y_pred, average=average, zero_division=0
                )
            elif metric == "recall":
                results[metric] = recall_score(
                    y_true, y_pred, average=average, zero_division=0
                )
            elif metric == "f1":
                results[metric] = f1_score(
                    y_true, y_pred, average=average, zero_division=0
                )
            elif metric == "roc_auc" and y_prob is not None:
                if n_classes == 2:
                    # Binary classification: use probability of positive class
                    results[metric] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    # Multi-class: use one-vs-rest
                    results[metric] = roc_auc_score(
                        y_true, y_prob, multi_class="ovr", average=average
                    )
            else:
                logger.warning(f"Metric '{metric}' not recognized or requires y_prob")
        except Exception as e:
            logger.warning(f"Failed to compute {metric}: {e}")

    return results


def stratified_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/test sets while preserving class distribution.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    test_size : float, default=0.2
        Proportion of data to use for testing (0.0 to 1.0).
    random_state : int, optional, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_train : np.ndarray
        Training features.
    X_test : np.ndarray
        Test features.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Test labels.

    Examples
    --------
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randint(0, 2, 100)
    >>> X_train, X_test, y_train, y_test = stratified_train_test_split(X, y)
    >>> print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    """
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def evaluate_model(
    model: BaseModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    metrics: Optional[List[str]] = None,
    return_report: bool = False,
) -> Dict[str, Any]:
    """Evaluate a trained model on a test set.

    Parameters
    ----------
    model : BaseModel
        Trained model to evaluate.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    metrics : list of str, optional
        Metrics to compute.
    return_report : bool, default=False
        Whether to include a detailed classification report.

    Returns
    -------
    dict
        Evaluation results including metrics and optionally a classification report.

    Examples
    --------
    >>> from neuros.models import SimpleClassifier
    >>> model = SimpleClassifier()
    >>> X_train = np.random.randn(100, 10)
    >>> y_train = np.random.randint(0, 2, 100)
    >>> model.train(X_train, y_train)
    >>> X_test = np.random.randn(20, 10)
    >>> y_test = np.random.randint(0, 2, 20)
    >>> results = evaluate_model(model, X_test, y_test, return_report=True)
    >>> print(f"Test accuracy: {results['accuracy']:.2f}")
    """
    y_pred = model.predict(X_test)

    # Get probabilities if available
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
        except Exception:
            pass

    # Compute metrics
    results = compute_metrics(y_test, y_pred, metrics=metrics, y_prob=y_prob)

    # Add confusion matrix
    results["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

    # Optionally add detailed classification report
    if return_report:
        report = classification_report(y_test, y_pred, output_dict=True)
        results["classification_report"] = report

    return results


def nested_cross_validation(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    *,
    outer_folds: int = 5,
    inner_folds: int = 3,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    stratified: bool = True,
    random_state: Optional[int] = 42,
) -> CVResults:
    """Perform nested cross-validation for unbiased performance estimation.

    Nested CV uses an outer loop for performance estimation and an inner loop
    for hyperparameter tuning, providing an unbiased estimate of generalization
    performance.

    Parameters
    ----------
    model : BaseModel
        The model to evaluate.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    outer_folds : int, default=5
        Number of outer CV folds.
    inner_folds : int, default=3
        Number of inner CV folds for hyperparameter tuning.
    param_grid : dict, optional
        Hyperparameter grid for tuning (not yet implemented).
    stratified : bool, default=True
        Whether to use stratified splits.
    random_state : int, optional, default=42
        Random seed.

    Returns
    -------
    CVResults
        Nested cross-validation results.

    Notes
    -----
    This is a simplified implementation. Full hyperparameter tuning support
    will be added in future versions via integration with Optuna.
    """
    # For now, just perform standard CV on outer loop
    # TODO: Add inner loop hyperparameter optimization
    logger.info(
        "Nested CV: outer_folds=%d, inner_folds=%d (inner loop not yet implemented)",
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
