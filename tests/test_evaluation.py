"""
Tests for cross-validation and evaluation utilities.
"""

import numpy as np
import pytest

from neuros.evaluation import (
    cross_validate_model,
    compute_metrics,
    stratified_train_test_split,
    evaluate_model,
    CVResults,
)
from neuros.models import SimpleClassifier, EEGNetModel


class TestComputeMetrics:
    """Test metric computation functions."""

    def test_compute_metrics_binary_classification(self):
        """Test metrics for binary classification."""
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])

        metrics = compute_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_compute_metrics_multiclass(self):
        """Test metrics for multi-class classification."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 2, 2, 1, 1])

        metrics = compute_metrics(y_true, y_pred, metrics=["accuracy", "f1"])

        assert "accuracy" in metrics
        assert "f1" in metrics
        assert metrics["accuracy"] == 0.75  # 6/8 correct

    def test_compute_metrics_with_probabilities(self):
        """Test ROC-AUC computation with probability predictions."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_prob = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.3, 0.7],
        ])

        metrics = compute_metrics(
            y_true, y_pred, metrics=["accuracy", "roc_auc"], y_prob=y_prob
        )

        assert "accuracy" in metrics
        assert "roc_auc" in metrics
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_compute_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])

        metrics = compute_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0


class TestStratifiedTrainTestSplit:
    """Test stratified data splitting."""

    def test_stratified_split_preserves_proportions(self):
        """Test that stratified split preserves class proportions."""
        # Create imbalanced dataset: 70% class 0, 30% class 1
        X = np.random.randn(100, 10)
        y = np.array([0] * 70 + [1] * 30)

        X_train, X_test, y_train, y_test = stratified_train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Check sizes
        assert len(X_train) == 80
        assert len(X_test) == 20

        # Check class proportions are preserved (approximately)
        train_ratio = np.sum(y_train == 0) / len(y_train)
        test_ratio = np.sum(y_test == 0) / len(y_test)

        assert abs(train_ratio - 0.7) < 0.1  # Within 10%
        assert abs(test_ratio - 0.7) < 0.2  # Within 20% (smaller sample)

    def test_stratified_split_reproducible(self):
        """Test that split is reproducible with same random_state."""
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        X_train1, X_test1, y_train1, y_test1 = stratified_train_test_split(
            X, y, random_state=42
        )
        X_train2, X_test2, y_train2, y_test2 = stratified_train_test_split(
            X, y, random_state=42
        )

        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)


class TestCrossValidateModel:
    """Test cross-validation functionality."""

    def test_cross_validation_simple_classifier(self):
        """Test CV with SimpleClassifier on 2D features."""
        model = SimpleClassifier()
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)

        results = cross_validate_model(model, X, y, n_folds=5, random_state=42)

        # Check that we got results for all folds
        assert len(results.fold_scores) == 5

        # Check that mean scores are computed
        assert "accuracy" in results.mean_scores
        assert "f1" in results.mean_scores

        # Check that std scores are computed
        assert "accuracy" in results.std_scores

        # Check that confusion matrices are stored
        assert len(results.confusion_matrices) == 5

    def test_cross_validation_eegnet_model(self):
        """Test CV with EEGNetModel on 2D flattened features."""
        # EEGNetModel uses MLPClassifier, so we need 2D input
        model = EEGNetModel(hidden_layer_sizes=(32, 16), max_iter=100)
        # Flattened EEG features: 80 samples, 40 features (e.g., 8 channels × 5 bands)
        X = np.random.randn(80, 40)
        y = np.random.randint(0, 2, 80)

        results = cross_validate_model(model, X, y, n_folds=4, random_state=42)

        assert len(results.fold_scores) == 4
        assert "accuracy" in results.mean_scores
        assert 0.0 <= results.mean_scores["accuracy"] <= 1.0

    def test_cross_validation_with_custom_metrics(self):
        """Test CV with custom metric selection."""
        model = SimpleClassifier()
        X = np.random.randn(60, 10)
        y = np.random.randint(0, 2, 60)

        results = cross_validate_model(
            model, X, y, n_folds=3, metrics=["accuracy", "precision", "recall"]
        )

        assert "accuracy" in results.mean_scores
        assert "precision" in results.mean_scores
        assert "recall" in results.mean_scores
        assert "f1" not in results.mean_scores  # Not requested

    def test_cross_validation_stratified(self):
        """Test that stratified CV preserves class distribution."""
        model = SimpleClassifier()
        # Imbalanced dataset
        X = np.random.randn(100, 10)
        y = np.array([0] * 80 + [1] * 20)

        results = cross_validate_model(
            model, X, y, n_folds=5, stratified=True, random_state=42
        )

        # Should complete successfully with all folds
        assert len(results.fold_scores) == 5

    def test_cross_validation_without_shuffle(self):
        """Test CV without shuffling data."""
        model = SimpleClassifier()
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        results = cross_validate_model(model, X, y, n_folds=5, shuffle=False)

        assert len(results.fold_scores) == 5

    def test_cross_validation_return_predictions(self):
        """Test that predictions can be returned."""
        model = SimpleClassifier()
        X = np.random.randn(40, 8)
        y = np.random.randint(0, 2, 40)

        results = cross_validate_model(
            model, X, y, n_folds=4, return_predictions=True
        )

        assert len(results.predictions) == 4
        # Each fold should have predictions
        for preds in results.predictions:
            assert len(preds) > 0


class TestCVResults:
    """Test CVResults dataclass."""

    def test_cv_results_summary(self):
        """Test summary generation."""
        results = CVResults(
            fold_scores=[
                {"accuracy": 0.8, "f1": 0.75},
                {"accuracy": 0.85, "f1": 0.80},
                {"accuracy": 0.82, "f1": 0.78},
            ],
            mean_scores={"accuracy": 0.823, "f1": 0.777},
            std_scores={"accuracy": 0.021, "f1": 0.021},
        )

        summary = results.summary()

        assert "Cross-Validation Results" in summary
        assert "accuracy" in summary
        assert "f1" in summary
        assert "0.823" in summary or "0.8233" in summary

    def test_cv_results_empty(self):
        """Test empty results."""
        results = CVResults()

        assert len(results.fold_scores) == 0
        assert len(results.mean_scores) == 0

        summary = results.summary()
        assert "Cross-Validation Results" in summary


class TestEvaluateModel:
    """Test single-shot model evaluation."""

    def test_evaluate_model_basic(self):
        """Test basic model evaluation."""
        model = SimpleClassifier()
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        model.train(X_train, y_train)

        X_test = np.random.randn(20, 10)
        y_test = np.random.randint(0, 2, 20)

        results = evaluate_model(model, X_test, y_test)

        assert "accuracy" in results
        assert "confusion_matrix" in results
        assert len(results["confusion_matrix"]) == 2  # 2x2 for binary

    def test_evaluate_model_with_report(self):
        """Test evaluation with detailed classification report."""
        model = SimpleClassifier()
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        model.train(X_train, y_train)

        X_test = np.random.randn(20, 10)
        y_test = np.random.randint(0, 2, 20)

        results = evaluate_model(model, X_test, y_test, return_report=True)

        assert "classification_report" in results
        assert "accuracy" in results["classification_report"]

    def test_evaluate_model_multiclass(self):
        """Test evaluation on multi-class problem."""
        model = SimpleClassifier()
        X_train = np.random.randn(150, 10)
        y_train = np.random.randint(0, 3, 150)  # 3 classes
        model.train(X_train, y_train)

        X_test = np.random.randn(30, 10)
        y_test = np.random.randint(0, 3, 30)

        results = evaluate_model(model, X_test, y_test)

        assert "accuracy" in results
        assert "confusion_matrix" in results
        assert len(results["confusion_matrix"]) == 3  # 3x3 matrix


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_cross_validation_with_too_few_samples(self):
        """Test CV behavior with very few samples."""
        model = SimpleClassifier()
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)

        # With 10 samples and 5 folds, each fold has only 2 samples
        results = cross_validate_model(model, X, y, n_folds=5)

        # Should still complete, though results may be unreliable
        assert len(results.fold_scores) <= 5

    def test_compute_metrics_with_unknown_metric(self):
        """Test that unknown metrics are gracefully skipped."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])

        metrics = compute_metrics(
            y_true, y_pred, metrics=["accuracy", "unknown_metric"]
        )

        # Should compute accuracy but skip unknown metric
        assert "accuracy" in metrics
        assert "unknown_metric" not in metrics
