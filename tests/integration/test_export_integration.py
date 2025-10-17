"""
Integration tests for model training, evaluation, and export workflows.

Tests the complete flow from training to evaluation to exporting results.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import json

from neuros.models.simple_classifier import SimpleClassifier
from neuros.models.svm_model import SVMModel
from neuros.models.random_forest_model import RandomForestModel
from neuros.models.knn_model import KNNModel
from neuros.evaluation import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class TestModelEvaluationIntegration:
    """Test model training to evaluation integration."""

    def test_train_evaluate_simple_classifier(self):
        """Test training and evaluating simple classifier."""
        # Generate data
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, size=100)
        X_test = np.random.randn(30, 20)
        y_test = np.random.randint(0, 2, size=30)

        # Train
        model = SimpleClassifier()
        model.train(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Verify metrics are valid
        assert 0 <= acc <= 1
        assert 0 <= prec <= 1
        assert 0 <= rec <= 1
        assert 0 <= f1 <= 1

    def test_train_evaluate_all_models(self):
        """Test training and evaluation for all model types."""
        models = [
            SimpleClassifier(),
            SVMModel(C=1.0, gamma="scale"),
            RandomForestModel(n_estimators=20),
            KNNModel(n_neighbors=3)
        ]

        X_train = np.random.randn(80, 15)
        y_train = np.random.randint(0, 2, size=80)
        X_test = np.random.randn(20, 15)
        y_test = np.random.randint(0, 2, size=20)

        for model in models:
            # Train
            model.train(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Evaluate
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            # Verify
            assert 0 <= acc <= 1
            assert cm.shape == (2, 2)
            assert np.sum(cm) == len(y_test)

    def test_confusion_matrix_integration(self):
        """Test confusion matrix calculation across models."""
        model = SVMModel(C=1.0, gamma="scale")

        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, size=100)
        X_test = np.random.randn(30, 20)
        y_test = np.random.randint(0, 2, size=30)

        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        # Verify structure
        assert cm.shape == (2, 2)
        assert np.all(cm >= 0)
        assert np.sum(cm) == len(y_test)

    def test_classification_report_integration(self):
        """Test full classification report generation."""
        model = RandomForestModel(n_estimators=30)

        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 2, size=100)
        X_test = np.random.randn(30, 20)
        y_test = np.random.randint(0, 2, size=30)

        model.train(X_train, y_train)
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred)

        # Verify report structure
        assert isinstance(report, dict)
        assert '0' in report
        assert '1' in report
        assert 'accuracy' in report
        assert 'macro avg' in report
        assert 'weighted avg' in report


class TestModelPersistenceExportIntegration:
    """Test model persistence and export workflows."""

    def test_train_save_load_evaluate(self):
        """Test complete workflow: train -> save -> load -> evaluate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"

            # Generate data
            X_train = np.random.randn(80, 20)
            y_train = np.random.randint(0, 2, size=80)
            X_test = np.random.randn(20, 20)
            y_test = np.random.randint(0, 2, size=20)

            # Train and save
            model1 = SVMModel(C=1.0, gamma="scale")
            model1.train(X_train, y_train)
            y_pred1 = model1.predict(X_test)
            acc1 = accuracy_score(y_test, y_pred1)
            model1.save(str(model_path))

            # Load and evaluate
            model2 = SVMModel.load(str(model_path))
            y_pred2 = model2.predict(X_test)
            acc2 = accuracy_score(y_test, y_pred2)

            # Predictions should be identical
            np.testing.assert_array_equal(y_pred1, y_pred2)
            assert acc1 == acc2

    def test_save_load_all_model_types(self):
        """Test save/load for all model types with evaluation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models = [
                (SimpleClassifier(), "simple.pkl"),
                (SVMModel(C=1.0, gamma="scale"), "svm.pkl"),
                (RandomForestModel(n_estimators=20), "rf.pkl"),
                (KNNModel(n_neighbors=3), "knn.pkl")
            ]

            X_train = np.random.randn(60, 15)
            y_train = np.random.randint(0, 2, size=60)
            X_test = np.random.randn(15, 15)
            y_test = np.random.randint(0, 2, size=15)

            for model, filename in models:
                model_path = Path(tmpdir) / filename

                # Train and save
                model.train(X_train, y_train)
                y_pred_before = model.predict(X_test)
                model.save(str(model_path))

                # Load and predict
                model_class = model.__class__
                loaded_model = model_class.load(str(model_path))
                y_pred_after = loaded_model.predict(X_test)

                # Predictions should match
                np.testing.assert_array_equal(y_pred_before, y_pred_after)

    def test_model_metadata_export(self):
        """Test exporting model metadata alongside model file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            metadata_path = Path(tmpdir) / "metadata.json"

            # Train model
            X_train = np.random.randn(80, 20)
            y_train = np.random.randint(0, 2, size=80)
            X_test = np.random.randn(20, 20)
            y_test = np.random.randint(0, 2, size=20)

            model = RandomForestModel(n_estimators=50)
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)

            # Save model
            model.save(str(model_path))

            # Create and save metadata
            metadata = {
                "model_type": model.__class__.__name__,
                "n_features": X_train.shape[1],
                "n_training_samples": len(X_train),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_score": float(f1_score(y_test, y_pred)),
                "hyperparameters": {
                    "n_estimators": 50
                }
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Verify metadata saved
            assert metadata_path.exists()
            with open(metadata_path, 'r') as f:
                loaded_metadata = json.load(f)

            assert loaded_metadata["model_type"] == "RandomForestModel"
            assert loaded_metadata["n_features"] == 20

    def test_export_evaluation_results(self):
        """Test exporting comprehensive evaluation results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "evaluation_results.json"

            # Train and evaluate
            X_train = np.random.randn(100, 20)
            y_train = np.random.randint(0, 2, size=100)
            X_test = np.random.randn(30, 20)
            y_test = np.random.randint(0, 2, size=30)

            model = SVMModel(C=1.0, gamma="scale")
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)

            # Compute all metrics
            results = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred)),
                "recall": float(recall_score(y_test, y_pred)),
                "f1": float(f1_score(y_test, y_pred)),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classification_report": classification_report(y_test, y_pred)
            }

            # Save results
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

            # Verify saved
            assert results_path.exists()
            with open(results_path, 'r') as f:
                loaded_results = json.load(f)

            assert "accuracy" in loaded_results
            assert "confusion_matrix" in loaded_results
            assert len(loaded_results["confusion_matrix"]) == 2


class TestDatasetExportIntegration:
    """Test dataset export and sharing workflows."""

    def test_export_training_data(self):
        """Test exporting training data for reproducibility."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "training_data.npz"

            # Create synthetic data
            X_train = np.random.randn(100, 20)
            y_train = np.random.randint(0, 2, size=100)

            # Save
            np.savez(data_path, X=X_train, y=y_train)

            # Load and verify
            loaded = np.load(data_path)
            np.testing.assert_array_equal(loaded['X'], X_train)
            np.testing.assert_array_equal(loaded['y'], y_train)

    def test_export_processed_features(self):
        """Test exporting processed features for analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features_path = Path(tmpdir) / "features.npz"

            # Simulate feature extraction
            raw_data = np.random.randn(50, 4, 250)  # 50 trials, 4 channels, 250 samples

            # Extract features (simplified)
            features = []
            for trial in raw_data:
                # Simple features: mean and std per channel
                trial_features = np.concatenate([
                    np.mean(trial, axis=1),
                    np.std(trial, axis=1)
                ])
                features.append(trial_features)

            features = np.array(features)
            labels = np.random.randint(0, 2, size=50)

            # Save
            np.savez(features_path, features=features, labels=labels,
                    metadata={'n_channels': 4, 'feature_type': 'mean_std'})

            # Load and verify
            loaded = np.load(features_path, allow_pickle=True)
            np.testing.assert_array_equal(loaded['features'], features)
            assert loaded['metadata'].item()['n_channels'] == 4

    def test_export_cross_validation_splits(self):
        """Test exporting cross-validation splits for reproducibility."""
        with tempfile.TemporaryDirectory() as tmpdir:
            splits_path = Path(tmpdir) / "cv_splits.npz"

            # Create data
            n_samples = 100
            X = np.random.randn(n_samples, 20)
            y = np.random.randint(0, 2, size=n_samples)

            # Create CV splits (simplified)
            n_folds = 5
            fold_size = n_samples // n_folds

            splits = {}
            for fold in range(n_folds):
                test_start = fold * fold_size
                test_end = test_start + fold_size
                test_idx = np.arange(test_start, test_end)
                train_idx = np.concatenate([
                    np.arange(0, test_start),
                    np.arange(test_end, n_samples)
                ])
                splits[f'fold_{fold}_train'] = train_idx
                splits[f'fold_{fold}_test'] = test_idx

            # Save
            np.savez(splits_path, **splits)

            # Load and verify
            loaded = np.load(splits_path)
            for fold in range(n_folds):
                assert f'fold_{fold}_train' in loaded
                assert f'fold_{fold}_test' in loaded


class TestBenchmarkResultsExport:
    """Test exporting benchmark and profiling results."""

    def test_export_benchmark_results(self):
        """Test exporting performance benchmark results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "benchmark_results.json"

            # Simulate benchmark
            benchmark_results = {
                "model": "SVMModel",
                "latency_ms": {
                    "mean": 2.5,
                    "median": 2.3,
                    "p95": 3.2,
                    "p99": 4.1
                },
                "throughput": {
                    "samples_per_second": 450.0
                },
                "memory_mb": {
                    "peak": 125.3,
                    "average": 98.7
                },
                "test_config": {
                    "n_samples": 1000,
                    "n_features": 20,
                    "duration_seconds": 5.0
                }
            }

            # Save
            with open(results_path, 'w') as f:
                json.dump(benchmark_results, f, indent=2)

            # Load and verify
            with open(results_path, 'r') as f:
                loaded = json.load(f)

            assert loaded["model"] == "SVMModel"
            assert loaded["latency_ms"]["mean"] == 2.5
            assert loaded["throughput"]["samples_per_second"] == 450.0

    def test_export_profiling_data(self):
        """Test exporting profiling data for optimization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "profile.json"

            # Simulate profiling data
            profiling_data = {
                "total_time_seconds": 5.23,
                "function_calls": [
                    {
                        "function": "predict",
                        "cumulative_time": 3.12,
                        "calls": 1000,
                        "time_per_call": 0.00312
                    },
                    {
                        "function": "feature_extraction",
                        "cumulative_time": 1.85,
                        "calls": 1000,
                        "time_per_call": 0.00185
                    },
                    {
                        "function": "preprocessing",
                        "cumulative_time": 0.26,
                        "calls": 1000,
                        "time_per_call": 0.00026
                    }
                ],
                "bottlenecks": [
                    "predict function takes 60% of total time",
                    "feature_extraction takes 35% of total time"
                ]
            }

            # Save
            with open(profile_path, 'w') as f:
                json.dump(profiling_data, f, indent=2)

            # Verify
            assert profile_path.exists()
            with open(profile_path, 'r') as f:
                loaded = json.load(f)

            assert loaded["total_time_seconds"] == 5.23
            assert len(loaded["function_calls"]) == 3
            assert len(loaded["bottlenecks"]) == 2

    def test_export_comparison_results(self):
        """Test exporting model comparison results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            comparison_path = Path(tmpdir) / "model_comparison.json"

            # Simulate comparison data
            comparison = {
                "models": [
                    {
                        "name": "SimpleClassifier",
                        "accuracy": 0.78,
                        "f1_score": 0.76,
                        "latency_ms": 1.2,
                        "training_time_s": 0.05
                    },
                    {
                        "name": "SVMModel",
                        "accuracy": 0.85,
                        "f1_score": 0.83,
                        "latency_ms": 2.5,
                        "training_time_s": 1.23
                    },
                    {
                        "name": "RandomForestModel",
                        "accuracy": 0.87,
                        "f1_score": 0.86,
                        "latency_ms": 5.3,
                        "training_time_s": 2.15
                    }
                ],
                "best_accuracy": "RandomForestModel",
                "best_speed": "SimpleClassifier",
                "recommended": "SVMModel"
            }

            # Save
            with open(comparison_path, 'w') as f:
                json.dump(comparison, f, indent=2)

            # Verify
            with open(comparison_path, 'r') as f:
                loaded = json.load(f)

            assert len(loaded["models"]) == 3
            assert loaded["best_accuracy"] == "RandomForestModel"
            assert loaded["best_speed"] == "SimpleClassifier"
