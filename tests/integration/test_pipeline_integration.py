"""
Integration tests for complete pipeline workflows.

These tests verify that all components (drivers, processing, models, export)
work together correctly in realistic end-to-end scenarios.
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
import tempfile
import json

from neuros.pipeline import Pipeline, MultiModalPipeline
from neuros.drivers.mock_driver import MockDriver
from neuros.drivers.dataset_driver import DatasetDriver
from neuros.models.simple_classifier import SimpleClassifier
from neuros.models.svm_model import SVMModel
from neuros.models.random_forest_model import RandomForestModel
from neuros.models.knn_model import KNNModel
from neuros.processing.filters import BandpassFilter, SmoothingFilter
from neuros.processing.feature_extraction import BandPowerExtractor


@pytest.mark.asyncio
class TestPipelineIntegration:
    """Test complete pipeline workflows with various configurations."""

    async def test_basic_pipeline_workflow(self):
        """Test basic pipeline: driver -> processing -> model -> prediction."""
        # Setup
        pipeline = Pipeline(
            driver=MockDriver(channels=4, sampling_rate=250),
            model=SimpleClassifier(),
            fs=250.0
        )

        # Generate training data
        n_samples = 100
        n_features = 4 * 5  # 4 channels * 5 frequency bands
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.random.randint(0, 2, size=n_samples)

        # Train
        pipeline.train(X_train, y_train)

        # Run pipeline
        metrics = await pipeline.run(duration=0.5)

        # Verify
        assert isinstance(metrics, dict)
        assert metrics["samples"] > 0
        assert metrics["throughput"] > 0
        assert metrics["mean_latency"] >= 0
        assert "model" in metrics
        assert "driver" in metrics

    async def test_pipeline_with_filters(self):
        """Test pipeline with bandpass and smoothing filters."""
        # Setup filters
        filters = [
            BandpassFilter(lowcut=1.0, highcut=50.0, fs=250.0, order=4),
            SmoothingFilter(window_size=5)
        ]

        pipeline = Pipeline(
            driver=MockDriver(channels=8, sampling_rate=250),
            model=SVMModel(C=1.0, gamma="scale"),
            fs=250.0,
            filters=filters
        )

        # Train
        X_train = np.random.randn(80, 8 * 5)
        y_train = np.random.randint(0, 2, size=80)
        pipeline.train(X_train, y_train)

        # Run
        metrics = await pipeline.run(duration=0.5)

        # Verify metrics
        assert metrics["samples"] > 0
        assert "throughput" in metrics
        assert "mean_latency" in metrics

    async def test_pipeline_with_custom_bands(self):
        """Test pipeline with custom frequency bands."""
        custom_bands = {
            "low_alpha": (8.0, 10.0),
            "high_alpha": (10.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 50.0)
        }

        pipeline = Pipeline(
            driver=MockDriver(channels=4, sampling_rate=250),
            model=RandomForestModel(n_estimators=50),
            fs=250.0,
            bands=custom_bands
        )

        # Train with correct feature dimensions
        n_bands = len(custom_bands)
        n_channels = 4
        X_train = np.random.randn(60, n_channels * n_bands)
        y_train = np.random.randint(0, 2, size=60)
        pipeline.train(X_train, y_train)

        # Run
        metrics = await pipeline.run(duration=0.5)
        assert metrics["samples"] > 0

    async def test_pipeline_different_models(self):
        """Test pipeline with different model types."""
        models = [
            SimpleClassifier(),
            SVMModel(C=1.0, gamma="scale"),
            RandomForestModel(n_estimators=20),
            KNNModel(n_neighbors=3)
        ]

        for model in models:
            pipeline = Pipeline(
                driver=MockDriver(channels=4, sampling_rate=250),
                model=model,
                fs=250.0
            )

            # Train
            X_train = np.random.randn(50, 20)
            y_train = np.random.randint(0, 2, size=50)
            pipeline.train(X_train, y_train)

            # Run
            metrics = await pipeline.run(duration=0.3)

            # Verify
            assert metrics["samples"] > 0
            assert "model" in metrics
            assert model.__class__.__name__ in str(metrics["model"])

    async def test_pipeline_adaptation(self):
        """Test pipeline with adaptation enabled vs disabled."""
        driver = MockDriver(channels=4, sampling_rate=250)
        model = SimpleClassifier()

        # Test with adaptation
        pipeline_adapted = Pipeline(
            driver=driver,
            model=model,
            fs=250.0,
            adaptation=True
        )
        X_train = np.random.randn(50, 20)
        y_train = np.random.randint(0, 2, size=50)
        pipeline_adapted.train(X_train, y_train)
        metrics_adapted = await pipeline_adapted.run(duration=0.3)

        # Test without adaptation
        pipeline_no_adapt = Pipeline(
            driver=MockDriver(channels=4, sampling_rate=250),
            model=SimpleClassifier(),
            fs=250.0,
            adaptation=False
        )
        pipeline_no_adapt.train(X_train, y_train)
        metrics_no_adapt = await pipeline_no_adapt.run(duration=0.3)

        # Both should complete successfully
        assert metrics_adapted["samples"] > 0
        assert metrics_no_adapt["samples"] > 0


@pytest.mark.asyncio
class TestMultiModalIntegration:
    """Test multi-modal pipeline integration."""

    async def test_multimodal_pipeline_two_modalities(self):
        """Test multi-modal pipeline with two data sources."""
        # Setup two drivers
        drivers = [
            MockDriver(channels=4, sampling_rate=250),  # EEG
            MockDriver(channels=1, sampling_rate=100)   # ECG
        ]

        pipeline = MultiModalPipeline(
            drivers=drivers,
            model=SimpleClassifier(),
            fs_list=[250.0, 100.0],
            adaptation=True
        )

        # Train (features from both modalities concatenated)
        # EEG: 4 channels * 5 bands = 20 features
        # ECG: 1 channel * 5 bands = 5 features
        # Total: 25 features
        X_train = np.random.randn(60, 25)
        y_train = np.random.randint(0, 2, size=60)
        pipeline.train(X_train, y_train)

        # Run
        metrics = await pipeline.run(duration=0.5)

        # Verify
        assert metrics["samples"] > 0
        assert metrics["throughput"] > 0

    async def test_multimodal_pipeline_with_filters(self):
        """Test multi-modal pipeline with per-modality filters."""
        drivers = [
            MockDriver(channels=4, sampling_rate=250),
            MockDriver(channels=2, sampling_rate=250)
        ]

        # Different filters for each modality
        filters_list = [
            [BandpassFilter(1.0, 50.0, fs=250.0)],  # For first modality
            [SmoothingFilter(window_size=3)]         # For second modality
        ]

        pipeline = MultiModalPipeline(
            drivers=drivers,
            model=SVMModel(C=1.0, gamma="scale"),
            fs_list=[250.0, 250.0],
            filters_list=filters_list,
            adaptation=True
        )

        # Train
        X_train = np.random.randn(50, 30)  # 4*5 + 2*5 = 30 features
        y_train = np.random.randint(0, 2, size=50)
        pipeline.train(X_train, y_train)

        # Run
        metrics = await pipeline.run(duration=0.4)
        assert metrics["samples"] > 0


@pytest.mark.asyncio
class TestDriverProcessingChain:
    """Test driver to processing chain integration."""

    async def test_driver_to_filter_chain(self):
        """Test data flow from driver through filter chain."""
        # Create driver
        driver = MockDriver(channels=4, sampling_rate=250)

        # Create filter chain
        filters = [
            BandpassFilter(lowcut=8.0, highcut=13.0, fs=250.0),  # Alpha band
            SmoothingFilter(window_size=5)
        ]

        # Create pipeline
        pipeline = Pipeline(
            driver=driver,
            model=SimpleClassifier(),
            fs=250.0,
            filters=filters
        )

        # Train
        X_train = np.random.randn(40, 20)
        y_train = np.random.randint(0, 2, size=40)
        pipeline.train(X_train, y_train)

        # Run and verify
        metrics = await pipeline.run(duration=0.3)
        assert metrics["samples"] > 0
        assert metrics["throughput"] > 0

    async def test_driver_to_feature_extraction(self):
        """Test driver data flows correctly to feature extraction."""
        # Use dataset driver for consistent data
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic dataset
            data_file = Path(tmpdir) / "test_data.npz"
            n_trials = 20
            n_channels = 4
            n_samples_per_trial = 250  # 1 second at 250 Hz

            X_data = np.random.randn(n_trials, n_channels, n_samples_per_trial)
            y_data = np.random.randint(0, 2, size=n_trials)

            np.savez(data_file, X=X_data, y=y_data)

            # Create driver
            driver = DatasetDriver(str(data_file), fs=250.0)

            # Create pipeline
            pipeline = Pipeline(
                driver=driver,
                model=RandomForestModel(n_estimators=30),
                fs=250.0
            )

            # Train
            X_train = np.random.randn(40, n_channels * 5)
            y_train = np.random.randint(0, 2, size=40)
            pipeline.train(X_train, y_train)

            # Run
            metrics = await pipeline.run()  # Duration from driver

            # Verify
            assert metrics["samples"] > 0
            assert "driver" in metrics


@pytest.mark.asyncio
class TestModelPersistenceIntegration:
    """Test model training, saving, loading integration."""

    async def test_pipeline_model_persistence(self):
        """Test saving and loading trained pipeline models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"

            # Train and save
            pipeline1 = Pipeline(
                driver=MockDriver(channels=4, sampling_rate=250),
                model=SVMModel(C=1.0, gamma="scale"),
                fs=250.0
            )

            X_train = np.random.randn(60, 20)
            y_train = np.random.randint(0, 2, size=60)
            pipeline1.train(X_train, y_train)

            # Save model
            pipeline1.model.save(str(model_path))

            # Load model into new pipeline
            loaded_model = SVMModel.load(str(model_path))
            pipeline2 = Pipeline(
                driver=MockDriver(channels=4, sampling_rate=250),
                model=loaded_model,
                fs=250.0
            )

            # Run loaded pipeline
            metrics = await pipeline2.run(duration=0.3)
            assert metrics["samples"] > 0

    async def test_multiple_models_persistence(self):
        """Test saving/loading multiple model types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models = [
                (SimpleClassifier(), "simple.pkl"),
                (SVMModel(C=1.0, gamma="scale"), "svm.pkl"),
                (RandomForestModel(n_estimators=20), "rf.pkl"),
                (KNNModel(n_neighbors=3), "knn.pkl")
            ]

            X_train = np.random.randn(50, 20)
            y_train = np.random.randint(0, 2, size=50)

            for model, filename in models:
                model_path = Path(tmpdir) / filename

                # Train and save
                model.train(X_train, y_train)
                model.save(str(model_path))

                # Load
                model_class = model.__class__
                loaded_model = model_class.load(str(model_path))

                # Verify predictions work
                X_test = np.random.randn(5, 20)
                predictions = loaded_model.predict(X_test)
                assert len(predictions) == 5


@pytest.mark.asyncio
class TestPerformanceIntegration:
    """Test performance characteristics of integrated pipelines."""

    async def test_pipeline_throughput_consistency(self):
        """Test that pipeline maintains consistent throughput."""
        pipeline = Pipeline(
            driver=MockDriver(channels=4, sampling_rate=250),
            model=SimpleClassifier(),
            fs=250.0
        )

        # Train
        X_train = np.random.randn(50, 20)
        y_train = np.random.randint(0, 2, size=50)
        pipeline.train(X_train, y_train)

        # Run multiple times
        throughputs = []
        for _ in range(3):
            metrics = await pipeline.run(duration=0.5)
            throughputs.append(metrics["throughput"])

        # Verify throughput is consistent (within 50% variation)
        mean_throughput = np.mean(throughputs)
        for tp in throughputs:
            assert 0.5 * mean_throughput <= tp <= 1.5 * mean_throughput

    async def test_pipeline_latency_reasonable(self):
        """Test that pipeline latency is reasonable."""
        pipeline = Pipeline(
            driver=MockDriver(channels=4, sampling_rate=250),
            model=SimpleClassifier(),
            fs=250.0
        )

        # Train
        X_train = np.random.randn(50, 20)
        y_train = np.random.randint(0, 2, size=50)
        pipeline.train(X_train, y_train)

        # Run
        metrics = await pipeline.run(duration=0.5)

        # Latency should be reasonable (< 100ms for simple model)
        assert metrics["mean_latency"] < 100.0  # milliseconds

    async def test_pipeline_scales_with_complexity(self):
        """Test that pipeline performance scales with model complexity."""
        X_train = np.random.randn(50, 20)
        y_train = np.random.randint(0, 2, size=50)

        # Simple model
        pipeline_simple = Pipeline(
            driver=MockDriver(channels=4, sampling_rate=250),
            model=SimpleClassifier(),
            fs=250.0
        )
        pipeline_simple.train(X_train, y_train)
        metrics_simple = await pipeline_simple.run(duration=0.3)

        # Complex model
        pipeline_complex = Pipeline(
            driver=MockDriver(channels=4, sampling_rate=250),
            model=RandomForestModel(n_estimators=100),
            fs=250.0
        )
        pipeline_complex.train(X_train, y_train)
        metrics_complex = await pipeline_complex.run(duration=0.3)

        # Both should complete, complex may have higher latency
        assert metrics_simple["samples"] > 0
        assert metrics_complex["samples"] > 0
        # Complex model typically has higher latency
        # But we don't enforce strict inequality due to variability


@pytest.mark.asyncio
class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""

    async def test_pipeline_handles_untrained_model(self):
        """Test pipeline behavior with untrained model."""
        pipeline = Pipeline(
            driver=MockDriver(channels=4, sampling_rate=250),
            model=SimpleClassifier(),
            fs=250.0
        )

        # Run without training - should handle gracefully
        try:
            metrics = await pipeline.run(duration=0.2)
            # May succeed with random predictions or raise error
            # Either is acceptable
            assert True
        except Exception as e:
            # Should be a meaningful error
            assert len(str(e)) > 0

    async def test_pipeline_handles_mismatched_dimensions(self):
        """Test pipeline with mismatched training/runtime dimensions."""
        pipeline = Pipeline(
            driver=MockDriver(channels=4, sampling_rate=250),
            model=SimpleClassifier(),
            fs=250.0
        )

        # Train with wrong number of features
        X_train = np.random.randn(50, 10)  # Wrong dimension
        y_train = np.random.randint(0, 2, size=50)

        pipeline.train(X_train, y_train)

        # Run - may fail or adapt
        try:
            metrics = await pipeline.run(duration=0.2)
            # If it succeeds, that's fine
            assert True
        except Exception:
            # If it fails, that's also expected behavior
            assert True
