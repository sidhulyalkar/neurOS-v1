"""
Tests for CEBRA (Learnable Latent Embeddings) foundation model.

These tests verify that CEBRA works correctly with different learning modes,
handles behavioral data, and provides sklearn-compatible API.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from neuros.foundation_models import CEBRAModel


class TestCEBRAInitialization:
    """Tests for CEBRA model initialization."""

    def test_basic_initialization(self):
        """Test basic CEBRA initialization."""
        model = CEBRAModel(input_dim=100, output_dim=3)
        assert model.input_dim == 100
        assert model.output_dim == 3
        assert model.learning_mode == "time"
        assert model.temperature == 0.1
        assert not model.is_trained

    def test_initialization_custom_params(self):
        """Test CEBRA initialization with custom parameters."""
        model = CEBRAModel(
            input_dim=200,
            output_dim=8,
            hidden_dims=[512, 256, 128],
            learning_mode="behavior",
            temperature=0.05,
            time_offset=20,
            dropout=0.2,
            max_iterations=10000,
            batch_size=1024,
        )
        assert model.input_dim == 200
        assert model.output_dim == 8
        assert model.hidden_dims == [512, 256, 128]
        assert model.learning_mode == "behavior"
        assert model.temperature == 0.05
        assert model.time_offset == 20
        assert model.dropout == 0.2
        assert model.max_iterations == 10000
        assert model.batch_size == 1024

    def test_invalid_learning_mode_raises(self):
        """Test that invalid learning mode raises an error."""
        with pytest.raises(ValueError, match="learning_mode must be"):
            CEBRAModel(input_dim=100, learning_mode="invalid")


class TestCEBRATraining:
    """Tests for CEBRA training."""

    def test_train_time_mode(self):
        """Test training with time-contrastive learning."""
        model = CEBRAModel(input_dim=100, output_dim=3, learning_mode="time")
        X = np.random.randn(500, 100)

        model.train(X)
        assert model.is_trained

    def test_train_behavior_mode(self):
        """Test training with behavior-contrastive learning."""
        model = CEBRAModel(input_dim=100, output_dim=3, learning_mode="behavior")
        X = np.random.randn(500, 100)
        behavior = np.random.randn(500, 2)  # 2D behavior (e.g., position)

        model.train(X, behavior=behavior)
        assert model.is_trained

    def test_train_hybrid_mode(self):
        """Test training with hybrid mode."""
        model = CEBRAModel(input_dim=100, output_dim=3, learning_mode="hybrid")
        X = np.random.randn(500, 100)
        behavior = np.random.randn(500, 2)

        model.train(X, behavior=behavior)
        assert model.is_trained

    def test_train_3d_input(self):
        """Test training with 3D input (time dimension)."""
        model = CEBRAModel(input_dim=50, output_dim=3)
        X = np.random.randn(100, 10, 50)  # (samples, timepoints, neurons)

        model.train(X)
        assert model.is_trained

    def test_train_behavior_mode_without_behavior_raises(self):
        """Test that behavior mode without behavior data raises error."""
        model = CEBRAModel(input_dim=100, learning_mode="behavior")
        X = np.random.randn(500, 100)

        with pytest.raises(ValueError, match="behavior data required"):
            model.train(X)

    def test_train_hybrid_mode_without_behavior_raises(self):
        """Test that hybrid mode without behavior data raises error."""
        model = CEBRAModel(input_dim=100, learning_mode="hybrid")
        X = np.random.randn(500, 100)

        with pytest.raises(ValueError, match="behavior data required"):
            model.train(X)


class TestCEBRAEncoding:
    """Tests for CEBRA encoding (transformation to latent space)."""

    def test_encode_basic(self):
        """Test basic encoding."""
        model = CEBRAModel(input_dim=100, output_dim=3)
        X_train = np.random.randn(500, 100)
        model.train(X_train)

        X_test = np.random.randn(50, 100)
        embeddings = model.encode(X_test)

        assert embeddings.shape == (50, 3)

    def test_encode_different_output_dims(self):
        """Test encoding with different output dimensions."""
        for output_dim in [3, 8, 16, 32]:
            model = CEBRAModel(input_dim=100, output_dim=output_dim)
            X = np.random.randn(500, 100)
            model.train(X)

            embeddings = model.encode(X[:50])
            assert embeddings.shape == (50, output_dim)

    def test_encode_untrained_raises(self):
        """Test that encoding before training raises an error."""
        model = CEBRAModel(input_dim=100, output_dim=3)
        X = np.random.randn(50, 100)

        with pytest.raises(ValueError, match="must be trained"):
            model.encode(X)

    def test_encode_3d_input(self):
        """Test encoding with 3D input."""
        model = CEBRAModel(input_dim=50, output_dim=3)
        X_train = np.random.randn(500, 50)
        model.train(X_train)

        X_test = np.random.randn(10, 5, 50)  # (samples, timepoints, neurons)
        embeddings = model.encode(X_test)

        # Should flatten time dimension: 10 * 5 = 50 samples
        assert embeddings.shape == (50, 3)


class TestCEBRAPredict:
    """Tests for CEBRA prediction (alias for encoding)."""

    def test_predict_is_encode_alias(self):
        """Test that predict is an alias for encode."""
        model = CEBRAModel(input_dim=100, output_dim=3)
        X = np.random.randn(500, 100)
        model.train(X)

        X_test = np.random.randn(50, 100)
        predictions = model.predict(X_test)
        encodings = model.encode(X_test)

        assert predictions.shape == encodings.shape


class TestCEBRADecode:
    """Tests for CEBRA decoding (inverse transformation)."""

    def test_decode_basic(self):
        """Test basic decoding."""
        model = CEBRAModel(input_dim=100, output_dim=3)
        X = np.random.randn(500, 100)
        model.train(X)

        latents = np.random.randn(50, 3)
        decoded = model.decode(latents)

        assert decoded.shape == (50, 100)

    def test_decode_untrained_raises(self):
        """Test that decoding before training raises an error."""
        model = CEBRAModel(input_dim=100, output_dim=3)
        latents = np.random.randn(50, 3)

        with pytest.raises(ValueError, match="must be trained"):
            model.decode(latents)


class TestCEBRASklearnAPI:
    """Tests for sklearn-compatible API."""

    def test_transform(self):
        """Test transform method (sklearn API)."""
        model = CEBRAModel(input_dim=100, output_dim=3)
        X = np.random.randn(500, 100)
        model.train(X)

        X_test = np.random.randn(50, 100)
        transformed = model.transform(X_test)

        assert transformed.shape == (50, 3)

    def test_fit_transform(self):
        """Test fit_transform method (sklearn API)."""
        model = CEBRAModel(input_dim=100, output_dim=3)
        X = np.random.randn(500, 100)

        embeddings = model.fit_transform(X)

        assert model.is_trained
        assert embeddings.shape == (500, 3)

    def test_fit_transform_with_behavior(self):
        """Test fit_transform with behavioral data."""
        model = CEBRAModel(input_dim=100, output_dim=3, learning_mode="behavior")
        X = np.random.randn(500, 100)
        behavior = np.random.randn(500, 2)

        embeddings = model.fit_transform(X, behavior=behavior)

        assert model.is_trained
        assert embeddings.shape == (500, 3)


class TestCEBRAConsistency:
    """Tests for consistency computation."""

    def test_compute_consistency(self):
        """Test consistency computation between datasets."""
        model = CEBRAModel(input_dim=100, output_dim=3)
        X = np.random.randn(500, 100)
        model.train(X)

        X1 = np.random.randn(50, 100)
        X2 = np.random.randn(50, 100)

        consistency = model.compute_consistency(X1, X2, n_neighbors=5)

        assert isinstance(consistency, (float, np.floating))
        assert 0 <= consistency <= 1

    def test_compute_consistency_untrained_raises(self):
        """Test that consistency before training raises error."""
        model = CEBRAModel(input_dim=100, output_dim=3)
        X1 = np.random.randn(50, 100)
        X2 = np.random.randn(50, 100)

        with pytest.raises(ValueError, match="must be trained"):
            model.compute_consistency(X1, X2)


class TestCEBRABehaviorDecoding:
    """Tests for behavior decoding from latent space."""

    def test_decode_behavior(self):
        """Test behavior decoding with cross-validation."""
        model = CEBRAModel(input_dim=100, output_dim=3)
        X = np.random.randn(500, 100)
        behavior = np.random.randn(500, 2)
        model.train(X, behavior=behavior)

        results = model.decode_behavior(X, behavior, n_folds=5)

        assert "r2_score" in results
        assert "mse" in results
        assert isinstance(results["r2_score"], (float, np.floating))
        assert isinstance(results["mse"], (float, np.floating))

    def test_decode_behavior_untrained_raises(self):
        """Test that behavior decoding before training raises error."""
        model = CEBRAModel(input_dim=100, output_dim=3)
        X = np.random.randn(500, 100)
        behavior = np.random.randn(500, 2)

        with pytest.raises(ValueError, match="must be trained"):
            model.decode_behavior(X, behavior)


class TestCEBRASaveLoad:
    """Tests for saving and loading CEBRA models."""

    def test_save_and_load(self):
        """Test saving and loading CEBRA model."""
        model = CEBRAModel(input_dim=100, output_dim=3, hidden_dims=[128, 64])
        X = np.random.randn(500, 100)
        model.train(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "cebra_model.pt"
            model.save_checkpoint(str(save_path))

            loaded_model = CEBRAModel.from_pretrained(
                str(save_path), input_dim=100, output_dim=3, hidden_dims=[128, 64]
            )
            assert loaded_model.is_trained
            assert loaded_model.input_dim == 100
            assert loaded_model.output_dim == 3


class TestCEBRAIntegration:
    """Integration tests for complete CEBRA workflows."""

    def test_complete_time_workflow(self):
        """Test complete workflow with time-contrastive learning."""
        # Create model
        model = CEBRAModel(input_dim=96, output_dim=3, learning_mode="time")

        # Generate synthetic neural data
        n_samples = 1000
        X = np.random.randn(n_samples, 96)

        # Train
        model.train(X)

        # Transform
        embeddings = model.transform(X)
        assert embeddings.shape == (n_samples, 3)

        # Test consistency
        X_test = np.random.randn(100, 96)
        consistency = model.compute_consistency(X[:100], X_test)
        assert 0 <= consistency <= 1

    def test_complete_behavior_workflow(self):
        """Test complete workflow with behavior-contrastive learning."""
        # Create model
        model = CEBRAModel(input_dim=96, output_dim=8, learning_mode="behavior")

        # Generate synthetic neural and behavioral data
        n_samples = 1000
        X = np.random.randn(n_samples, 96)
        behavior = np.random.randn(n_samples, 2)  # 2D position

        # Fit and transform
        embeddings = model.fit_transform(X, behavior=behavior)
        assert embeddings.shape == (n_samples, 8)

        # Decode behavior
        results = model.decode_behavior(X, behavior, n_folds=3)
        assert "r2_score" in results

    def test_cross_session_consistency(self):
        """Test consistency across multiple sessions."""
        model = CEBRAModel(input_dim=100, output_dim=3)

        # Session 1
        X_session1 = np.random.randn(500, 100)
        model.train(X_session1)

        # Session 2
        X_session2 = np.random.randn(500, 100)

        # Compute cross-session consistency
        consistency = model.compute_consistency(
            X_session1[:100], X_session2[:100], n_neighbors=5
        )

        assert isinstance(consistency, (float, np.floating))
