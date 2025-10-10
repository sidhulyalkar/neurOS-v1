"""Tests for LSTM model."""
import pytest
import numpy as np


class TestLSTMModel:
    """Tests for LSTMModel."""

    def test_lstm_model_train_predict(self):
        """Test LSTM model training and prediction."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch not installed")

        from neuros.models import LSTMModel

        # Create model
        model = LSTMModel(
            n_channels=4,
            n_timepoints=100,
            n_classes=2,
            lstm_units=32,
            n_lstm_layers=1,
            n_epochs=5,  # Quick training for test
            batch_size=8,
        )

        # Generate synthetic data (samples, channels, timepoints)
        X_train = np.random.randn(40, 4, 100)
        y_train = np.random.randint(0, 2, 40)

        # Train
        model.train(X_train, y_train)
        assert model.is_trained

        # Predict
        X_test = np.random.randn(10, 4, 100)
        predictions = model.predict(X_test)

        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)

    def test_lstm_model_predict_proba(self):
        """Test LSTM probability predictions."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch not installed")

        from neuros.models import LSTMModel

        model = LSTMModel(
            n_channels=4,
            n_timepoints=50,
            n_classes=3,
            lstm_units=16,
            n_lstm_layers=1,
            n_epochs=3,
        )

        X_train = np.random.randn(30, 4, 50)
        y_train = np.random.randint(0, 3, 30)

        model.train(X_train, y_train)

        X_test = np.random.randn(5, 4, 50)
        probas = model.predict_proba(X_test)

        assert probas.shape == (5, 3)
        # Check probabilities sum to 1
        np.testing.assert_array_almost_equal(probas.sum(axis=1), np.ones(5), decimal=5)
        # Check all probabilities in [0, 1]
        assert np.all((probas >= 0) & (probas <= 1))

    def test_lstm_model_partial_fit(self):
        """Test incremental learning with partial_fit."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch not installed")

        from neuros.models import LSTMModel

        model = LSTMModel(
            n_channels=3,
            n_timepoints=50,
            n_classes=2,
            lstm_units=16,
            n_epochs=3,
        )

        # Initial training
        X_train = np.random.randn(20, 3, 50)
        y_train = np.random.randint(0, 2, 20)
        model.train(X_train, y_train)

        # Incremental update
        X_new = np.random.randn(10, 3, 50)
        y_new = np.random.randint(0, 2, 10)
        model.partial_fit(X_new, y_new)

        # Should still work
        X_test = np.random.randn(5, 3, 50)
        predictions = model.predict(X_test)
        assert len(predictions) == 5

    def test_lstm_model_invalid_shape(self):
        """Test that model raises error on invalid input shape."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch not installed")

        from neuros.models import LSTMModel

        model = LSTMModel(n_channels=4, n_timepoints=100, n_classes=2)

        # Wrong number of dimensions
        X_wrong = np.random.randn(20, 100)  # Missing channel dimension
        y = np.random.randint(0, 2, 20)

        with pytest.raises(ValueError, match="3 dimensions"):
            model.train(X_wrong, y)

    def test_lstm_model_predict_before_train(self):
        """Test that predict raises error if model not trained."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch not installed")

        from neuros.models import LSTMModel

        model = LSTMModel(n_channels=4, n_timepoints=100, n_classes=2)
        X_test = np.random.randn(5, 4, 100)

        with pytest.raises(ValueError, match="must be trained"):
            model.predict(X_test)

    @pytest.mark.slow
    def test_lstm_model_multiclass(self):
        """Test LSTM on multi-class problem."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch not installed")

        from neuros.models import LSTMModel

        # 5-class problem
        model = LSTMModel(
            n_channels=8,
            n_timepoints=200,
            n_classes=5,
            lstm_units=64,
            n_lstm_layers=2,
            n_epochs=10,
        )

        X_train = np.random.randn(100, 8, 200)
        y_train = np.random.randint(0, 5, 100)

        model.train(X_train, y_train)

        X_test = np.random.randn(20, 8, 200)
        predictions = model.predict(X_test)

        assert len(predictions) == 20
        assert all(p in range(5) for p in predictions)
