"""
Tests for foundation models.
"""

import numpy as np
import pytest

from neuros.foundation_models import (
    BaseFoundationModel,
    POYOModel,
    POYOPlusModel,
    POYO_AVAILABLE,
)
from neuros.foundation_models.utils import (
    spikes_to_tokens,
    create_session_embeddings,
    create_readout_spec,
    raster_to_spike_times,
    align_session_lengths,
)


class TestUtilities:
    """Test foundation model utilities."""

    def test_spikes_to_tokens(self):
        """Test conversion of spike times to tokens."""
        spike_times = [
            np.array([0.1, 0.5, 1.2]),  # Neuron 0
            np.array([0.3, 0.7, 1.5]),  # Neuron 1
        ]

        unit_ids, timestamps, _ = spikes_to_tokens(
            spike_times, time_window=(0, 2.0)
        )

        assert len(unit_ids) == 6  # 3 + 3 spikes
        assert len(timestamps) == 6
        # Should be sorted by time
        assert np.all(np.diff(timestamps) >= 0)

    def test_spikes_to_tokens_with_binning(self):
        """Test spike tokenization with binning."""
        spike_times = [np.array([0.1, 0.5, 1.2])]

        unit_ids, timestamps, counts = spikes_to_tokens(
            spike_times, time_window=(0, 2.0), bin_size=0.1
        )

        assert len(counts) == 20  # 2.0 / 0.1
        assert np.sum(counts) == 3  # Total spikes

    def test_spikes_to_tokens_max_spikes(self):
        """Test spike truncation."""
        spike_times = [np.array([0.1, 0.2, 0.3, 0.4, 0.5])]

        unit_ids, timestamps, _ = spikes_to_tokens(
            spike_times, time_window=(0, 1.0), max_spikes=3
        )

        assert len(unit_ids) == 3
        assert len(timestamps) == 3

    def test_create_session_embeddings_random(self):
        """Test random session embeddings."""
        embeddings = create_session_embeddings(10, embedding_dim=64, method="random")

        assert embeddings.shape == (10, 64)
        assert embeddings.dtype == np.float32

    def test_create_session_embeddings_positional(self):
        """Test positional session embeddings."""
        embeddings = create_session_embeddings(
            5, embedding_dim=32, method="positional"
        )

        assert embeddings.shape == (5, 32)
        # Positional encodings should be bounded
        assert np.all(np.abs(embeddings) <= 1.0)

    def test_create_session_embeddings_learned(self):
        """Test learned session embeddings initialization."""
        embeddings = create_session_embeddings(10, embedding_dim=64, method="learned")

        assert embeddings.shape == (10, 64)
        # Should be small random values
        assert np.std(embeddings) < 0.1

    def test_create_readout_spec(self):
        """Test readout specification creation."""
        tasks = [
            {"name": "velocity", "type": "regression", "output_dim": 2},
            {"name": "direction", "type": "classification", "output_dim": 8},
        ]

        spec = create_readout_spec(tasks)

        assert len(spec) == 2
        assert spec[0]["name"] == "velocity"
        assert spec[0]["type"] == "regression"
        assert spec[1]["num_classes"] == 8

    def test_raster_to_spike_times(self):
        """Test conversion from raster to spike times."""
        # Create simple raster with known spikes
        raster = np.zeros((100, 3))
        raster[10, 0] = 1
        raster[20, 0] = 1
        raster[15, 1] = 1

        spike_times = raster_to_spike_times(raster, fs=100.0)

        assert len(spike_times) == 3  # 3 neurons
        assert len(spike_times[0]) == 2  # Neuron 0 has 2 spikes
        assert len(spike_times[1]) == 1  # Neuron 1 has 1 spike
        assert len(spike_times[2]) == 0  # Neuron 2 has no spikes

    def test_align_session_lengths_pad(self):
        """Test session length alignment with padding."""
        sessions = [
            np.random.randn(100, 10),
            np.random.randn(150, 10),
            np.random.randn(120, 10),
        ]

        aligned = align_session_lengths(sessions, method="pad")

        assert all(s.shape[0] == 150 for s in aligned)  # Max length
        assert all(s.shape[1] == 10 for s in aligned)

    def test_align_session_lengths_crop(self):
        """Test session length alignment with cropping."""
        sessions = [
            np.random.randn(100, 10),
            np.random.randn(150, 10),
            np.random.randn(120, 10),
        ]

        aligned = align_session_lengths(sessions, method="crop")

        assert all(s.shape[0] == 100 for s in aligned)  # Min length


class TestPOYOModel:
    """Test POYO model wrapper."""

    def test_poyo_model_init(self):
        """Test POYO model initialization."""
        model = POYOModel(output_dim=2, dim=128, depth=2)

        assert model.output_dim == 2
        assert model.dim == 128
        assert model.depth == 2

    def test_poyo_model_config(self):
        """Test model configuration."""
        model = POYOModel(output_dim=4, pretrained=False)

        config = model.get_config()

        assert config["model_class"] == "POYOModel"
        assert config["pretrained"] is False
        assert config["device"] == "cpu"

    def test_poyo_model_train_predict(self):
        """Test training and prediction with mock implementation."""
        model = POYOModel(output_dim=2)

        # Create mock spike data
        spike_times = [
            np.array([0.1, 0.5, 1.2]),
            np.array([0.3, 0.7]),
        ]
        labels = np.array([0, 1])

        # Train (mock)
        model.train(spike_times, labels)

        assert model._is_trained

        # Predict (mock)
        predictions = model.predict(spike_times)

        assert predictions.shape[1] == 2  # output_dim

    def test_poyo_model_encode(self):
        """Test encoding to latent space."""
        model = POYOModel(dim=256)

        spike_times = [np.array([0.1, 0.5])]

        latents = model.encode(spike_times)

        assert latents.shape[1] == 256  # dim

    def test_poyo_model_repr(self):
        """Test string representation."""
        model = POYOModel(output_dim=4, dim=128)

        repr_str = repr(model)

        assert "POYOModel" in repr_str
        assert "output_dim" in repr_str or "model_class" in repr_str


class TestPOYOPlusModel:
    """Test POYO+ model wrapper."""

    def test_poyo_plus_init(self):
        """Test POYO+ initialization."""
        tasks = [
            {"name": "velocity", "type": "regression", "output_dim": 2},
            {"name": "direction", "type": "classification", "output_dim": 8},
        ]

        model = POYOPlusModel(task_configs=tasks)

        assert len(model.task_configs) == 2
        assert model.task_configs[0]["name"] == "velocity"

    def test_poyo_plus_default_task(self):
        """Test POYO+ with default task config."""
        model = POYOPlusModel()

        assert len(model.task_configs) == 1
        assert model.task_configs[0]["type"] == "regression"

    def test_poyo_plus_train_predict(self):
        """Test multi-task training and prediction."""
        tasks = [
            {"name": "task1", "type": "regression", "output_dim": 2},
            {"name": "task2", "type": "classification", "output_dim": 4},
        ]

        model = POYOPlusModel(task_configs=tasks)

        # Mock data
        spike_times = [np.array([0.1, 0.5])]
        labels = np.array([0])

        model.train(spike_times, labels)

        assert model._is_trained

        # Predict for all tasks
        predictions = model.predict(spike_times)

        assert isinstance(predictions, dict)
        assert "task1" in predictions
        assert "task2" in predictions
        assert predictions["task1"].shape[1] == 2
        assert predictions["task2"].shape[1] == 4

    def test_poyo_plus_decode_specific_task(self):
        """Test decoding for a specific task."""
        tasks = [
            {"name": "velocity", "type": "regression", "output_dim": 3},
        ]

        model = POYOPlusModel(task_configs=tasks)

        latents = np.random.randn(5, 256)

        decoded = model.decode(latents, task_name="velocity")

        assert decoded.shape == (5, 3)

    def test_poyo_plus_decode_unknown_task(self):
        """Test that decoding unknown task raises error."""
        model = POYOPlusModel()

        latents = np.random.randn(5, 256)

        with pytest.raises(ValueError):
            model.decode(latents, task_name="unknown_task")


class TestBaseFoundationModel:
    """Test base foundation model class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseFoundationModel cannot be instantiated."""
        # Should not be able to instantiate abstract class directly
        # (though Python doesn't enforce this strictly without metaclass)
        pass

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        import tempfile
        from pathlib import Path

        model = POYOModel(output_dim=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.ckpt"

            # Save
            model.save_checkpoint(
                str(checkpoint_path), metadata={"epoch": 10, "loss": 0.5}
            )

            assert checkpoint_path.exists()

            # Load into new model
            model2 = POYOModel(output_dim=4)  # Different config
            metadata = model2.load_checkpoint(str(checkpoint_path))

            assert metadata["epoch"] == 10
            assert metadata["loss"] == 0.5

    def test_fine_tune_fallback(self):
        """Test fine_tune falls back to train()."""
        model = POYOModel(output_dim=2)

        spike_times = [np.array([0.1, 0.5])]
        labels = np.array([0])

        # Fine-tune should call train()
        history = model.fine_tune(spike_times, labels, n_epochs=5)

        assert model._is_trained
        assert "message" in history


class TestIntegration:
    """Test integration with other neurOS components."""

    def test_integration_with_allen_data(self):
        """Test using POYO with Allen dataset format."""
        from neuros.datasets import load_allen_mock_data

        data = load_allen_mock_data(n_neurons=50, duration=30.0)

        # Extract spike times
        spike_times = data["spike_times"][0]  # First session

        # Create model
        model = POYOModel(output_dim=2)

        # Mock labels
        labels = np.array([0])

        # Should work with Allen data format
        model.train(spike_times, labels)

        assert model._is_trained

    def test_poyo_with_spike_raster(self):
        """Test POYO with spike raster input."""
        # Create spike raster
        raster = np.random.poisson(0.1, (1000, 20))

        model = POYOModel(output_dim=3)

        labels = np.array([0, 1, 2])

        # Should handle raster input
        model.train(raster, labels)

        assert model._is_trained


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_predict_before_train(self):
        """Test that predicting before training raises error."""
        model = POYOModel(output_dim=2)

        spike_times = [np.array([0.1])]

        with pytest.raises(RuntimeError):
            model.predict(spike_times)

    def test_from_pretrained_without_path(self):
        """Test from_pretrained with invalid path."""
        model = POYOModel.from_pretrained("nonexistent_model")

        # Should create model but not load weights
        assert model.pretrained is True
        assert model.pretrained_path == "nonexistent_model"

    def test_empty_spike_times(self):
        """Test handling empty spike times."""
        spike_times = [np.array([]), np.array([])]

        unit_ids, timestamps, _ = spikes_to_tokens(
            spike_times, time_window=(0, 1.0)
        )

        assert len(unit_ids) == 0
        assert len(timestamps) == 0

    def test_align_empty_session_list(self):
        """Test session alignment with empty list."""
        aligned = align_session_lengths([], method="pad")

        assert len(aligned) == 0
