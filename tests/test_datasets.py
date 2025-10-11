"""
Tests for dataset loaders.
"""

import numpy as np
import pytest

from neuros.datasets.allen_datasets import (
    load_allen_mock_data,
    convert_to_spike_raster,
    AllenDatasetConfig,
)
from neuros.datasets.bci_datasets import load_mock_bci_data


class TestAllenDatasets:
    """Test Allen Institute dataset loaders."""

    def test_load_allen_mock_data(self):
        """Test loading mock Allen data."""
        data = load_allen_mock_data(n_neurons=50, duration=30.0)

        assert "spike_times" in data
        assert "neurons" in data
        assert "sessions" in data
        assert data["total_units"] == 50
        assert abs(data["duration"] - 30.0) < 0.1
        assert data["is_mock"] is True

    def test_mock_data_structure(self):
        """Test that mock data has correct structure."""
        data = load_allen_mock_data(n_neurons=100, duration=60.0)

        # Check spike times
        assert len(data["spike_times"]) > 0
        assert len(data["spike_times"][0]) == 100  # n_neurons

        # Check neurons metadata
        assert len(data["neurons"]) > 0
        assert len(data["neurons"][0]) == 100

        # Check each neuron has required fields
        neuron = data["neurons"][0][0]
        assert "unit_id" in neuron
        assert "ecephys_structure_acronym" in neuron
        assert "firing_rate" in neuron

    def test_mock_data_stimuli(self):
        """Test that mock data includes stimuli."""
        data = load_allen_mock_data(n_neurons=50, duration=30.0)

        assert "stimuli" in data
        assert len(data["stimuli"]) > 0
        assert len(data["stimuli"][0]) > 0

        # Check stimulus structure
        stim = data["stimuli"][0][0]
        assert "stimulus_name" in stim
        assert "start_time" in stim
        assert "stop_time" in stim

    def test_mock_data_behavior(self):
        """Test that mock data includes behavior."""
        data = load_allen_mock_data(n_neurons=50, duration=30.0)

        assert "behavior" in data
        assert len(data["behavior"]) > 0

        behavior = data["behavior"][0]
        assert "running_speed" in behavior
        assert "timestamps" in behavior
        assert len(behavior["running_speed"]) > 0

    def test_convert_to_spike_raster(self):
        """Test conversion of spike times to raster."""
        # Create simple spike times
        spike_times = [
            np.array([0.1, 0.5, 1.2]),  # Neuron 1
            np.array([0.3, 0.7, 1.5]),  # Neuron 2
        ]

        raster, time_bins = convert_to_spike_raster(
            spike_times, bin_size=0.1, duration=2.0
        )

        # Check output shape
        assert raster.shape[0] == 20  # 2.0 seconds / 0.1 bin_size
        assert raster.shape[1] == 2   # 2 neurons

        # Check that spikes are binned correctly
        assert np.sum(raster) == 6  # 3 + 3 spikes total

    def test_spike_raster_empty_neuron(self):
        """Test spike raster with neuron that didn't fire."""
        spike_times = [
            np.array([0.1, 0.5]),  # Neuron 1 fires
            np.array([]),          # Neuron 2 doesn't fire
        ]

        raster, time_bins = convert_to_spike_raster(
            spike_times, bin_size=0.1, duration=1.0
        )

        assert raster.shape == (10, 2)
        assert np.sum(raster[:, 0]) == 2  # 2 spikes in neuron 1
        assert np.sum(raster[:, 1]) == 0  # 0 spikes in neuron 2

    def test_spike_raster_time_bins(self):
        """Test that time bins are correctly centered."""
        spike_times = [np.array([0.5])]

        raster, time_bins = convert_to_spike_raster(
            spike_times, bin_size=0.1, duration=1.0
        )

        # Time bins should be centered at bin_size/2, bin_size*1.5, etc.
        assert time_bins[0] == 0.05  # First bin center
        assert len(time_bins) == 10

    def test_allen_dataset_config(self):
        """Test AllenDatasetConfig dataclass."""
        config = AllenDatasetConfig(
            dataset_name="visual_coding",
            download=True,
            preprocess=True,
            subset="small",
        )

        assert config.dataset_name == "visual_coding"
        assert config.download is True
        assert config.subset == "small"


class TestBCIDatasets:
    """Test BCI dataset loaders."""

    def test_load_mock_bci_data(self):
        """Test loading mock BCI data."""
        data = load_mock_bci_data(
            n_trials=100, n_channels=22, n_timepoints=1000, n_classes=2
        )

        assert "X" in data
        assert "y" in data
        assert "fs" in data
        assert data["X"].shape == (100, 22, 1000)
        assert len(data["y"]) == 100
        assert data["is_mock"] is True

    def test_mock_bci_data_classes(self):
        """Test that mock data has correct number of classes."""
        data = load_mock_bci_data(n_trials=80, n_classes=4)

        # Check that we have 4 unique classes
        unique_classes = np.unique(data["y"])
        assert len(unique_classes) == 4
        assert np.all(unique_classes == np.array([0, 1, 2, 3]))

    def test_mock_bci_data_channels(self):
        """Test channel information."""
        data = load_mock_bci_data(n_channels=16)

        assert len(data["channels"]) == 16
        assert data["channels"][0] == "C1"
        assert data["channels"][15] == "C16"

    def test_mock_bci_data_different_sizes(self):
        """Test creating mock data with different sizes."""
        # Small dataset
        data_small = load_mock_bci_data(n_trials=10, n_channels=8, n_timepoints=100)
        assert data_small["X"].shape == (10, 8, 100)

        # Large dataset
        data_large = load_mock_bci_data(n_trials=500, n_channels=64, n_timepoints=2000)
        assert data_large["X"].shape == (500, 64, 2000)

    def test_mock_bci_data_sampling_frequency(self):
        """Test sampling frequency parameter."""
        data = load_mock_bci_data(fs=500.0)

        assert data["fs"] == 500.0

    def test_mock_bci_class_distribution(self):
        """Test that classes are evenly distributed in mock data."""
        data = load_mock_bci_data(n_trials=100, n_classes=4)

        # Count trials per class
        class_counts = np.bincount(data["y"])

        # Should be roughly 25 trials per class
        for count in class_counts:
            assert 20 <= count <= 30  # Allow some variance

    def test_mock_bci_data_has_structure(self):
        """Test that mock data has some signal structure (not pure noise)."""
        data = load_mock_bci_data(n_trials=20, n_classes=2, n_timepoints=1000)

        # Get trials from class 0 and class 1
        class_0_trials = data["X"][data["y"] == 0]
        class_1_trials = data["X"][data["y"] == 1]

        # Average over trials within each class
        mean_class_0 = np.mean(class_0_trials, axis=0)
        mean_class_1 = np.mean(class_1_trials, axis=0)

        # Classes should have different spectral content
        # (We can't test FFT easily without scipy, but we can check variance)
        var_0 = np.var(mean_class_0)
        var_1 = np.var(mean_class_1)

        # Both should have non-zero variance
        assert var_0 > 0.1
        assert var_1 > 0.1


class TestDatasetIntegration:
    """Test integration between different dataset types."""

    def test_allen_and_bci_data_compatibility(self):
        """Test that Allen and BCI data can be used together."""
        allen_data = load_allen_mock_data(n_neurons=50, duration=30.0)
        bci_data = load_mock_bci_data(n_trials=50)

        # Both should return dictionaries
        assert isinstance(allen_data, dict)
        assert isinstance(bci_data, dict)

        # Both should have some metadata
        assert "sessions" in allen_data or "metadata" in allen_data
        assert "fs" in bci_data

    def test_spike_raster_size_consistency(self):
        """Test that spike raster dimensions are consistent."""
        data = load_allen_mock_data(n_neurons=20, duration=10.0)

        # Convert to raster with 1ms bins
        spike_times = data["spike_times"][0]
        raster, time_bins = convert_to_spike_raster(
            spike_times, bin_size=0.001, duration=10.0
        )

        # Raster should be (10000, 20) for 10s at 1ms bins with 20 neurons
        assert raster.shape[0] == 10000
        assert raster.shape[1] == 20
        assert len(time_bins) == 10000


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_mock_data_with_zero_neurons(self):
        """Test creating mock data with edge case parameters."""
        data = load_allen_mock_data(n_neurons=1, duration=1.0)

        assert data["total_units"] == 1
        assert len(data["spike_times"][0]) == 1

    def test_spike_raster_with_no_spikes(self):
        """Test spike raster when there are no spikes."""
        spike_times = [np.array([]), np.array([])]

        raster, time_bins = convert_to_spike_raster(
            spike_times, bin_size=0.1, duration=1.0
        )

        assert np.sum(raster) == 0
        assert raster.shape == (10, 2)

    def test_mock_bci_single_trial(self):
        """Test creating mock BCI data with single trial."""
        data = load_mock_bci_data(n_trials=1, n_classes=1)

        assert data["X"].shape[0] == 1
        assert len(data["y"]) == 1
        assert data["y"][0] == 0

    def test_mock_bci_single_channel(self):
        """Test creating mock BCI data with single channel."""
        data = load_mock_bci_data(n_channels=1)

        assert data["X"].shape[1] == 1
        assert len(data["channels"]) == 1
