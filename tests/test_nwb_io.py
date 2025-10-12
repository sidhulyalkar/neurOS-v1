"""
Tests for NWB file I/O functionality.

Tests the NWBLoader and NWBWriter classes for reading and writing
neural data in the Neurodata Without Borders (NWB) format.
"""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from neuros.io.nwb_loader import NWB_AVAILABLE

if NWB_AVAILABLE:
    from pynwb import NWBHDF5IO, NWBFile, TimeSeries
    from pynwb.ecephys import ElectricalSeries, LFP
    from pynwb.behavior import BehavioralTimeSeries, Position, SpatialSeries
    from pynwb.file import Subject

    from neuros.io.nwb_loader import NWBLoader, NWBWriter


# Skip all tests if pynwb not available
pytestmark = pytest.mark.skipif(
    not NWB_AVAILABLE,
    reason="pynwb not installed"
)


@pytest.fixture
def temp_nwb_path():
    """Create a temporary file path for NWB files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.nwb"


@pytest.fixture
def sample_spike_data():
    """Generate sample spike data for testing."""
    np.random.seed(42)
    return {
        "unit_0": np.sort(np.random.uniform(0, 10, 50)),
        "unit_1": np.sort(np.random.uniform(0, 10, 75)),
        "unit_2": np.sort(np.random.uniform(0, 10, 100)),
    }


@pytest.fixture
def sample_lfp_data():
    """Generate sample LFP data for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_channels = 4
    return np.random.randn(n_samples, n_channels) * 0.001  # Scale to realistic LFP values


@pytest.fixture
def sample_behavior_data():
    """Generate sample behavioral data for testing."""
    np.random.seed(42)
    n_samples = 100
    return {
        "position": np.random.randn(n_samples, 2),  # 2D position
        "velocity": np.random.randn(n_samples, 2),  # 2D velocity
    }


@pytest.fixture
def basic_nwb_file(temp_nwb_path, sample_spike_data, sample_lfp_data):
    """Create a basic NWB file with spike and LFP data."""
    # Create NWB file
    nwbfile = NWBFile(
        session_description="Test session",
        identifier="test_001",
        session_start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        experimenter="Test User",
        lab="Test Lab",
        institution="Test Institution",
    )

    # Add subject information
    nwbfile.subject = Subject(
        subject_id="mouse_001",
        species="Mus musculus",
        age="P90D",
    )

    # Add spike data with integer IDs
    for idx, (unit_name, spike_times) in enumerate(sample_spike_data.items()):
        nwbfile.add_unit(spike_times=spike_times, id=idx)

    # Add LFP data
    device = nwbfile.create_device(name="test_device")
    electrode_group = nwbfile.create_electrode_group(
        name="test_electrodes",
        description="Test electrode group",
        location="hippocampus",
        device=device,
    )

    for i in range(sample_lfp_data.shape[1]):
        nwbfile.add_electrode(
            id=i,
            x=0.0, y=0.0, z=0.0,
            imp=np.nan,
            location="hippocampus",
            filtering="bandpass 0.1-300 Hz",
            group=electrode_group,
        )

    electrode_table_region = nwbfile.create_electrode_table_region(
        region=list(range(sample_lfp_data.shape[1])),
        description="All electrodes",
    )

    lfp_series = ElectricalSeries(
        name="LFP",
        data=sample_lfp_data,
        electrodes=electrode_table_region,
        starting_time=0.0,
        rate=1000.0,  # 1 kHz sampling rate
    )

    ecephys_module = nwbfile.create_processing_module(
        name="ecephys",
        description="Extracellular electrophysiology",
    )
    ecephys_module.add(LFP(electrical_series=lfp_series))

    # Save to file
    with NWBHDF5IO(str(temp_nwb_path), "w") as io:
        io.write(nwbfile)

    return temp_nwb_path


@pytest.fixture
def nwb_file_with_behavior(temp_nwb_path, sample_behavior_data):
    """Create an NWB file with behavioral data."""
    nwbfile = NWBFile(
        session_description="Behavior test session",
        identifier="test_002",
        session_start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )

    # Add position data
    position = Position(name="Position")
    spatial_series = SpatialSeries(
        name="position",
        data=sample_behavior_data["position"],
        reference_frame="arena",
        starting_time=0.0,
        rate=30.0,  # 30 Hz
    )
    position.add_spatial_series(spatial_series)

    behavior_module = nwbfile.create_processing_module(
        name="behavior",
        description="Behavioral data",
    )
    behavior_module.add(position)

    # Add velocity as behavioral time series
    velocity_ts = TimeSeries(
        name="velocity",
        data=sample_behavior_data["velocity"],
        unit="m/s",
        starting_time=0.0,
        rate=30.0,
    )

    behavioral_ts = BehavioralTimeSeries(name="BehavioralTimeSeries")
    behavioral_ts.add_timeseries(velocity_ts)
    behavior_module.add(behavioral_ts)

    # Save to file
    with NWBHDF5IO(str(temp_nwb_path), "w") as io:
        io.write(nwbfile)

    return temp_nwb_path


@pytest.fixture
def nwb_file_with_trials(temp_nwb_path):
    """Create an NWB file with trial information."""
    nwbfile = NWBFile(
        session_description="Trial test session",
        identifier="test_003",
        session_start_time=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )

    # Add trials
    nwbfile.add_trial_column(name="correct", description="Whether trial was correct")
    nwbfile.add_trial_column(name="condition", description="Trial condition")

    for i in range(10):
        nwbfile.add_trial(
            start_time=float(i * 2.0),
            stop_time=float(i * 2.0 + 1.5),
            correct=bool(i % 2),
            condition=f"condition_{i % 3}",
        )

    # Save to file
    with NWBHDF5IO(str(temp_nwb_path), "w") as io:
        io.write(nwbfile)

    return temp_nwb_path


class TestNWBLoader:
    """Tests for NWBLoader class."""

    def test_loader_init(self, basic_nwb_file):
        """Test NWBLoader initialization."""
        loader = NWBLoader(basic_nwb_file)
        assert loader.file_path == basic_nwb_file
        assert loader.nwbfile is not None
        assert loader.io is not None
        loader.close()

    def test_loader_init_nonexistent_file(self):
        """Test NWBLoader with non-existent file."""
        with pytest.raises(FileNotFoundError):
            NWBLoader("/nonexistent/path/file.nwb")

    def test_loader_context_manager(self, basic_nwb_file):
        """Test NWBLoader as context manager."""
        with NWBLoader(basic_nwb_file) as loader:
            assert loader.nwbfile is not None
        # File should be closed after context exit

    def test_load_spikes_all_units(self, basic_nwb_file, sample_spike_data):
        """Test loading all spike units."""
        with NWBLoader(basic_nwb_file) as loader:
            spikes = loader.load_spikes()

        assert len(spikes) == len(sample_spike_data)
        for unit_id in range(len(sample_spike_data)):
            assert f"unit_{unit_id}" in spikes
            assert isinstance(spikes[f"unit_{unit_id}"], np.ndarray)

    def test_load_spikes_specific_units(self, basic_nwb_file):
        """Test loading specific spike units."""
        with NWBLoader(basic_nwb_file) as loader:
            spikes = loader.load_spikes(unit_ids=[0, 2])

        assert len(spikes) == 2
        assert "unit_0" in spikes
        assert "unit_2" in spikes
        assert "unit_1" not in spikes

    def test_load_spikes_empty_file(self, temp_nwb_path):
        """Test loading spikes from file without units."""
        # Create minimal NWB file without units
        nwbfile = NWBFile(
            session_description="Empty session",
            identifier="test_empty",
            session_start_time=datetime(2025, 1, 1, 12, 0, 0),
        )

        with NWBHDF5IO(str(temp_nwb_path), "w") as io:
            io.write(nwbfile)

        with NWBLoader(temp_nwb_path) as loader:
            spikes = loader.load_spikes()

        assert len(spikes) == 0

    def test_load_lfp(self, basic_nwb_file, sample_lfp_data):
        """Test loading LFP data."""
        with NWBLoader(basic_nwb_file) as loader:
            lfp = loader.load_lfp()

        assert "data" in lfp
        assert "timestamps" in lfp
        assert "rate" in lfp
        assert "electrode_ids" in lfp
        assert "series_name" in lfp

        assert lfp["data"].shape == sample_lfp_data.shape
        assert lfp["rate"] == 1000.0
        assert len(lfp["electrode_ids"]) == sample_lfp_data.shape[1]

    def test_load_lfp_specific_electrodes(self, basic_nwb_file):
        """Test loading LFP from specific electrodes."""
        electrode_ids = [0, 2]
        with NWBLoader(basic_nwb_file) as loader:
            lfp = loader.load_lfp(electrode_ids=electrode_ids)

        assert lfp["data"].shape[1] == len(electrode_ids)
        assert lfp["electrode_ids"] == electrode_ids

    def test_load_lfp_empty_file(self, temp_nwb_path):
        """Test loading LFP from file without LFP data."""
        nwbfile = NWBFile(
            session_description="No LFP session",
            identifier="test_no_lfp",
            session_start_time=datetime(2025, 1, 1, 12, 0, 0),
        )

        with NWBHDF5IO(str(temp_nwb_path), "w") as io:
            io.write(nwbfile)

        with NWBLoader(temp_nwb_path) as loader:
            lfp = loader.load_lfp()

        assert len(lfp) == 0

    def test_load_behavior(self, nwb_file_with_behavior, sample_behavior_data):
        """Test loading behavioral data."""
        with NWBLoader(nwb_file_with_behavior) as loader:
            behavior = loader.load_behavior()

        assert len(behavior) > 0

        # Check position data
        position_keys = [k for k in behavior.keys() if "position" in k.lower()]
        assert len(position_keys) > 0

        for key in position_keys:
            assert "data" in behavior[key]
            assert "timestamps" in behavior[key]
            assert isinstance(behavior[key]["data"], np.ndarray)

    def test_load_behavior_empty_file(self, temp_nwb_path):
        """Test loading behavior from file without behavioral data."""
        nwbfile = NWBFile(
            session_description="No behavior session",
            identifier="test_no_behavior",
            session_start_time=datetime(2025, 1, 1, 12, 0, 0),
        )

        with NWBHDF5IO(str(temp_nwb_path), "w") as io:
            io.write(nwbfile)

        with NWBLoader(temp_nwb_path) as loader:
            behavior = loader.load_behavior()

        assert len(behavior) == 0

    def test_load_trials(self, nwb_file_with_trials):
        """Test loading trial information."""
        with NWBLoader(nwb_file_with_trials) as loader:
            trials = loader.load_trials()

        assert trials is not None
        assert "start_time" in trials
        assert "stop_time" in trials
        assert "correct" in trials
        assert "condition" in trials

        assert len(trials["start_time"]) == 10
        assert isinstance(trials["start_time"], np.ndarray)

    def test_load_trials_empty_file(self, temp_nwb_path):
        """Test loading trials from file without trial data."""
        nwbfile = NWBFile(
            session_description="No trials session",
            identifier="test_no_trials",
            session_start_time=datetime(2025, 1, 1, 12, 0, 0),
        )

        with NWBHDF5IO(str(temp_nwb_path), "w") as io:
            io.write(nwbfile)

        with NWBLoader(temp_nwb_path) as loader:
            trials = loader.load_trials()

        assert trials is None

    def test_get_session_metadata(self, basic_nwb_file):
        """Test retrieving session metadata."""
        with NWBLoader(basic_nwb_file) as loader:
            metadata = loader.get_session_metadata()

        assert "session_description" in metadata
        assert "session_start_time" in metadata
        assert "experimenter" in metadata
        assert "lab" in metadata
        assert "institution" in metadata
        assert "subject" in metadata

        assert metadata["session_description"] == "Test session"
        assert metadata["experimenter"] == ("Test User",)
        assert metadata["lab"] == "Test Lab"
        assert metadata["institution"] == "Test Institution"

        # Check subject info
        assert metadata["subject"]["subject_id"] == "mouse_001"
        assert metadata["subject"]["species"] == "Mus musculus"


class TestNWBWriter:
    """Tests for NWBWriter class."""

    def test_writer_init(self, temp_nwb_path):
        """Test NWBWriter initialization."""
        writer = NWBWriter(
            temp_nwb_path,
            session_description="Test session",
            experimenter="Test User",
        )

        assert writer.file_path == temp_nwb_path
        assert writer.nwbfile is not None
        assert writer.nwbfile.session_description == "Test session"

    def test_writer_init_default_start_time(self, temp_nwb_path):
        """Test NWBWriter with default session start time."""
        writer = NWBWriter(temp_nwb_path, session_description="Test")

        assert writer.nwbfile.session_start_time is not None
        # Should be close to current time (comparing timezone-aware datetimes)
        time_diff = datetime.now(timezone.utc) - writer.nwbfile.session_start_time
        assert abs(time_diff.total_seconds()) < 5  # Within 5 seconds

    def test_add_spikes(self, temp_nwb_path, sample_spike_data):
        """Test adding spike data to NWB file."""
        writer = NWBWriter(temp_nwb_path, session_description="Spike test")
        writer.add_spikes(sample_spike_data)
        writer.save()

        # Verify by loading
        with NWBLoader(temp_nwb_path) as loader:
            spikes = loader.load_spikes()

        assert len(spikes) == len(sample_spike_data)
        for unit_name in sample_spike_data.keys():
            assert unit_name in spikes
            # Check spike times are approximately equal
            np.testing.assert_array_almost_equal(
                spikes[unit_name],
                sample_spike_data[unit_name],
                decimal=6,
            )

    def test_add_lfp(self, temp_nwb_path, sample_lfp_data):
        """Test adding LFP data to NWB file."""
        writer = NWBWriter(temp_nwb_path, session_description="LFP test")
        writer.add_lfp(sample_lfp_data, sampling_rate=1000.0)
        writer.save()

        # Verify by loading
        with NWBLoader(temp_nwb_path) as loader:
            lfp = loader.load_lfp()

        assert lfp["data"].shape == sample_lfp_data.shape
        assert lfp["rate"] == 1000.0
        np.testing.assert_array_almost_equal(
            lfp["data"],
            sample_lfp_data,
            decimal=10,
        )

    def test_add_lfp_with_timestamps(self, temp_nwb_path, sample_lfp_data):
        """Test adding LFP data with explicit timestamps."""
        timestamps = np.arange(sample_lfp_data.shape[0]) / 1000.0  # 1 kHz

        writer = NWBWriter(temp_nwb_path, session_description="LFP timestamp test")
        writer.add_lfp(sample_lfp_data, sampling_rate=1000.0, timestamps=timestamps)
        writer.save()

        # Verify by loading
        with NWBLoader(temp_nwb_path) as loader:
            lfp = loader.load_lfp()

        assert lfp["timestamps"] is not None
        np.testing.assert_array_almost_equal(
            lfp["timestamps"],
            timestamps,
            decimal=10,
        )

    def test_add_behavior(self, temp_nwb_path, sample_behavior_data):
        """Test adding behavioral data to NWB file."""
        writer = NWBWriter(temp_nwb_path, session_description="Behavior test")
        writer.add_behavior(sample_behavior_data, sampling_rate=30.0)
        writer.save()

        # Verify by loading
        with NWBLoader(temp_nwb_path) as loader:
            behavior = loader.load_behavior()

        assert len(behavior) > 0

        # Check that we can find our data
        for key in sample_behavior_data.keys():
            matching_keys = [k for k in behavior.keys() if key in k]
            assert len(matching_keys) > 0

    def test_add_multiple_data_types(self, temp_nwb_path, sample_spike_data,
                                     sample_lfp_data, sample_behavior_data):
        """Test adding multiple data types to single NWB file."""
        writer = NWBWriter(
            temp_nwb_path,
            session_description="Multi-modal test",
            experimenter="Test User",
            lab="Test Lab",
        )

        writer.add_spikes(sample_spike_data)
        writer.add_lfp(sample_lfp_data, sampling_rate=1000.0)
        writer.add_behavior(sample_behavior_data, sampling_rate=30.0)
        writer.save()

        # Verify all data types
        with NWBLoader(temp_nwb_path) as loader:
            spikes = loader.load_spikes()
            lfp = loader.load_lfp()
            behavior = loader.load_behavior()
            metadata = loader.get_session_metadata()

        assert len(spikes) == len(sample_spike_data)
        assert lfp["data"].shape == sample_lfp_data.shape
        assert len(behavior) > 0
        assert metadata["experimenter"] == ("Test User",)
        assert metadata["lab"] == "Test Lab"

    def test_writer_context_manager(self, temp_nwb_path, sample_spike_data):
        """Test NWBWriter as context manager."""
        with NWBWriter(temp_nwb_path, session_description="Context test") as writer:
            writer.add_spikes(sample_spike_data)
        # File should be saved automatically on context exit

        assert temp_nwb_path.exists()

        # Verify data was saved
        with NWBLoader(temp_nwb_path) as loader:
            spikes = loader.load_spikes()

        assert len(spikes) == len(sample_spike_data)

    def test_save_creates_file(self, temp_nwb_path):
        """Test that save() creates the file on disk."""
        writer = NWBWriter(temp_nwb_path, session_description="Save test")

        assert not temp_nwb_path.exists()

        writer.save()

        assert temp_nwb_path.exists()
        assert temp_nwb_path.stat().st_size > 0


class TestNWBIntegration:
    """Integration tests for NWB I/O."""

    def test_round_trip_spikes(self, temp_nwb_path, sample_spike_data):
        """Test writing and reading spike data (round trip)."""
        # Write
        with NWBWriter(temp_nwb_path, session_description="Round trip test") as writer:
            writer.add_spikes(sample_spike_data)

        # Read
        with NWBLoader(temp_nwb_path) as loader:
            loaded_spikes = loader.load_spikes()

        # Compare
        assert len(loaded_spikes) == len(sample_spike_data)
        for unit_name, original_times in sample_spike_data.items():
            assert unit_name in loaded_spikes
            np.testing.assert_array_almost_equal(
                loaded_spikes[unit_name],
                original_times,
                decimal=6,
            )

    def test_round_trip_lfp(self, temp_nwb_path, sample_lfp_data):
        """Test writing and reading LFP data (round trip)."""
        # Write
        with NWBWriter(temp_nwb_path, session_description="LFP round trip") as writer:
            writer.add_lfp(sample_lfp_data, sampling_rate=1000.0)

        # Read
        with NWBLoader(temp_nwb_path) as loader:
            loaded_lfp = loader.load_lfp()

        # Compare
        np.testing.assert_array_almost_equal(
            loaded_lfp["data"],
            sample_lfp_data,
            decimal=10,
        )
        assert loaded_lfp["rate"] == 1000.0

    def test_round_trip_multimodal(self, temp_nwb_path, sample_spike_data,
                                   sample_lfp_data, sample_behavior_data):
        """Test writing and reading all data types (round trip)."""
        # Write
        with NWBWriter(
            temp_nwb_path,
            session_description="Multi-modal round trip",
            experimenter="Test User",
        ) as writer:
            writer.add_spikes(sample_spike_data)
            writer.add_lfp(sample_lfp_data, sampling_rate=1000.0)
            writer.add_behavior(sample_behavior_data, sampling_rate=30.0)

        # Read
        with NWBLoader(temp_nwb_path) as loader:
            spikes = loader.load_spikes()
            lfp = loader.load_lfp()
            behavior = loader.load_behavior()
            metadata = loader.get_session_metadata()

        # Verify all data types
        assert len(spikes) == len(sample_spike_data)
        assert lfp["data"].shape == sample_lfp_data.shape
        assert len(behavior) > 0
        assert metadata["session_description"] == "Multi-modal round trip"


class TestNWBErrorHandling:
    """Test error handling in NWB I/O."""

    def test_loader_invalid_unit_ids(self, basic_nwb_file):
        """Test loading spikes with invalid unit IDs."""
        with NWBLoader(basic_nwb_file) as loader:
            # Request non-existent units
            spikes = loader.load_spikes(unit_ids=[999, 1000])

        # Should handle gracefully and return empty or available data
        # (behavior depends on implementation)

    def test_loader_invalid_electrode_ids(self, basic_nwb_file):
        """Test loading LFP with invalid electrode IDs."""
        with NWBLoader(basic_nwb_file) as loader:
            # This should either fail gracefully or raise an appropriate error
            with pytest.raises((IndexError, ValueError, Exception)):
                loader.load_lfp(electrode_ids=[999, 1000])
