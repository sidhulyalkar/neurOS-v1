"""
Allen Institute dataset loaders for neural foundation models.

Provides interfaces to Allen Institute datasets including Visual Coding,
Neuropixels, and other large-scale neural recordings suitable for training
foundation models like POYO, NDT, and CEBRA.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AllenDatasetConfig:
    """Configuration for Allen Institute dataset loading.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset ('visual_coding', 'neuropixels', etc.).
    cache_dir : Path or str, optional
        Directory for caching downloaded data.
    download : bool, default=True
        Whether to download data if not cached.
    preprocess : bool, default=True
        Whether to apply standard preprocessing.
    subset : str, optional
        Load only a subset of data ('train', 'test', 'small', etc.).
    """

    dataset_name: str
    cache_dir: Optional[Path] = None
    download: bool = True
    preprocess: bool = True
    subset: Optional[str] = None


def load_allen_visual_coding(
    cache_dir: Optional[str] = None,
    download: bool = True,
    subset: Optional[str] = None,
) -> Dict[str, Any]:
    """Load Allen Visual Coding Neuropixels dataset.

    This dataset contains high-density neural recordings from mouse visual cortex
    during presentation of various visual stimuli. Suitable for training foundation
    models that learn visual cortex representations.

    Parameters
    ----------
    cache_dir : str, optional
        Directory to cache downloaded data. Defaults to ~/.neuros/datasets/allen/.
    download : bool, default=True
        Whether to download data if not already cached.
    subset : str, optional
        Load a subset: 'small' (demo), 'train', 'test', or None (all).

    Returns
    -------
    dict
        Dictionary containing:
        - 'spike_times': List of spike time arrays per neuron
        - 'spike_clusters': Cluster assignments for each spike
        - 'neurons': Metadata for each neuron (brain region, cell type, etc.)
        - 'stimuli': Information about visual stimuli presented
        - 'behavior': Behavioral data (running speed, pupil size, etc.)
        - 'sessions': Session metadata

    Examples
    --------
    >>> data = load_allen_visual_coding(subset='small')
    >>> print(f"Loaded {len(data['spike_times'])} neurons")
    >>> print(f"Recording duration: {data['duration']:.1f} seconds")

    References
    ----------
    - Allen Institute Visual Coding - Neuropixels:
      https://portal.brain-map.org/explore/circuits/visual-coding-neuropixels

    Notes
    -----
    This function requires the AllenSDK package:
        pip install allensdk

    For the full dataset, you'll need significant disk space (~100GB+).
    Use subset='small' for testing and development.
    """
    try:
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import (
            EcephysProjectCache,
        )
    except ImportError:
        raise ImportError(
            "AllenSDK is required to load Allen Institute datasets. "
            "Install with: pip install allensdk"
        )

    # Set up cache directory
    if cache_dir is None:
        cache_dir = Path.home() / ".neuros" / "datasets" / "allen" / "visual_coding"
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Allen Visual Coding dataset to {cache_dir}")

    # Initialize cache
    manifest_path = cache_dir / "manifest.json"
    cache = EcephysProjectCache.from_warehouse(manifest=str(manifest_path))

    # Get sessions
    sessions = cache.get_session_table()

    if subset == "small":
        # Load just one session for demo/testing
        sessions = sessions.head(1)
        logger.info("Loading small subset (1 session)")
    elif subset == "train":
        # Use first 80% for training
        n_train = int(len(sessions) * 0.8)
        sessions = sessions.head(n_train)
        logger.info(f"Loading training subset ({n_train} sessions)")
    elif subset == "test":
        # Use last 20% for testing
        n_train = int(len(sessions) * 0.8)
        sessions = sessions.tail(len(sessions) - n_train)
        logger.info(f"Loading test subset ({len(sessions) - n_train} sessions)")

    # Load data from sessions
    all_spike_times = []
    all_spike_clusters = []
    all_neurons = []
    all_stimuli = []
    all_behavior = []
    session_metadata = []

    for session_id in sessions.index:
        logger.info(f"Loading session {session_id}")

        try:
            session = cache.get_session_data(session_id)

            # Extract spike times and clusters
            spike_times = session.spike_times
            units = session.units

            # Store per-session data
            session_metadata.append({
                "session_id": session_id,
                "n_units": len(units),
                "duration": float(np.max(spike_times)) if len(spike_times) > 0 else 0,
            })

            all_spike_times.append(spike_times)
            all_spike_clusters.append(units.index.values)
            all_neurons.append(units.to_dict("records"))

            # Store stimulus information
            if hasattr(session, "stimulus_presentations"):
                all_stimuli.append(session.stimulus_presentations.to_dict("records"))

            # Store behavioral data
            if hasattr(session, "running_speed"):
                all_behavior.append({
                    "running_speed": session.running_speed.values,
                    "timestamps": session.running_speed.index.values,
                })

        except Exception as e:
            logger.warning(f"Failed to load session {session_id}: {e}")
            continue

    logger.info(f"Successfully loaded {len(session_metadata)} sessions")

    return {
        "spike_times": all_spike_times,
        "spike_clusters": all_spike_clusters,
        "neurons": all_neurons,
        "stimuli": all_stimuli,
        "behavior": all_behavior,
        "sessions": session_metadata,
        "duration": sum(s["duration"] for s in session_metadata),
        "total_units": sum(s["n_units"] for s in session_metadata),
    }


def load_allen_neuropixels(
    cache_dir: Optional[str] = None,
    download: bool = True,
    brain_regions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load Allen Neuropixels dataset with optional brain region filtering.

    Parameters
    ----------
    cache_dir : str, optional
        Directory to cache downloaded data.
    download : bool, default=True
        Whether to download data if not already cached.
    brain_regions : list of str, optional
        Filter to specific brain regions (e.g., ['VISp', 'VISl', 'CA1']).

    Returns
    -------
    dict
        Dictionary with neural recordings and metadata.

    Examples
    --------
    >>> # Load only visual cortex recordings
    >>> data = load_allen_neuropixels(brain_regions=['VISp', 'VISl'])
    >>> print(f"Loaded {data['total_units']} units from visual cortex")
    """
    data = load_allen_visual_coding(cache_dir=cache_dir, download=download)

    # Filter by brain region if specified
    if brain_regions is not None:
        filtered_neurons = []
        filtered_spike_times = []
        filtered_spike_clusters = []

        for neurons, spikes, clusters in zip(
            data["neurons"], data["spike_times"], data["spike_clusters"]
        ):
            # Filter neurons by brain region
            keep_indices = [
                i for i, n in enumerate(neurons)
                if n.get("ecephys_structure_acronym") in brain_regions
            ]

            if keep_indices:
                filtered_neurons.append([neurons[i] for i in keep_indices])
                # Note: This is simplified - in practice you'd need to filter spike times too
                filtered_spike_times.append(spikes)
                filtered_spike_clusters.append(clusters)

        data["neurons"] = filtered_neurons
        data["spike_times"] = filtered_spike_times
        data["spike_clusters"] = filtered_spike_clusters
        data["brain_regions"] = brain_regions

        logger.info(f"Filtered to brain regions: {brain_regions}")

    return data


def load_allen_mock_data(n_neurons: int = 100, duration: float = 60.0) -> Dict[str, Any]:
    """Generate mock Allen-style data for testing without downloading.

    Parameters
    ----------
    n_neurons : int, default=100
        Number of neurons to simulate.
    duration : float, default=60.0
        Recording duration in seconds.

    Returns
    -------
    dict
        Mock dataset with same structure as real Allen data.

    Examples
    --------
    >>> data = load_allen_mock_data(n_neurons=50, duration=30.0)
    >>> print(f"Mock data: {data['total_units']} units, {data['duration']:.1f}s")
    """
    rng = np.random.default_rng(42)

    # Generate spike times (Poisson process with varying rates)
    spike_times_list = []
    for _ in range(n_neurons):
        # Random firing rate between 0.5 and 20 Hz
        rate = rng.uniform(0.5, 20.0)
        n_spikes = int(rate * duration)
        spike_times = np.sort(rng.uniform(0, duration, n_spikes))
        spike_times_list.append(spike_times)

    # Generate neuron metadata
    brain_regions = ["VISp", "VISl", "VISal", "VISpm", "CA1", "DG", "LGd"]
    cell_types = ["excitatory", "inhibitory", "unclassified"]

    neurons = []
    for i in range(n_neurons):
        neurons.append({
            "unit_id": i,
            "ecephys_structure_acronym": rng.choice(brain_regions),
            "cell_type": rng.choice(cell_types),
            "firing_rate": len(spike_times_list[i]) / duration,
            "quality": rng.choice(["good", "ok"]),
        })

    # Generate mock stimuli
    n_stimuli = 50
    stimuli = []
    for i in range(n_stimuli):
        start_time = (duration / n_stimuli) * i
        stimuli.append({
            "stimulus_name": f"stimulus_{i % 5}",
            "start_time": start_time,
            "stop_time": start_time + (duration / n_stimuli) * 0.8,
            "orientation": rng.uniform(0, 180) if i % 5 < 3 else None,
        })

    # Generate mock behavior (running speed)
    n_behavior_samples = int(duration * 60)  # 60 Hz sampling
    behavior = {
        "running_speed": rng.uniform(0, 50, n_behavior_samples),
        "timestamps": np.linspace(0, duration, n_behavior_samples),
    }

    return {
        "spike_times": [spike_times_list],
        "spike_clusters": [np.arange(n_neurons)],
        "neurons": [neurons],
        "stimuli": [stimuli],
        "behavior": [behavior],
        "sessions": [{
            "session_id": "mock_session_001",
            "n_units": n_neurons,
            "duration": duration,
        }],
        "duration": duration,
        "total_units": n_neurons,
        "is_mock": True,
    }


def convert_to_spike_raster(
    spike_times: List[np.ndarray],
    bin_size: float = 0.001,
    duration: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert spike times to binned spike raster.

    Parameters
    ----------
    spike_times : list of np.ndarray
        Spike times for each neuron.
    bin_size : float, default=0.001
        Bin size in seconds (default 1ms).
    duration : float, optional
        Total duration. If None, use max spike time.

    Returns
    -------
    spike_raster : np.ndarray
        Binned spike counts, shape (n_bins, n_neurons).
    time_bins : np.ndarray
        Time bin centers.

    Examples
    --------
    >>> spike_times = [np.array([0.1, 0.5, 1.2]), np.array([0.3, 0.7])]
    >>> raster, times = convert_to_spike_raster(spike_times, bin_size=0.1)
    >>> raster.shape
    (15, 2)  # depends on duration
    """
    if duration is None:
        duration = max(np.max(st) if len(st) > 0 else 0 for st in spike_times)

    n_bins = int(np.ceil(duration / bin_size))
    n_neurons = len(spike_times)

    spike_raster = np.zeros((n_bins, n_neurons))

    for neuron_id, times in enumerate(spike_times):
        if len(times) > 0:
            bins = (times / bin_size).astype(int)
            bins = bins[bins < n_bins]  # Clip to valid range
            np.add.at(spike_raster[:, neuron_id], bins, 1)

    time_bins = np.arange(n_bins) * bin_size + bin_size / 2

    return spike_raster, time_bins


# ==============================================================================
# Validation Extensions for SAE Feature Analysis
# ==============================================================================

from .base_dataset import BaseNeuralDataset, NeuralWindow
from scipy.stats import circmean


class AllenVisualCodingValidator(BaseNeuralDataset):
    """
    Extended Allen Visual Coding dataset loader for SAE validation experiments.

    Focuses on orientation tuning validation using drifting gratings stimulus.
    Implements the BaseNeuralDataset interface for consistent cross-modal analysis.

    Parameters
    ----------
    session_id : int, optional
        Specific session ID to load. If None, auto-selects a good session.
    data_path : str, default='./allen_data'
        Path to store Allen data cache.
    cache_dir : str, default='./cache'
        Directory for caching processed data.
    brain_areas : List[str], default=['V1']
        Brain areas to include in analysis.
    min_units : int, default=100
        Minimum number of good units required for session selection.
    use_all_units : bool, default=False
        If True, use all units. If False, use only "good quality" units.
        Setting to True can increase sample size but may include noisy units.

    Examples
    --------
    >>> validator = AllenVisualCodingValidator(brain_areas=['V1', 'LM'])
    >>> windows = validator.get_neural_windows(window_length=1.0, stride=0.5, bin_size=0.02)
    >>> labels = validator.get_task_labels()
    >>> print(f"Extracted {len(windows)} windows with orientations {labels['orientation']}")

    References
    ----------
    Allen Institute Visual Coding - Neuropixels:
    https://portal.brain-map.org/explore/circuits/visual-coding-neuropixels

    Notes
    -----
    Requires allensdk: pip install allensdk
    """

    def __init__(
        self,
        session_id: Optional[int] = None,
        data_path: str = "./allen_data",
        cache_dir: str = "./cache",
        brain_areas: List[str] = None,
        min_units: int = 100,
        use_all_units: bool = False
    ):
        super().__init__(data_path, cache_dir)

        if brain_areas is None:
            brain_areas = ["V1"]

        self.session_id = session_id
        self.brain_areas = brain_areas
        self.min_units = min_units
        self.use_all_units = use_all_units

        try:
            from allensdk.brain_observatory.ecephys.ecephys_project_cache import (
                EcephysProjectCache,
            )
        except ImportError:
            raise ImportError(
                "AllenSDK is required. Install with: pip install allensdk"
            )

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.cache = EcephysProjectCache.from_warehouse(
            manifest=str(Path(cache_dir) / "manifest.json")
        )

        self.session = None
        self.stimulus_table = None
        self.units = None
        self._load_session()

    def _load_session(self):
        """Load Allen session data with quality filtering."""
        if self.session_id is None:
            # Auto-select good session for validation
            sessions = self.cache.get_session_table()

            # Filter to appropriate sessions
            valid_sessions = sessions[
                sessions.session_type == 'brain_observatory_1.1'
            ]

            # Find session with good unit count in target areas
            for _, session_row in valid_sessions.iterrows():
                try:
                    test_session = self.cache.get_session_data(session_row.name)
                    test_units = test_session.units[
                        test_session.units.ecephys_structure_acronym.isin(self.brain_areas)
                    ]

                    # Try to filter by quality if available
                    if 'quality' in test_units.columns:
                        good_units = test_units[test_units.quality == 'good']
                    else:
                        # If no quality column, use all units
                        good_units = test_units

                    if len(good_units) >= self.min_units:
                        self.session_id = session_row.name
                        logger.info(
                            f"Auto-selected session {self.session_id} with "
                            f"{len(good_units)} units"
                        )
                        break
                except Exception as e:
                    logger.warning(f"Failed to load session {session_row.name}: {e}")
                    continue

            if self.session_id is None:
                raise ValueError(
                    f"No suitable sessions found with {self.min_units}+ units "
                    f"in {self.brain_areas}"
                )

        self.session = self.cache.get_session_data(self.session_id)

        # Get drifting gratings stimulus table
        stim_tables = self.session.stimulus_presentations
        drifting_gratings = stim_tables[
            stim_tables.stimulus_name == 'drifting_gratings'
        ]
        self.stimulus_table = drifting_gratings

        # Filter to units in target brain areas
        units_in_areas = self.session.units[
            self.session.units.ecephys_structure_acronym.isin(self.brain_areas)
        ]

        # Filter by quality if requested (default: only use good quality)
        if self.use_all_units:
            self.units = units_in_areas
            quality_note = "all"
        else:
            if 'quality' in units_in_areas.columns:
                self.units = units_in_areas[units_in_areas.quality == 'good']
                quality_note = "good quality"
            else:
                self.units = units_in_areas
                quality_note = "all (no quality column)"

        logger.info(
            f"Loaded session {self.session_id} with {len(self.units)} {quality_note} units, "
            f"{len(self.stimulus_table)} drifting grating presentations"
        )

    def get_neural_windows(
        self,
        window_length: float = 1.0,
        stride: float = 0.5,
        bin_size: float = 0.02
    ) -> List[NeuralWindow]:
        """Extract spike windows aligned to drifting gratings stimulus.

        Parameters
        ----------
        window_length : float, default=1.0
            Window duration in seconds.
        stride : float, default=0.5
            Step size in seconds.
        bin_size : float, default=0.02
            Spike binning resolution (20ms default).

        Returns
        -------
        List[NeuralWindow]
            List of windows with spike data and orientation labels.
        """
        # Get spike times for all good units
        spike_times = {}
        for unit_id in self.units.index:
            spikes = self.session.spike_times[unit_id]
            spike_times[unit_id] = spikes

        # Process each stimulus presentation
        windows = []
        for _, stim in self.stimulus_table.iterrows():
            stim_start = stim.start_time
            stim_stop = stim.stop_time
            stim_duration = stim_stop - stim_start

            # Skip short stimuli
            if stim_duration < window_length:
                continue

            # Extract windows with stride
            window_start = stim_start
            while window_start + window_length <= stim_stop:
                window_end = window_start + window_length

                # Bin spikes for all units in this window
                spike_counts = []
                time_bins = np.arange(window_start, window_end + bin_size, bin_size)

                for unit_id in self.units.index:
                    unit_spikes = spike_times[unit_id]
                    window_spikes = unit_spikes[
                        (unit_spikes >= window_start) &
                        (unit_spikes < window_end)
                    ]

                    # Bin spike counts
                    counts, _ = np.histogram(window_spikes, bins=time_bins)
                    spike_counts.append(counts)

                spike_data = np.array(spike_counts).T  # [time_bins, n_units]

                # Create window object
                window = NeuralWindow(
                    data=spike_data,
                    labels=np.array([
                        stim.orientation,
                        stim.temporal_frequency if 'temporal_frequency' in stim else 0,
                    ]),
                    metadata={
                        'stimulus_condition_id': stim.stimulus_condition_id if 'stimulus_condition_id' in stim else None,
                        'session_id': self.session_id,
                        'start_time': window_start,
                        'end_time': window_end,
                        'orientation': stim.orientation,
                        'temporal_frequency': stim.temporal_frequency if 'temporal_frequency' in stim else None,
                        'brain_area': self.brain_areas[0] if len(self.brain_areas) == 1 else 'mixed',
                        'n_units': len(self.units)
                    },
                    window_id=f"session_{self.session_id}_stim_{stim.name}_t_{window_start:.2f}"
                )
                windows.append(window)

                window_start += stride

        logger.info(f"Extracted {len(windows)} neural windows")
        return windows

    def get_task_labels(self) -> Dict[str, np.ndarray]:
        """Return orientation and other stimulus labels.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with keys: orientation, temporal_frequency,
            orientation_sin, orientation_cos (for circular correlation).
        """
        orientations = self.stimulus_table.orientation.values

        # Convert to numeric and remove NaN values
        orientations = pd.to_numeric(orientations, errors='coerce')
        valid_mask = ~np.isnan(orientations)
        orientations = orientations[valid_mask]

        result = {
            'orientation': orientations,
            'orientation_sin': np.sin(np.deg2rad(orientations * 2)),
            'orientation_cos': np.cos(np.deg2rad(orientations * 2))
        }

        if 'temporal_frequency' in self.stimulus_table.columns:
            temp_freqs = pd.to_numeric(self.stimulus_table.temporal_frequency.values, errors='coerce')
            result['temporal_frequency'] = temp_freqs

        return result

    def get_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        temporal_split: bool = True
    ) -> Tuple[List[int], List[int], List[int]]:
        """Split by stimulus trials to avoid temporal leakage.

        Parameters
        ----------
        train_ratio : float, default=0.7
            Fraction for training.
        val_ratio : float, default=0.15
            Fraction for validation.
        temporal_split : bool, default=True
            If True, split by trial order (recommended).

        Returns
        -------
        Tuple[List[int], List[int], List[int]]
            Train, validation, and test indices.
        """
        n_trials = len(self.stimulus_table)

        if temporal_split:
            # Split by trial order (temporal)
            train_end = int(n_trials * train_ratio)
            val_end = int(n_trials * (train_ratio + val_ratio))

            train_trials = list(range(train_end))
            val_trials = list(range(train_end, val_end))
            test_trials = list(range(val_end, n_trials))
        else:
            # Random split (not recommended but available)
            indices = np.random.permutation(n_trials)
            train_end = int(n_trials * train_ratio)
            val_end = int(n_trials * (train_ratio + val_ratio))

            train_trials = indices[:train_end].tolist()
            val_trials = indices[train_end:val_end].tolist()
            test_trials = indices[val_end:].tolist()

        return train_trials, val_trials, test_trials

    def get_neural_properties(self) -> Dict[str, Any]:
        """Return Allen-specific neural properties.

        Returns
        -------
        Dict[str, Any]
            Neural recording properties including sampling rate, modality,
            species, and recording details.
        """
        return {
            'n_units': len(self.units),
            'brain_regions': self.brain_areas,
            'sampling_rate': 30000.0,  # Allen Neuropixels sampling rate
            'modality': 'extracellular_spikes',
            'species': 'mouse',
            'recording_type': 'neuropixels',
            'session_id': self.session_id,
            'stimulus_type': 'drifting_gratings'
        }

    def compute_orientation_tuning_curves(
        self,
        unit_responses: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute orientation tuning curves for validation.

        Parameters
        ----------
        unit_responses : np.ndarray
            Response matrix with shape [n_stimuli, n_units].

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with tuning curve statistics including preferred
            orientations and selectivity metrics.

        Examples
        --------
        >>> windows = validator.get_neural_windows()
        >>> responses = np.array([w.data.mean(axis=0) for w in windows])
        >>> tuning = validator.compute_orientation_tuning_curves(responses)
        >>> print(f"Preferred orientations: {tuning['preferred_orientations']}")
        """
        orientations = self.stimulus_table.orientation.values
        # Remove NaN orientations
        valid_mask = ~np.isnan(orientations)
        orientations = orientations[valid_mask]

        # Handle shape mismatch
        if len(orientations) > len(unit_responses):
            orientations = orientations[:len(unit_responses)]
        elif len(unit_responses) > len(orientations):
            unit_responses = unit_responses[:len(orientations)]

        unique_orientations = np.unique(orientations)
        n_units = unit_responses.shape[1] if len(unit_responses.shape) > 1 else 1

        tuning_curves = np.zeros((n_units, len(unique_orientations)))

        for i, ori in enumerate(unique_orientations):
            ori_mask = orientations == ori
            if len(unit_responses.shape) > 1:
                tuning_curves[:, i] = np.mean(unit_responses[ori_mask], axis=0)
            else:
                tuning_curves[0, i] = np.mean(unit_responses[ori_mask])

        # Compute orientation selectivity metrics
        preferred_orientations = unique_orientations[np.argmax(tuning_curves, axis=1)]
        selectivity = np.max(tuning_curves, axis=1) - np.mean(tuning_curves, axis=1)

        return {
            'tuning_curves': tuning_curves,
            'preferred_orientations': preferred_orientations,
            'selectivity': selectivity,
            'orientations': unique_orientations
        }
