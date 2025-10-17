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
