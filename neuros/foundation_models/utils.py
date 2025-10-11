"""
Utility functions for foundation models.

Provides data conversion, preprocessing, and helper functions for working with
foundation models in neurOS.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


def spikes_to_tokens(
    spike_times: List[np.ndarray],
    time_window: Tuple[float, float],
    *,
    bin_size: Optional[float] = None,
    max_spikes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert spike times to token format for transformer models.

    Parameters
    ----------
    spike_times : list of np.ndarray
        Spike times for each neuron.
    time_window : tuple of float
        (start_time, end_time) for the window of interest.
    bin_size : float, optional
        Bin size for discretizing time. If None, use continuous times.
    max_spikes : int, optional
        Maximum number of spikes to include (for truncation).

    Returns
    -------
    unit_ids : np.ndarray
        Unit ID for each spike (which neuron fired).
    spike_timestamps : np.ndarray
        Timestamp for each spike.
    spike_counts : np.ndarray
        Count of spikes in each bin (if using binning).

    Examples
    --------
    >>> spike_times = [np.array([0.1, 0.5, 1.2]), np.array([0.3, 0.7])]
    >>> unit_ids, timestamps, counts = spikes_to_tokens(
    ...     spike_times, time_window=(0, 2.0), bin_size=0.1
    ... )
    """
    start_time, end_time = time_window
    all_unit_ids = []
    all_timestamps = []

    for unit_id, times in enumerate(spike_times):
        # Filter to time window
        mask = (times >= start_time) & (times < end_time)
        times_in_window = times[mask]

        if len(times_in_window) == 0:
            continue

        # Add unit IDs and timestamps
        all_unit_ids.extend([unit_id] * len(times_in_window))
        all_timestamps.extend(times_in_window.tolist())

    # Convert to arrays
    unit_ids = np.array(all_unit_ids, dtype=np.int64)
    timestamps = np.array(all_timestamps, dtype=np.float32)

    # Sort by time
    sort_indices = np.argsort(timestamps)
    unit_ids = unit_ids[sort_indices]
    timestamps = timestamps[sort_indices]

    # Truncate if needed
    if max_spikes is not None and len(unit_ids) > max_spikes:
        unit_ids = unit_ids[:max_spikes]
        timestamps = timestamps[:max_spikes]
        logger.warning(
            f"Truncated from {len(sort_indices)} to {max_spikes} spikes"
        )

    # Bin if requested
    if bin_size is not None:
        n_bins = int(np.ceil((end_time - start_time) / bin_size))
        spike_counts = np.zeros(n_bins, dtype=np.int32)

        bin_indices = ((timestamps - start_time) / bin_size).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        np.add.at(spike_counts, bin_indices, 1)

        return unit_ids, timestamps, spike_counts

    return unit_ids, timestamps, np.array([])


def create_session_embeddings(
    n_sessions: int,
    embedding_dim: int = 64,
    *,
    method: str = "learned",
) -> np.ndarray:
    """Create session embeddings for multi-session models.

    Parameters
    ----------
    n_sessions : int
        Number of sessions.
    embedding_dim : int, default=64
        Dimensionality of embeddings.
    method : str, default='learned'
        Method for creating embeddings ('learned', 'random', 'positional').

    Returns
    -------
    np.ndarray
        Session embeddings of shape (n_sessions, embedding_dim).

    Examples
    --------
    >>> session_emb = create_session_embeddings(10, embedding_dim=64)
    >>> session_emb.shape
    (10, 64)
    """
    if method == "random":
        # Random Gaussian embeddings
        rng = np.random.default_rng(42)
        embeddings = rng.normal(0, 1, (n_sessions, embedding_dim))
    elif method == "positional":
        # Sinusoidal positional encodings
        position = np.arange(n_sessions)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim)
        )

        embeddings = np.zeros((n_sessions, embedding_dim))
        embeddings[:, 0::2] = np.sin(position * div_term)
        embeddings[:, 1::2] = np.cos(position * div_term)
    elif method == "learned":
        # Initialize with small random values (will be learned during training)
        rng = np.random.default_rng(42)
        embeddings = rng.normal(0, 0.02, (n_sessions, embedding_dim))
    else:
        raise ValueError(f"Unknown method: {method}")

    return embeddings.astype(np.float32)


def create_readout_spec(
    task_configs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Create readout specification for multi-task models.

    Parameters
    ----------
    task_configs : list of dict
        List of task configurations, each with:
        - 'name': str, task name
        - 'type': str, one of {'regression', 'classification', 'segmentation'}
        - 'output_dim': int, output dimensionality
        - 'loss_weight': float, optional weighting in multi-task loss

    Returns
    -------
    list of dict
        Readout specifications formatted for the model.

    Examples
    --------
    >>> tasks = [
    ...     {'name': 'velocity', 'type': 'regression', 'output_dim': 2},
    ...     {'name': 'direction', 'type': 'classification', 'output_dim': 8},
    ... ]
    >>> spec = create_readout_spec(tasks)
    """
    readout_spec = []

    for task in task_configs:
        spec_entry = {
            "name": task["name"],
            "type": task["type"],
            "output_dim": task["output_dim"],
            "loss_weight": task.get("loss_weight", 1.0),
        }

        # Add task-specific parameters
        if task["type"] == "classification":
            spec_entry["num_classes"] = task["output_dim"]
        elif task["type"] == "regression":
            spec_entry["continuous"] = True

        readout_spec.append(spec_entry)

    return readout_spec


def raster_to_spike_times(
    spike_raster: np.ndarray,
    fs: float,
    *,
    threshold: float = 0.5,
) -> List[np.ndarray]:
    """Convert spike raster back to spike times.

    Parameters
    ----------
    spike_raster : np.ndarray
        Binned spike counts, shape (n_bins, n_neurons).
    fs : float
        Sampling frequency in Hz.
    threshold : float, default=0.5
        Threshold for detecting spikes (for continuous rasters).

    Returns
    -------
    list of np.ndarray
        Spike times for each neuron.

    Examples
    --------
    >>> raster = np.random.poisson(0.1, (1000, 50))  # 1000 bins, 50 neurons
    >>> spike_times = raster_to_spike_times(raster, fs=1000.0)
    >>> len(spike_times)
    50
    """
    n_bins, n_neurons = spike_raster.shape
    dt = 1.0 / fs

    spike_times_list = []

    for neuron_idx in range(n_neurons):
        # Find bins with spikes
        spike_bins = np.where(spike_raster[:, neuron_idx] > threshold)[0]

        # Convert bin indices to times
        times = spike_bins * dt

        # If multiple spikes in a bin, replicate times
        if spike_raster.dtype in [np.int32, np.int64]:
            # Integer counts: replicate times
            counts = spike_raster[spike_bins, neuron_idx].astype(int)
            times = np.repeat(times, counts)

        spike_times_list.append(times)

    return spike_times_list


def align_session_lengths(
    data_list: List[np.ndarray],
    *,
    method: str = "pad",
    target_length: Optional[int] = None,
) -> List[np.ndarray]:
    """Align sessions to same length for batch processing.

    Parameters
    ----------
    data_list : list of np.ndarray
        List of data arrays with potentially different first dimensions.
    method : str, default='pad'
        Alignment method ('pad', 'crop', 'resample').
    target_length : int, optional
        Target length. If None, use max length (for pad) or min (for crop).

    Returns
    -------
    list of np.ndarray
        Aligned data arrays.

    Examples
    --------
    >>> sessions = [np.random.randn(100, 10), np.random.randn(150, 10)]
    >>> aligned = align_session_lengths(sessions, method='pad')
    >>> [s.shape for s in aligned]
    [(150, 10), (150, 10)]
    """
    if not data_list:
        return data_list

    lengths = [len(d) for d in data_list]

    if target_length is None:
        if method == "pad":
            target_length = max(lengths)
        elif method == "crop":
            target_length = min(lengths)
        else:
            target_length = int(np.mean(lengths))

    aligned = []

    for data in data_list:
        current_length = len(data)

        if current_length == target_length:
            aligned.append(data)
        elif current_length < target_length:
            if method == "pad":
                # Pad with zeros
                pad_width = [(0, target_length - current_length)]
                pad_width.extend([(0, 0)] * (data.ndim - 1))
                aligned.append(np.pad(data, pad_width, mode='constant'))
            elif method == "resample":
                # Simple linear interpolation
                from scipy.interpolate import interp1d

                x_old = np.linspace(0, 1, current_length)
                x_new = np.linspace(0, 1, target_length)

                if data.ndim == 1:
                    f = interp1d(x_old, data, kind='linear')
                    aligned.append(f(x_new))
                else:
                    resampled = np.zeros((target_length,) + data.shape[1:])
                    for i in range(data.shape[1]):
                        f = interp1d(x_old, data[:, i], kind='linear')
                        resampled[:, i] = f(x_new)
                    aligned.append(resampled)
            else:  # crop
                aligned.append(data[:target_length])
        else:  # current_length > target_length
            if method in ["pad", "crop"]:
                aligned.append(data[:target_length])
            else:  # resample
                from scipy.interpolate import interp1d

                x_old = np.linspace(0, 1, current_length)
                x_new = np.linspace(0, 1, target_length)

                if data.ndim == 1:
                    f = interp1d(x_old, data, kind='linear')
                    aligned.append(f(x_new))
                else:
                    resampled = np.zeros((target_length,) + data.shape[1:])
                    for i in range(data.shape[1]):
                        f = interp1d(x_old, data[:, i], kind='linear')
                        resampled[:, i] = f(x_new)
                    aligned.append(resampled)

    return aligned
