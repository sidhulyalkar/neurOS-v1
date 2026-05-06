"""
Allen Brain Observatory data loader for neuros-astro.

This module provides utilities to load and process Allen Visual Coding
2-photon calcium imaging data for astrocyte event detection.
"""

import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    HAS_ALLENSDK = True
except ImportError:
    HAS_ALLENSDK = False
    warnings.warn(
        "allensdk not installed. Install with: pip install allensdk"
    )


def load_allen_session_from_npz(
    session_path: str | Path,
) -> Tuple[np.ndarray, dict]:
    """
    Load Allen session from preprocessed NPZ file.

    This loads trial-aligned data that was already preprocessed.
    For continuous traces, you'll need to access the Allen cache directly.

    Args:
        session_path: Path to NPZ file

    Returns:
        Tuple of (responses, metadata)
        - responses: [n_trials, n_cells] array
        - metadata: dict with session information

    Example:
        >>> data, metadata = load_allen_session_from_npz("2p_session_545446482.npz")
        >>> print(f"Shape: {data.shape}, Cells: {metadata.get('n_cells', 'unknown')}")
    """
    session_path = Path(session_path)

    if not session_path.exists():
        raise FileNotFoundError(f"Session file not found: {session_path}")

    data = np.load(session_path, allow_pickle=True)

    # Extract response matrix (usually 'X' in your format)
    if 'X' in data:
        responses = data['X']  # [n_trials, n_cells]
    else:
        raise ValueError(f"Expected 'X' key in NPZ file, found: {list(data.keys())}")

    # Build metadata dict
    metadata = {
        'session_path': str(session_path),
        'session_id': session_path.stem,
        'n_trials': responses.shape[0] if len(responses.shape) > 0 else 0,
        'n_cells': responses.shape[1] if len(responses.shape) > 1 else 0,
    }

    # Add other available metadata
    for key in data.keys():
        if key != 'X' and not key.startswith('_'):
            try:
                metadata[key] = data[key]
            except:
                pass  # Skip if can't convert

    return responses, metadata


def load_allen_continuous_traces(
    session_id: int,
    cache_dir: str | Path = "./allen_cache",
    stimulus_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load continuous fluorescence traces from Allen Brain Observatory.

    This requires AllenSDK and downloads data if not cached.

    Args:
        session_id: Allen session ID (e.g., 501498760)
        cache_dir: Path to Allen cache directory
        stimulus_name: Optional stimulus to filter by

    Returns:
        Tuple of (dff_traces, timestamps, metadata)
        - dff_traces: [n_cells, n_timepoints] array of dF/F
        - timestamps: [n_timepoints] array of timestamps
        - metadata: dict with session information

    Example:
        >>> traces, time, meta = load_allen_continuous_traces(501498760)
        >>> print(f"Loaded {traces.shape[0]} cells, {traces.shape[1]} timepoints")
        >>> print(f"Duration: {time[-1] - time[0]:.1f}s")
    """
    if not HAS_ALLENSDK:
        raise ImportError(
            "allensdk is required. Install with: pip install allensdk"
        )

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)

    # Initialize Brain Observatory cache
    print(f"📦 Loading Allen session {session_id}...")
    boc = BrainObservatoryCache(manifest_file=str(cache_dir / 'manifest.json'))

    # Get dataset
    dataset = boc.get_ophys_experiment_data(session_id)

    # Get dF/F traces
    print("  ✓ Extracting dF/F traces...")
    _, dff_traces = dataset.get_dff_traces()  # Returns (cell_specimen_ids, dff_traces)
    # dff_traces shape: [n_cells, n_timepoints]

    # Get timestamps
    timestamps = dataset.get_fluorescence_timestamps()

    # Get metadata
    metadata = {
        'session_id': session_id,
        'n_cells': dff_traces.shape[0],
        'n_timepoints': dff_traces.shape[1],
        'duration_s': timestamps[-1] - timestamps[0],
        'frame_rate_hz': 1.0 / np.median(np.diff(timestamps)),
        'stimulus_name': stimulus_name,
    }

    # Add experiment metadata
    experiment_info = boc.get_ophys_experiments(experiment_container_ids=[session_id])
    if len(experiment_info) > 0:
        exp = experiment_info[0]
        metadata['cre_line'] = exp.get('cre_line', 'unknown')
        metadata['imaging_depth'] = exp.get('imaging_depth_um', 'unknown')
        metadata['targeted_structure'] = exp.get('targeted_structure', 'unknown')

    print(f"  ✓ Loaded {metadata['n_cells']} cells, "
          f"{metadata['n_timepoints']} timepoints")
    print(f"  ✓ Duration: {metadata['duration_s']:.1f}s, "
          f"Frame rate: {metadata['frame_rate_hz']:.2f} Hz")

    return dff_traces, timestamps, metadata


def convert_trial_aligned_to_continuous(
    trial_responses: np.ndarray,
    trial_duration_s: float = 0.5,
    frame_rate_hz: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert trial-aligned responses to pseudo-continuous traces.

    This is a workaround for when you only have trial-aligned data
    but want to run event detection.

    Args:
        trial_responses: [n_trials, n_cells] array
        trial_duration_s: Duration of each trial window
        frame_rate_hz: Sampling rate

    Returns:
        Tuple of (continuous_traces, timestamps)
        - continuous_traces: [n_cells, n_timepoints] array
        - timestamps: [n_timepoints] array

    Warning:
        This is an approximation! Real continuous traces are better.
        Adjacent trials may have discontinuities.

    Example:
        >>> # From trial-aligned data
        >>> trial_data, _ = load_allen_session_from_npz("session.npz")
        >>> traces, time = convert_trial_aligned_to_continuous(trial_data)
    """
    n_trials, n_cells = trial_responses.shape
    frames_per_trial = int(trial_duration_s * frame_rate_hz)

    # Create continuous traces by concatenating trials
    # This is an approximation - there may be gaps between trials!
    continuous_traces = np.zeros((n_cells, n_trials * frames_per_trial))

    for trial_idx in range(n_trials):
        start_frame = trial_idx * frames_per_trial
        end_frame = start_frame + frames_per_trial

        # Expand single trial value to multiple frames (crude approximation)
        for cell_idx in range(n_cells):
            continuous_traces[cell_idx, start_frame:end_frame] = trial_responses[trial_idx, cell_idx]

    # Create timestamps
    total_duration = n_trials * trial_duration_s
    timestamps = np.linspace(0, total_duration, continuous_traces.shape[1])

    return continuous_traces, timestamps


def select_candidate_astrocyte_cells(
    dff_traces: np.ndarray,
    min_event_duration_s: float = 1.0,
    max_event_rate_hz: float = 0.5,
    frame_rate_hz: float = 30.0,
) -> np.ndarray:
    """
    Heuristically select cells that may be astrocytes based on activity.

    WARNING: This is purely heuristic! Without genetic labeling,
    we cannot definitively identify astrocytes. Use with caution.

    Astrocytes typically have:
    - Slower calcium events (> 1s duration)
    - Lower event rates (< 0.5 Hz)
    - Smaller amplitude events than neurons

    Args:
        dff_traces: [n_cells, n_timepoints] dF/F traces
        min_event_duration_s: Minimum expected event duration
        max_event_rate_hz: Maximum expected event rate
        frame_rate_hz: Sampling rate

    Returns:
        Boolean array indicating candidate astrocyte cells

    Example:
        >>> traces, time, meta = load_allen_continuous_traces(session_id)
        >>> astro_mask = select_candidate_astrocyte_cells(traces, frame_rate_hz=meta['frame_rate_hz'])
        >>> astro_traces = traces[astro_mask]
        >>> print(f"Selected {astro_mask.sum()} / {len(astro_mask)} cells as candidates")
    """
    n_cells, n_timepoints = dff_traces.shape

    # Simple heuristic: look for slow, infrequent events
    # This is NOT definitive - just a starting point for exploration

    candidate_mask = np.ones(n_cells, dtype=bool)

    for cell_idx in range(n_cells):
        trace = dff_traces[cell_idx]

        # Compute rough event rate using zero crossings of z-scored trace
        z_trace = (trace - np.median(trace)) / (np.median(np.abs(trace - np.median(trace))) + 1e-10)
        above_threshold = z_trace > 2.0

        # Count transitions
        transitions = np.diff(above_threshold.astype(int))
        n_onsets = np.sum(transitions > 0)

        event_rate = n_onsets / (n_timepoints / frame_rate_hz)

        # Exclude if event rate is too high (likely neuron)
        if event_rate > max_event_rate_hz:
            candidate_mask[cell_idx] = False

    return candidate_mask
