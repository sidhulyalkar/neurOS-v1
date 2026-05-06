"""Functional connectivity analysis for astrocyte networks."""

import numpy as np
from neuros_astro.metadata.schema import AstroEvent, AstroGraph


def events_to_binary_matrix(
    events: list[AstroEvent],
    frame_rate_hz: float,
    bin_size_s: float = 1.0,
    duration_s: float | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Convert event list to binary activity matrix.

    Args:
        events: List of AstroEvent objects
        frame_rate_hz: Frame rate for time conversion
        bin_size_s: Time bin size in seconds
        duration_s: Total duration (if None, inferred from events)

    Returns:
        binary_matrix: Binary matrix [n_bins, n_regions]
        region_ids: List of region IDs
        bin_times_s: Array of bin center times
    """
    if len(events) == 0:
        return np.array([]), [], np.array([])

    # Extract unique regions
    region_ids = sorted(set(e.region_id for e in events if e.region_id is not None))

    if len(region_ids) == 0:
        return np.array([]), [], np.array([])

    region_to_idx = {rid: idx for idx, rid in enumerate(region_ids)}

    # Determine duration
    if duration_s is None:
        max_offset_frame = max(e.offset_frame for e in events)
        duration_s = (max_offset_frame + 1) / frame_rate_hz

    # Create time bins
    n_bins = int(np.ceil(duration_s / bin_size_s))
    binary_matrix = np.zeros((n_bins, len(region_ids)), dtype=np.int8)
    bin_times_s = np.arange(n_bins) * bin_size_s + bin_size_s / 2

    # Fill binary matrix
    for event in events:
        if event.region_id is None or event.region_id not in region_to_idx:
            continue

        region_idx = region_to_idx[event.region_id]

        # Convert frame indices to time
        onset_s = event.onset_frame / frame_rate_hz
        offset_s = event.offset_frame / frame_rate_hz

        # Find bins that overlap with event
        start_bin = int(onset_s / bin_size_s)
        end_bin = int(offset_s / bin_size_s)

        # Mark bins as active
        for bin_idx in range(start_bin, min(end_bin + 1, n_bins)):
            binary_matrix[bin_idx, region_idx] = 1

    return binary_matrix, region_ids, bin_times_s


def build_event_coactivation_graph(
    events: list[AstroEvent],
    session_id: str,
    frame_rate_hz: float,
    window_size_s: float = 30.0,
    stride_s: float = 5.0,
    bin_size_s: float = 1.0,
    min_edge_weight: float = 0.1,
) -> list[AstroGraph]:
    """
    Build astrocyte coactivation graphs using sliding windows.

    Uses Jaccard similarity of event coactivation:
        edge_weight = (bins where both active) / (bins where either active)

    Args:
        events: List of AstroEvent objects
        session_id: Session identifier
        frame_rate_hz: Frame rate
        window_size_s: Window size in seconds
        stride_s: Stride between windows in seconds
        bin_size_s: Time bin size for activity matrix
        min_edge_weight: Minimum edge weight threshold

    Returns:
        List of AstroGraph objects (one per window)
    """
    if len(events) == 0:
        return []

    # Convert to binary matrix
    binary_matrix, region_ids, bin_times_s = events_to_binary_matrix(
        events=events,
        frame_rate_hz=frame_rate_hz,
        bin_size_s=bin_size_s,
    )

    if len(binary_matrix) == 0 or len(region_ids) == 0:
        return []

    # Determine windows
    total_duration_s = bin_times_s[-1] + bin_size_s / 2
    window_starts = np.arange(0, max(0.1, total_duration_s - window_size_s + stride_s), stride_s)

    graphs = []

    for window_start_s in window_starts:
        window_end_s = window_start_s + window_size_s

        # Find bins in this window
        in_window = (bin_times_s >= window_start_s) & (bin_times_s < window_end_s)
        window_matrix = binary_matrix[in_window, :]

        if window_matrix.shape[0] == 0:
            continue

        # Compute Jaccard coactivation between all pairs
        edges = []
        edge_weights = []

        n_regions = len(region_ids)

        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                activity_i = window_matrix[:, i]
                activity_j = window_matrix[:, j]

                # Jaccard: intersection / union
                both_active = np.sum((activity_i == 1) & (activity_j == 1))
                either_active = np.sum((activity_i == 1) | (activity_j == 1))

                if either_active > 0:
                    jaccard = both_active / either_active

                    if jaccard >= min_edge_weight:
                        edges.append((region_ids[i], region_ids[j]))
                        edge_weights.append(float(jaccard))

        # Create graph
        graph = AstroGraph(
            session_id=session_id,
            window_start_s=float(window_start_s),
            window_end_s=float(window_end_s),
            nodes=region_ids,
            edges=edges,
            edge_weights=edge_weights,
            edge_metric="event_coactivation",
        )

        graphs.append(graph)

    return graphs
