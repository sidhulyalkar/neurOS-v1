"""Event detection from astrocyte calcium traces and movies."""

import numpy as np
from scipy import ndimage
from typing import Any
from neuros_astro.metadata.schema import AstroEvent


def robust_zscore(trace: np.ndarray) -> np.ndarray:
    """
    Compute robust z-score using median and MAD (median absolute deviation).

    This is more robust to outliers than mean/std z-score.

    Args:
        trace: 1D array

    Returns:
        z_trace: Robust z-scored trace
    """
    trace = np.asarray(trace, dtype=np.float32)

    # Handle NaNs
    valid_mask = ~np.isnan(trace)
    if not np.any(valid_mask):
        return np.zeros_like(trace)

    median = np.median(trace[valid_mask])
    mad = np.median(np.abs(trace[valid_mask] - median))

    # Avoid divide by zero
    if mad < 1e-10:
        return np.zeros_like(trace)

    # Scale MAD to match std (for normal distribution)
    # MAD * 1.4826 ≈ std
    z_trace = (trace - median) / (mad * 1.4826)

    # Set NaN positions back to 0
    z_trace[~valid_mask] = 0

    return z_trace


def detect_events_from_trace(
    trace: np.ndarray,
    frame_rate_hz: float,
    session_id: str,
    region_id: str | None = None,
    z_threshold: float = 2.0,
    min_duration_s: float = 1.0,
    merge_gap_s: float = 0.5,
) -> list[AstroEvent]:
    """
    Detect astrocyte calcium events from a single trace.

    Uses robust z-scoring and contiguous threshold crossings.

    Args:
        trace: 1D fluorescence trace (dF/F or raw)
        frame_rate_hz: Imaging frame rate
        session_id: Session identifier
        region_id: Region/ROI identifier
        z_threshold: Z-score threshold for detection
        min_duration_s: Minimum event duration in seconds
        merge_gap_s: Merge events separated by less than this gap

    Returns:
        List of AstroEvent objects
    """
    if len(trace) == 0:
        return []

    # Compute robust z-score
    z_trace = robust_zscore(trace)

    # Find threshold crossings
    above_threshold = z_trace > z_threshold

    # Find contiguous regions
    labeled, n_regions = ndimage.label(above_threshold)

    if n_regions == 0:
        return []

    # Extract events
    events = []
    dt = 1.0 / frame_rate_hz

    for region_idx in range(1, n_regions + 1):
        mask = labeled == region_idx
        indices = np.where(mask)[0]

        if len(indices) == 0:
            continue

        onset_frame = int(indices[0])
        offset_frame = int(indices[-1])
        duration_s = (offset_frame - onset_frame + 1) * dt

        # Filter by minimum duration
        if duration_s < min_duration_s:
            continue

        # Find peak within event
        event_trace = trace[onset_frame : offset_frame + 1]
        peak_idx = np.argmax(event_trace)
        peak_frame = onset_frame + peak_idx
        peak_dff = float(event_trace[peak_idx])

        events.append({
            "onset_frame": onset_frame,
            "offset_frame": offset_frame,
            "peak_frame": peak_frame,
            "duration_s": duration_s,
            "peak_dff": peak_dff,
        })

    # Merge close events
    if merge_gap_s > 0:
        events = _merge_close_events(events, frame_rate_hz, merge_gap_s)

    # Convert to AstroEvent objects
    astro_events = []
    for idx, event in enumerate(events):
        event_id = f"{session_id}_{region_id or 'unknown'}_evt_{idx:04d}"

        astro_events.append(
            AstroEvent(
                event_id=event_id,
                session_id=session_id,
                region_id=region_id,
                onset_frame=event["onset_frame"],
                offset_frame=event["offset_frame"],
                peak_frame=event["peak_frame"],
                duration_s=event["duration_s"],
                peak_dff=event["peak_dff"],
                confidence=0.8,  # Default confidence
            )
        )

    return astro_events


def detect_events_from_traces(
    traces: np.ndarray,
    frame_rate_hz: float,
    session_id: str,
    region_ids: list[str] | None = None,
    z_threshold: float = 2.0,
    min_duration_s: float = 1.0,
    merge_gap_s: float = 0.5,
) -> list[AstroEvent]:
    """
    Detect events from multiple traces (batch processing).

    Args:
        traces: 2D array of shape [n_regions, n_frames]
        frame_rate_hz: Imaging frame rate
        session_id: Session identifier
        region_ids: List of region IDs (optional)
        z_threshold: Z-score threshold
        min_duration_s: Minimum event duration
        merge_gap_s: Merge gap in seconds

    Returns:
        List of AstroEvent objects from all regions
    """
    n_regions = traces.shape[0]

    if region_ids is None:
        region_ids = [f"roi_{i:03d}" for i in range(n_regions)]

    all_events = []

    for region_idx, region_id in enumerate(region_ids):
        trace = traces[region_idx, :]
        events = detect_events_from_trace(
            trace=trace,
            frame_rate_hz=frame_rate_hz,
            session_id=session_id,
            region_id=region_id,
            z_threshold=z_threshold,
            min_duration_s=min_duration_s,
            merge_gap_s=merge_gap_s,
        )
        all_events.extend(events)

    return all_events


def detect_candidate_events_from_movie(
    movie: np.ndarray,
    frame_rate_hz: float,
    session_id: str,
    z_threshold: float = 3.0,
    min_area_px: int = 10,
    min_duration_s: float = 0.5,
    max_distance_px: float = 20.0,
    max_events: int | None = None,
) -> list[AstroEvent]:
    """
    Detect candidate spatiotemporal events from calcium imaging movie.

    Uses per-pixel z-scoring, connected components, and temporal linking.

    Args:
        movie: 3D array [n_frames, height, width]
        frame_rate_hz: Imaging frame rate
        session_id: Session identifier
        z_threshold: Z-score threshold for detection
        min_area_px: Minimum event area in pixels
        min_duration_s: Minimum event duration in seconds
        max_distance_px: Maximum centroid distance for linking across frames
        max_events: Maximum number of events to return (for memory safety)

    Returns:
        List of AstroEvent objects
    """
    if movie.ndim != 3:
        raise ValueError(f"Movie must be 3D [frames, height, width], got shape {movie.shape}")

    n_frames, height, width = movie.shape

    # Compute per-pixel baseline and noise
    baseline = np.median(movie, axis=0)
    mad = np.median(np.abs(movie - baseline[None, :, :]), axis=0)

    # Avoid divide by zero
    mad = np.maximum(mad, 1e-10)

    # Compute z-movie
    z_movie = (movie - baseline[None, :, :]) / (mad[None, :, :] * 1.4826)

    # Threshold and find connected components per frame
    frame_components = []

    for frame_idx in range(n_frames):
        binary = z_movie[frame_idx, :, :] > z_threshold
        labeled, n_comps = ndimage.label(binary)

        components = []
        for comp_idx in range(1, n_comps + 1):
            mask = labeled == comp_idx
            area = np.sum(mask)

            if area < min_area_px:
                continue

            # Compute centroid
            y_coords, x_coords = np.where(mask)
            centroid_y = float(np.mean(y_coords))
            centroid_x = float(np.mean(x_coords))

            # Compute peak value
            peak_val = float(np.max(movie[frame_idx, :, :][mask]))

            components.append({
                "frame": frame_idx,
                "area": area,
                "centroid": (centroid_y, centroid_x),
                "peak": peak_val,
                "mask": mask,
            })

        frame_components.append(components)

    # Link components across time
    event_tracks = _link_components_across_time(
        frame_components, max_distance_px=max_distance_px
    )

    # Convert to AstroEvent objects
    min_duration_frames = int(min_duration_s * frame_rate_hz)
    dt = 1.0 / frame_rate_hz

    astro_events = []

    for track_idx, track in enumerate(event_tracks):
        if len(track) < min_duration_frames:
            continue

        # Extract event properties
        onset_frame = track[0]["frame"]
        offset_frame = track[-1]["frame"]
        duration_s = (offset_frame - onset_frame + 1) * dt

        # Find peak frame
        peaks = [comp["peak"] for comp in track]
        peak_idx = np.argmax(peaks)
        peak_frame = track[peak_idx]["frame"]
        peak_dff = track[peak_idx]["peak"]

        # Average centroid
        centroids = [comp["centroid"] for comp in track]
        mean_centroid = (
            float(np.mean([c[0] for c in centroids])),
            float(np.mean([c[1] for c in centroids])),
        )

        # Average area
        mean_area = float(np.mean([comp["area"] for comp in track]))

        event_id = f"{session_id}_movie_evt_{track_idx:04d}"

        astro_events.append(
            AstroEvent(
                event_id=event_id,
                session_id=session_id,
                region_id=None,
                onset_frame=onset_frame,
                offset_frame=offset_frame,
                peak_frame=peak_frame,
                duration_s=duration_s,
                peak_dff=peak_dff,
                area_px=mean_area,
                centroid_yx=mean_centroid,
                confidence=0.7,  # Lower confidence for automated spatial detection
            )
        )

        if max_events is not None and len(astro_events) >= max_events:
            break

    return astro_events


def _merge_close_events(
    events: list[dict[str, Any]], frame_rate_hz: float, merge_gap_s: float
) -> list[dict[str, Any]]:
    """Merge events separated by less than merge_gap_s."""
    if len(events) <= 1:
        return events

    # Sort by onset
    events = sorted(events, key=lambda e: e["onset_frame"])

    merge_gap_frames = int(merge_gap_s * frame_rate_hz)
    merged = []
    current = events[0].copy()

    for next_event in events[1:]:
        gap = next_event["onset_frame"] - current["offset_frame"]

        if gap <= merge_gap_frames:
            # Merge
            current["offset_frame"] = next_event["offset_frame"]
            current["duration_s"] = (
                current["offset_frame"] - current["onset_frame"] + 1
            ) / frame_rate_hz

            # Update peak if needed
            if next_event["peak_dff"] > current["peak_dff"]:
                current["peak_frame"] = next_event["peak_frame"]
                current["peak_dff"] = next_event["peak_dff"]
        else:
            # Save current and start new
            merged.append(current)
            current = next_event.copy()

    merged.append(current)
    return merged


def _link_components_across_time(
    frame_components: list[list[dict[str, Any]]], max_distance_px: float
) -> list[list[dict[str, Any]]]:
    """
    Link components across adjacent frames to form event tracks.

    Simple greedy linking based on centroid distance.
    """
    tracks = []
    active_tracks = []

    for frame_idx, components in enumerate(frame_components):
        # Try to extend active tracks
        matched_components = set()
        extended_tracks = []

        for track in active_tracks:
            last_comp = track[-1]
            last_centroid = last_comp["centroid"]

            # Find nearest unmatched component in current frame
            best_match = None
            best_distance = float("inf")

            for comp_idx, comp in enumerate(components):
                if comp_idx in matched_components:
                    continue

                distance = np.sqrt(
                    (comp["centroid"][0] - last_centroid[0]) ** 2
                    + (comp["centroid"][1] - last_centroid[1]) ** 2
                )

                if distance < best_distance and distance < max_distance_px:
                    best_distance = distance
                    best_match = comp_idx

            if best_match is not None:
                # Extend track
                track.append(components[best_match])
                matched_components.add(best_match)
                extended_tracks.append(track)
            else:
                # Track ended
                tracks.append(track)

        # Start new tracks for unmatched components
        for comp_idx, comp in enumerate(components):
            if comp_idx not in matched_components:
                extended_tracks.append([comp])

        active_tracks = extended_tracks

    # Add remaining active tracks
    tracks.extend(active_tracks)

    return tracks
