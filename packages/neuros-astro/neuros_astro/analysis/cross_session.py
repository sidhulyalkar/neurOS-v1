"""
Cross-session analysis tools for comparing astrocyte dynamics across recordings.

Enables analysis of stability, drift, and reproducibility across sessions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from neuros_astro.metadata.schema import AstroEvent, AstroGraph
from neuros_astro.analysis.statistics import compute_event_statistics, EventStatistics


@dataclass
class SessionAlignment:
    """Result of aligning two sessions."""

    session_a_id: str
    session_b_id: str
    n_matched_regions: int
    region_mapping: Dict[str, str]  # region_a -> region_b
    similarity_score: float


@dataclass
class CrossSessionMetrics:
    """Metrics comparing multiple sessions."""

    n_sessions: int
    session_ids: List[str]

    # Event statistics per session
    event_stats_per_session: Dict[str, EventStatistics]

    # Cross-session similarity
    mean_similarity: float
    min_similarity: float
    max_similarity: float

    # Stability metrics
    coefficient_of_variation: Dict[str, float]  # feature -> CV
    consistent_features: List[str]  # Features with low CV

    # Network stability (if available)
    network_stability_mean: Optional[float] = None


def align_sessions(
    events_a: List[AstroEvent],
    events_b: List[AstroEvent],
    method: str = "spatial_correlation",
) -> SessionAlignment:
    """
    Align two sessions to identify corresponding regions.

    This is useful when the same tissue is imaged across sessions
    but ROI IDs may differ.

    Args:
        events_a: Events from session A
        events_b: Events from session B
        method: Alignment method ('spatial_correlation', 'activity_correlation')

    Returns:
        SessionAlignment object

    Example:
        >>> alignment = align_sessions(session1_events, session2_events)
        >>> print(f"Matched {alignment.n_matched_regions} regions")
        >>> print(f"Similarity: {alignment.similarity_score:.3f}")
    """
    # Get unique regions
    regions_a = sorted(set(e.region_id for e in events_a))
    regions_b = sorted(set(e.region_id for e in events_b))

    if method == "spatial_correlation":
        # Simple heuristic: match by spatial position if available
        # For now, use a simple ID-based matching as placeholder
        region_mapping = {}
        for ra in regions_a:
            if ra in regions_b:
                region_mapping[ra] = ra

        n_matched = len(region_mapping)
        similarity = n_matched / max(len(regions_a), len(regions_b))

    elif method == "activity_correlation":
        # Match regions by similar event patterns
        # This is more complex - placeholder for now
        region_mapping = {}
        similarity = 0.0

        # Build activity vectors for each region
        activity_a = {}
        for ra in regions_a:
            region_events = [e for e in events_a if e.region_id == ra]
            if region_events:
                # Simple feature: [n_events, mean_duration, mean_amplitude]
                activity_a[ra] = np.array([
                    len(region_events),
                    np.mean([e.duration_s for e in region_events]),
                    np.mean([e.peak_dff for e in region_events]),
                ])

        activity_b = {}
        for rb in regions_b:
            region_events = [e for e in events_b if e.region_id == rb]
            if region_events:
                activity_b[rb] = np.array([
                    len(region_events),
                    np.mean([e.duration_s for e in region_events]),
                    np.mean([e.peak_dff for e in region_events]),
                ])

        # Match regions by correlation
        for ra in activity_a:
            best_match = None
            best_corr = -1

            for rb in activity_b:
                if rb in region_mapping.values():
                    continue  # Already matched

                # Compute correlation
                corr = np.corrcoef(activity_a[ra], activity_b[rb])[0, 1]

                if not np.isnan(corr) and corr > best_corr and corr > 0.5:
                    best_corr = corr
                    best_match = rb

            if best_match is not None:
                region_mapping[ra] = best_match

        n_matched = len(region_mapping)
        similarity = n_matched / max(len(regions_a), len(regions_b))

    else:
        raise ValueError(f"Unknown method: {method}")

    # Extract session IDs
    session_a_id = events_a[0].session_id if events_a else "unknown"
    session_b_id = events_b[0].session_id if events_b else "unknown"

    return SessionAlignment(
        session_a_id=session_a_id,
        session_b_id=session_b_id,
        n_matched_regions=n_matched,
        region_mapping=region_mapping,
        similarity_score=similarity,
    )


def compute_cross_session_similarity(
    all_events: Dict[str, List[AstroEvent]],
    recording_durations: Optional[Dict[str, float]] = None,
) -> CrossSessionMetrics:
    """
    Compute similarity metrics across multiple sessions.

    Args:
        all_events: Dict mapping session_id -> list of events
        recording_durations: Optional dict of session_id -> duration (seconds)

    Returns:
        CrossSessionMetrics object

    Example:
        >>> all_events = {
        ...     'session_1': events_1,
        ...     'session_2': events_2,
        ...     'session_3': events_3,
        ... }
        >>> metrics = compute_cross_session_similarity(all_events)
        >>> print(f"Mean similarity: {metrics.mean_similarity:.3f}")
        >>> print(f"Consistent features: {metrics.consistent_features}")
    """
    session_ids = list(all_events.keys())
    n_sessions = len(session_ids)

    if n_sessions < 2:
        raise ValueError("Need at least 2 sessions for cross-session analysis")

    # Compute stats for each session
    event_stats_per_session = {}

    for session_id, events in all_events.items():
        duration = recording_durations.get(session_id) if recording_durations else None
        stats = compute_event_statistics(events, recording_duration_s=duration)
        event_stats_per_session[session_id] = stats

    # Compute pairwise similarities
    similarities = []

    for i in range(n_sessions):
        for j in range(i + 1, n_sessions):
            sid_a = session_ids[i]
            sid_b = session_ids[j]

            alignment = align_sessions(
                all_events[sid_a],
                all_events[sid_b],
                method='spatial_correlation'
            )

            similarities.append(alignment.similarity_score)

    mean_similarity = float(np.mean(similarities))
    min_similarity = float(np.min(similarities))
    max_similarity = float(np.max(similarities))

    # Compute coefficient of variation for each feature
    feature_names = [
        'duration_mean', 'duration_median',
        'amplitude_mean', 'amplitude_median',
        'confidence_mean',
    ]

    coefficient_of_variation = {}

    for feature in feature_names:
        values = [getattr(event_stats_per_session[sid], feature)
                 for sid in session_ids]

        if np.mean(values) > 0:
            cv = np.std(values) / np.mean(values)
        else:
            cv = 0.0

        coefficient_of_variation[feature] = float(cv)

    # Identify consistent features (CV < 0.3)
    consistent_features = [f for f, cv in coefficient_of_variation.items() if cv < 0.3]

    return CrossSessionMetrics(
        n_sessions=n_sessions,
        session_ids=session_ids,
        event_stats_per_session=event_stats_per_session,
        mean_similarity=mean_similarity,
        min_similarity=min_similarity,
        max_similarity=max_similarity,
        coefficient_of_variation=coefficient_of_variation,
        consistent_features=consistent_features,
        network_stability_mean=None,
    )


def identify_stable_patterns(
    all_events: Dict[str, List[AstroEvent]],
    consistency_threshold: float = 0.7,
) -> Dict[str, List[str]]:
    """
    Identify event patterns that are stable across sessions.

    Args:
        all_events: Dict mapping session_id -> list of events
        consistency_threshold: Minimum consistency score (0-1)

    Returns:
        Dict mapping pattern_type -> list of session_ids where pattern is present

    Example:
        >>> stable = identify_stable_patterns(all_events, consistency_threshold=0.7)
        >>> print(f"Stable patterns: {stable.keys()}")
        >>> if 'long_duration_events' in stable:
        ...     print(f"Long duration events found in: {stable['long_duration_events']}")
    """
    session_ids = list(all_events.keys())
    patterns = {}

    # Pattern 1: Long duration events (> 5s)
    long_duration_sessions = []
    for session_id, events in all_events.items():
        durations = [e.duration_s for e in events]
        if durations and np.mean(durations) > 5.0:
            long_duration_sessions.append(session_id)

    if len(long_duration_sessions) / len(session_ids) >= consistency_threshold:
        patterns['long_duration_events'] = long_duration_sessions

    # Pattern 2: High amplitude events (> 0.5 dF/F)
    high_amplitude_sessions = []
    for session_id, events in all_events.items():
        amplitudes = [e.peak_dff for e in events]
        if amplitudes and np.mean(amplitudes) > 0.5:
            high_amplitude_sessions.append(session_id)

    if len(high_amplitude_sessions) / len(session_ids) >= consistency_threshold:
        patterns['high_amplitude_events'] = high_amplitude_sessions

    # Pattern 3: High event rate (> 0.1 Hz)
    high_rate_sessions = []
    for session_id, events in all_events.items():
        # Assume 600s recording for rate calculation
        event_rate = len(events) / 600.0
        if event_rate > 0.1:
            high_rate_sessions.append(session_id)

    if len(high_rate_sessions) / len(session_ids) >= consistency_threshold:
        patterns['high_event_rate'] = high_rate_sessions

    return patterns


def compute_session_drift(
    events_per_session: Dict[str, List[AstroEvent]],
    feature: str = 'duration_mean',
) -> Tuple[np.ndarray, float]:
    """
    Compute temporal drift of a feature across sessions.

    Args:
        events_per_session: Dict of session_id -> events (in temporal order)
        feature: Feature to track ('duration_mean', 'amplitude_mean', etc.)

    Returns:
        Tuple of (feature_values, drift_coefficient)
        - feature_values: Array of feature values across sessions
        - drift_coefficient: Linear regression slope (drift per session)

    Example:
        >>> sessions_ordered = {f'session_{i}': events_list[i] for i in range(5)}
        >>> values, drift = compute_session_drift(sessions_ordered, feature='duration_mean')
        >>> if drift > 0.1:
        ...     print("Positive drift detected")
    """
    session_ids = list(events_per_session.keys())
    feature_values = []

    for session_id in session_ids:
        events = events_per_session[session_id]
        stats = compute_event_statistics(events)
        value = getattr(stats, feature)
        feature_values.append(value)

    feature_values = np.array(feature_values)

    # Compute linear drift using least squares
    x = np.arange(len(feature_values))
    coeffs = np.polyfit(x, feature_values, deg=1)
    drift_coefficient = float(coeffs[0])  # Slope

    return feature_values, drift_coefficient
