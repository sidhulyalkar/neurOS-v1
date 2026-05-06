"""
Statistical analysis utilities for astrocyte event data.

Provides functions for computing statistics, comparing distributions,
and performing hypothesis tests on event features.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from neuros_astro.metadata.schema import AstroEvent


@dataclass
class EventStatistics:
    """Statistical summary of astrocyte events."""

    n_events: int
    n_regions: int

    # Duration statistics (seconds)
    duration_mean: float
    duration_std: float
    duration_median: float
    duration_iqr: Tuple[float, float]
    duration_range: Tuple[float, float]

    # Amplitude statistics (dF/F)
    amplitude_mean: float
    amplitude_std: float
    amplitude_median: float
    amplitude_iqr: Tuple[float, float]
    amplitude_range: Tuple[float, float]

    # Confidence statistics
    confidence_mean: float
    confidence_std: float
    confidence_median: float

    # Spatial statistics (if available)
    area_mean: Optional[float] = None
    area_std: Optional[float] = None
    area_median: Optional[float] = None

    # Event rate
    event_rate_hz: Optional[float] = None
    events_per_region: Optional[float] = None


def compute_event_statistics(
    events: List[AstroEvent],
    recording_duration_s: Optional[float] = None,
) -> EventStatistics:
    """
    Compute comprehensive statistics for astrocyte events.

    Args:
        events: List of AstroEvent objects
        recording_duration_s: Optional total recording duration for rate calculation

    Returns:
        EventStatistics object with all computed metrics

    Example:
        >>> stats = compute_event_statistics(events, recording_duration_s=600.0)
        >>> print(f"Mean duration: {stats.duration_mean:.2f}s")
        >>> print(f"Event rate: {stats.event_rate_hz:.3f} Hz")
    """
    if len(events) == 0:
        raise ValueError("Cannot compute statistics on empty event list")

    # Extract features
    durations = np.array([e.duration_s for e in events])
    amplitudes = np.array([e.peak_dff for e in events])
    confidences = np.array([e.confidence for e in events])

    # Count unique regions
    unique_regions = len(set(e.region_id for e in events))

    # Compute duration statistics
    duration_mean = float(np.mean(durations))
    duration_std = float(np.std(durations))
    duration_median = float(np.median(durations))
    duration_iqr = (float(np.percentile(durations, 25)),
                    float(np.percentile(durations, 75)))
    duration_range = (float(np.min(durations)), float(np.max(durations)))

    # Compute amplitude statistics
    amplitude_mean = float(np.mean(amplitudes))
    amplitude_std = float(np.std(amplitudes))
    amplitude_median = float(np.median(amplitudes))
    amplitude_iqr = (float(np.percentile(amplitudes, 25)),
                     float(np.percentile(amplitudes, 75)))
    amplitude_range = (float(np.min(amplitudes)), float(np.max(amplitudes)))

    # Compute confidence statistics
    confidence_mean = float(np.mean(confidences))
    confidence_std = float(np.std(confidences))
    confidence_median = float(np.median(confidences))

    # Spatial statistics (if available)
    areas = [e.area_px for e in events if e.area_px is not None]
    if areas:
        area_mean = float(np.mean(areas))
        area_std = float(np.std(areas))
        area_median = float(np.median(areas))
    else:
        area_mean = None
        area_std = None
        area_median = None

    # Event rate
    event_rate_hz = None
    if recording_duration_s is not None and recording_duration_s > 0:
        event_rate_hz = len(events) / recording_duration_s

    events_per_region = len(events) / unique_regions if unique_regions > 0 else 0

    return EventStatistics(
        n_events=len(events),
        n_regions=unique_regions,
        duration_mean=duration_mean,
        duration_std=duration_std,
        duration_median=duration_median,
        duration_iqr=duration_iqr,
        duration_range=duration_range,
        amplitude_mean=amplitude_mean,
        amplitude_std=amplitude_std,
        amplitude_median=amplitude_median,
        amplitude_iqr=amplitude_iqr,
        amplitude_range=amplitude_range,
        confidence_mean=confidence_mean,
        confidence_std=confidence_std,
        confidence_median=confidence_median,
        area_mean=area_mean,
        area_std=area_std,
        area_median=area_median,
        event_rate_hz=event_rate_hz,
        events_per_region=events_per_region,
    )


@dataclass
class ComparisonResult:
    """Result of comparing two event distributions."""

    feature_name: str
    group_a_mean: float
    group_b_mean: float
    mean_difference: float
    effect_size: float  # Cohen's d
    p_value: float
    significant: bool
    test_name: str


def compare_event_distributions(
    events_a: List[AstroEvent],
    events_b: List[AstroEvent],
    feature: str = "duration_s",
    alpha: float = 0.05,
    n_permutations: int = 10000,
) -> ComparisonResult:
    """
    Compare a feature distribution between two groups of events.

    Uses permutation testing for robustness to non-normal distributions.

    Args:
        events_a: First group of events
        events_b: Second group of events
        feature: Feature to compare ('duration_s', 'peak_dff', 'confidence')
        alpha: Significance level
        n_permutations: Number of permutations for test

    Returns:
        ComparisonResult object

    Example:
        >>> # Compare baseline vs stimulation periods
        >>> result = compare_event_distributions(baseline_events, stim_events,
        ...                                      feature='duration_s')
        >>> print(f"p-value: {result.p_value:.4f}, Effect size: {result.effect_size:.2f}")
        >>> print(f"Significant: {result.significant}")
    """
    # Extract feature values
    values_a = np.array([getattr(e, feature) for e in events_a])
    values_b = np.array([getattr(e, feature) for e in events_b])

    # Compute means
    mean_a = float(np.mean(values_a))
    mean_b = float(np.mean(values_b))
    mean_diff = mean_a - mean_b

    # Compute effect size (Cohen's d)
    pooled_std = np.sqrt(((len(values_a) - 1) * np.var(values_a, ddof=1) +
                          (len(values_b) - 1) * np.var(values_b, ddof=1)) /
                         (len(values_a) + len(values_b) - 2))

    if pooled_std > 0:
        effect_size = mean_diff / pooled_std
    else:
        effect_size = 0.0

    # Permutation test
    p_value = permutation_test(values_a, values_b, n_permutations=n_permutations)

    return ComparisonResult(
        feature_name=feature,
        group_a_mean=mean_a,
        group_b_mean=mean_b,
        mean_difference=mean_diff,
        effect_size=float(effect_size),
        p_value=p_value,
        significant=p_value < alpha,
        test_name="permutation_test",
    )


def permutation_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_permutations: int = 10000,
    seed: Optional[int] = None,
) -> float:
    """
    Two-sample permutation test for difference in means.

    Args:
        group_a: First group values
        group_b: Second group values
        n_permutations: Number of permutations
        seed: Random seed for reproducibility

    Returns:
        Two-tailed p-value

    Example:
        >>> p_val = permutation_test(condition_a, condition_b, n_permutations=10000)
        >>> print(f"p-value: {p_val:.4f}")
    """
    if seed is not None:
        np.random.seed(seed)

    # Observed difference
    observed_diff = np.mean(group_a) - np.mean(group_b)

    # Combine data
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    # Permutation distribution
    perm_diffs = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Shuffle and split
        np.random.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]

        perm_diffs[i] = np.mean(perm_a) - np.mean(perm_b)

    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    return float(p_value)


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: str = "mean",
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: Input data array
        statistic: Statistic to compute ('mean', 'median', 'std')
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)

    Example:
        >>> durations = np.array([e.duration_s for e in events])
        >>> mean, lower, upper = bootstrap_confidence_interval(durations,
        ...                                                    statistic='mean')
        >>> print(f"Mean: {mean:.2f}, 95% CI: [{lower:.2f}, {upper:.2f}]")
    """
    if seed is not None:
        np.random.seed(seed)

    # Choose statistic function
    if statistic == "mean":
        stat_func = np.mean
    elif statistic == "median":
        stat_func = np.median
    elif statistic == "std":
        stat_func = np.std
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    # Compute point estimate
    point_estimate = float(stat_func(data))

    # Bootstrap
    bootstrap_stats = np.zeros(n_bootstrap)
    n = len(data)

    for i in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = stat_func(sample)

    # Compute percentiles for CI
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = float(np.percentile(bootstrap_stats, lower_percentile))
    upper_bound = float(np.percentile(bootstrap_stats, upper_percentile))

    return point_estimate, lower_bound, upper_bound


def effect_size_cohens_d(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> float:
    """
    Compute Cohen's d effect size for two groups.

    Interpretation:
        - |d| < 0.2: negligible
        - 0.2 <= |d| < 0.5: small
        - 0.5 <= |d| < 0.8: medium
        - |d| >= 0.8: large

    Args:
        group_a: First group values
        group_b: Second group values

    Returns:
        Cohen's d effect size

    Example:
        >>> d = effect_size_cohens_d(baseline_durations, stim_durations)
        >>> print(f"Cohen's d: {d:.2f}")
        >>> if abs(d) >= 0.8:
        ...     print("Large effect size")
    """
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)

    # Pooled standard deviation
    n_a = len(group_a)
    n_b = len(group_b)

    pooled_std = np.sqrt(((n_a - 1) * np.var(group_a, ddof=1) +
                          (n_b - 1) * np.var(group_b, ddof=1)) /
                         (n_a + n_b - 2))

    if pooled_std == 0:
        return 0.0

    d = (mean_a - mean_b) / pooled_std

    return float(d)


def compute_event_rate_by_time_window(
    events: List[AstroEvent],
    window_size_s: float,
    total_duration_s: float,
    frame_rate_hz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute event rate in sliding time windows.

    Args:
        events: List of AstroEvent objects
        window_size_s: Window size in seconds
        total_duration_s: Total recording duration
        frame_rate_hz: Frame rate for time conversion

    Returns:
        Tuple of (window_centers, event_rates)
        - window_centers: Array of window center times
        - event_rates: Array of event rates (Hz) in each window

    Example:
        >>> times, rates = compute_event_rate_by_time_window(
        ...     events, window_size_s=10.0, total_duration_s=600.0,
        ...     frame_rate_hz=10.0
        ... )
        >>> plt.plot(times, rates)
        >>> plt.xlabel('Time (s)')
        >>> plt.ylabel('Event Rate (Hz)')
    """
    # Create time bins
    n_windows = int(total_duration_s / window_size_s)
    window_edges = np.linspace(0, total_duration_s, n_windows + 1)
    window_centers = (window_edges[:-1] + window_edges[1:]) / 2

    # Count events in each window
    event_times = np.array([e.peak_frame / frame_rate_hz for e in events])
    event_counts, _ = np.histogram(event_times, bins=window_edges)

    # Convert to rates (Hz)
    event_rates = event_counts / window_size_s

    return window_centers, event_rates
