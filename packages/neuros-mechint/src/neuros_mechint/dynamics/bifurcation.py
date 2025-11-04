"""
Bifurcation Detection and Analysis

This module provides tools for detecting and analyzing bifurcations and
critical transitions in dynamical systems.

Key capabilities:
- Bifurcation point detection
- Critical slowing down indicators
- Early warning signals for transitions
- Bifurcation diagram construction
- Stability analysis near bifurcations
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
import numpy as np
from scipy import stats
from scipy.signal import find_peaks, detrend
from scipy.optimize import fsolve
import logging

logger = logging.getLogger(__name__)


@dataclass
class BifurcationPoint:
    """A detected bifurcation point."""

    location: int  # Time index of bifurcation
    parameter_value: Optional[float] = None  # Parameter value at bifurcation
    bifurcation_type: str = "unknown"  # Type of bifurcation
    pre_state: Optional[np.ndarray] = None  # State before bifurcation
    post_state: Optional[np.ndarray] = None  # State after bifurcation
    confidence: float = 0.0  # Confidence score [0, 1]


@dataclass
class EarlyWarningSignals:
    """Early warning signals for critical transitions."""

    # Statistical moments
    variance: np.ndarray  # Time-varying variance
    autocorrelation: np.ndarray  # Time-varying autocorrelation at lag 1
    skewness: np.ndarray  # Time-varying skewness
    kurtosis: np.ndarray  # Time-varying kurtosis

    # Trend indicators
    variance_trend: float  # Trend in variance (slope)
    autocorr_trend: float  # Trend in autocorrelation (slope)

    # Spectral indicators
    spectral_density_ratio: Optional[np.ndarray] = None  # Low/high frequency ratio

    # Composite score
    warning_score: float = 0.0  # Overall warning score [0, 1]


@dataclass
class BifurcationResult:
    """Results from bifurcation analysis."""

    bifurcation_points: List[BifurcationPoint]  # Detected bifurcations
    n_bifurcations: int  # Number of detected bifurcations

    early_warning_signals: Optional[EarlyWarningSignals] = None  # EWS analysis
    bifurcation_diagram: Optional[np.ndarray] = None  # Bifurcation diagram


class BifurcationDetector:
    """
    Detect and analyze bifurcations in dynamical systems.

    Bifurcations are qualitative changes in system behavior as parameters vary,
    often associated with critical transitions.
    """

    def __init__(
        self,
        dt: float = 0.01,
        window_size: int = 100,
        detection_threshold: float = 2.0,
        verbose: bool = True
    ):
        """
        Initialize bifurcation detector.

        Args:
            dt: Time step
            window_size: Window size for statistical indicators
            detection_threshold: Threshold for bifurcation detection (in std devs)
            verbose: Whether to log information
        """
        self.dt = dt
        self.window_size = window_size
        self.detection_threshold = detection_threshold
        self.verbose = verbose

    def detect(
        self,
        trajectories: np.ndarray,
        parameter_values: Optional[np.ndarray] = None,
        compute_early_warnings: bool = True
    ) -> BifurcationResult:
        """
        Detect bifurcations in trajectory data.

        Args:
            trajectories: Trajectory data (n_timesteps, n_features)
            parameter_values: Optional parameter values at each time step
            compute_early_warnings: Whether to compute early warning signals

        Returns:
            BifurcationResult with detected bifurcations
        """
        if trajectories.ndim == 3:
            trajectories = trajectories.reshape(-1, trajectories.shape[-1])

        # Detect bifurcation points
        bifurcation_points = self._detect_bifurcation_points(
            trajectories,
            parameter_values
        )

        # Compute early warning signals
        if compute_early_warnings:
            early_warning_signals = self._compute_early_warning_signals(trajectories)
        else:
            early_warning_signals = None

        return BifurcationResult(
            bifurcation_points=bifurcation_points,
            n_bifurcations=len(bifurcation_points),
            early_warning_signals=early_warning_signals
        )

    def _detect_bifurcation_points(
        self,
        X: np.ndarray,
        parameter_values: Optional[np.ndarray] = None
    ) -> List[BifurcationPoint]:
        """
        Detect bifurcation points using multiple indicators.

        Args:
            X: Trajectory data (n_timesteps, n_features)
            parameter_values: Optional parameter values

        Returns:
            List of BifurcationPoint objects
        """
        n_timesteps, n_features = X.shape
        bifurcation_points = []

        # Method 1: Variance change detection
        variance_changes = self._detect_variance_changes(X)

        # Method 2: State space distance detection
        state_changes = self._detect_state_changes(X)

        # Method 3: Spectral changes
        spectral_changes = self._detect_spectral_changes(X)

        # Combine detection methods
        all_detections = set(variance_changes) | set(state_changes) | set(spectral_changes)

        for t in all_detections:
            # Compute confidence based on agreement between methods
            confidence = 0.0
            if t in variance_changes:
                confidence += 0.33
            if t in state_changes:
                confidence += 0.33
            if t in spectral_changes:
                confidence += 0.34

            # Determine bifurcation type
            bif_type = self._classify_bifurcation_type(
                X,
                t,
                parameter_values[t] if parameter_values is not None else None
            )

            # Get pre and post states
            pre_state = X[max(0, t - 10):t].mean(axis=0) if t > 10 else None
            post_state = X[t:min(n_timesteps, t + 10)].mean(axis=0) if t < n_timesteps - 10 else None

            bifurcation_point = BifurcationPoint(
                location=t,
                parameter_value=parameter_values[t] if parameter_values is not None else None,
                bifurcation_type=bif_type,
                pre_state=pre_state,
                post_state=post_state,
                confidence=confidence
            )

            bifurcation_points.append(bifurcation_point)

        return bifurcation_points

    def _detect_variance_changes(self, X: np.ndarray) -> List[int]:
        """Detect bifurcations via sudden variance changes."""
        n_timesteps = len(X)
        window = self.window_size

        # Compute rolling variance
        variances = []
        for t in range(window, n_timesteps - window):
            var = np.var(X[t - window:t + window], axis=0).mean()
            variances.append(var)

        variances = np.array(variances)

        # Detect jumps in variance
        variance_diff = np.abs(np.diff(variances))
        threshold = np.mean(variance_diff) + self.detection_threshold * np.std(variance_diff)

        change_points = np.where(variance_diff > threshold)[0] + window

        return list(change_points)

    def _detect_state_changes(self, X: np.ndarray) -> List[int]:
        """Detect bifurcations via sudden state space changes."""
        n_timesteps = len(X)
        window = self.window_size

        # Compute rolling mean position
        means = []
        for t in range(window, n_timesteps - window):
            mean_pos = X[t - window:t + window].mean(axis=0)
            means.append(mean_pos)

        means = np.array(means)

        # Detect jumps in mean position
        mean_distances = np.linalg.norm(np.diff(means, axis=0), axis=1)
        threshold = np.mean(mean_distances) + self.detection_threshold * np.std(mean_distances)

        change_points = np.where(mean_distances > threshold)[0] + window

        return list(change_points)

    def _detect_spectral_changes(self, X: np.ndarray) -> List[int]:
        """Detect bifurcations via spectral changes."""
        n_timesteps, n_features = X.shape
        window = self.window_size

        # Compute rolling spectral properties
        spectral_entropies = []

        for t in range(window, n_timesteps - window):
            segment = X[t - window:t + window, 0]  # Use first feature

            # Compute power spectral density
            from scipy.signal import periodogram
            freqs, psd = periodogram(segment, fs=1.0 / self.dt)

            # Spectral entropy
            psd_norm = psd / np.sum(psd)
            entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))

            spectral_entropies.append(entropy)

        spectral_entropies = np.array(spectral_entropies)

        # Detect jumps
        entropy_diff = np.abs(np.diff(spectral_entropies))
        threshold = np.mean(entropy_diff) + self.detection_threshold * np.std(entropy_diff)

        change_points = np.where(entropy_diff > threshold)[0] + window

        return list(change_points)

    def _classify_bifurcation_type(
        self,
        X: np.ndarray,
        t: int,
        parameter_value: Optional[float] = None
    ) -> str:
        """
        Classify the type of bifurcation.

        Args:
            X: Trajectory data
            t: Time index of bifurcation
            parameter_value: Parameter value

        Returns:
            Bifurcation type string
        """
        window = min(50, t, len(X) - t - 1)

        if window < 10:
            return "unknown"

        # Analyze before and after segments
        before = X[max(0, t - window):t]
        after = X[t:min(len(X), t + window)]

        # Check for transitions between different attractor types
        var_before = np.var(before, axis=0).mean()
        var_after = np.var(after, axis=0).mean()

        mean_before = before.mean(axis=0)
        mean_after = after.mean(axis=0)
        mean_shift = np.linalg.norm(mean_after - mean_before)

        # Classification heuristics
        if var_before < 0.01 and var_after > 0.1:
            return "supercritical_hopf"  # Fixed point → oscillation
        elif var_before > 0.1 and var_after < 0.01:
            return "subcritical_hopf"  # Oscillation → fixed point
        elif mean_shift > 1.0 and var_after / var_before < 2:
            return "saddle_node"  # Jump to different fixed point
        elif var_after / var_before > 5:
            return "period_doubling"  # Onset of complexity
        else:
            return "unknown"

    def _compute_early_warning_signals(
        self,
        X: np.ndarray
    ) -> EarlyWarningSignals:
        """
        Compute early warning signals for critical transitions.

        Args:
            X: Trajectory data (n_timesteps, n_features)

        Returns:
            EarlyWarningSignals object
        """
        n_timesteps = len(X)
        window = self.window_size

        # Use first feature for univariate indicators
        x = X[:, 0]

        # Detrend
        x_detrended = detrend(x)

        # Rolling statistics
        variance = []
        autocorr = []
        skewness = []
        kurtosis = []

        for t in range(window, n_timesteps):
            segment = x_detrended[t - window:t]

            # Variance
            var = np.var(segment)
            variance.append(var)

            # Autocorrelation at lag 1
            ac = np.corrcoef(segment[:-1], segment[1:])[0, 1]
            autocorr.append(ac)

            # Skewness
            skew = stats.skew(segment)
            skewness.append(skew)

            # Kurtosis
            kurt = stats.kurtosis(segment)
            kurtosis.append(kurt)

        variance = np.array(variance)
        autocorr = np.array(autocorr)
        skewness = np.array(skewness)
        kurtosis = np.array(kurtosis)

        # Compute trends (using Kendall's tau for robustness)
        time_indices = np.arange(len(variance))

        variance_trend = stats.kendalltau(time_indices, variance)[0]
        autocorr_trend = stats.kendalltau(time_indices, autocorr)[0]

        # Spectral density ratio (low freq / high freq)
        from scipy.signal import periodogram
        freqs, psd = periodogram(x_detrended, fs=1.0 / self.dt)

        # Split into low and high frequency bands
        mid_freq = len(freqs) // 2
        low_freq_power = np.sum(psd[:mid_freq])
        high_freq_power = np.sum(psd[mid_freq:])

        spectral_ratio = low_freq_power / (high_freq_power + 1e-10)

        # Composite warning score
        # Positive trends in variance and autocorr indicate approaching transition
        warning_score = 0.0

        if variance_trend > 0.2:
            warning_score += 0.4
        if autocorr_trend > 0.2:
            warning_score += 0.4
        if spectral_ratio > 2.0:
            warning_score += 0.2

        warning_score = min(1.0, warning_score)

        return EarlyWarningSignals(
            variance=variance,
            autocorrelation=autocorr,
            skewness=skewness,
            kurtosis=kurtosis,
            variance_trend=variance_trend,
            autocorr_trend=autocorr_trend,
            spectral_density_ratio=np.array([spectral_ratio]),
            warning_score=warning_score
        )

    def construct_bifurcation_diagram(
        self,
        system_function: Callable,
        parameter_range: Tuple[float, float],
        initial_state: np.ndarray,
        n_parameters: int = 100,
        n_transient: int = 1000,
        n_sample: int = 100
    ) -> np.ndarray:
        """
        Construct a bifurcation diagram by varying a parameter.

        Args:
            system_function: Function f(x, p) computing dx/dt
            parameter_range: (p_min, p_max) range for parameter
            initial_state: Initial state
            n_parameters: Number of parameter values to sample
            n_transient: Number of transient steps to discard
            n_sample: Number of points to sample per parameter value

        Returns:
            Bifurcation diagram (n_parameters, n_sample, n_features)
        """
        p_min, p_max = parameter_range
        parameters = np.linspace(p_min, p_max, n_parameters)

        n_features = len(initial_state)
        diagram = np.zeros((n_parameters, n_sample, n_features))

        x = initial_state.copy()

        for i, p in enumerate(parameters):
            # Integrate to remove transients
            for _ in range(n_transient):
                dx = system_function(x, p)
                x = x + self.dt * dx

            # Sample points
            for j in range(n_sample):
                diagram[i, j] = x
                dx = system_function(x, p)
                x = x + self.dt * dx

        return diagram

    def analyze_critical_slowing_down(
        self,
        X: np.ndarray,
        perturbation_indices: Optional[List[int]] = None
    ) -> dict:
        """
        Analyze critical slowing down near bifurcations.

        Critical slowing down is characterized by slower recovery from perturbations
        as a bifurcation is approached.

        Args:
            X: Trajectory data (n_timesteps, n_features)
            perturbation_indices: Indices where perturbations occurred

        Returns:
            Dictionary with recovery time analysis
        """
        if perturbation_indices is None or len(perturbation_indices) == 0:
            # Estimate recovery from natural fluctuations
            recovery_times = self._estimate_recovery_from_fluctuations(X)
        else:
            # Compute recovery times after perturbations
            recovery_times = []

            for pert_idx in perturbation_indices:
                recovery_time = self._compute_recovery_time(X, pert_idx)
                recovery_times.append(recovery_time)

            recovery_times = np.array(recovery_times)

        # Fit trend in recovery times
        if len(recovery_times) > 1:
            time_indices = np.arange(len(recovery_times))
            slope, _ = np.polyfit(time_indices, recovery_times, 1)

            # Positive slope indicates critical slowing down
            is_slowing_down = slope > 0
        else:
            slope = 0.0
            is_slowing_down = False

        return {
            'recovery_times': recovery_times,
            'trend_slope': slope,
            'is_slowing_down': is_slowing_down,
            'mean_recovery_time': np.mean(recovery_times)
        }

    def _compute_recovery_time(
        self,
        X: np.ndarray,
        perturbation_idx: int,
        threshold: float = 0.1
    ) -> float:
        """
        Compute recovery time after a perturbation.

        Args:
            X: Trajectory data
            perturbation_idx: Index of perturbation
            threshold: Threshold for considering system recovered

        Returns:
            Recovery time in time units
        """
        if perturbation_idx >= len(X) - 10:
            return 0.0

        # Baseline: state before perturbation
        baseline = X[max(0, perturbation_idx - 10):perturbation_idx].mean(axis=0)

        # Find when system returns to within threshold of baseline
        post_perturbation = X[perturbation_idx:]

        distances = np.linalg.norm(post_perturbation - baseline, axis=1)
        baseline_distance = np.linalg.norm(X[max(0, perturbation_idx - 10):perturbation_idx].std(axis=0))

        recovery_indices = np.where(distances < threshold * baseline_distance)[0]

        if len(recovery_indices) > 0:
            recovery_steps = recovery_indices[0]
            return recovery_steps * self.dt
        else:
            return (len(post_perturbation) - 1) * self.dt

    def _estimate_recovery_from_fluctuations(
        self,
        X: np.ndarray,
        n_samples: int = 10
    ) -> np.ndarray:
        """
        Estimate recovery times from natural fluctuations.

        Args:
            X: Trajectory data
            n_samples: Number of fluctuation events to sample

        Returns:
            Array of estimated recovery times
        """
        # Find large fluctuations
        x = X[:, 0]  # Use first feature
        x_smoothed = np.convolve(x, np.ones(10) / 10, mode='same')

        deviations = np.abs(x - x_smoothed)
        threshold = np.percentile(deviations, 90)

        fluctuation_indices = np.where(deviations > threshold)[0]

        # Sample subset
        if len(fluctuation_indices) > n_samples:
            sampled_indices = np.random.choice(fluctuation_indices, n_samples, replace=False)
        else:
            sampled_indices = fluctuation_indices

        # Compute recovery times
        recovery_times = []
        for idx in sampled_indices:
            recovery_time = self._compute_recovery_time(X, idx, threshold=0.5)
            recovery_times.append(recovery_time)

        return np.array(recovery_times)
