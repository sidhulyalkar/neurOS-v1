"""
Neural Criticality Analysis

Methods for detecting and quantifying critical dynamics in neural systems,
including avalanche analysis, branching ratios, and distance from criticality.

Neural systems near criticality exhibit optimal information processing,
dynamic range, and sensitivity to inputs.

References:
- Beggs & Plenz (2003): Neuronal avalanches in neocortical circuits
- Shew et al. (2009): Neuronal avalanches imply maximum dynamic range
- Hahn et al. (2017): Spontaneous cortical activity is transiently poised close to criticality
- Wilting & Priesemann (2019): Between perfectly critical and fully irregular
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import powerlaw, linregress
from scipy.optimize import curve_fit


@dataclass
class AvalancheStatistics:
    """Statistics from neuronal avalanche analysis."""
    sizes: np.ndarray  # Avalanche sizes
    durations: np.ndarray  # Avalanche durations
    size_exponent: float  # Power law exponent for sizes (α)
    duration_exponent: float  # Power law exponent for durations (β)
    size_duration_exponent: float  # Exponent relating size to duration (γ)
    branching_ratio: float  # Average branching ratio
    distance_from_criticality: float  # |m - 1| where m is branching ratio
    kappa: float  # Shape collapse parameter


@dataclass
class CriticalityMetrics:
    """Comprehensive criticality metrics."""
    branching_ratio: float
    autocorrelation_time: float
    susceptibility: float
    largest_eigenvalue: float
    avalanche_stats: Optional[AvalancheStatistics] = None
    is_critical: bool = False
    criticality_score: float = 0.0


class NeuronalAvalanche:
    """
    Detect and analyze neuronal avalanches.

    Avalanches are cascades of neural activity that exhibit
    power-law distributed sizes and durations at criticality.
    """

    def __init__(
        self,
        threshold: float = 2.0,  # Standard deviations above mean
        dt: float = 1.0,  # Time bin (ms)
        min_size: int = 2  # Minimum avalanche size
    ):
        self.threshold = threshold
        self.dt = dt
        self.min_size = min_size

    def detect_avalanches(
        self,
        activity: np.ndarray
    ) -> List[Dict[str, np.ndarray]]:
        """
        Detect avalanches in population activity.

        Args:
            activity: Neural activity (time, neurons) - binary spike trains

        Returns:
            List of avalanche dictionaries with 'times', 'neurons', 'size', 'duration'
        """
        # Sum over neurons to get population activity
        pop_activity = activity.sum(axis=1)

        # Threshold for avalanche activity
        threshold_value = pop_activity.mean() + self.threshold * pop_activity.std()

        # Binarize
        active = pop_activity > threshold_value

        # Find avalanches (connected components of activity)
        avalanches = []
        in_avalanche = False
        current_avalanche = {'times': [], 'size': 0}

        for t in range(len(active)):
            if active[t] and not in_avalanche:
                # Start new avalanche
                in_avalanche = True
                current_avalanche = {'times': [t], 'size': int(pop_activity[t])}
            elif active[t] and in_avalanche:
                # Continue avalanche
                current_avalanche['times'].append(t)
                current_avalanche['size'] += int(pop_activity[t])
            elif not active[t] and in_avalanche:
                # End avalanche
                in_avalanche = False
                current_avalanche['duration'] = len(current_avalanche['times'])

                if current_avalanche['size'] >= self.min_size:
                    avalanches.append(current_avalanche)

        return avalanches

    def compute_statistics(
        self,
        avalanches: List[Dict]
    ) -> AvalancheStatistics:
        """
        Compute avalanche statistics and power law fits.

        Args:
            avalanches: List of avalanche dictionaries

        Returns:
            Avalanche statistics
        """
        if len(avalanches) == 0:
            raise ValueError("No avalanches detected")

        sizes = np.array([a['size'] for a in avalanches])
        durations = np.array([a['duration'] for a in avalanches])

        # Fit power laws
        # P(s) ~ s^(-α)
        size_exponent = self._fit_powerlaw(sizes)
        duration_exponent = self._fit_powerlaw(durations)

        # Size-duration relationship: s ~ d^γ
        log_sizes = np.log(sizes)
        log_durations = np.log(durations)
        slope, _, _, _, _ = linregress(log_durations, log_sizes)
        size_duration_exponent = slope

        # Branching ratio (from size exponent)
        # At criticality: α ≈ 1.5, β ≈ 2, γ ≈ 2
        branching_ratio = self._estimate_branching_ratio(sizes, durations)

        # Distance from criticality
        distance = abs(branching_ratio - 1.0)

        # Shape collapse parameter (checks if s ~ d^γ collapses onto universal curve)
        kappa = self._compute_shape_collapse(sizes, durations, size_duration_exponent)

        return AvalancheStatistics(
            sizes=sizes,
            durations=durations,
            size_exponent=size_exponent,
            duration_exponent=duration_exponent,
            size_duration_exponent=size_duration_exponent,
            branching_ratio=branching_ratio,
            distance_from_criticality=distance,
            kappa=kappa
        )

    def _fit_powerlaw(self, data: np.ndarray) -> float:
        """
        Fit power law exponent using linear regression in log-log space.

        Args:
            data: Data to fit

        Returns:
            Power law exponent
        """
        # Bin data
        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 20)
        hist, bin_edges = np.histogram(data, bins=bins)

        # Remove zeros
        mask = hist > 0
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        log_bins = np.log(bin_centers[mask])
        log_hist = np.log(hist[mask])

        # Linear fit
        slope, _, _, _, _ = linregress(log_bins, log_hist)

        return -slope  # Power law exponent is negative of slope

    def _estimate_branching_ratio(
        self,
        sizes: np.ndarray,
        durations: np.ndarray
    ) -> float:
        """
        Estimate branching ratio from avalanche statistics.

        Args:
            sizes: Avalanche sizes
            durations: Avalanche durations

        Returns:
            Branching ratio estimate
        """
        # Mean avalanche size per time step
        mean_size_per_time = sizes.mean() / durations.mean()

        # Branching ratio approximation
        # At criticality, each active neuron activates on average 1 other neuron
        return mean_size_per_time

    def _compute_shape_collapse(
        self,
        sizes: np.ndarray,
        durations: np.ndarray,
        gamma: float
    ) -> float:
        """
        Compute shape collapse parameter.

        Tests if rescaled avalanche shapes collapse onto universal curve.

        Args:
            sizes: Avalanche sizes
            durations: Avalanche durations
            gamma: Size-duration exponent

        Returns:
            Shape collapse quality metric
        """
        # Rescale: s / d^γ should be constant at criticality
        rescaled = sizes / (durations ** gamma)

        # Coefficient of variation (lower is better collapse)
        cv = rescaled.std() / rescaled.mean()

        # Quality: 1 - CV (higher is better)
        kappa = max(0, 1 - cv)

        return kappa


class BranchingProcess:
    """
    Analyze branching process dynamics.

    At criticality, the branching ratio m = 1.
    m < 1: subcritical (dying activity)
    m > 1: supercritical (exponentially growing activity)
    """

    def __init__(self, dt: float = 1.0):
        self.dt = dt

    def estimate_branching_ratio(
        self,
        spike_trains: np.ndarray,
        max_lag: int = 10
    ) -> Tuple[float, float]:
        """
        Estimate branching ratio from spike trains.

        Uses autocorrelation method: m = A(1) where A is autocorrelation.

        Args:
            spike_trains: Binary spike trains (time, neurons)
            max_lag: Maximum time lag for autocorrelation

        Returns:
            (branching_ratio, std_error)
        """
        # Population activity
        pop_activity = spike_trains.sum(axis=1)

        # Autocorrelation
        autocorr = self._autocorrelation(pop_activity, max_lag)

        # Branching ratio is autocorr at lag 1
        m = autocorr[1]

        # Error estimate from variance
        std_error = autocorr.std() / np.sqrt(len(pop_activity))

        return m, std_error

    def _autocorrelation(
        self,
        signal: np.ndarray,
        max_lag: int
    ) -> np.ndarray:
        """
        Compute autocorrelation function.

        Args:
            signal: Time series
            max_lag: Maximum lag

        Returns:
            Autocorrelation (max_lag + 1,)
        """
        signal = signal - signal.mean()
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr[:max_lag + 1]
        autocorr = autocorr / autocorr[0]  # Normalize

        return autocorr

    def estimate_distance_from_criticality(
        self,
        spike_trains: np.ndarray
    ) -> Dict[str, float]:
        """
        Estimate how far system is from criticality.

        Args:
            spike_trains: Binary spike trains

        Returns:
            Dictionary with distance metrics
        """
        m, m_std = self.estimate_branching_ratio(spike_trains)

        # Distance from criticality
        distance = abs(m - 1.0)

        # Autocorrelation time (diverges at criticality)
        autocorr = self._autocorrelation(spike_trains.sum(axis=1), max_lag=100)

        # Find when autocorr drops below 1/e
        threshold = 1.0 / np.e
        tau_indices = np.where(autocorr < threshold)[0]
        tau = tau_indices[0] if len(tau_indices) > 0 else 100

        # Susceptibility (variance, diverges at criticality)
        pop_activity = spike_trains.sum(axis=1)
        susceptibility = pop_activity.var()

        return {
            'branching_ratio': m,
            'branching_ratio_std': m_std,
            'distance': distance,
            'autocorrelation_time': tau,
            'susceptibility': susceptibility,
            'regime': 'subcritical' if m < 1 else ('supercritical' if m > 1 else 'critical')
        }


class CriticalityDetector:
    """
    Comprehensive criticality detection using multiple methods.

    Combines avalanche analysis, branching process, eigenvalue analysis,
    and fluctuation scaling.
    """

    def __init__(self):
        self.avalanche_analyzer = NeuronalAvalanche()
        self.branching_analyzer = BranchingProcess()

    def analyze(
        self,
        spike_trains: np.ndarray,
        connectivity: Optional[np.ndarray] = None
    ) -> CriticalityMetrics:
        """
        Comprehensive criticality analysis.

        Args:
            spike_trains: Binary spike trains (time, neurons)
            connectivity: Connectivity matrix (neurons, neurons)

        Returns:
            Criticality metrics
        """
        # Branching process analysis
        branching_metrics = self.branching_analyzer.estimate_distance_from_criticality(
            spike_trains
        )

        # Avalanche analysis
        avalanches = self.avalanche_analyzer.detect_avalanches(spike_trains)

        if len(avalanches) > 10:
            avalanche_stats = self.avalanche_analyzer.compute_statistics(avalanches)
        else:
            avalanche_stats = None

        # Eigenvalue analysis (if connectivity provided)
        if connectivity is not None:
            eigenvalues = np.linalg.eigvals(connectivity)
            largest_eigenvalue = np.max(np.abs(eigenvalues))
        else:
            largest_eigenvalue = np.nan

        # Overall criticality score
        # Combines multiple indicators
        score = self._compute_criticality_score(
            branching_metrics,
            avalanche_stats,
            largest_eigenvalue
        )

        # Determine if critical
        is_critical = self._is_critical(
            branching_metrics['branching_ratio'],
            avalanche_stats,
            largest_eigenvalue
        )

        return CriticalityMetrics(
            branching_ratio=branching_metrics['branching_ratio'],
            autocorrelation_time=branching_metrics['autocorrelation_time'],
            susceptibility=branching_metrics['susceptibility'],
            largest_eigenvalue=largest_eigenvalue,
            avalanche_stats=avalanche_stats,
            is_critical=is_critical,
            criticality_score=score
        )

    def _compute_criticality_score(
        self,
        branching_metrics: Dict,
        avalanche_stats: Optional[AvalancheStatistics],
        largest_eigenvalue: float
    ) -> float:
        """
        Compute overall criticality score [0, 1].

        Args:
            branching_metrics: Branching process metrics
            avalanche_stats: Avalanche statistics
            largest_eigenvalue: Largest eigenvalue of connectivity

        Returns:
            Criticality score (1 = perfectly critical)
        """
        scores = []

        # Branching ratio score (1 - distance from 1)
        m_score = 1.0 - min(1.0, abs(branching_metrics['branching_ratio'] - 1.0))
        scores.append(m_score)

        # Avalanche exponent score
        if avalanche_stats is not None:
            # At criticality: α ≈ 1.5, β ≈ 2
            alpha_score = 1.0 - min(1.0, abs(avalanche_stats.size_exponent - 1.5) / 1.5)
            beta_score = 1.0 - min(1.0, abs(avalanche_stats.duration_exponent - 2.0) / 2.0)
            scores.extend([alpha_score, beta_score])

        # Eigenvalue score (critical when λ_max ≈ 1)
        if not np.isnan(largest_eigenvalue):
            lambda_score = 1.0 - min(1.0, abs(largest_eigenvalue - 1.0))
            scores.append(lambda_score)

        return np.mean(scores)

    def _is_critical(
        self,
        branching_ratio: float,
        avalanche_stats: Optional[AvalancheStatistics],
        largest_eigenvalue: float,
        tolerance: float = 0.1
    ) -> bool:
        """
        Determine if system is critical.

        Args:
            branching_ratio: Branching ratio
            avalanche_stats: Avalanche statistics
            largest_eigenvalue: Largest eigenvalue
            tolerance: Tolerance for criticality

        Returns:
            True if critical
        """
        # Check branching ratio
        if abs(branching_ratio - 1.0) > tolerance:
            return False

        # Check avalanche exponents if available
        if avalanche_stats is not None:
            if abs(avalanche_stats.size_exponent - 1.5) > tolerance:
                return False

        # Check eigenvalue if available
        if not np.isnan(largest_eigenvalue):
            if abs(largest_eigenvalue - 1.0) > tolerance:
                return False

        return True


class SelfOrganizedCriticality:
    """
    Test for self-organized criticality (SOC).

    SOC systems naturally evolve toward critical state without tuning.
    """

    def __init__(self):
        pass

    def temporal_evolution(
        self,
        spike_trains: np.ndarray,
        window_size: int = 1000,
        step_size: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Track criticality metrics over time.

        Args:
            spike_trains: Spike trains (time, neurons)
            window_size: Window size for analysis
            step_size: Step size for sliding window

        Returns:
            Time-varying criticality metrics
        """
        n_time = spike_trains.shape[0]
        n_windows = (n_time - window_size) // step_size

        branching_ratios = np.zeros(n_windows)
        distances = np.zeros(n_windows)

        analyzer = BranchingProcess()

        for i in range(n_windows):
            start = i * step_size
            end = start + window_size

            window = spike_trains[start:end]

            m, _ = analyzer.estimate_branching_ratio(window)
            branching_ratios[i] = m
            distances[i] = abs(m - 1.0)

        return {
            'time': np.arange(n_windows) * step_size,
            'branching_ratio': branching_ratios,
            'distance_from_criticality': distances,
            'mean_distance': distances.mean(),
            'std_distance': distances.std()
        }

    def test_convergence(
        self,
        temporal_metrics: Dict[str, np.ndarray],
        threshold: float = 0.05
    ) -> bool:
        """
        Test if system converges to criticality.

        Args:
            temporal_metrics: Time-varying metrics from temporal_evolution
            threshold: Convergence threshold

        Returns:
            True if converges to critical state
        """
        distances = temporal_metrics['distance_from_criticality']

        # Test if final distances are below threshold
        final_portion = distances[-len(distances)//4:]  # Last 25%

        return final_portion.mean() < threshold
