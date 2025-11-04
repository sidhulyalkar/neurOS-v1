"""
Recurrence Analysis

This module provides tools for recurrence plot analysis and recurrence
quantification analysis (RQA) for nonlinear time series.

Key capabilities:
- Recurrence plot construction
- Recurrence quantification analysis (RQA)
- Cross-recurrence plots
- Joint recurrence plots
- Recurrence network analysis
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import logging

logger = logging.getLogger(__name__)


@dataclass
class RecurrenceResult:
    """Results from recurrence analysis."""

    # Recurrence plot
    recurrence_matrix: np.ndarray  # Binary recurrence matrix
    recurrence_rate: float  # Fraction of recurrent points

    # RQA metrics
    determinism: float  # Fraction of recurrent points in diagonal lines
    average_diagonal_line: float  # Average length of diagonal lines
    max_diagonal_line: float  # Longest diagonal line
    entropy_diagonal_lines: float  # Shannon entropy of diagonal line distribution
    laminarity: float  # Fraction of recurrent points in vertical lines
    trapping_time: float  # Average length of vertical lines

    # Dynamical properties
    is_periodic: bool  # Whether system exhibits periodicity
    period_estimate: Optional[float] = None  # Estimated period if periodic

    # Additional metrics
    recurrence_time: Optional[float] = None  # Mean recurrence time
    clustering_coefficient: Optional[float] = None  # Recurrence network clustering


class RecurrenceAnalyzer:
    """
    Recurrence analysis for dynamical systems.

    Recurrence plots visualize when a trajectory returns close to
    previous states, revealing periodicities and patterns.
    """

    def __init__(
        self,
        dt: float = 0.01,
        threshold: Optional[float] = None,
        threshold_method: str = "fixed",
        verbose: bool = True
    ):
        """
        Initialize recurrence analyzer.

        Args:
            dt: Time step
            threshold: Recurrence threshold (distance)
            threshold_method: Method to determine threshold ("fixed", "adaptive", "fan")
            verbose: Whether to log information
        """
        self.dt = dt
        self.threshold = threshold
        self.threshold_method = threshold_method
        self.verbose = verbose

    def analyze(
        self,
        trajectories: np.ndarray,
        compute_network: bool = False
    ) -> RecurrenceResult:
        """
        Perform recurrence analysis.

        Args:
            trajectories: Trajectory data (n_timesteps, n_features)
            compute_network: Whether to compute recurrence network properties

        Returns:
            RecurrenceResult
        """
        if trajectories.ndim == 3:
            trajectories = trajectories.reshape(-1, trajectories.shape[-1])

        # Compute recurrence matrix
        recurrence_matrix = self._compute_recurrence_matrix(trajectories)

        # Compute RQA metrics
        rqa_metrics = self._compute_rqa_metrics(recurrence_matrix)

        # Check for periodicity
        is_periodic, period = self._detect_periodicity(recurrence_matrix)

        # Recurrence time
        recurrence_time = self._compute_recurrence_time(recurrence_matrix)

        # Network metrics
        if compute_network:
            clustering = self._compute_clustering_coefficient(recurrence_matrix)
        else:
            clustering = None

        return RecurrenceResult(
            recurrence_matrix=recurrence_matrix,
            recurrence_rate=rqa_metrics['recurrence_rate'],
            determinism=rqa_metrics['determinism'],
            average_diagonal_line=rqa_metrics['avg_diagonal'],
            max_diagonal_line=rqa_metrics['max_diagonal'],
            entropy_diagonal_lines=rqa_metrics['entropy'],
            laminarity=rqa_metrics['laminarity'],
            trapping_time=rqa_metrics['trapping_time'],
            is_periodic=is_periodic,
            period_estimate=period,
            recurrence_time=recurrence_time,
            clustering_coefficient=clustering
        )

    def _compute_recurrence_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute recurrence matrix.

        Args:
            X: Trajectory data (n_timesteps, n_features)

        Returns:
            Binary recurrence matrix (n_timesteps, n_timesteps)
        """
        # Compute distance matrix
        dists = distance_matrix(X, X)

        # Determine threshold
        if self.threshold is None:
            if self.threshold_method == "adaptive":
                # Use percentile of distances
                threshold = np.percentile(dists, 10)
            elif self.threshold_method == "fan":
                # FAN (Fixed Amount of Neighbors): ensure each point has k neighbors
                k = max(1, int(0.05 * len(X)))
                threshold = np.median([np.sort(row)[k] for row in dists])
            else:
                # Default: use standard deviation of distances
                threshold = np.std(dists) * 0.1
        else:
            threshold = self.threshold

        # Create binary recurrence matrix
        recurrence_matrix = (dists < threshold).astype(int)

        # Remove main diagonal (trivial recurrences)
        np.fill_diagonal(recurrence_matrix, 0)

        return recurrence_matrix

    def _compute_rqa_metrics(self, R: np.ndarray) -> dict:
        """
        Compute Recurrence Quantification Analysis metrics.

        Args:
            R: Recurrence matrix

        Returns:
            Dictionary of RQA metrics
        """
        n = R.shape[0]

        # Recurrence rate
        recurrence_rate = np.sum(R) / (n * (n - 1))

        # Diagonal lines (determinism)
        diagonal_lines = self._extract_diagonal_lines(R, min_length=2)

        if len(diagonal_lines) > 0:
            total_diagonal_points = sum(diagonal_lines)
            determinism = total_diagonal_points / max(1, np.sum(R))
            avg_diagonal = np.mean(diagonal_lines)
            max_diagonal = np.max(diagonal_lines)

            # Entropy of diagonal line distribution
            hist, _ = np.histogram(diagonal_lines, bins=20)
            hist = hist / np.sum(hist)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log(hist))
        else:
            determinism = 0.0
            avg_diagonal = 0.0
            max_diagonal = 0.0
            entropy = 0.0

        # Vertical lines (laminarity)
        vertical_lines = self._extract_vertical_lines(R, min_length=2)

        if len(vertical_lines) > 0:
            total_vertical_points = sum(vertical_lines)
            laminarity = total_vertical_points / max(1, np.sum(R))
            trapping_time = np.mean(vertical_lines)
        else:
            laminarity = 0.0
            trapping_time = 0.0

        return {
            'recurrence_rate': recurrence_rate,
            'determinism': determinism,
            'avg_diagonal': avg_diagonal,
            'max_diagonal': max_diagonal,
            'entropy': entropy,
            'laminarity': laminarity,
            'trapping_time': trapping_time
        }

    def _extract_diagonal_lines(self, R: np.ndarray, min_length: int = 2) -> list:
        """
        Extract diagonal line lengths from recurrence matrix.

        Args:
            R: Recurrence matrix
            min_length: Minimum line length to consider

        Returns:
            List of diagonal line lengths
        """
        n = R.shape[0]
        line_lengths = []

        # Check all diagonals (excluding main diagonal)
        for offset in range(-n + 1, n):
            if offset == 0:
                continue

            diagonal = np.diagonal(R, offset=offset)

            # Find consecutive 1s
            current_length = 0
            for val in diagonal:
                if val == 1:
                    current_length += 1
                else:
                    if current_length >= min_length:
                        line_lengths.append(current_length)
                    current_length = 0

            if current_length >= min_length:
                line_lengths.append(current_length)

        return line_lengths

    def _extract_vertical_lines(self, R: np.ndarray, min_length: int = 2) -> list:
        """
        Extract vertical line lengths from recurrence matrix.

        Args:
            R: Recurrence matrix
            min_length: Minimum line length

        Returns:
            List of vertical line lengths
        """
        line_lengths = []

        # Check each column
        for col in range(R.shape[1]):
            column = R[:, col]

            # Find consecutive 1s
            current_length = 0
            for val in column:
                if val == 1:
                    current_length += 1
                else:
                    if current_length >= min_length:
                        line_lengths.append(current_length)
                    current_length = 0

            if current_length >= min_length:
                line_lengths.append(current_length)

        return line_lengths

    def _detect_periodicity(
        self,
        R: np.ndarray,
        threshold: float = 0.1
    ) -> Tuple[bool, Optional[float]]:
        """
        Detect periodicity from recurrence matrix.

        Args:
            R: Recurrence matrix
            threshold: Threshold for considering diagonal line significant

        Returns:
            Tuple of (is_periodic, period_in_time_units)
        """
        n = R.shape[0]

        # Sum along diagonals to find periodic structure
        diagonal_sums = []
        for offset in range(1, n // 2):
            diagonal = np.diagonal(R, offset=offset)
            diagonal_sums.append(np.sum(diagonal))

        diagonal_sums = np.array(diagonal_sums)

        if len(diagonal_sums) == 0:
            return False, None

        # Normalize
        diagonal_sums = diagonal_sums / np.max(diagonal_sums)

        # Find peaks (potential periods)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(diagonal_sums, height=threshold, distance=5)

        if len(peaks) > 0:
            # Primary period is first peak
            period_steps = peaks[0] + 1
            period_time = period_steps * self.dt

            return True, period_time
        else:
            return False, None

    def _compute_recurrence_time(self, R: np.ndarray) -> float:
        """
        Compute mean recurrence time.

        Args:
            R: Recurrence matrix

        Returns:
            Mean recurrence time in time units
        """
        n = R.shape[0]
        recurrence_times = []

        for i in range(n):
            # Find next recurrence
            recurrences = np.where(R[i, i+1:] == 1)[0]

            if len(recurrences) > 0:
                recurrence_time = recurrences[0] + 1
                recurrence_times.append(recurrence_time)

        if len(recurrence_times) > 0:
            mean_recurrence_time = np.mean(recurrence_times) * self.dt
        else:
            mean_recurrence_time = np.inf

        return mean_recurrence_time

    def _compute_clustering_coefficient(self, R: np.ndarray) -> float:
        """
        Compute clustering coefficient of recurrence network.

        Args:
            R: Recurrence matrix (adjacency matrix)

        Returns:
            Clustering coefficient
        """
        n = R.shape[0]

        # Compute local clustering coefficients
        clustering_coeffs = []

        for i in range(n):
            neighbors = np.where(R[i] == 1)[0]
            k = len(neighbors)

            if k < 2:
                continue

            # Count triangles
            triangles = 0
            for j in range(len(neighbors)):
                for l in range(j + 1, len(neighbors)):
                    if R[neighbors[j], neighbors[l]] == 1:
                        triangles += 1

            # Clustering coefficient
            max_triangles = k * (k - 1) / 2
            if max_triangles > 0:
                clustering_coeffs.append(triangles / max_triangles)

        if len(clustering_coeffs) > 0:
            return np.mean(clustering_coeffs)
        else:
            return 0.0

    def cross_recurrence(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> np.ndarray:
        """
        Compute cross-recurrence plot between two time series.

        Args:
            X: First time series (n_timesteps, n_features)
            Y: Second time series (m_timesteps, n_features)

        Returns:
            Cross-recurrence matrix (n_timesteps, m_timesteps)
        """
        # Compute distance matrix
        dists = distance_matrix(X, Y)

        # Determine threshold
        if self.threshold is None:
            threshold = np.std(dists) * 0.1
        else:
            threshold = self.threshold

        # Binary cross-recurrence matrix
        cross_recurrence = (dists < threshold).astype(int)

        return cross_recurrence

    def joint_recurrence(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> np.ndarray:
        """
        Compute joint recurrence plot (both systems must recur simultaneously).

        Args:
            X: First time series (n_timesteps, n_features)
            Y: Second time series (n_timesteps, n_features)

        Returns:
            Joint recurrence matrix (n_timesteps, n_timesteps)
        """
        assert len(X) == len(Y), "Time series must have same length"

        # Compute individual recurrence matrices
        R_X = self._compute_recurrence_matrix(X)
        R_Y = self._compute_recurrence_matrix(Y)

        # Joint recurrence: element-wise AND
        joint_R = R_X * R_Y

        return joint_R
