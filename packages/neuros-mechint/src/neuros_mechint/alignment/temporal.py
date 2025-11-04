"""
Temporal Alignment Methods

Dynamic time warping, temporal CCA, and other methods for aligning
time-varying neural representations.

References:
- Haxby et al. (2011): A common, high-dimensional model of the brain
- Hasson et al. (2004): Intersubject synchronization
- Dinstein et al. (2015): Neural variability
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


@dataclass
class TemporalAlignmentResult:
    """Results from temporal alignment."""
    aligned_a: np.ndarray
    aligned_b: np.ndarray
    alignment_path: List[Tuple[int, int]]
    alignment_cost: float
    time_warping_function: Optional[np.ndarray] = None


class DynamicTimeWarping:
    """
    Dynamic Time Warping (DTW) for temporal alignment.

    Finds optimal alignment between two time series allowing
    for temporal distortions.
    """

    def __init__(
        self,
        distance_metric: str = 'euclidean',
        window_size: Optional[int] = None
    ):
        self.distance_metric = distance_metric
        self.window_size = window_size

    def align(
        self,
        sequence_a: np.ndarray,
        sequence_b: np.ndarray
    ) -> TemporalAlignmentResult:
        """
        Align two time series using DTW.

        Args:
            sequence_a: Time series A (time_a, features)
            sequence_b: Time series B (time_b, features)

        Returns:
            Alignment results
        """
        # Use fastdtw for efficiency
        distance, path = fastdtw(
            sequence_a,
            sequence_b,
            dist=euclidean if self.distance_metric == 'euclidean' else None,
            radius=self.window_size or 1
        )

        # Extract aligned sequences
        indices_a = [p[0] for p in path]
        indices_b = [p[1] for p in path]

        aligned_a = sequence_a[indices_a]
        aligned_b = sequence_b[indices_b]

        # Compute time warping function
        time_warp = np.array([[i, j] for i, j in path])

        return TemporalAlignmentResult(
            aligned_a=aligned_a,
            aligned_b=aligned_b,
            alignment_path=path,
            alignment_cost=distance,
            time_warping_function=time_warp
        )

    def multi_sequence_alignment(
        self,
        sequences: List[np.ndarray],
        reference_idx: int = 0
    ) -> List[TemporalAlignmentResult]:
        """
        Align multiple sequences to a reference.

        Args:
            sequences: List of time series
            reference_idx: Index of reference sequence

        Returns:
            List of alignment results
        """
        reference = sequences[reference_idx]
        results = []

        for i, seq in enumerate(sequences):
            if i == reference_idx:
                continue

            result = self.align(reference, seq)
            results.append(result)

        return results


class InterSubjectSynchronization:
    """
    Measure temporal synchronization across subjects/models.

    Quantifies how aligned neural responses are across time.
    """

    def __init__(self, method: str = 'correlation'):
        self.method = method

    def compute_isc(
        self,
        responses: List[np.ndarray],
        window_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute inter-subject correlation.

        Args:
            responses: List of (time, features) arrays
            window_size: Sliding window size for temporal resolution

        Returns:
            ISC values over time (time,) or (time, features)
        """
        n_subjects = len(responses)
        n_time = responses[0].shape[0]
        n_features = responses[0].shape[1]

        if window_size is None:
            # Global ISC per feature
            isc = np.zeros(n_features)

            for feat in range(n_features):
                # Extract feature across subjects
                feature_data = np.stack([r[:, feat] for r in responses], axis=1)

                # Leave-one-out correlation
                corrs = []
                for i in range(n_subjects):
                    left_out = feature_data[:, i]
                    others_mean = feature_data[:, [j for j in range(n_subjects) if j != i]].mean(1)

                    corr = np.corrcoef(left_out, others_mean)[0, 1]
                    corrs.append(corr)

                isc[feat] = np.mean(corrs)

        else:
            # Time-resolved ISC
            isc = np.zeros((n_time - window_size + 1, n_features))

            for t in range(n_time - window_size + 1):
                window_data = [r[t:t+window_size] for r in responses]

                for feat in range(n_features):
                    feature_windows = np.stack([w[:, feat] for w in window_data], axis=1)

                    corrs = []
                    for i in range(n_subjects):
                        left_out = feature_windows[:, i]
                        others_mean = feature_windows[:, [j for j in range(n_subjects) if j != i]].mean(1)

                        corr = np.corrcoef(left_out, others_mean)[0, 1]
                        corrs.append(corr)

                    isc[t, feat] = np.mean(corrs)

        return isc

    def phase_synchronization(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray
    ) -> float:
        """
        Compute phase synchronization between two signals.

        Uses Hilbert transform to extract instantaneous phase.

        Args:
            signal_a: Time series A (time,)
            signal_b: Time series B (time,)

        Returns:
            Phase locking value [0, 1]
        """
        from scipy.signal import hilbert

        # Hilbert transform
        analytic_a = hilbert(signal_a)
        analytic_b = hilbert(signal_b)

        # Instantaneous phase
        phase_a = np.angle(analytic_a)
        phase_b = np.angle(analytic_b)

        # Phase difference
        phase_diff = phase_a - phase_b

        # Phase locking value
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))

        return plv


class TimeResolvedCCA:
    """
    Time-resolved Canonical Correlation Analysis.

    Tracks how shared representations evolve over time.
    """

    def __init__(
        self,
        n_components: int = 5,
        window_size: int = 100,
        step_size: int = 10
    ):
        self.n_components = n_components
        self.window_size = window_size
        self.step_size = step_size

    def fit_transform(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
        """
        Fit time-resolved CCA.

        Args:
            X: Time series X (time, features_x)
            Y: Time series Y (time, features_y)

        Returns:
            (X_canonical_list, Y_canonical_list, correlations_over_time)
        """
        from sklearn.cross_decomposition import CCA

        n_time = X.shape[0]
        n_windows = (n_time - self.window_size) // self.step_size + 1

        X_canonical_list = []
        Y_canonical_list = []
        correlations_over_time = np.zeros((n_windows, self.n_components))

        for w in range(n_windows):
            start = w * self.step_size
            end = start + self.window_size

            X_window = X[start:end]
            Y_window = Y[start:end]

            # Fit CCA
            cca = CCA(n_components=self.n_components)
            X_c, Y_c = cca.fit_transform(X_window, Y_window)

            X_canonical_list.append(X_c)
            Y_canonical_list.append(Y_c)

            # Correlations
            for comp in range(self.n_components):
                correlations_over_time[w, comp] = np.corrcoef(
                    X_c[:, comp],
                    Y_c[:, comp]
                )[0, 1]

        return X_canonical_list, Y_canonical_list, correlations_over_time


class TemporalReceptiveField:
    """
    Temporal Receptive Field (TRF) modeling.

    Models how neural responses are predicted by stimuli over time.
    Uses ridge regression with temporal lags.
    """

    def __init__(
        self,
        tmin: float = 0.0,
        tmax: float = 0.5,
        dt: float = 0.01,
        alpha: float = 1.0
    ):
        """
        Args:
            tmin: Minimum lag (s)
            tmax: Maximum lag (s)
            dt: Time step (s)
            alpha: Ridge regularization
        """
        self.tmin = tmin
        self.tmax = tmax
        self.dt = dt
        self.alpha = alpha

        self.lags = np.arange(int(tmin / dt), int(tmax / dt) + 1)
        self.weights = None

    def create_lagged_features(
        self,
        stimulus: np.ndarray
    ) -> np.ndarray:
        """
        Create lagged feature matrix.

        Args:
            stimulus: Stimulus time series (time, features)

        Returns:
            Lagged features (time, features * n_lags)
        """
        n_time, n_features = stimulus.shape
        n_lags = len(self.lags)

        lagged = np.zeros((n_time, n_features * n_lags))

        for i, lag in enumerate(self.lags):
            if lag >= 0:
                lagged[lag:, i*n_features:(i+1)*n_features] = stimulus[:-lag or None]
            else:
                lagged[:lag, i*n_features:(i+1)*n_features] = stimulus[-lag:]

        return lagged

    def fit(
        self,
        stimulus: np.ndarray,
        response: np.ndarray
    ):
        """
        Fit TRF model.

        Args:
            stimulus: Stimulus (time, stim_features)
            response: Neural response (time, response_features)
        """
        from sklearn.linear_model import Ridge

        # Create lagged stimulus
        X = self.create_lagged_features(stimulus)
        y = response

        # Trim to valid time points
        valid_start = max(0, -self.lags.min())
        valid_end = min(X.shape[0], X.shape[0] - self.lags.max())

        X_valid = X[valid_start:valid_end]
        y_valid = y[valid_start:valid_end]

        # Ridge regression
        ridge = Ridge(alpha=self.alpha)
        ridge.fit(X_valid, y_valid)

        self.weights = ridge.coef_

    def predict(
        self,
        stimulus: np.ndarray
    ) -> np.ndarray:
        """
        Predict response from stimulus.

        Args:
            stimulus: Stimulus (time, features)

        Returns:
            Predicted response (time, response_features)
        """
        X = self.create_lagged_features(stimulus)

        valid_start = max(0, -self.lags.min())
        valid_end = min(X.shape[0], X.shape[0] - self.lags.max())

        X_valid = X[valid_start:valid_end]

        prediction = X_valid @ self.weights.T

        # Pad with zeros
        full_prediction = np.zeros((X.shape[0], self.weights.shape[0]))
        full_prediction[valid_start:valid_end] = prediction

        return full_prediction

    def get_trf(self, feature_idx: int) -> np.ndarray:
        """
        Get TRF for a specific response feature.

        Args:
            feature_idx: Index of response feature

        Returns:
            TRF (n_lags, n_stim_features)
        """
        if self.weights is None:
            raise ValueError("Model not fitted")

        n_stim_features = self.weights.shape[1] // len(self.lags)
        trf = self.weights[feature_idx].reshape(len(self.lags), n_stim_features)

        return trf


class PhasePrecession:
    """
    Analyze phase precession in temporal sequences.

    Measures how neural representations shift phase within
    oscillatory cycles.
    """

    def __init__(self, frequency_band: Tuple[float, float] = (4, 12)):
        """
        Args:
            frequency_band: (low, high) frequency in Hz
        """
        self.frequency_band = frequency_band

    def extract_phase(
        self,
        signal: np.ndarray,
        fs: float
    ) -> np.ndarray:
        """
        Extract instantaneous phase in frequency band.

        Args:
            signal: Time series (time,)
            fs: Sampling frequency (Hz)

        Returns:
            Instantaneous phase (time,)
        """
        from scipy.signal import butter, filtfilt, hilbert

        # Bandpass filter
        nyq = fs / 2
        low = self.frequency_band[0] / nyq
        high = self.frequency_band[1] / nyq

        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)

        # Hilbert transform
        analytic = hilbert(filtered)
        phase = np.angle(analytic)

        return phase

    def measure_precession(
        self,
        position: np.ndarray,
        spikes: np.ndarray,
        phase: np.ndarray
    ) -> Tuple[float, float]:
        """
        Measure phase precession slope.

        Args:
            position: Position in environment (time,)
            spikes: Spike train (time,)
            phase: Theta phase (time,)

        Returns:
            (slope, r_squared): Precession slope and fit quality
        """
        # Extract spike positions and phases
        spike_times = np.where(spikes > 0)[0]
        spike_positions = position[spike_times]
        spike_phases = phase[spike_times]

        # Linear regression
        from scipy.stats import linregress

        slope, intercept, r_value, p_value, std_err = linregress(
            spike_positions,
            spike_phases
        )

        return slope, r_value ** 2
