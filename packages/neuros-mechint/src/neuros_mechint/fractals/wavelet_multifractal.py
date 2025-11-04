"""
Wavelet-Based Multifractal Analysis

Wavelet transform modulus maxima (WTMM) method for multifractal analysis.
Provides more robust and localized fractal characterization than traditional methods.

References:
- Mallat & Hwang (1992): Singularity detection and processing with wavelets
- Muzy et al. (1991): Wavelets and multifractal formalism
- Ivanov et al. (1999): Multifractality in human heartbeat dynamics
- Wendt et al. (2007): Bootstrap for wavelet-based multifractal analysis
"""

import numpy as np
import pywt
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from scipy.interpolate import interp1d


@dataclass
class MultifractalSpectrum:
    """Multifractal spectrum from WTMM analysis."""
    q_values: np.ndarray  # Moment orders
    tau_q: np.ndarray  # Scaling exponents τ(q)
    h_q: np.ndarray  # Hölder exponents h(q)
    D_h: np.ndarray  # Singularity spectrum D(h)
    width: float  # Spectrum width (multifractality measure)
    asymmetry: float  # Spectrum asymmetry
    max_alpha: float  # Maximum singularity strength
    min_alpha: float  # Minimum singularity strength


class WaveletMultifractal:
    """
    Wavelet-based multifractal analysis using WTMM method.

    Analyzes scale-invariant fluctuations and singularity structure
    of time series or spatial patterns.
    """

    def __init__(
        self,
        wavelet: str = 'mexh',  # Mexican hat wavelet
        scales: Optional[np.ndarray] = None,
        q_range: Tuple[float, float] = (-5, 5),
        n_q: int = 21
    ):
        """
        Args:
            wavelet: Wavelet type ('mexh', 'gaus1', 'morl')
            scales: Wavelet scales to use
            q_range: Range of moment orders
            n_q: Number of moment orders
        """
        self.wavelet = wavelet
        self.scales = scales
        self.q_values = np.linspace(q_range[0], q_range[1], n_q)

    def analyze(
        self,
        signal: np.ndarray,
        return_coefficients: bool = False
    ) -> MultifractalSpectrum:
        """
        Perform multifractal analysis on signal.

        Args:
            signal: 1D time series
            return_coefficients: Whether to return wavelet coefficients

        Returns:
            Multifractal spectrum
        """
        # Continuous wavelet transform
        if self.scales is None:
            # Auto-generate scales
            max_scale = min(len(signal) // 4, 1024)
            self.scales = 2 ** np.arange(1, np.log2(max_scale))

        coeffs, freqs = pywt.cwt(signal, self.scales, self.wavelet)

        # Partition function (sum of q-th moments)
        Z_q = self._partition_function(coeffs, self.q_values)

        # Scaling exponents τ(q)
        tau_q = self._scaling_exponents(Z_q, self.scales, self.q_values)

        # Singularity spectrum via Legendre transform
        h_q, D_h = self._singularity_spectrum(tau_q, self.q_values)

        # Spectrum characteristics
        width = h_q.max() - h_q.min()
        asymmetry = (h_q[len(h_q)//2] - h_q.min()) / width

        result = MultifractalSpectrum(
            q_values=self.q_values,
            tau_q=tau_q,
            h_q=h_q,
            D_h=D_h,
            width=width,
            asymmetry=asymmetry,
            max_alpha=h_q.max(),
            min_alpha=h_q.min()
        )

        return result

    def _partition_function(
        self,
        coeffs: np.ndarray,
        q_values: np.ndarray
    ) -> np.ndarray:
        """
        Compute partition function Z(q, a) for wavelet coefficients.

        Z(q, a) = Σ |W(a, b)|^q

        Args:
            coeffs: Wavelet coefficients (scales, time)
            q_values: Moment orders

        Returns:
            Partition function (n_scales, n_q)
        """
        n_scales = coeffs.shape[0]
        n_q = len(q_values)

        Z_q = np.zeros((n_scales, n_q))

        for i, q in enumerate(q_values):
            if q == 0:
                # Special case: count non-zero coefficients
                Z_q[:, i] = np.sum(np.abs(coeffs) > 0, axis=1)
            else:
                # Sum of q-th powers
                Z_q[:, i] = np.sum(np.abs(coeffs) ** q, axis=1)

        return Z_q

    def _scaling_exponents(
        self,
        Z_q: np.ndarray,
        scales: np.ndarray,
        q_values: np.ndarray
    ) -> np.ndarray:
        """
        Estimate scaling exponents τ(q) from partition function.

        Z(q, a) ~ a^τ(q)

        Args:
            Z_q: Partition function (n_scales, n_q)
            scales: Wavelet scales
            q_values: Moment orders

        Returns:
            Scaling exponents τ(q)
        """
        n_q = len(q_values)
        tau_q = np.zeros(n_q)

        log_scales = np.log2(scales)

        for i in range(n_q):
            # Avoid log of zero
            mask = Z_q[:, i] > 0
            log_Z = np.log2(Z_q[mask, i])

            # Linear regression in log-log space
            if len(log_Z) > 2:
                tau_q[i] = np.polyfit(log_scales[mask], log_Z, 1)[0]
            else:
                tau_q[i] = np.nan

        return tau_q

    def _singularity_spectrum(
        self,
        tau_q: np.ndarray,
        q_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute singularity spectrum D(h) via Legendre transform.

        h(q) = dτ(q)/dq
        D(h) = q*h(q) - τ(q)

        Args:
            tau_q: Scaling exponents
            q_values: Moment orders

        Returns:
            (h_q, D_h): Hölder exponents and singularity spectrum
        """
        # Numerical derivative
        h_q = np.gradient(tau_q, q_values)

        # Legendre transform
        D_h = q_values * h_q - tau_q

        return h_q, D_h

    def local_hurst_exponent(
        self,
        signal: np.ndarray,
        window_size: int = 256,
        step_size: int = 64
    ) -> np.ndarray:
        """
        Compute time-varying local Hurst exponent.

        Args:
            signal: Time series
            window_size: Window size for local analysis
            step_size: Step size for sliding window

        Returns:
            Local Hurst exponents over time
        """
        n_time = len(signal)
        n_windows = (n_time - window_size) // step_size

        hurst_local = np.zeros(n_windows)

        for i in range(n_windows):
            start = i * step_size
            end = start + window_size

            window = signal[start:end]

            # Analyze window
            # Use q=2 (second moment) for Hurst exponent
            spectrum = self.analyze(window)

            # Hurst exponent is h(2)
            q2_idx = np.argmin(np.abs(spectrum.q_values - 2))
            hurst_local[i] = spectrum.h_q[q2_idx]

        return hurst_local


class MultifractalDetrendedFluctuationAnalysis:
    """
    Multifractal Detrended Fluctuation Analysis (MF-DFA).

    More robust to trends than standard DFA, provides
    multifractal characterization.

    References:
        Kantelhardt et al. (2002): Multifractal detrended fluctuation analysis
    """

    def __init__(
        self,
        q_range: Tuple[float, float] = (-5, 5),
        n_q: int = 21,
        polynomial_order: int = 1
    ):
        self.q_values = np.linspace(q_range[0], q_range[1], n_q)
        self.polynomial_order = polynomial_order

    def analyze(
        self,
        signal: np.ndarray,
        scales: Optional[np.ndarray] = None
    ) -> MultifractalSpectrum:
        """
        Perform MF-DFA.

        Args:
            signal: Time series
            scales: Scales for analysis

        Returns:
            Multifractal spectrum
        """
        N = len(signal)

        if scales is None:
            # Logarithmically spaced scales
            min_scale = 16
            max_scale = N // 4
            scales = np.unique(np.logspace(
                np.log10(min_scale),
                np.log10(max_scale),
                20
            ).astype(int))

        # Profile (cumulative sum)
        profile = np.cumsum(signal - signal.mean())

        # Fluctuation function for each scale and q
        F_q = np.zeros((len(scales), len(self.q_values)))

        for i, scale in enumerate(scales):
            # Detrended fluctuations
            fluctuations = self._detrend_fluctuations(profile, scale)

            # q-th order fluctuation function
            for j, q in enumerate(self.q_values):
                if q == 0:
                    # Logarithmic averaging
                    F_q[i, j] = np.exp(0.5 * np.mean(np.log(fluctuations + 1e-10)))
                else:
                    # q-th moment
                    F_q[i, j] = (np.mean(fluctuations ** (q / 2))) ** (1 / q)

        # Scaling exponents
        h_q = self._generalized_hurst_exponents(F_q, scales)

        # Multifractal spectrum via Legendre transform
        tau_q = self.q_values * h_q - 1

        # Singularity spectrum
        alpha = h_q
        f_alpha = self.q_values * h_q - tau_q

        width = alpha.max() - alpha.min()
        asymmetry = (alpha[len(alpha)//2] - alpha.min()) / width

        return MultifractalSpectrum(
            q_values=self.q_values,
            tau_q=tau_q,
            h_q=h_q,
            D_h=f_alpha,
            width=width,
            asymmetry=asymmetry,
            max_alpha=alpha.max(),
            min_alpha=alpha.min()
        )

    def _detrend_fluctuations(
        self,
        profile: np.ndarray,
        scale: int
    ) -> np.ndarray:
        """
        Compute detrended fluctuations at given scale.

        Args:
            profile: Cumulative profile
            scale: Scale (window size)

        Returns:
            Detrended variance for each segment
        """
        N = len(profile)
        n_segments = N // scale

        fluctuations = []

        for v in range(n_segments):
            start = v * scale
            end = (v + 1) * scale

            segment = profile[start:end]
            t = np.arange(len(segment))

            # Polynomial fit
            poly_fit = np.polyfit(t, segment, self.polynomial_order)
            trend = np.polyval(poly_fit, t)

            # Variance
            variance = np.mean((segment - trend) ** 2)
            fluctuations.append(variance)

        return np.array(fluctuations)

    def _generalized_hurst_exponents(
        self,
        F_q: np.ndarray,
        scales: np.ndarray
    ) -> np.ndarray:
        """
        Estimate generalized Hurst exponents.

        F(q, s) ~ s^h(q)

        Args:
            F_q: Fluctuation function (n_scales, n_q)
            scales: Scales

        Returns:
            Generalized Hurst exponents h(q)
        """
        n_q = F_q.shape[1]
        h_q = np.zeros(n_q)

        log_scales = np.log(scales)

        for i in range(n_q):
            log_F = np.log(F_q[:, i] + 1e-10)

            # Linear regression
            h_q[i] = np.polyfit(log_scales, log_F, 1)[0]

        return h_q


class MultifractalTemporalCorrelation:
    """
    Analyze multifractal temporal correlations in neural activity.

    Combines multifractal analysis with correlation analysis to
    understand scale-dependent memory effects.
    """

    def __init__(self):
        self.mfdfa = MultifractalDetrendedFluctuationAnalysis()

    def correlation_dimension(
        self,
        signal: np.ndarray,
        embedding_dim: int = 5,
        time_delay: int = 1
    ) -> float:
        """
        Estimate correlation dimension (fractal dimension of attractor).

        Args:
            signal: Time series
            embedding_dim: Embedding dimension
            time_delay: Time delay for embedding

        Returns:
            Correlation dimension
        """
        # Time-delay embedding
        embedded = self._time_delay_embedding(signal, embedding_dim, time_delay)

        # Correlation sum
        distances = self._pairwise_distances(embedded)

        # Estimate scaling
        r_values = np.logspace(-2, 0, 20)
        C_r = np.array([np.mean(distances < r) for r in r_values])

        # Linear regression in log-log
        mask = (C_r > 0) & (C_r < 1)
        if np.sum(mask) > 2:
            log_r = np.log(r_values[mask])
            log_C = np.log(C_r[mask])

            correlation_dim = np.polyfit(log_r, log_C, 1)[0]
        else:
            correlation_dim = np.nan

        return correlation_dim

    def _time_delay_embedding(
        self,
        signal: np.ndarray,
        embedding_dim: int,
        time_delay: int
    ) -> np.ndarray:
        """
        Create time-delay embedding.

        Args:
            signal: 1D time series
            embedding_dim: Embedding dimension
            time_delay: Time delay

        Returns:
            Embedded vectors (n_vectors, embedding_dim)
        """
        N = len(signal)
        n_vectors = N - (embedding_dim - 1) * time_delay

        embedded = np.zeros((n_vectors, embedding_dim))

        for i in range(embedding_dim):
            start = i * time_delay
            end = start + n_vectors
            embedded[:, i] = signal[start:end]

        return embedded

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances."""
        from scipy.spatial.distance import pdist

        return pdist(X)
