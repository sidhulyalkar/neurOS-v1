"""
Spectral Analysis Methods

This module provides advanced spectral analysis tools for dynamical systems.

Key capabilities:
- Power spectral density
- Spectral entropy
- Dominant frequencies
- Wavelet analysis
- Spectral clustering
- Laplacian eigenmap
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.linalg import eigh
from scipy.sparse import csgraph
from scipy.spatial.distance import pdist, squareform
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpectralResult:
    """Results from spectral analysis."""

    # Frequency domain
    frequencies: np.ndarray  # Frequency values
    power_spectral_density: np.ndarray  # PSD
    dominant_frequencies: np.ndarray  # Dominant frequency peaks
    spectral_entropy: float  # Shannon entropy of spectrum

    # Wavelets (if computed)
    wavelet_coefficients: Optional[np.ndarray] = None
    wavelet_frequencies: Optional[np.ndarray] = None

    # Spectral embedding (if computed)
    laplacian_eigenvectors: Optional[np.ndarray] = None
    laplacian_eigenvalues: Optional[np.ndarray] = None


class SpectralAnalyzer:
    """
    Advanced spectral analysis for dynamical systems.

    Spectral methods reveal the frequency content and harmonic structure
    of time series and spatial patterns.
    """

    def __init__(
        self,
        dt: float = 0.01,
        verbose: bool = True
    ):
        """
        Initialize spectral analyzer.

        Args:
            dt: Time step
            verbose: Whether to log information
        """
        self.dt = dt
        self.fs = 1.0 / dt  # Sampling frequency
        self.verbose = verbose

    def analyze(
        self,
        time_series: np.ndarray,
        compute_wavelets: bool = False,
        compute_embedding: bool = False
    ) -> SpectralResult:
        """
        Comprehensive spectral analysis.

        Args:
            time_series: Time series data (n_timesteps,) or (n_timesteps, n_features)
            compute_wavelets: Whether to compute wavelet transform
            compute_embedding: Whether to compute spectral embedding

        Returns:
            SpectralResult
        """
        # Handle multi-dimensional
        if time_series.ndim == 2:
            # Use first feature for frequency analysis
            ts = time_series[:, 0]
        else:
            ts = time_series

        # Power spectral density
        freqs, psd = self._compute_psd(ts)

        # Dominant frequencies
        dominant_freqs = self._find_dominant_frequencies(freqs, psd)

        # Spectral entropy
        spectral_entropy = self._compute_spectral_entropy(psd)

        # Wavelets
        if compute_wavelets:
            wavelet_coeffs, wavelet_freqs = self._wavelet_transform(ts)
        else:
            wavelet_coeffs = None
            wavelet_freqs = None

        # Spectral embedding
        if compute_embedding and time_series.ndim == 2:
            lap_eigenvectors, lap_eigenvalues = self._spectral_embedding(time_series)
        else:
            lap_eigenvectors = None
            lap_eigenvalues = None

        return SpectralResult(
            frequencies=freqs,
            power_spectral_density=psd,
            dominant_frequencies=dominant_freqs,
            spectral_entropy=spectral_entropy,
            wavelet_coefficients=wavelet_coeffs,
            wavelet_frequencies=wavelet_freqs,
            laplacian_eigenvectors=lap_eigenvectors,
            laplacian_eigenvalues=lap_eigenvalues
        )

    def _compute_psd(self, ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density using Welch's method.

        Args:
            ts: Time series (n_timesteps,)

        Returns:
            Tuple of (frequencies, psd)
        """
        # Welch's method for smoother estimate
        freqs, psd = signal.welch(
            ts,
            fs=self.fs,
            nperseg=min(256, len(ts) // 4),
            scaling='density'
        )

        return freqs, psd

    def _find_dominant_frequencies(
        self,
        freqs: np.ndarray,
        psd: np.ndarray,
        n_peaks: int = 5
    ) -> np.ndarray:
        """
        Find dominant frequency peaks.

        Args:
            freqs: Frequency array
            psd: Power spectral density
            n_peaks: Number of peaks to return

        Returns:
            Array of dominant frequencies
        """
        # Find peaks
        peaks, properties = signal.find_peaks(psd, height=np.max(psd) * 0.1)

        if len(peaks) == 0:
            return np.array([])

        # Sort by height
        peak_heights = properties['peak_heights']
        sorted_indices = np.argsort(peak_heights)[::-1]

        # Get top n_peaks
        top_peaks = peaks[sorted_indices[:n_peaks]]
        dominant_freqs = freqs[top_peaks]

        return dominant_freqs

    def _compute_spectral_entropy(self, psd: np.ndarray) -> float:
        """
        Compute spectral entropy.

        Args:
            psd: Power spectral density

        Returns:
            Spectral entropy (bits)
        """
        # Normalize to probability distribution
        psd_normalized = psd / np.sum(psd)

        # Shannon entropy
        entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-10))

        return entropy

    def _wavelet_transform(
        self,
        ts: np.ndarray,
        wavelet: str = 'morlet',
        scales: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute continuous wavelet transform.

        Args:
            ts: Time series
            wavelet: Wavelet type
            scales: Wavelet scales

        Returns:
            Tuple of (coefficients, frequencies)
        """
        if scales is None:
            # Default scales (logarithmic spacing)
            scales = np.logspace(0, 3, 50)

        # Compute CWT using scipy
        if wavelet == 'morlet':
            # Morlet wavelet
            coefficients = np.zeros((len(scales), len(ts)), dtype=complex)

            for i, scale in enumerate(scales):
                # Create Morlet wavelet
                wavelet_length = min(10 * scale, len(ts))
                wavelet_time = np.arange(-wavelet_length // 2, wavelet_length // 2)
                wavelet_func = np.exp(1j * 2 * np.pi * wavelet_time / scale) * np.exp(-(wavelet_time ** 2) / (2 * scale ** 2))

                # Convolve
                coefficients[i] = np.convolve(ts, wavelet_func, mode='same')

        else:
            # Use scipy's cwt for other wavelets
            from scipy import signal as scipy_signal
            coefficients = scipy_signal.cwt(ts, scipy_signal.ricker, scales)

        # Convert scales to frequencies
        frequencies = self.fs / scales

        return coefficients, frequencies

    def _spectral_embedding(
        self,
        X: np.ndarray,
        n_components: int = 3,
        method: str = "laplacian"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute spectral embedding (Laplacian eigenmap).

        Args:
            X: Data points (n_samples, n_features)
            n_components: Number of components
            method: Embedding method ("laplacian", "normalized")

        Returns:
            Tuple of (eigenvectors, eigenvalues)
        """
        # Construct affinity matrix (Gaussian kernel)
        distances = squareform(pdist(X, metric='euclidean'))
        sigma = np.median(distances)
        affinity = np.exp(-distances ** 2 / (2 * sigma ** 2))

        # Construct graph Laplacian
        degree_matrix = np.diag(np.sum(affinity, axis=1))

        if method == "laplacian":
            laplacian = degree_matrix - affinity
        elif method == "normalized":
            # Normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
            degree_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degree_matrix) + 1e-10))
            laplacian = np.eye(len(X)) - degree_inv_sqrt @ affinity @ degree_inv_sqrt
        else:
            raise ValueError(f"Unknown method: {method}")

        # Eigendecomposition
        eigenvalues, eigenvectors = eigh(laplacian)

        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx[:n_components]]
        eigenvectors = eigenvectors[:, idx[:n_components]]

        return eigenvectors, eigenvalues

    def spectral_clustering(
        self,
        X: np.ndarray,
        n_clusters: int = 3
    ) -> np.ndarray:
        """
        Perform spectral clustering.

        Args:
            X: Data points (n_samples, n_features)
            n_clusters: Number of clusters

        Returns:
            Cluster labels
        """
        # Spectral embedding
        eigenvectors, _ = self._spectral_embedding(X, n_components=n_clusters)

        # K-means on embedding
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(eigenvectors)

        return labels

    def cross_spectral_density(
        self,
        ts1: np.ndarray,
        ts2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cross-spectral density between two time series.

        Args:
            ts1: First time series
            ts2: Second time series

        Returns:
            Tuple of (frequencies, cross_spectrum)
        """
        # Use Welch's method
        freqs, cross_spectrum = signal.csd(
            ts1,
            ts2,
            fs=self.fs,
            nperseg=min(256, len(ts1) // 4)
        )

        return freqs, cross_spectrum

    def coherence(
        self,
        ts1: np.ndarray,
        ts2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute magnitude-squared coherence.

        Args:
            ts1: First time series
            ts2: Second time series

        Returns:
            Tuple of (frequencies, coherence)
        """
        freqs, coherence = signal.coherence(
            ts1,
            ts2,
            fs=self.fs,
            nperseg=min(256, len(ts1) // 4)
        )

        return freqs, coherence

    def multitaper_spectrum(
        self,
        ts: np.ndarray,
        n_tapers: int = 5,
        bandwidth: float = 4.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute multitaper power spectrum for reduced variance.

        Args:
            ts: Time series
            n_tapers: Number of tapers
            bandwidth: Time-bandwidth product

        Returns:
            Tuple of (frequencies, spectrum)
        """
        n = len(ts)

        # Generate Slepian (DPSS) tapers
        from scipy.signal import windows
        tapers = windows.dpss(n, bandwidth, n_tapers)

        # Compute spectrum for each taper
        spectra = []
        for taper in tapers:
            windowed = ts * taper
            spectrum = np.abs(fft(windowed)) ** 2
            spectra.append(spectrum)

        # Average spectra
        mean_spectrum = np.mean(spectra, axis=0)

        # Frequencies
        freqs = fftfreq(n, d=self.dt)

        # Return positive frequencies only
        positive_freqs = freqs[:n // 2]
        positive_spectrum = mean_spectrum[:n // 2]

        return positive_freqs, positive_spectrum

    def spectrogram(
        self,
        ts: np.ndarray,
        window_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram (time-frequency representation).

        Args:
            ts: Time series
            window_length: Length of window for STFT

        Returns:
            Tuple of (times, frequencies, spectrogram)
        """
        if window_length is None:
            window_length = min(256, len(ts) // 8)

        # Short-time Fourier transform
        freqs, times, Sxx = signal.spectrogram(
            ts,
            fs=self.fs,
            nperseg=window_length,
            noverlap=window_length // 2
        )

        return times, freqs, Sxx
