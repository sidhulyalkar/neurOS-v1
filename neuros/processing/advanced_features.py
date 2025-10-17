"""
Advanced signal processing features for BCI applications.

This module implements state-of-the-art spatial filtering and feature extraction
techniques used in competitive BCI benchmarks:
- Common Spatial Patterns (CSP)
- Riemannian geometry features
- Wavelet decomposition
- Time-frequency analysis
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import signal
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin


class CommonSpatialPatterns(BaseEstimator, TransformerMixin):
    """
    Common Spatial Patterns (CSP) for EEG spatial filtering.

    CSP is one of the most effective techniques for motor imagery BCI,
    often achieving state-of-the-art performance on benchmark datasets.

    Parameters
    ----------
    n_components : int
        Number of CSP components to retain (default: 4)
    reg : float or None
        Regularization parameter for covariance estimation (default: None)
    log : bool
        If True, apply log transform to variance features (default: True)

    References
    ----------
    Ramoser, H., Muller-Gerking, J., & Pfurtscheller, G. (2000).
    "Optimal spatial filtering of single trial EEG during imagined hand movement."
    IEEE Transactions on Rehabilitation Engineering.
    """

    def __init__(self, n_components: int = 4, reg: Optional[float] = None,
                 log: bool = True):
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.filters_ = None
        self.patterns_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'CommonSpatialPatterns':
        """
        Fit CSP filters to training data.

        Parameters
        ----------
        X : np.ndarray
            EEG data of shape (n_trials, n_channels, n_samples)
        y : np.ndarray
            Binary class labels (0 or 1)

        Returns
        -------
        self : CommonSpatialPatterns
        """
        # Separate classes
        X_class_0 = X[y == 0]
        X_class_1 = X[y == 1]

        # Compute covariance matrices
        cov_0 = self._compute_covariance(X_class_0)
        cov_1 = self._compute_covariance(X_class_1)

        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = eigh(cov_0, cov_0 + cov_1)

        # Sort by eigenvalues
        ix = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[ix]
        eigenvectors = eigenvectors[:, ix]

        # Select most discriminative components
        self.filters_ = eigenvectors.T
        self.patterns_ = np.linalg.pinv(self.filters_).T

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply CSP spatial filtering and extract features.

        Parameters
        ----------
        X : np.ndarray
            EEG data of shape (n_trials, n_channels, n_samples)

        Returns
        -------
        features : np.ndarray
            CSP features of shape (n_trials, n_components)
        """
        if self.filters_ is None:
            raise RuntimeError("CSP not fitted. Call fit() first.")

        n_trials = X.shape[0]
        features = np.zeros((n_trials, self.n_components))

        for i in range(n_trials):
            # Apply spatial filter
            X_filtered = self.filters_[:self.n_components] @ X[i]

            # Compute variance of each component
            variances = np.var(X_filtered, axis=1)

            # Normalize by total variance
            features[i] = variances / np.sum(variances)

        # Apply log transform for better classification
        if self.log:
            features = np.log(features + 1e-8)

        return features

    def _compute_covariance(self, X: np.ndarray) -> np.ndarray:
        """Compute average covariance matrix for a class."""
        n_trials, n_channels, n_samples = X.shape
        cov = np.zeros((n_channels, n_channels))

        for trial in X:
            # Normalize by trace for robustness
            trial_cov = np.cov(trial)
            cov += trial_cov / np.trace(trial_cov)

        cov /= n_trials

        # Add regularization if specified
        if self.reg is not None:
            cov += self.reg * np.eye(n_channels)

        return cov


class RiemannianFeatures(BaseEstimator, TransformerMixin):
    """
    Riemannian geometry features for covariance matrices.

    Represents EEG trials as points on the manifold of symmetric positive
    definite (SPD) matrices, which often outperforms Euclidean methods.

    Parameters
    ----------
    metric : str
        Distance metric on SPD manifold: 'riemann', 'euclid', 'logeuclid'
    n_jobs : int
        Number of parallel jobs (default: 1)

    References
    ----------
    Barachant, A., Bonnet, S., Congedo, M., & Jutten, C. (2012).
    "Multiclass brain-computer interface classification by Riemannian geometry."
    IEEE Transactions on Biomedical Engineering.
    """

    def __init__(self, metric: str = 'riemann', n_jobs: int = 1):
        self.metric = metric
        self.n_jobs = n_jobs
        self.mean_cov_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RiemannianFeatures':
        """
        Fit Riemannian mean for each class.

        Parameters
        ----------
        X : np.ndarray
            EEG data of shape (n_trials, n_channels, n_samples)
        y : np.ndarray
            Class labels

        Returns
        -------
        self : RiemannianFeatures
        """
        # Compute covariance matrices
        covmats = self._compute_covariances(X)

        # Compute Riemannian mean for each class
        classes = np.unique(y)
        self.mean_cov_ = {}

        for cls in classes:
            class_covs = covmats[y == cls]
            self.mean_cov_[cls] = self._riemannian_mean(class_covs)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extract Riemannian distance features.

        Parameters
        ----------
        X : np.ndarray
            EEG data of shape (n_trials, n_channels, n_samples)

        Returns
        -------
        features : np.ndarray
            Distance features of shape (n_trials, n_classes)
        """
        if self.mean_cov_ is None:
            raise RuntimeError("RiemannianFeatures not fitted.")

        # Compute covariance matrices
        covmats = self._compute_covariances(X)

        # Compute distances to each class mean
        n_trials = len(covmats)
        n_classes = len(self.mean_cov_)
        features = np.zeros((n_trials, n_classes))

        for i, covmat in enumerate(covmats):
            for j, (cls, mean_cov) in enumerate(self.mean_cov_.items()):
                features[i, j] = self._distance(covmat, mean_cov)

        return features

    def _compute_covariances(self, X: np.ndarray) -> np.ndarray:
        """Compute covariance matrix for each trial."""
        n_trials = X.shape[0]
        n_channels = X.shape[1]
        covmats = np.zeros((n_trials, n_channels, n_channels))

        for i in range(n_trials):
            covmats[i] = np.cov(X[i])

        return covmats

    def _riemannian_mean(self, covmats: np.ndarray, max_iter: int = 50,
                         tol: float = 1e-6) -> np.ndarray:
        """Compute Riemannian mean of covariance matrices."""
        # Initialize with arithmetic mean
        mean = np.mean(covmats, axis=0)

        for _ in range(max_iter):
            # Compute geometric mean using fixed-point iteration
            mean_sqrt = self._sqrtm(mean)
            mean_inv_sqrt = np.linalg.inv(mean_sqrt)

            # Accumulate log-Euclidean updates
            accum = np.zeros_like(mean)
            for cov in covmats:
                temp = mean_inv_sqrt @ cov @ mean_inv_sqrt
                accum += self._logm(temp)

            accum /= len(covmats)

            # Update mean
            new_mean = mean_sqrt @ self._expm(accum) @ mean_sqrt

            # Check convergence
            if np.linalg.norm(new_mean - mean) < tol:
                break

            mean = new_mean

        return mean

    def _distance(self, A: np.ndarray, B: np.ndarray) -> float:
        """Compute Riemannian distance between two SPD matrices."""
        if self.metric == 'riemann':
            # Affine-invariant Riemannian metric
            A_inv_sqrt = np.linalg.inv(self._sqrtm(A))
            temp = A_inv_sqrt @ B @ A_inv_sqrt
            eigvals = np.linalg.eigvalsh(temp)
            return np.sqrt(np.sum(np.log(eigvals) ** 2))
        elif self.metric == 'logeuclid':
            # Log-Euclidean metric
            return np.linalg.norm(self._logm(A) - self._logm(B), 'fro')
        else:  # euclid
            # Euclidean metric
            return np.linalg.norm(A - B, 'fro')

    def _sqrtm(self, A: np.ndarray) -> np.ndarray:
        """Matrix square root."""
        eigvals, eigvecs = eigh(A)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    def _logm(self, A: np.ndarray) -> np.ndarray:
        """Matrix logarithm."""
        eigvals, eigvecs = eigh(A)
        return eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T

    def _expm(self, A: np.ndarray) -> np.ndarray:
        """Matrix exponential."""
        eigvals, eigvecs = eigh(A)
        return eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.T


class WaveletFeatures:
    """
    Wavelet-based time-frequency features.

    Uses continuous or discrete wavelet transform to extract
    time-frequency features from EEG signals.

    Parameters
    ----------
    wavelet : str
        Wavelet family ('db4', 'sym4', 'coif3', etc.)
    scales : array-like
        Scales for continuous wavelet transform
    """

    def __init__(self, wavelet: str = 'db4', scales: Optional[np.ndarray] = None):
        self.wavelet = wavelet
        self.scales = scales if scales is not None else np.arange(1, 32)

    def extract(self, data: np.ndarray) -> np.ndarray:
        """
        Extract wavelet features from EEG data.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_channels, n_samples) or (n_samples,)

        Returns
        -------
        features : np.ndarray
            Wavelet coefficient statistics
        """
        try:
            import pywt
        except ImportError:
            raise ImportError("PyWavelets is required. Install with: pip install PyWavelets")

        if data.ndim == 1:
            data = data[np.newaxis, :]

        features_list = []

        for ch_data in data:
            # Continuous wavelet transform
            coeffs, freqs = pywt.cwt(ch_data, self.scales, self.wavelet)

            # Extract statistical features from coefficients
            mean_coeffs = np.mean(np.abs(coeffs), axis=1)
            std_coeffs = np.std(coeffs, axis=1)
            energy = np.sum(coeffs ** 2, axis=1)

            ch_features = np.concatenate([mean_coeffs, std_coeffs, energy])
            features_list.append(ch_features)

        return np.concatenate(features_list)


class TimeFrequencyFeatures:
    """
    Advanced time-frequency analysis features.

    Extracts features using Short-Time Fourier Transform (STFT)
    and spectrograms for dynamic frequency analysis.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz
    window : str
        Window function ('hann', 'hamming', 'blackman')
    nperseg : int
        Length of each segment for STFT
    """

    def __init__(self, fs: float, window: str = 'hann', nperseg: int = 256):
        self.fs = fs
        self.window = window
        self.nperseg = nperseg

    def extract(self, data: np.ndarray) -> np.ndarray:
        """
        Extract time-frequency features using STFT.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_channels, n_samples)

        Returns
        -------
        features : np.ndarray
            Time-frequency feature vector
        """
        if data.ndim == 1:
            data = data[np.newaxis, :]

        features_list = []

        for ch_data in data:
            # Compute spectrogram
            f, t, Sxx = signal.spectrogram(
                ch_data, self.fs,
                window=self.window,
                nperseg=self.nperseg
            )

            # Extract features from spectrogram
            mean_power = np.mean(Sxx, axis=1)
            max_power = np.max(Sxx, axis=1)
            spectral_entropy = self._spectral_entropy(Sxx)

            ch_features = np.concatenate([
                mean_power,
                max_power,
                [spectral_entropy]
            ])
            features_list.append(ch_features)

        return np.concatenate(features_list)

    def _spectral_entropy(self, Sxx: np.ndarray) -> float:
        """Compute spectral entropy of power spectrum."""
        # Normalize to probability distribution
        psd = np.mean(Sxx, axis=1)
        psd = psd / np.sum(psd)

        # Compute entropy
        entropy = -np.sum(psd * np.log2(psd + 1e-12))
        return entropy


class SpatioTemporalPatterns:
    """
    Spatio-temporal pattern extraction for EEG.

    Combines spatial and temporal filtering for enhanced
    feature discrimination in BCI tasks.
    """

    def __init__(self, n_spatial_filters: int = 4, n_temporal_filters: int = 3):
        self.n_spatial_filters = n_spatial_filters
        self.n_temporal_filters = n_temporal_filters
        self.spatial_filters_ = None
        self.temporal_filters_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SpatioTemporalPatterns':
        """
        Learn spatio-temporal filters from training data.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_trials, n_channels, n_samples)
        y : np.ndarray
            Binary labels

        Returns
        -------
        self : SpatioTemporalPatterns
        """
        # First learn spatial filters (CSP-like)
        csp = CommonSpatialPatterns(n_components=self.n_spatial_filters)
        csp.fit(X, y)
        self.spatial_filters_ = csp.filters_[:self.n_spatial_filters]

        # Then learn temporal filters on spatially filtered data
        X_spatial = np.array([self.spatial_filters_ @ trial for trial in X])

        # Simple temporal filtering using bandpass filters
        # In practice, could use more sophisticated methods
        self.temporal_filters_ = self._learn_temporal_filters(X_spatial, y)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply spatio-temporal filtering and extract features.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_trials, n_channels, n_samples)

        Returns
        -------
        features : np.ndarray
            Spatio-temporal features
        """
        if self.spatial_filters_ is None:
            raise RuntimeError("Not fitted.")

        # Apply spatial filters
        X_spatial = np.array([self.spatial_filters_ @ trial for trial in X])

        # Apply temporal filters and extract features
        features_list = []
        for trial in X_spatial:
            trial_features = []
            for filt in self.temporal_filters_:
                filtered = signal.filtfilt(filt['b'], filt['a'], trial, axis=1)
                # Extract variance as feature
                var = np.var(filtered, axis=1)
                trial_features.extend(var)
            features_list.append(trial_features)

        return np.array(features_list)

    def _learn_temporal_filters(self, X: np.ndarray, y: np.ndarray) -> List[Dict]:
        """Learn temporal filters (simplified implementation)."""
        # Use predefined frequency bands for temporal filtering
        bands = [
            (8, 13),   # Alpha
            (13, 30),  # Beta
            (4, 8),    # Theta
        ]

        filters = []
        for low, high in bands[:self.n_temporal_filters]:
            sos = signal.butter(4, [low, high], btype='band',
                               fs=250, output='sos')
            b, a = signal.butter(4, [low, high], btype='band', fs=250)
            filters.append({'b': b, 'a': a, 'sos': sos})

        return filters
