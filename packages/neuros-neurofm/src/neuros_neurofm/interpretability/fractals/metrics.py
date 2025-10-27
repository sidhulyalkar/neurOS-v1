"""
Fractal Metrics for Neural Time Series and Graphs

GPU-accelerated implementations of fractal dimension estimators and scaling analysis.
All methods support batched computation and automatic differentiation.

References:
    - Higuchi (1988): Approach to an irregular time series on the basis of the fractal theory
    - Peng et al. (1994): Mosaic organization of DNA nucleotides
    - Mandelbrot & Van Ness (1968): Fractional Brownian motions
    - Song et al. (2005): Self-similarity of complex networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HiguchiFractalDimension:
    """
    Higuchi Fractal Dimension (HFD) for time series.

    Measures complexity/irregularity of signals via self-similarity across scales.
    FD ∈ [1, 2] where 1=simple line, 2=space-filling curve.

    Algorithm:
        1. Construct k-subseries with delay k: X_k^m = [x(m), x(m+k), x(m+2k), ...]
        2. Compute curve length L_m(k) for each subseries
        3. Average: L(k) = mean_m(L_m(k))
        4. Fit log(L(k)) vs log(1/k): slope = FD

    Args:
        k_max: Maximum delay (default: 10)
        device: torch device (default: cuda if available)

    Returns:
        Fractal dimension for each sequence [batch_size]

    Example:
        >>> hfd = HiguchiFractalDimension(k_max=10)
        >>> X = torch.randn(32, 1000)  # 32 sequences of length 1000
        >>> fd = hfd.compute(X)
        >>> print(fd.shape)  # torch.Size([32])
    """

    def __init__(self, k_max: int = 10, device: Optional[str] = None):
        self.k_max = k_max
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def compute(self, X: Tensor, k_max: Optional[int] = None) -> Tensor:
        """
        Compute Higuchi fractal dimension.

        Args:
            X: Time series [batch_size, seq_len] or [seq_len]
            k_max: Override default k_max

        Returns:
            Fractal dimensions [batch_size]
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)

        X = X.to(self.device)
        batch_size, N = X.shape
        k_max = k_max or self.k_max

        # Storage for curve lengths
        L_k = []
        k_values = torch.arange(1, k_max + 1, device=self.device, dtype=torch.float32)

        for k in range(1, k_max + 1):
            # Compute curve lengths for all m in [1, k]
            L_m_k = []
            for m in range(1, k + 1):
                # Subseries X_k^m
                indices = torch.arange(m - 1, N, k, device=self.device)
                if len(indices) < 2:
                    continue

                X_k_m = X[:, indices]  # [batch_size, len(indices)]

                # Curve length: sum of absolute differences
                diffs = torch.abs(X_k_m[:, 1:] - X_k_m[:, :-1])  # [batch_size, len-1]
                L_m = diffs.sum(dim=1)  # [batch_size]

                # Normalize by (N-1) / (floor((N-m)/k) * k)
                n_points = len(indices)
                normalization = (N - 1) / (k * (n_points - 1))
                L_m = L_m * normalization

                L_m_k.append(L_m)

            if L_m_k:
                # Average over all m
                L_k.append(torch.stack(L_m_k, dim=0).mean(dim=0))  # [batch_size]

        # Stack all L(k) values
        L_k = torch.stack(L_k, dim=1)  # [batch_size, k_max]

        # Fit log(L(k)) vs log(1/k) to get slope (fractal dimension)
        log_L = torch.log(L_k + 1e-10)  # Add epsilon for stability
        log_inv_k = torch.log(1.0 / k_values).unsqueeze(0)  # [1, k_max]

        # Linear regression: slope = cov(x,y) / var(x)
        mean_log_inv_k = log_inv_k.mean(dim=1, keepdim=True)
        mean_log_L = log_L.mean(dim=1, keepdim=True)

        cov = ((log_inv_k - mean_log_inv_k) * (log_L - mean_log_L)).sum(dim=1)
        var = ((log_inv_k - mean_log_inv_k) ** 2).sum(dim=1)

        fd = cov / (var + 1e-10)

        return fd


class DetrendedFluctuationAnalysis:
    """
    Detrended Fluctuation Analysis (DFA) for scaling exponent estimation.

    Quantifies long-range correlations in non-stationary time series.

    Scaling exponent α interpretation:
        - α = 0.5: Uncorrelated (white noise)
        - α < 0.5: Anti-correlated (anti-persistent)
        - α > 0.5: Correlated (persistent)
        - α = 1.0: 1/f noise (pink noise)
        - α = 1.5: Brownian noise (Brownian motion)

    Args:
        min_window: Minimum window size (default: 10)
        max_window: Maximum window size (default: None, auto-set to N//4)
        n_windows: Number of window sizes to test (default: 20)

    Returns:
        alpha: DFA scaling exponent [batch_size]
        fluctuations: Fluctuation function F(n) for all window sizes [batch_size, n_windows]

    Example:
        >>> dfa = DetrendedFluctuationAnalysis(min_window=10, max_window=100)
        >>> X = torch.randn(16, 2000)
        >>> alpha, F_n = dfa.compute(X)
        >>> print(alpha)  # Scaling exponent for each sequence
    """

    def __init__(
        self,
        min_window: int = 10,
        max_window: Optional[int] = None,
        n_windows: int = 20,
        device: Optional[str] = None,
    ):
        self.min_window = min_window
        self.max_window = max_window
        self.n_windows = n_windows
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def compute(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Compute DFA scaling exponent.

        Args:
            X: Time series [batch_size, seq_len]

        Returns:
            alpha: Scaling exponents [batch_size]
            F_n: Fluctuation function [batch_size, n_windows]
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)

        X = X.to(self.device)
        batch_size, N = X.shape

        # Integrate the signal (cumulative sum)
        Y = torch.cumsum(X - X.mean(dim=1, keepdim=True), dim=1)

        # Window sizes (log-spaced)
        max_window = self.max_window or (N // 4)
        window_sizes = torch.logspace(
            np.log10(self.min_window),
            np.log10(max_window),
            self.n_windows,
            device=self.device,
        ).long()

        F_n = []

        for n in window_sizes:
            n = n.item()
            # Number of segments
            n_segments = N // n

            if n_segments < 1:
                F_n.append(torch.zeros(batch_size, device=self.device))
                continue

            # Reshape into segments
            Y_segments = Y[:, :n_segments * n].reshape(batch_size, n_segments, n)

            # Fit polynomial trend (degree 1) to each segment
            # Create time index
            t = torch.arange(n, device=self.device, dtype=torch.float32)

            # Polynomial fitting: y = a + b*t
            # Solve normal equations
            t_mean = t.mean()
            Y_mean = Y_segments.mean(dim=2, keepdim=True)

            # b = cov(t, y) / var(t)
            cov = ((t - t_mean) * (Y_segments - Y_mean)).sum(dim=2, keepdim=True)
            var_t = ((t - t_mean) ** 2).sum()
            b = cov / var_t

            # a = mean(y) - b * mean(t)
            a = Y_mean - b * t_mean

            # Fitted trend
            trend = a + b * t.unsqueeze(0).unsqueeze(0)

            # Detrended signal
            detrended = Y_segments - trend

            # Fluctuation: RMS of detrended signal
            fluctuation = torch.sqrt((detrended ** 2).mean(dim=2))  # [batch, n_segments]

            # Average over segments
            F = fluctuation.mean(dim=1)  # [batch]
            F_n.append(F)

        F_n = torch.stack(F_n, dim=1)  # [batch, n_windows]

        # Fit log(F(n)) vs log(n) to get scaling exponent α
        log_F = torch.log(F_n + 1e-10)
        log_n = torch.log(window_sizes.float()).unsqueeze(0)

        # Linear regression
        mean_log_n = log_n.mean(dim=1, keepdim=True)
        mean_log_F = log_F.mean(dim=1, keepdim=True)

        cov = ((log_n - mean_log_n) * (log_F - mean_log_F)).sum(dim=1)
        var = ((log_n - mean_log_n) ** 2).sum(dim=1)

        alpha = cov / (var + 1e-10)

        return alpha, F_n


class HurstExponent:
    """
    Hurst exponent estimation via multiple methods.

    Characterizes long-term memory and self-similarity.

    H interpretation:
        - H = 0.5: Random walk (no correlation)
        - H < 0.5: Anti-persistent (mean-reverting)
        - H > 0.5: Persistent (trending)

    Relation to fractal dimension: FD = 2 - H (for time series)
    Relation to DFA: H ≈ α (DFA exponent)

    Methods:
        - 'rs': Rescaled Range (R/S) analysis (Hurst's original method)
        - 'dfa': Detrended Fluctuation Analysis
        - 'wavelets': Wavelet-based estimation

    Example:
        >>> hurst = HurstExponent(method='rs')
        >>> X = torch.randn(8, 5000)
        >>> H = hurst.compute(X)
    """

    def __init__(self, method: str = 'rs', device: Optional[str] = None):
        self.method = method
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if method == 'dfa':
            self.dfa = DetrendedFluctuationAnalysis(device=device)

    def compute(self, X: Tensor) -> Tensor:
        """
        Compute Hurst exponent.

        Args:
            X: Time series [batch_size, seq_len]

        Returns:
            H: Hurst exponents [batch_size]
        """
        if self.method == 'rs':
            return self._rs_method(X)
        elif self.method == 'dfa':
            alpha, _ = self.dfa.compute(X)
            return alpha  # H ≈ α for DFA
        elif self.method == 'wavelets':
            return self._wavelet_method(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _rs_method(self, X: Tensor) -> Tensor:
        """Rescaled Range (R/S) analysis"""
        if X.dim() == 1:
            X = X.unsqueeze(0)

        X = X.to(self.device)
        batch_size, N = X.shape

        # Divide into segments of varying lengths
        min_len = 10
        segment_lengths = torch.logspace(
            np.log10(min_len),
            np.log10(N // 2),
            20,
            device=self.device,
        ).long().unique()

        RS_values = []

        for seg_len in segment_lengths:
            n_segments = N // seg_len
            if n_segments < 1:
                continue

            # Reshape into segments
            segments = X[:, :n_segments * seg_len].reshape(batch_size, n_segments, seg_len)

            # Mean-adjusted cumulative sum
            mean_seg = segments.mean(dim=2, keepdim=True)
            Y = torch.cumsum(segments - mean_seg, dim=2)

            # Range
            R = Y.max(dim=2)[0] - Y.min(dim=2)[0]  # [batch, n_segments]

            # Standard deviation
            S = segments.std(dim=2)  # [batch, n_segments]

            # Rescaled range
            RS = R / (S + 1e-10)

            # Average over segments
            RS_avg = RS.mean(dim=1)  # [batch]
            RS_values.append(RS_avg)

        RS_values = torch.stack(RS_values, dim=1)  # [batch, n_lengths]

        # Fit log(R/S) vs log(n): slope = H
        log_RS = torch.log(RS_values + 1e-10)
        log_n = torch.log(segment_lengths.float()).unsqueeze(0)

        # Linear regression
        mean_log_n = log_n.mean(dim=1, keepdim=True)
        mean_log_RS = log_RS.mean(dim=1, keepdim=True)

        cov = ((log_n - mean_log_n) * (log_RS - mean_log_RS)).sum(dim=1)
        var = ((log_n - mean_log_n) ** 2).sum(dim=1)

        H = cov / (var + 1e-10)

        return H

    def _wavelet_method(self, X: Tensor) -> Tensor:
        """Wavelet-based Hurst estimation"""
        # Simplified wavelet-based method using Haar wavelets
        if X.dim() == 1:
            X = X.unsqueeze(0)

        X = X.to(self.device)
        batch_size, N = X.shape

        # Multi-scale wavelet variance
        scales = []
        variances = []

        for j in range(1, int(np.log2(N // 10))):
            scale = 2 ** j
            # Downsample
            downsampled = F.avg_pool1d(
                X.unsqueeze(1),
                kernel_size=scale,
                stride=scale,
            ).squeeze(1)

            # Wavelet coefficients (differences)
            coeffs = downsampled[:, 1:] - downsampled[:, :-1]
            var = coeffs.var(dim=1)

            scales.append(scale)
            variances.append(var)

        scales = torch.tensor(scales, device=self.device, dtype=torch.float32)
        variances = torch.stack(variances, dim=1)  # [batch, n_scales]

        # Fit log(var) vs log(scale): slope = 2H
        log_var = torch.log(variances + 1e-10)
        log_scale = torch.log(scales).unsqueeze(0)

        mean_log_scale = log_scale.mean(dim=1, keepdim=True)
        mean_log_var = log_var.mean(dim=1, keepdim=True)

        cov = ((log_scale - mean_log_scale) * (log_var - mean_log_var)).sum(dim=1)
        var_scale = ((log_scale - mean_log_scale) ** 2).sum(dim=1)

        slope = cov / (var_scale + 1e-10)
        H = slope / 2.0  # Since variance scales as scale^(2H)

        return H


class SpectralSlope:
    """
    Power spectral density (PSD) slope estimation for 1/f^β scaling.

    Biological neural signals often exhibit 1/f^β scaling (scale-free dynamics):
        - β ≈ 0: White noise
        - β ≈ 1: Pink noise (1/f, prevalent in brain)
        - β ≈ 2: Brown noise (Brownian motion)

    Args:
        freq_range: Frequency range for fitting [f_min, f_max] (Hz)
        sampling_rate: Sampling rate (Hz)
        nperseg: FFT segment length (default: 256)

    Returns:
        beta: Spectral slope [batch_size]
        freqs: Frequency vector
        psd: Power spectral density [batch_size, n_freqs]

    Example:
        >>> slope_est = SpectralSlope(freq_range=(1.0, 50.0), sampling_rate=1000)
        >>> X = torch.randn(16, 10000)
        >>> beta, freqs, psd = slope_est.compute(X)
    """

    def __init__(
        self,
        freq_range: Tuple[float, float] = (1.0, 100.0),
        sampling_rate: float = 1000.0,
        nperseg: int = 256,
        device: Optional[str] = None,
    ):
        self.freq_range = freq_range
        self.sampling_rate = sampling_rate
        self.nperseg = nperseg
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def compute(self, X: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute spectral slope.

        Args:
            X: Time series [batch_size, seq_len]

        Returns:
            beta: Spectral slopes [batch_size]
            freqs: Frequency vector
            psd: Power spectral density [batch_size, n_freqs]
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)

        X = X.to(self.device)
        batch_size, N = X.shape

        # Welch's method for PSD estimation
        # Use overlapping segments
        noverlap = self.nperseg // 2
        n_segments = (N - noverlap) // (self.nperseg - noverlap)

        psds = []
        for i in range(n_segments):
            start = i * (self.nperseg - noverlap)
            segment = X[:, start:start + self.nperseg]

            # Apply Hann window
            window = torch.hann_window(self.nperseg, device=self.device)
            segment = segment * window.unsqueeze(0)

            # FFT
            fft = torch.fft.rfft(segment, dim=1)
            psd_segment = (fft.abs() ** 2) / self.nperseg
            psds.append(psd_segment)

        # Average across segments
        psd = torch.stack(psds, dim=0).mean(dim=0)  # [batch, n_freqs]

        # Frequency vector
        freqs = torch.fft.rfftfreq(self.nperseg, d=1.0 / self.sampling_rate)
        freqs = freqs.to(self.device)

        # Select frequency range
        freq_mask = (freqs >= self.freq_range[0]) & (freqs <= self.freq_range[1])
        freqs_fit = freqs[freq_mask]
        psd_fit = psd[:, freq_mask]

        # Fit log(PSD) vs log(freq): slope = -β
        log_psd = torch.log(psd_fit + 1e-10)
        log_freq = torch.log(freqs_fit + 1e-10).unsqueeze(0)

        # Linear regression
        mean_log_freq = log_freq.mean(dim=1, keepdim=True)
        mean_log_psd = log_psd.mean(dim=1, keepdim=True)

        cov = ((log_freq - mean_log_freq) * (log_psd - mean_log_psd)).sum(dim=1)
        var = ((log_freq - mean_log_freq) ** 2).sum(dim=1)

        slope = cov / (var + 1e-10)
        beta = -slope  # PSD ~ 1/f^β, so slope = -β

        return beta, freqs, psd


class GraphFractalDimension:
    """
    Graph fractal dimension via box-covering algorithm.

    Measures self-similarity in network structure. Applicable to:
        - Attention weight matrices (as adjacency matrices)
        - Functional connectivity graphs
        - Structural connectomes

    Algorithm: Song-Havlin-Makse box-covering
        1. Cover graph with minimal boxes of size l_B
        2. Count number of boxes N_B(l_B)
        3. Fractal dimension: d_B = -d(log N_B)/d(log l_B)

    Args:
        min_box: Minimum box size (default: 2)
        max_box: Maximum box size (default: 10)
        threshold: Edge weight threshold for binarization (default: 0.1)

    Returns:
        Fractal dimension [batch_size]

    Example:
        >>> graph_fd = GraphFractalDimension(min_box=2, max_box=10)
        >>> # Attention weights as adjacency matrix
        >>> attn = torch.softmax(torch.randn(8, 100, 100), dim=-1)
        >>> fd = graph_fd.compute(attn)
    """

    def __init__(
        self,
        min_box: int = 2,
        max_box: int = 10,
        threshold: float = 0.1,
        device: Optional[str] = None,
    ):
        self.min_box = min_box
        self.max_box = max_box
        self.threshold = threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def compute(self, adj_matrix: Tensor) -> Tensor:
        """
        Compute graph fractal dimension.

        Args:
            adj_matrix: Adjacency matrix [batch_size, n_nodes, n_nodes]
                       or [n_nodes, n_nodes]

        Returns:
            Fractal dimensions [batch_size]
        """
        if adj_matrix.dim() == 2:
            adj_matrix = adj_matrix.unsqueeze(0)

        adj_matrix = adj_matrix.to(self.device)
        batch_size, n_nodes, _ = adj_matrix.shape

        # Binarize adjacency matrix
        adj_binary = (adj_matrix > self.threshold).float()

        box_sizes = torch.arange(self.min_box, self.max_box + 1, device=self.device)
        n_boxes_list = []

        for l_B in box_sizes:
            # Greedy box-covering algorithm
            n_boxes = self._box_covering(adj_binary, l_B)
            n_boxes_list.append(n_boxes)

        n_boxes_list = torch.stack(n_boxes_list, dim=1)  # [batch, n_box_sizes]

        # Fit log(N_B) vs log(l_B): slope = -d_B
        log_n_boxes = torch.log(n_boxes_list.float() + 1e-10)
        log_box_sizes = torch.log(box_sizes.float()).unsqueeze(0)

        # Linear regression
        mean_log_l = log_box_sizes.mean(dim=1, keepdim=True)
        mean_log_n = log_n_boxes.mean(dim=1, keepdim=True)

        cov = ((log_box_sizes - mean_log_l) * (log_n_boxes - mean_log_n)).sum(dim=1)
        var = ((log_box_sizes - mean_log_l) ** 2).sum(dim=1)

        slope = cov / (var + 1e-10)
        d_B = -slope

        return d_B

    def _box_covering(self, adj: Tensor, l_B: int) -> Tensor:
        """
        Greedy box-covering algorithm.

        Args:
            adj: Binary adjacency matrices [batch, n, n]
            l_B: Box size (diameter)

        Returns:
            Number of boxes [batch]
        """
        batch_size, n_nodes, _ = adj.shape

        n_boxes = torch.zeros(batch_size, device=self.device, dtype=torch.long)

        for b in range(batch_size):
            # Compute shortest path distances (BFS-like, simplified)
            # For efficiency, use hop-distance instead of true shortest path
            dist = self._hop_distance(adj[b])

            uncovered = torch.ones(n_nodes, device=self.device, dtype=torch.bool)
            boxes = 0

            while uncovered.any():
                # Select uncovered node with most uncovered neighbors within l_B
                uncovered_indices = uncovered.nonzero(as_tuple=True)[0]

                # For each uncovered node, count uncovered neighbors within l_B
                scores = torch.zeros(len(uncovered_indices), device=self.device)
                for i, node in enumerate(uncovered_indices):
                    # Nodes within distance l_B from node
                    within_box = (dist[node] < l_B) & uncovered
                    scores[i] = within_box.sum()

                # Select node with max score
                best_idx = scores.argmax()
                center = uncovered_indices[best_idx]

                # Cover all nodes within l_B from center
                covered_by_box = (dist[center] < l_B)
                uncovered = uncovered & (~covered_by_box)
                boxes += 1

            n_boxes[b] = boxes

        return n_boxes

    def _hop_distance(self, adj: Tensor, max_hops: int = 10) -> Tensor:
        """
        Compute hop distance (shortest path length) between all nodes.

        Args:
            adj: Adjacency matrix [n, n]
            max_hops: Maximum hops to consider

        Returns:
            Distance matrix [n, n]
        """
        n = adj.size(0)
        dist = torch.full((n, n), float('inf'), device=adj.device)
        dist.fill_diagonal_(0)
        dist[adj > 0] = 1

        # Floyd-Warshall (simplified for max_hops)
        for _ in range(max_hops):
            dist = torch.min(dist, dist.unsqueeze(1) + dist.unsqueeze(0))

        return dist


class MultifractalSpectrum:
    """
    Multifractal spectrum analysis via moment method.

    Characterizes heterogeneity in scaling properties. Useful for:
        - Detecting multiscale structure in neural activity
        - Quantifying complexity beyond monofractal measures
        - Identifying criticality and phase transitions

    Computes:
        - τ(q): Mass exponent function
        - α: Hölder exponent (singularity strength)
        - f(α): Multifractal spectrum (dimension of set with exponent α)

    For monofractal: f(α) is a point
    For multifractal: f(α) is a concave spectrum

    Args:
        q_range: Range of moments to compute (default: -5 to 5)
        n_q: Number of q values (default: 21)
        box_sizes: Box sizes for partitioning (default: auto)

    Returns:
        tau: Mass exponent τ(q) [batch_size, n_q]
        alpha: Hölder exponents α [batch_size, n_q]
        f_alpha: Spectrum f(α) [batch_size, n_q]

    Example:
        >>> mf = MultifractalSpectrum(q_range=(-5, 5), n_q=21)
        >>> X = torch.randn(4, 10000)
        >>> result = mf.compute(X)
        >>> print(result['f_alpha'].shape)  # torch.Size([4, 21])
    """

    def __init__(
        self,
        q_range: Tuple[float, float] = (-5.0, 5.0),
        n_q: int = 21,
        box_sizes: Optional[List[int]] = None,
        device: Optional[str] = None,
    ):
        self.q_values = torch.linspace(q_range[0], q_range[1], n_q)
        self.box_sizes = box_sizes
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def compute(self, X: Tensor) -> Dict[str, Tensor]:
        """
        Compute multifractal spectrum.

        Args:
            X: Time series [batch_size, seq_len]

        Returns:
            Dictionary with keys:
                - 'tau': Mass exponent τ(q) [batch, n_q]
                - 'alpha': Hölder exponent α(q) [batch, n_q]
                - 'f_alpha': Multifractal spectrum f(α) [batch, n_q]
                - 'q': Moment orders [n_q]
        """
        if X.dim() == 1:
            X = X.unsqueeze(0)

        X = X.to(self.device)
        q_values = self.q_values.to(self.device)
        batch_size, N = X.shape

        # Auto-select box sizes if not provided
        if self.box_sizes is None:
            box_sizes = [2 ** i for i in range(4, int(np.log2(N // 10)))]
        else:
            box_sizes = self.box_sizes

        box_sizes = torch.tensor(box_sizes, device=self.device)

        # Compute partition function Z(q, ε) for each box size ε
        tau_q = []

        for q in q_values:
            log_Z_eps = []

            for eps in box_sizes:
                eps = eps.item()
                # Partition signal into boxes
                n_boxes = N // eps
                boxes = X[:, :n_boxes * eps].reshape(batch_size, n_boxes, eps)

                # Measure in each box (sum of absolute values)
                mu_i = boxes.abs().sum(dim=2)  # [batch, n_boxes]

                # Normalize
                mu_i = mu_i / (mu_i.sum(dim=1, keepdim=True) + 1e-10)

                # Partition function Z(q, ε) = sum_i μ_i^q
                if abs(q) < 1e-6:  # q ≈ 0
                    Z_q_eps = (mu_i > 1e-10).float().sum(dim=1)  # Count non-zero boxes
                else:
                    Z_q_eps = (mu_i ** q).sum(dim=1)

                log_Z_eps.append(torch.log(Z_q_eps + 1e-10))

            log_Z_eps = torch.stack(log_Z_eps, dim=1)  # [batch, n_box_sizes]

            # Fit log(Z) vs log(ε): slope = τ(q)
            log_eps = torch.log(box_sizes.float()).unsqueeze(0)

            mean_log_eps = log_eps.mean(dim=1, keepdim=True)
            mean_log_Z = log_Z_eps.mean(dim=1, keepdim=True)

            cov = ((log_eps - mean_log_eps) * (log_Z_eps - mean_log_Z)).sum(dim=1)
            var = ((log_eps - mean_log_eps) ** 2).sum(dim=1)

            tau = cov / (var + 1e-10)
            tau_q.append(tau)

        tau_q = torch.stack(tau_q, dim=1)  # [batch, n_q]

        # Compute α(q) and f(α) via Legendre transform
        # α(q) = dτ/dq
        # f(α) = q*α - τ

        # Numerical derivative
        dq = q_values[1] - q_values[0]
        alpha = torch.gradient(tau_q, spacing=(dq,), dim=1)[0]  # [batch, n_q]

        # f(α) = q*α - τ
        f_alpha = q_values.unsqueeze(0) * alpha - tau_q

        return {
            'tau': tau_q,
            'alpha': alpha,
            'f_alpha': f_alpha,
            'q': q_values,
        }


class FractalMetricsBundle:
    """
    Convenience class to compute all fractal metrics at once.

    Example:
        >>> bundle = FractalMetricsBundle()
        >>> X = torch.randn(16, 5000)
        >>> metrics = bundle.compute_all(X)
        >>> print(metrics.keys())
        >>> # dict_keys(['higuchi_fd', 'dfa_alpha', 'hurst', 'spectral_beta', 'multifractal'])
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.higuchi = HiguchiFractalDimension(device=self.device)
        self.dfa = DetrendedFluctuationAnalysis(device=self.device)
        self.hurst = HurstExponent(method='rs', device=self.device)
        self.spectral = SpectralSlope(device=self.device)
        self.multifractal = MultifractalSpectrum(device=self.device)

    def compute_all(self, X: Tensor, include_multifractal: bool = False) -> Dict[str, Tensor]:
        """
        Compute all fractal metrics.

        Args:
            X: Time series [batch_size, seq_len]
            include_multifractal: Whether to compute multifractal spectrum (slower)

        Returns:
            Dictionary with all metrics
        """
        results = {}

        # Higuchi FD
        results['higuchi_fd'] = self.higuchi.compute(X)

        # DFA
        results['dfa_alpha'], results['dfa_fluctuations'] = self.dfa.compute(X)

        # Hurst exponent
        results['hurst'] = self.hurst.compute(X)

        # Spectral slope
        results['spectral_beta'], results['freqs'], results['psd'] = self.spectral.compute(X)

        # Multifractal spectrum (optional, slower)
        if include_multifractal:
            results['multifractal'] = self.multifractal.compute(X)

        return results

    def to_dataframe(self, metrics: Dict[str, Tensor]) -> 'pd.DataFrame':
        """Convert metrics to pandas DataFrame for easy analysis."""
        import pandas as pd

        scalar_metrics = {
            'higuchi_fd': metrics['higuchi_fd'].cpu().numpy(),
            'dfa_alpha': metrics['dfa_alpha'].cpu().numpy(),
            'hurst': metrics['hurst'].cpu().numpy(),
            'spectral_beta': metrics['spectral_beta'].cpu().numpy(),
        }

        return pd.DataFrame(scalar_metrics)
