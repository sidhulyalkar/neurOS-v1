"""
LFP tokenizer for NeuroFM-X.

Converts local field potential (LFP) or EEG signals into tokens by
extracting spectral features or using convolutional encoders.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LFPTokenizer(nn.Module):
    """Tokenize LFP/EEG signals using convolutional encoder.

    Processes continuous LFP signals using 1D convolutions to extract
    temporal features at multiple scales.

    Parameters
    ----------
    n_channels : int
        Number of LFP channels.
    d_model : int
        Embedding dimension for output tokens.
    kernel_sizes : list of int, optional
        Kernel sizes for multi-scale convolutions.
        Default: [3, 5, 7].
    n_filters : int, optional
        Number of filters per kernel size.
        Default: 64.
    pool_size : int, optional
        Pooling size to reduce temporal resolution.
        Default: 4.
    use_spectral_features : bool, optional
        Extract spectral features (band powers) in addition to temporal.
        Default: True.
    freq_bands : dict, optional
        Frequency bands for spectral features.
        Default: standard EEG bands (delta, theta, alpha, beta, gamma).
    dropout : float, optional
        Dropout probability.
        Default: 0.1.
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int,
        kernel_sizes: list = [3, 5, 7],
        n_filters: int = 64,
        pool_size: int = 4,
        use_spectral_features: bool = True,
        freq_bands: Optional[dict] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.kernel_sizes = kernel_sizes
        self.n_filters = n_filters
        self.pool_size = pool_size
        self.use_spectral_features = use_spectral_features

        # Default frequency bands (in Hz)
        if freq_bands is None:
            self.freq_bands = {
                "delta": (0.5, 4.0),
                "theta": (4.0, 8.0),
                "alpha": (8.0, 13.0),
                "beta": (13.0, 30.0),
                "gamma": (30.0, 100.0),
            }
        else:
            self.freq_bands = freq_bands

        # Multi-scale convolutional encoder
        self.conv_blocks = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        n_channels,
                        n_filters,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(n_filters),
                    nn.ReLU(),
                    nn.Conv1d(
                        n_filters,
                        n_filters,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(n_filters),
                    nn.ReLU(),
                )
            )

        # Pooling layer
        self.pool = nn.AvgPool1d(pool_size)

        # Calculate total feature dimension
        temporal_dim = n_filters * len(kernel_sizes)
        spectral_dim = len(self.freq_bands) * n_channels if use_spectral_features else 0
        total_dim = temporal_dim + spectral_dim

        # Project to d_model
        self.projection = nn.Linear(total_dim, d_model)

        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Positional encoding
        self.register_buffer("_positional_encoding_cache", None)

    def _extract_band_power(
        self,
        lfp: torch.Tensor,
        fs: float,
    ) -> torch.Tensor:
        """Extract band power features from LFP.

        Parameters
        ----------
        lfp : torch.Tensor
            LFP signal, shape (batch, n_channels, time_points).
        fs : float
            Sampling frequency in Hz.

        Returns
        -------
        torch.Tensor
            Band powers, shape (batch, n_channels * n_bands, seq_length).
        """
        batch_size, n_channels, time_points = lfp.shape

        # Use FFT to compute power spectrum
        # Apply Hann window
        window = torch.hann_window(time_points, device=lfp.device)
        windowed = lfp * window.unsqueeze(0).unsqueeze(0)

        # Compute FFT
        fft = torch.fft.rfft(windowed, dim=-1)
        power = torch.abs(fft) ** 2

        # Frequency bins
        freqs = torch.fft.rfftfreq(time_points, 1.0 / fs)

        # Extract power in each band
        band_powers = []
        for band_name, (low, high) in self.freq_bands.items():
            mask = (freqs >= low) & (freqs <= high)
            band_power = power[:, :, mask].mean(dim=-1, keepdim=True)
            band_powers.append(band_power)

        # Concatenate all bands
        band_powers = torch.cat(band_powers, dim=-1)
        # band_powers: (batch, n_channels, n_bands)

        # Reshape to (batch, n_channels * n_bands)
        band_powers = band_powers.view(batch_size, -1)

        return band_powers

    def _get_positional_encoding(
        self,
        seq_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate sinusoidal positional encoding."""
        if (self._positional_encoding_cache is not None and
            self._positional_encoding_cache.shape[0] >= seq_length):
            return self._positional_encoding_cache[:seq_length]

        position = torch.arange(seq_length, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device) *
            -(torch.log(torch.tensor(10000.0)) / self.d_model)
        )

        pe = torch.zeros(seq_length, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("_positional_encoding_cache", pe)
        return pe

    def forward(
        self,
        lfp: torch.Tensor,
        fs: float = 250.0,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize LFP signals.

        Parameters
        ----------
        lfp : torch.Tensor
            LFP signal, shape (batch, n_channels, time_points).
        fs : float, optional
            Sampling frequency in Hz.
            Default: 250.0.
        attention_mask : torch.Tensor, optional
            Mask for valid time points (1 = valid, 0 = padding),
            shape (batch, time_points).

        Returns
        -------
        tokens : torch.Tensor
            LFP tokens, shape (batch, seq_length, d_model).
        attention_mask : torch.Tensor
            Attention mask, shape (batch, seq_length).
        """
        batch_size, n_channels, time_points = lfp.shape

        if n_channels != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {n_channels}"
            )

        # Extract temporal features using multi-scale convolutions
        temporal_features = []
        for conv_block in self.conv_blocks:
            features = conv_block(lfp)
            # features: (batch, n_filters, time_points)
            temporal_features.append(features)

        # Concatenate features from all scales
        temporal_features = torch.cat(temporal_features, dim=1)
        # temporal_features: (batch, n_filters * n_scales, time_points)

        # Apply pooling to reduce temporal resolution
        temporal_features = self.pool(temporal_features)
        # temporal_features: (batch, n_filters * n_scales, seq_length)

        seq_length = temporal_features.shape[-1]

        # Transpose to (batch, seq_length, n_filters * n_scales)
        temporal_features = temporal_features.transpose(1, 2)

        # Extract spectral features if enabled
        if self.use_spectral_features:
            spectral_features = self._extract_band_power(lfp, fs)
            # spectral_features: (batch, n_channels * n_bands)

            # Expand to match sequence length
            spectral_features = spectral_features.unsqueeze(1).repeat(1, seq_length, 1)
            # spectral_features: (batch, seq_length, n_channels * n_bands)

            # Concatenate temporal and spectral
            combined = torch.cat([temporal_features, spectral_features], dim=-1)
        else:
            combined = temporal_features

        # Project to d_model
        tokens = self.projection(combined)
        # tokens: (batch, seq_length, d_model)

        # Add positional encoding
        pe = self._get_positional_encoding(seq_length, tokens.device)
        tokens = tokens + pe.unsqueeze(0)

        # Normalize and dropout
        tokens = self.layer_norm(tokens)
        tokens = self.dropout(tokens)

        # Adjust attention mask for pooling
        if attention_mask is not None:
            # Pool the mask as well
            attention_mask = attention_mask.float().unsqueeze(1)
            attention_mask = F.avg_pool1d(
                attention_mask,
                kernel_size=self.pool_size,
                stride=self.pool_size,
            )
            attention_mask = (attention_mask > 0.5).squeeze(1)
        else:
            attention_mask = torch.ones(
                batch_size, seq_length,
                dtype=torch.bool,
                device=tokens.device,
            )

        return tokens, attention_mask
