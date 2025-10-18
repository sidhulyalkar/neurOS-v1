"""
Calcium imaging tokenizer for NeuroFM-X.

Handles calcium fluorescence traces (dF/F) from two-photon imaging,
miniscopes, or fiber photometry.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CalciumTokenizer(nn.Module):
    """Tokenize calcium imaging traces.

    Processes calcium fluorescence signals (dF/F) using temporal convolutions
    and event detection.

    Parameters
    ----------
    n_neurons : int
        Number of imaged neurons/ROIs.
    d_model : int
        Output embedding dimension.
    kernel_sizes : list of int, optional
        Temporal convolution kernel sizes.
        Default: [3, 5, 7].
    n_filters : int, optional
        Number of filters per kernel.
        Default: 64.
    pool_size : int, optional
        Temporal pooling size.
        Default: 4.
    detect_events : bool, optional
        Perform calcium event detection.
        Default: True.
    event_threshold : float, optional
        Threshold for event detection (in std units).
        Default: 2.0.
    deconvolve : bool, optional
        Apply deconvolution (simplified OASIS-like).
        Default: False.
    dropout : float, optional
        Dropout probability.
        Default: 0.1.
    """

    def __init__(
        self,
        n_neurons: int,
        d_model: int,
        kernel_sizes: list = [3, 5, 7],
        n_filters: int = 64,
        pool_size: int = 4,
        detect_events: bool = True,
        event_threshold: float = 2.0,
        deconvolve: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.d_model = d_model
        self.kernel_sizes = kernel_sizes
        self.pool_size = pool_size
        self.detect_events = detect_events
        self.event_threshold = event_threshold
        self.deconvolve = deconvolve

        # Multi-scale temporal convolutions
        self.conv_blocks = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        n_neurons,
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

        # Temporal pooling
        self.pool = nn.AvgPool1d(pool_size)

        # Event detection network (if enabled)
        if detect_events:
            self.event_detector = nn.Sequential(
                nn.Conv1d(n_neurons, n_neurons, kernel_size=5, padding=2),
                nn.Sigmoid(),
            )

        # Projection to d_model
        temporal_dim = n_filters * len(kernel_sizes)
        event_dim = n_neurons if detect_events else 0
        total_dim = temporal_dim + event_dim

        self.projection = nn.Linear(total_dim, d_model)

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Positional encoding
        self.register_buffer("_positional_encoding_cache", None)

    def _detect_events(self, calcium: torch.Tensor) -> torch.Tensor:
        """Detect calcium events using thresholding.

        Parameters
        ----------
        calcium : torch.Tensor
            Calcium traces, shape (batch, n_neurons, time).

        Returns
        -------
        torch.Tensor
            Event probabilities, same shape.
        """
        if self.event_detector is not None:
            # Use learned event detector
            events = self.event_detector(calcium)
        else:
            # Simple threshold-based detection
            # Standardize
            mean = calcium.mean(dim=-1, keepdim=True)
            std = calcium.std(dim=-1, keepdim=True) + 1e-8
            z_scores = (calcium - mean) / std

            # Threshold
            events = (z_scores > self.event_threshold).float()

        return events

    def _simple_deconvolve(self, calcium: torch.Tensor) -> torch.Tensor:
        """Simplified deconvolution (OASIS-like).

        Estimates underlying spike rates from calcium traces.

        Parameters
        ----------
        calcium : torch.Tensor
            Calcium traces, shape (batch, n_neurons, time).

        Returns
        -------
        torch.Tensor
            Deconvolved spikes, same shape.
        """
        # Simplified deconvolution using first-order difference
        # Real OASIS would use constrained optimization
        diff = torch.diff(calcium, dim=-1, prepend=calcium[:, :, :1])
        spikes = F.relu(diff)  # Only keep positive changes
        return spikes

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
        calcium: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize calcium imaging data.

        Parameters
        ----------
        calcium : torch.Tensor
            Calcium fluorescence traces (dF/F),
            shape (batch, n_neurons, time_points).
        attention_mask : torch.Tensor, optional
            Mask for valid time points (1 = valid, 0 = padding),
            shape (batch, time_points).

        Returns
        -------
        tokens : torch.Tensor
            Calcium tokens, shape (batch, seq_length, d_model).
        attention_mask : torch.Tensor
            Attention mask, shape (batch, seq_length).
        """
        batch_size, n_neurons, time_points = calcium.shape

        if n_neurons != self.n_neurons:
            raise ValueError(
                f"Expected {self.n_neurons} neurons, got {n_neurons}"
            )

        # Optional: Deconvolve to get spike estimates
        if self.deconvolve:
            calcium = self._simple_deconvolve(calcium)

        # Extract temporal features using multi-scale convolutions
        temporal_features = []
        for conv_block in self.conv_blocks:
            features = conv_block(calcium)
            temporal_features.append(features)

        # Concatenate features from all scales
        temporal_features = torch.cat(temporal_features, dim=1)
        # temporal_features: (batch, n_filters * n_scales, time_points)

        # Apply pooling
        temporal_features = self.pool(temporal_features)
        seq_length = temporal_features.shape[-1]

        # Transpose to (batch, seq_length, features)
        temporal_features = temporal_features.transpose(1, 2)

        # Event detection features
        if self.detect_events:
            events = self._detect_events(calcium)
            events = self.pool(events)  # Pool to match seq_length
            events = events.transpose(1, 2)
            # events: (batch, seq_length, n_neurons)

            # Concatenate temporal and event features
            combined = torch.cat([temporal_features, events], dim=-1)
        else:
            combined = temporal_features

        # Project to d_model
        tokens = self.projection(combined)

        # Add positional encoding
        pe = self._get_positional_encoding(seq_length, tokens.device)
        tokens = tokens + pe.unsqueeze(0)

        # Normalize and dropout
        tokens = self.layer_norm(tokens)
        tokens = self.dropout(tokens)

        # Adjust attention mask for pooling
        if attention_mask is not None:
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


class TwoPhotonTokenizer(CalciumTokenizer):
    """Specialized tokenizer for two-photon calcium imaging.

    Pre-configured for typical two-photon imaging parameters.
    """

    def __init__(
        self,
        n_neurons: int,
        d_model: int,
        imaging_rate: float = 30.0,  # Hz
        **kwargs
    ):
        # Two-photon specific defaults
        kwargs.setdefault('detect_events', True)
        kwargs.setdefault('deconvolve', False)  # Can enable for spike inference
        kwargs.setdefault('event_threshold', 2.5)
        kwargs.setdefault('pool_size', 2)

        super().__init__(n_neurons, d_model, **kwargs)
        self.imaging_rate = imaging_rate


class MiniscopeTokenizer(CalciumTokenizer):
    """Specialized tokenizer for miniscope calcium imaging.

    Pre-configured for miniscope (lower SNR, higher motion artifacts).
    """

    def __init__(
        self,
        n_neurons: int,
        d_model: int,
        imaging_rate: float = 20.0,  # Hz
        **kwargs
    ):
        # Miniscope specific defaults (more aggressive processing)
        kwargs.setdefault('detect_events', True)
        kwargs.setdefault('event_threshold', 3.0)  # Higher threshold due to noise
        kwargs.setdefault('pool_size', 4)  # More pooling

        super().__init__(n_neurons, d_model, **kwargs)
        self.imaging_rate = imaging_rate
