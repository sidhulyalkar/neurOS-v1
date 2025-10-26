"""
Spike tokenizer for NeuroFM-X.

Converts spike trains into discrete tokens suitable for SSM processing.
Implements "spikes-as-tokens" approach where each spike event becomes a token
with unit identity, timestamp, and optional features.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from neuros_neurofm.tokenizers.base_tokenizer import BaseTokenizer, TokenizedSequence


class SpikeTokenizer(BaseTokenizer):
    """Tokenize spike trains into discrete events.

    This tokenizer converts spike trains from multiple units into a sequence of
    tokens, where each token represents a spike event with:
    - Unit identity (which neuron fired)
    - Relative timestamp (when it fired)
    - Optional waveform features

    Parameters
    ----------
    n_units : int
        Maximum number of units to handle.
    d_model : int
        Embedding dimension for output tokens.
    bin_size_ms : float, optional
        Bin size in milliseconds for discretizing timestamps.
        Default: 1.0 ms.
    max_sequence_length : int, optional
        Maximum number of spikes in a sequence.
        Default: 10000.
    use_waveform_features : bool, optional
        Whether to include waveform features in embeddings.
        Default: False.
    waveform_dim : int, optional
        Dimension of waveform features if used.
        Default: 32.
    learnable_unit_embeddings : bool, optional
        Whether unit identity embeddings are learnable.
        Default: True.
    """

    def __init__(
        self,
        n_units: int,
        d_model: int,
        bin_size_ms: float = 1.0,
        max_sequence_length: int = 10000,
        use_waveform_features: bool = False,
        waveform_dim: int = 32,
        learnable_unit_embeddings: bool = True,
    ):
        super().__init__(d_model=d_model)
        self.n_units = n_units
        self.bin_size_ms = bin_size_ms
        self.max_sequence_length = max_sequence_length
        self.use_waveform_features = use_waveform_features
        self.waveform_dim = waveform_dim
        self.default_sampling_rate = 1000.0 / bin_size_ms  # Convert ms to Hz

        # Unit identity embeddings (one per neuron)
        self.unit_embeddings = nn.Embedding(
            n_units,
            d_model,
        )
        if not learnable_unit_embeddings:
            self.unit_embeddings.weight.requires_grad = False

        # Timestamp encoding (sinusoidal positional encoding)
        self.register_buffer(
            "timestamp_freqs",
            self._get_timestamp_frequencies(d_model),
        )

        # Optional waveform feature encoder
        if use_waveform_features:
            self.waveform_encoder = nn.Sequential(
                nn.Linear(waveform_dim, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, d_model),
            )
        else:
            self.waveform_encoder = None

        # Final projection to combine unit + time + waveform
        self.output_projection = nn.Linear(
            d_model if not use_waveform_features else d_model * 2,
            d_model,
        )

    def _get_timestamp_frequencies(self, d_model: int) -> torch.Tensor:
        """Generate frequencies for sinusoidal timestamp encoding."""
        # Use frequencies spanning from 1ms to ~10s
        min_period = self.bin_size_ms
        max_period = 10000.0  # 10 seconds

        freqs = torch.exp(
            torch.linspace(
                np.log(1.0 / max_period),
                np.log(1.0 / min_period),
                d_model // 2,
            )
        )
        return freqs

    def _encode_timestamps(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode timestamps using sinusoidal features.

        Parameters
        ----------
        timestamps : torch.Tensor
            Spike timestamps in milliseconds, shape (batch, n_spikes).

        Returns
        -------
        torch.Tensor
            Timestamp encodings, shape (batch, n_spikes, d_model).
        """
        # timestamps: (batch, n_spikes)
        # freqs: (d_model // 2,)
        angles = timestamps.unsqueeze(-1) * self.timestamp_freqs.unsqueeze(0).unsqueeze(0)
        # angles: (batch, n_spikes, d_model // 2)

        # Concatenate sin and cos
        encodings = torch.cat([
            torch.sin(angles),
            torch.cos(angles),
        ], dim=-1)
        # encodings: (batch, n_spikes, d_model)

        return encodings

    def forward(
        self,
        spike_times: torch.Tensor,
        spike_units: torch.Tensor,
        spike_waveforms: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        t0: float = 0.0,
        return_sequence: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[TokenizedSequence]]:
        """Tokenize spike trains.

        Parameters
        ----------
        spike_times : torch.Tensor
            Spike timestamps in milliseconds, shape (batch, n_spikes).
        spike_units : torch.Tensor
            Unit indices for each spike, shape (batch, n_spikes).
            Values should be in [0, n_units).
        spike_waveforms : torch.Tensor, optional
            Spike waveform features, shape (batch, n_spikes, waveform_dim).
        attention_mask : torch.Tensor, optional
            Mask for valid spikes (1 = valid, 0 = padding),
            shape (batch, n_spikes).
        t0 : float, optional
            Start time in seconds (default: 0.0).
        return_sequence : bool, optional
            If True, also return TokenizedSequence (default: True).

        Returns
        -------
        tokens : torch.Tensor
            Spike tokens, shape (batch, n_spikes, d_model).
        attention_mask : torch.Tensor
            Attention mask, shape (batch, n_spikes).
        sequence : TokenizedSequence or None
            TokenizedSequence if return_sequence=True, else None.
        """
        batch_size, n_spikes = spike_times.shape

        # Get unit embeddings
        unit_embs = self.unit_embeddings(spike_units)
        # unit_embs: (batch, n_spikes, d_model)

        # Encode timestamps
        time_embs = self._encode_timestamps(spike_times)
        # time_embs: (batch, n_spikes, d_model)

        # Combine unit and time information
        combined = unit_embs + time_embs

        # Add waveform features if provided
        if self.use_waveform_features and spike_waveforms is not None:
            waveform_embs = self.waveform_encoder(spike_waveforms)
            # waveform_embs: (batch, n_spikes, d_model)
            combined = torch.cat([combined, waveform_embs], dim=-1)
            # combined: (batch, n_spikes, d_model * 2)

        # Project to final dimension
        tokens = self.output_projection(combined)
        # tokens: (batch, n_spikes, d_model)

        # Create attention mask if not provided
        if attention_mask is None:
            # Mark all spikes as valid (no padding)
            attention_mask = torch.ones(
                batch_size, n_spikes,
                dtype=torch.bool,
                device=tokens.device,
            )

        # Create TokenizedSequence if requested
        sequence = None
        if return_sequence:
            # For spike data, dt is the bin size
            dt = self.bin_size_ms / 1000.0  # Convert ms to seconds

            sequence = self.create_sequence(
                tokens=tokens,
                t0=t0,
                dt=dt,
                mask=attention_mask,
                metadata={
                    'modality': 'spike',
                    'n_units': self.n_units,
                    'bin_size_ms': self.bin_size_ms,
                    'use_waveform_features': self.use_waveform_features
                }
            )

        return tokens, attention_mask, sequence

    def from_binned_spikes(
        self,
        binned_spikes: np.ndarray,
        bin_size_ms: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert binned spike counts to spike events.

        Parameters
        ----------
        binned_spikes : np.ndarray
            Binned spike counts, shape (n_units, n_bins).
        bin_size_ms : float
            Size of each time bin in milliseconds.

        Returns
        -------
        spike_times : torch.Tensor
            Spike timestamps, shape (n_spikes,).
        spike_units : torch.Tensor
            Unit indices, shape (n_spikes,).
        attention_mask : torch.Tensor
            All ones, shape (n_spikes,).
        """
        spike_times_list = []
        spike_units_list = []

        n_units, n_bins = binned_spikes.shape

        for unit_idx in range(n_units):
            for bin_idx in range(n_bins):
                count = int(binned_spikes[unit_idx, bin_idx])
                if count > 0:
                    # Add 'count' spikes at this bin
                    for _ in range(count):
                        spike_times_list.append(bin_idx * bin_size_ms)
                        spike_units_list.append(unit_idx)

        spike_times = torch.tensor(spike_times_list, dtype=torch.float32)
        spike_units = torch.tensor(spike_units_list, dtype=torch.long)
        attention_mask = torch.ones(len(spike_times_list), dtype=torch.bool)

        return spike_times, spike_units, attention_mask
