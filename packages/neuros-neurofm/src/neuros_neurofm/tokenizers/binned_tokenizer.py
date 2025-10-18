"""
Binned tokenizer for NeuroFM-X.

Converts binned neural data (spike counts, firing rates) into continuous
tokens suitable for SSM processing. This is simpler than spike-level tokenization
and works well for high-dimensional population recordings.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class BinnedTokenizer(nn.Module):
    """Tokenize binned neural data.

    Converts binned spike counts or firing rates into token embeddings.
    Each time bin becomes a token representing the population activity
    at that moment.

    Parameters
    ----------
    n_units : int
        Number of neural units/channels.
    d_model : int
        Embedding dimension for output tokens.
    bin_size_ms : float, optional
        Size of each time bin in milliseconds.
        Default: 10.0 ms.
    use_sqrt_transform : bool, optional
        Apply sqrt transform to stabilize firing rates.
        Default: True.
    use_positional_encoding : bool, optional
        Add sinusoidal positional encodings.
        Default: True.
    dropout : float, optional
        Dropout probability.
        Default: 0.1.
    """

    def __init__(
        self,
        n_units: int,
        d_model: int,
        bin_size_ms: float = 10.0,
        use_sqrt_transform: bool = True,
        use_positional_encoding: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_units = n_units
        self.d_model = d_model
        self.bin_size_ms = bin_size_ms
        self.use_sqrt_transform = use_sqrt_transform
        self.use_positional_encoding = use_positional_encoding

        # Linear projection from n_units to d_model
        self.input_projection = nn.Linear(n_units, d_model)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Positional encoding
        if use_positional_encoding:
            # Will be initialized in forward based on sequence length
            self.register_buffer("_positional_encoding_cache", None)

    def _get_positional_encoding(
        self,
        seq_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate sinusoidal positional encoding.

        Parameters
        ----------
        seq_length : int
            Length of the sequence.
        device : torch.device
            Device to create encoding on.

        Returns
        -------
        torch.Tensor
            Positional encoding, shape (seq_length, d_model).
        """
        # Check cache
        if (self._positional_encoding_cache is not None and
            self._positional_encoding_cache.shape[0] >= seq_length):
            return self._positional_encoding_cache[:seq_length]

        # Create new encoding
        position = torch.arange(seq_length, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device) *
            -(torch.log(torch.tensor(10000.0)) / self.d_model)
        )

        pe = torch.zeros(seq_length, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Cache for future use
        self.register_buffer("_positional_encoding_cache", pe)

        return pe

    def forward(
        self,
        binned_data: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize binned neural data.

        Parameters
        ----------
        binned_data : torch.Tensor
            Binned spike counts or firing rates,
            shape (batch, seq_length, n_units).
        attention_mask : torch.Tensor, optional
            Mask for valid time bins (1 = valid, 0 = padding),
            shape (batch, seq_length).

        Returns
        -------
        tokens : torch.Tensor
            Binned tokens, shape (batch, seq_length, d_model).
        attention_mask : torch.Tensor
            Attention mask, shape (batch, seq_length).
        """
        batch_size, seq_length, n_units = binned_data.shape

        if n_units != self.n_units:
            raise ValueError(
                f"Expected {self.n_units} units, got {n_units}"
            )

        # Apply sqrt transform for variance stabilization
        if self.use_sqrt_transform:
            # sqrt(x + 0.5) for Anscombe transform
            data = torch.sqrt(binned_data + 0.5)
        else:
            data = binned_data

        # Project to d_model
        tokens = self.input_projection(data)
        # tokens: (batch, seq_length, d_model)

        # Add positional encoding
        if self.use_positional_encoding:
            pe = self._get_positional_encoding(seq_length, tokens.device)
            tokens = tokens + pe.unsqueeze(0)

        # Normalize and dropout
        tokens = self.layer_norm(tokens)
        tokens = self.dropout(tokens)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, seq_length,
                dtype=torch.bool,
                device=tokens.device,
            )

        return tokens, attention_mask


class MultiRateBinnedTokenizer(nn.Module):
    """Multi-rate binned tokenizer for hierarchical temporal modeling.

    Creates multiple token sequences at different temporal resolutions
    (e.g., 10ms, 40ms, 160ms bins) to capture both fast and slow dynamics.

    Parameters
    ----------
    n_units : int
        Number of neural units.
    d_model : int
        Embedding dimension per rate stream.
    bin_sizes_ms : list of float
        Bin sizes for each rate stream (e.g., [10, 40, 160]).
    use_sqrt_transform : bool, optional
        Apply sqrt transform to stabilize firing rates.
        Default: True.
    dropout : float, optional
        Dropout probability.
        Default: 0.1.
    """

    def __init__(
        self,
        n_units: int,
        d_model: int,
        bin_sizes_ms: list = [10.0, 40.0, 160.0],
        use_sqrt_transform: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_units = n_units
        self.d_model = d_model
        self.bin_sizes_ms = bin_sizes_ms
        self.n_rates = len(bin_sizes_ms)

        # Create a tokenizer for each rate
        self.tokenizers = nn.ModuleList([
            BinnedTokenizer(
                n_units=n_units,
                d_model=d_model,
                bin_size_ms=bin_size,
                use_sqrt_transform=use_sqrt_transform,
                use_positional_encoding=True,
                dropout=dropout,
            )
            for bin_size in bin_sizes_ms
        ])

    def forward(
        self,
        binned_data_list: list,
        attention_masks: Optional[list] = None,
    ) -> Tuple[list, list]:
        """Tokenize multi-rate binned data.

        Parameters
        ----------
        binned_data_list : list of torch.Tensor
            List of binned data at different rates.
            Each tensor has shape (batch, seq_length_i, n_units).
        attention_masks : list of torch.Tensor, optional
            List of attention masks for each rate.

        Returns
        -------
        tokens_list : list of torch.Tensor
            List of token sequences, one per rate.
        masks_list : list of torch.Tensor
            List of attention masks, one per rate.
        """
        if attention_masks is None:
            attention_masks = [None] * self.n_rates

        tokens_list = []
        masks_list = []

        for i, (data, mask) in enumerate(zip(binned_data_list, attention_masks)):
            tokens, mask_out = self.tokenizers[i](data, mask)
            tokens_list.append(tokens)
            masks_list.append(mask_out)

        return tokens_list, masks_list
