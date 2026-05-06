"""
Astrocyte signal tokenizer for neuroFMx.

Handles astrocyte calcium imaging, event sequences, and network state features.
Designed to capture slow-timescale glial dynamics that complement fast neural signals.
"""

import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, Dict, Any
import numpy as np

from neuros_neurofm.tokenizers.base_tokenizer import BaseTokenizer, TokenizedSequence


class AstroTokenizer(BaseTokenizer):
    """
    Tokenize astrocyte calcium imaging or derived event features.

    Astrocyte dynamics are typically slower than neuronal (1-10s vs ms timescales),
    so this tokenizer emphasizes:
    - Slower temporal filters
    - Spatial network structure
    - Event-based representations

    Args:
        n_astrocytes: Number of astrocyte regions/ROIs
        d_model: Model dimension for output tokens
        sampling_rate: Sampling rate in Hz (default: 10 Hz for typical calcium imaging)
        use_events: Whether input is event-based (True) or continuous traces (False)
        pool_size: Temporal pooling factor
        event_threshold: Threshold for event detection (if use_events=False)
        dropout: Dropout rate

    Input shapes:
        - Continuous: (batch, time, n_astrocytes) - dF/F traces
        - Event-based: (batch, n_events, event_features) - from neuros-astro

    Output:
        - tokens: (batch, seq_len, d_model)
        - TokenizedSequence with temporal metadata

    Example:
        >>> tokenizer = AstroTokenizer(n_astrocytes=100, d_model=512)
        >>> traces = torch.randn(4, 1000, 100)  # (batch, time, astrocytes)
        >>> tokens, sequence = tokenizer(traces, return_sequence=True)
        >>> print(tokens.shape)  # (4, 250, 512)
    """

    def __init__(
        self,
        n_astrocytes: int,
        d_model: int,
        sampling_rate: float = 10.0,
        use_events: bool = False,
        pool_size: int = 4,
        event_threshold: float = 2.0,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(d_model=d_model)

        self.n_astrocytes = n_astrocytes
        self.default_sampling_rate = sampling_rate
        self.use_events = use_events
        self.pool_size = pool_size
        self.event_threshold = event_threshold

        if use_events:
            # Event-based: Input is already tokenized by neuros-astro
            # Just need to project to d_model
            # Typical input: (batch, n_events, 10) for event features
            self.event_projection = nn.Sequential(
                nn.Linear(10, d_model // 2),  # Assuming 10 event features from neuros-astro
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model),
                nn.LayerNorm(d_model)
            )
        else:
            # Continuous trace processing
            # Multi-scale temporal convolutions (slower than neural)
            # Kernels: 5, 11, 21 (vs 3,7,15 for neurons) to capture slow dynamics
            self.conv_blocks = nn.ModuleList()
            kernel_sizes = [5, 11, 21]

            for kernel_size in kernel_sizes:
                block = nn.Sequential(
                    nn.Conv1d(
                        n_astrocytes,
                        d_model // len(kernel_sizes),
                        kernel_size=kernel_size,
                        padding=kernel_size // 2
                    ),
                    nn.BatchNorm1d(d_model // len(kernel_sizes)),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
                self.conv_blocks.append(block)

            # Temporal pooling (downsample slow signals)
            self.pool = nn.AvgPool1d(pool_size, stride=pool_size)

            # Event detection features (optional, learned)
            self.event_detector = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=7, padding=3),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=1)
            )

            # Final projection
            self.projection = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model)
            )

        # Positional encoding (learned, for slow timescales)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model) * 0.02)

        # Modality embedding (added in model)
        self.register_buffer('_dummy', torch.zeros(1))

    def forward(
        self,
        astro_data: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        t0: float = 0.0,
        return_sequence: bool = True
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], TokenizedSequence]:
        """
        Tokenize astrocyte data.

        Args:
            astro_data: Astrocyte signals
                - If use_events=False: (batch, time, n_astrocytes) continuous traces
                - If use_events=True: (batch, n_events, event_features) event tokens
            mask: Optional attention mask (batch, time)
            t0: Start time in seconds
            return_sequence: Whether to return TokenizedSequence

        Returns:
            If return_sequence=True:
                - tokens: (batch, seq_len, d_model)
                - sequence: TokenizedSequence object
            Else:
                - tokens: (batch, seq_len, d_model)
                - mask: (batch, seq_len)
        """
        batch_size = astro_data.shape[0]

        if self.use_events:
            # Event-based processing
            # Input: (batch, n_events, event_features)
            tokens = self.event_projection(astro_data)  # → (batch, n_events, d_model)
            seq_len = tokens.shape[1]

            # For events, timestamps are irregular (from neuros-astro)
            # Use mean sampling rate for dt calculation
            dt = 1.0 / self.default_sampling_rate

        else:
            # Continuous trace processing
            # Input: (batch, time, n_astrocytes)
            x = astro_data.transpose(1, 2)  # → (batch, n_astrocytes, time)

            # Multi-scale convolutions
            conv_outputs = []
            for conv_block in self.conv_blocks:
                conv_out = conv_block(x)  # → (batch, d_model/3, time)
                conv_outputs.append(conv_out)

            # Concatenate multi-scale features
            x = torch.cat(conv_outputs, dim=1)  # → (batch, d_model, time)

            # Temporal pooling
            x = self.pool(x)  # → (batch, d_model, time/pool_size)

            # Event detection features
            x = x + self.event_detector(x)

            # Transpose back
            x = x.transpose(1, 2)  # → (batch, seq_len, d_model)

            # Projection
            tokens = self.projection(x)  # → (batch, seq_len, d_model)

            seq_len = tokens.shape[1]
            dt = (self.pool_size / self.default_sampling_rate)

        # Add positional encoding
        if seq_len <= self.pos_encoding.shape[1]:
            tokens = tokens + self.pos_encoding[:, :seq_len, :]
        else:
            # Interpolate if sequence is longer
            pos_enc = torch.nn.functional.interpolate(
                self.pos_encoding.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            tokens = tokens + pos_enc

        # Create or update mask
        if mask is None:
            mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=tokens.device)
        elif mask.shape[1] != seq_len:
            # Downsample mask to match pooled sequence
            mask = mask[:, ::self.pool_size] if not self.use_events else mask

        if return_sequence:
            sequence = self.create_sequence(
                tokens=tokens,
                t0=t0,
                dt=dt,
                mask=mask,
                metadata={
                    'modality': 'astro',
                    'n_astrocytes': self.n_astrocytes,
                    'sampling_rate': self.default_sampling_rate,
                    'use_events': self.use_events,
                    'pool_size': self.pool_size,
                    'event_threshold': self.event_threshold,
                }
            )
            return tokens, sequence

        return tokens, mask

    @torch.no_grad()
    def from_neuros_astro_tokens(
        self,
        event_tokens: np.ndarray,
        timestamps: np.ndarray,
        max_events: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert neuros-astro event tokens to torch tensors.

        Args:
            event_tokens: Event feature array from neuros-astro (n_events, n_features)
            timestamps: Event timestamps in seconds (n_events,)
            max_events: Maximum events to use (for memory efficiency)

        Returns:
            - event_tensor: (1, n_events, n_features) torch tensor
            - timestamp_tensor: (1, n_events) torch tensor

        Example:
            >>> # After running neuros-astro pipeline
            >>> tokens_npz = np.load('astro_tokens.npz')
            >>> event_tokens = tokens_npz['tokens']  # (n_events, 10)
            >>> timestamps = tokens_npz['timestamps_s']  # (n_events,)
            >>> event_tensor, ts_tensor = tokenizer.from_neuros_astro_tokens(
            ...     event_tokens, timestamps
            ... )
            >>> output, sequence = tokenizer(event_tensor, return_sequence=True)
        """
        if max_events is not None and len(event_tokens) > max_events:
            # Sample evenly or take first max_events
            indices = np.linspace(0, len(event_tokens)-1, max_events, dtype=int)
            event_tokens = event_tokens[indices]
            timestamps = timestamps[indices]

        # Convert to torch tensors
        event_tensor = torch.tensor(event_tokens, dtype=torch.float32).unsqueeze(0)  # (1, n_events, features)
        timestamp_tensor = torch.tensor(timestamps, dtype=torch.float32).unsqueeze(0)  # (1, n_events)

        return event_tensor, timestamp_tensor


class AstroEventTokenizer(AstroTokenizer):
    """
    Convenience class for event-based astrocyte tokenization.
    Pre-configured for neuros-astro event outputs.
    """

    def __init__(
        self,
        n_astrocytes: int,
        d_model: int,
        sampling_rate: float = 10.0,
        **kwargs
    ):
        super().__init__(
            n_astrocytes=n_astrocytes,
            d_model=d_model,
            sampling_rate=sampling_rate,
            use_events=True,  # Always use event-based mode
            **kwargs
        )


class AstroContinuousTokenizer(AstroTokenizer):
    """
    Convenience class for continuous astrocyte trace tokenization.
    Pre-configured for continuous calcium imaging.
    """

    def __init__(
        self,
        n_astrocytes: int,
        d_model: int,
        sampling_rate: float = 10.0,
        pool_size: int = 4,
        **kwargs
    ):
        super().__init__(
            n_astrocytes=n_astrocytes,
            d_model=d_model,
            sampling_rate=sampling_rate,
            use_events=False,  # Continuous mode
            pool_size=pool_size,
            **kwargs
        )
