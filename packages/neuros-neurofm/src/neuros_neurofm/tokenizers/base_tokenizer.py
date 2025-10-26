"""
Base tokenizer classes and data structures for NeuroFMx.

Provides the foundational TokenizedSequence dataclass and base tokenizer
interface for all neural data modalities.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union, List
import torch
import torch.nn as nn


@dataclass
class TokenizedSequence:
    """
    Unified representation for tokenized neural data across modalities.

    This dataclass provides a common interface for temporal neural signals,
    enabling multi-modal alignment and synchronization.

    Attributes:
        tokens: Tokenized representation of shape (B, T, D) where:
            - B is batch size
            - T is temporal sequence length
            - D is embedding dimension
        t0: Start time of the sequence in seconds (absolute or relative)
        dt: Sampling interval in seconds (time between consecutive tokens)
        mask: Boolean mask of shape (B, T) indicating valid positions
            - True/1 = valid token
            - False/0 = padding or invalid
        metadata: Dictionary containing modality-specific information such as:
            - 'modality': str (e.g., 'eeg', 'spike', 'video', 'fmri')
            - 'sampling_rate': float (original sampling rate in Hz)
            - 'channels': int or List[str] (channel count or names)
            - 'subject_id': str
            - 'session_id': str
            - 'task': str
            - Any other modality-specific fields

    Properties:
        duration: Total duration of the sequence in seconds
        end_time: End time of the sequence (t0 + duration)
        sampling_rate: Effective sampling rate in Hz (1/dt)
        timestamps: Array of timestamps for each token
        valid_tokens: Number of valid (non-masked) tokens per batch element

    Examples:
        >>> # EEG data: 128 Hz, 2 seconds, 64 channels
        >>> eeg_seq = TokenizedSequence(
        ...     tokens=torch.randn(4, 256, 512),  # 4 batch, 256 tokens, 512-dim
        ...     t0=0.0,
        ...     dt=1/128.0,  # 128 Hz
        ...     mask=torch.ones(4, 256, dtype=torch.bool),
        ...     metadata={'modality': 'eeg', 'sampling_rate': 128.0, 'channels': 64}
        ... )

        >>> # Spike trains: variable rate events
        >>> spike_seq = TokenizedSequence(
        ...     tokens=torch.randn(4, 1000, 512),  # 1000 spike events
        ...     t0=0.0,
        ...     dt=0.001,  # 1ms resolution
        ...     mask=torch.ones(4, 1000, dtype=torch.bool),
        ...     metadata={'modality': 'spike', 'n_units': 96}
        ... )

        >>> # Video: 30 fps
        >>> video_seq = TokenizedSequence(
        ...     tokens=torch.randn(4, 60, 512),  # 2 seconds at 30fps
        ...     t0=0.0,
        ...     dt=1/30.0,  # 30 fps
        ...     mask=torch.ones(4, 60, dtype=torch.bool),
        ...     metadata={'modality': 'video', 'fps': 30, 'resolution': (224, 224)}
        ... )
    """

    tokens: torch.Tensor  # (B, T, D)
    t0: float  # Start time in seconds
    dt: float  # Sampling interval in seconds
    mask: torch.Tensor  # (B, T) boolean mask
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the TokenizedSequence after initialization."""
        # Validate tokens shape
        if self.tokens.dim() != 3:
            raise ValueError(
                f"tokens must be 3D (B, T, D), got shape {self.tokens.shape}"
            )

        batch_size, seq_len, _ = self.tokens.shape

        # Validate mask shape
        if self.mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"mask shape {self.mask.shape} doesn't match tokens "
                f"(batch={batch_size}, seq_len={seq_len})"
            )

        # Validate temporal parameters
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")

        # Ensure mask is boolean
        if self.mask.dtype not in [torch.bool, torch.uint8]:
            self.mask = self.mask.bool()

    @property
    def batch_size(self) -> int:
        """Batch size."""
        return self.tokens.shape[0]

    @property
    def seq_len(self) -> int:
        """Sequence length (number of tokens)."""
        return self.tokens.shape[1]

    @property
    def d_model(self) -> int:
        """Embedding dimension."""
        return self.tokens.shape[2]

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return self.seq_len * self.dt

    @property
    def end_time(self) -> float:
        """End time in seconds (t0 + duration)."""
        return self.t0 + self.duration

    @property
    def sampling_rate(self) -> float:
        """Effective sampling rate in Hz."""
        return 1.0 / self.dt

    @property
    def timestamps(self) -> torch.Tensor:
        """
        Generate timestamps for each token.

        Returns:
            Tensor of shape (seq_len,) with timestamps in seconds
        """
        return torch.arange(
            self.seq_len,
            dtype=torch.float32,
            device=self.tokens.device
        ) * self.dt + self.t0

    @property
    def valid_tokens(self) -> torch.Tensor:
        """
        Count valid (non-masked) tokens per batch element.

        Returns:
            Tensor of shape (batch_size,) with counts
        """
        return self.mask.sum(dim=1)

    def to(self, device: Union[str, torch.device]) -> 'TokenizedSequence':
        """
        Move all tensors to the specified device.

        Args:
            device: Target device

        Returns:
            New TokenizedSequence on the target device
        """
        return TokenizedSequence(
            tokens=self.tokens.to(device),
            t0=self.t0,
            dt=self.dt,
            mask=self.mask.to(device),
            metadata=self.metadata.copy()
        )

    def clone(self) -> 'TokenizedSequence':
        """
        Create a deep copy of this TokenizedSequence.

        Returns:
            Cloned TokenizedSequence
        """
        return TokenizedSequence(
            tokens=self.tokens.clone(),
            t0=self.t0,
            dt=self.dt,
            mask=self.mask.clone(),
            metadata=self.metadata.copy()
        )

    def slice_time(self, start_time: float, end_time: float) -> 'TokenizedSequence':
        """
        Extract a temporal slice of the sequence.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            Sliced TokenizedSequence
        """
        # Find token indices
        timestamps = self.timestamps
        start_idx = torch.searchsorted(timestamps, start_time).item()
        end_idx = torch.searchsorted(timestamps, end_time).item()

        # Clamp to valid range
        start_idx = max(0, start_idx)
        end_idx = min(self.seq_len, end_idx)

        if start_idx >= end_idx:
            raise ValueError(
                f"Invalid time range [{start_time}, {end_time}] for sequence "
                f"[{self.t0}, {self.end_time}]"
            )

        return TokenizedSequence(
            tokens=self.tokens[:, start_idx:end_idx, :],
            t0=self.t0 + start_idx * self.dt,
            dt=self.dt,
            mask=self.mask[:, start_idx:end_idx],
            metadata=self.metadata.copy()
        )

    def __repr__(self) -> str:
        """String representation."""
        modality = self.metadata.get('modality', 'unknown')
        return (
            f"TokenizedSequence("
            f"modality={modality}, "
            f"batch={self.batch_size}, "
            f"seq_len={self.seq_len}, "
            f"d_model={self.d_model}, "
            f"t0={self.t0:.3f}s, "
            f"dt={self.dt:.4f}s, "
            f"duration={self.duration:.3f}s, "
            f"sampling_rate={self.sampling_rate:.1f}Hz"
            f")"
        )


class BaseTokenizer(nn.Module):
    """
    Base class for all neural data tokenizers.

    Provides common interface and utilities for converting raw neural
    recordings into TokenizedSequence format.

    All tokenizer implementations should:
    1. Inherit from this class
    2. Implement the forward() method
    3. Return TokenizedSequence objects
    4. Set self.d_model in __init__
    5. Set self.default_sampling_rate if applicable

    Attributes:
        d_model: Output embedding dimension
        default_sampling_rate: Default sampling rate for this modality (Hz)
    """

    def __init__(self, d_model: int):
        """
        Initialize base tokenizer.

        Args:
            d_model: Output embedding dimension
        """
        super().__init__()
        self.d_model = d_model
        self.default_sampling_rate: Optional[float] = None

    def forward(self, *args, **kwargs) -> TokenizedSequence:
        """
        Convert raw data to TokenizedSequence.

        This method should be implemented by subclasses.

        Returns:
            TokenizedSequence containing tokenized data
        """
        raise NotImplementedError(
            "Subclasses must implement forward() method"
        )

    def create_sequence(
        self,
        tokens: torch.Tensor,
        t0: float,
        dt: float,
        mask: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TokenizedSequence:
        """
        Helper method to create a TokenizedSequence.

        Args:
            tokens: Token embeddings (B, T, D)
            t0: Start time in seconds
            dt: Sampling interval in seconds
            mask: Optional boolean mask (B, T). If None, all tokens are valid
            metadata: Optional metadata dictionary

        Returns:
            TokenizedSequence
        """
        batch_size, seq_len, _ = tokens.shape

        # Create default mask if not provided
        if mask is None:
            mask = torch.ones(
                batch_size, seq_len,
                dtype=torch.bool,
                device=tokens.device
            )

        # Create default metadata if not provided
        if metadata is None:
            metadata = {}

        # Add default sampling rate to metadata if available
        if 'sampling_rate' not in metadata and self.default_sampling_rate is not None:
            metadata['sampling_rate'] = self.default_sampling_rate

        return TokenizedSequence(
            tokens=tokens,
            t0=t0,
            dt=dt,
            mask=mask,
            metadata=metadata
        )

    def get_output_dim(self) -> int:
        """
        Get the output embedding dimension.

        Returns:
            Embedding dimension
        """
        return self.d_model


# Utility functions for working with TokenizedSequences
def concatenate_sequences(
    sequences: List[TokenizedSequence],
    dim: int = 1
) -> TokenizedSequence:
    """
    Concatenate multiple TokenizedSequences along the temporal dimension.

    All sequences must have:
    - Same batch size
    - Same embedding dimension
    - Same dt (sampling interval)
    - Consecutive time ranges (within tolerance)

    Args:
        sequences: List of TokenizedSequence objects to concatenate
        dim: Dimension to concatenate along (default=1 for temporal)

    Returns:
        Concatenated TokenizedSequence

    Raises:
        ValueError: If sequences are not compatible
    """
    if not sequences:
        raise ValueError("Cannot concatenate empty sequence list")

    if len(sequences) == 1:
        return sequences[0].clone()

    # Validate compatibility
    batch_size = sequences[0].batch_size
    d_model = sequences[0].d_model
    dt = sequences[0].dt

    for i, seq in enumerate(sequences[1:], 1):
        if seq.batch_size != batch_size:
            raise ValueError(
                f"Batch size mismatch: seq[0]={batch_size}, seq[{i}]={seq.batch_size}"
            )
        if seq.d_model != d_model:
            raise ValueError(
                f"Embedding dim mismatch: seq[0]={d_model}, seq[{i}]={seq.d_model}"
            )
        if abs(seq.dt - dt) > 1e-6:
            raise ValueError(
                f"Sampling interval mismatch: seq[0]={dt}, seq[{i}]={seq.dt}"
            )

    # Concatenate tokens and masks
    tokens = torch.cat([seq.tokens for seq in sequences], dim=dim)
    masks = torch.cat([seq.mask for seq in sequences], dim=dim)

    # Use first sequence's start time
    t0 = sequences[0].t0

    # Merge metadata (prioritize first sequence)
    metadata = sequences[0].metadata.copy()

    return TokenizedSequence(
        tokens=tokens,
        t0=t0,
        dt=dt,
        mask=masks,
        metadata=metadata
    )


def batch_sequences(
    sequences: List[TokenizedSequence],
    padding_value: float = 0.0
) -> TokenizedSequence:
    """
    Batch multiple TokenizedSequences with different lengths.

    Pads shorter sequences to match the longest sequence length.
    All sequences must have the same dt and d_model.

    Args:
        sequences: List of TokenizedSequence objects
        padding_value: Value to use for padding tokens

    Returns:
        Batched TokenizedSequence
    """
    if not sequences:
        raise ValueError("Cannot batch empty sequence list")

    if len(sequences) == 1:
        return sequences[0].clone()

    # Validate compatibility
    d_model = sequences[0].d_model
    dt = sequences[0].dt

    for i, seq in enumerate(sequences[1:], 1):
        if seq.d_model != d_model:
            raise ValueError(
                f"Embedding dim mismatch: seq[0]={d_model}, seq[{i}]={seq.d_model}"
            )
        if abs(seq.dt - dt) > 1e-6:
            raise ValueError(
                f"Sampling interval mismatch: seq[0]={dt}, seq[{i}]={seq.dt}"
            )

    # Find max sequence length
    max_len = max(seq.seq_len for seq in sequences)

    # Pad sequences
    padded_tokens = []
    padded_masks = []

    for seq in sequences:
        if seq.seq_len < max_len:
            # Pad tokens
            pad_len = max_len - seq.seq_len
            padding = torch.full(
                (seq.batch_size, pad_len, d_model),
                padding_value,
                dtype=seq.tokens.dtype,
                device=seq.tokens.device
            )
            tokens = torch.cat([seq.tokens, padding], dim=1)

            # Pad mask (padded positions are False)
            mask_padding = torch.zeros(
                (seq.batch_size, pad_len),
                dtype=torch.bool,
                device=seq.mask.device
            )
            mask = torch.cat([seq.mask, mask_padding], dim=1)
        else:
            tokens = seq.tokens
            mask = seq.mask

        padded_tokens.append(tokens)
        padded_masks.append(mask)

    # Stack along batch dimension
    batched_tokens = torch.cat(padded_tokens, dim=0)
    batched_masks = torch.cat(padded_masks, dim=0)

    # Use first sequence's parameters
    t0 = sequences[0].t0
    metadata = sequences[0].metadata.copy()

    return TokenizedSequence(
        tokens=batched_tokens,
        t0=t0,
        dt=dt,
        mask=batched_masks,
        metadata=metadata
    )
