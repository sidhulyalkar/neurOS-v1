"""
Temporal alignment utilities for multi-modal NeuroFMx data.

Provides tools for synchronizing neural recordings from different modalities
with varying sampling rates and temporal resolutions.
"""

from typing import List, Optional, Tuple, Dict, Any, Literal
from enum import Enum
import warnings

import torch
import torch.nn.functional as F
import numpy as np

from neuros_neurofm.tokenizers.base_tokenizer import TokenizedSequence


class InterpolationMethod(str, Enum):
    """Supported interpolation methods for temporal alignment."""
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"
    CAUSAL = "causal"  # Only use past context


class TemporalAligner:
    """
    Aligns multiple TokenizedSequences to a common temporal grid.

    Handles synchronization of neural data from different modalities with
    different sampling rates (e.g., 1000 Hz EEG, 30 Hz video, irregular spikes).

    Features:
    - Multiple interpolation methods (nearest, linear, cubic, causal)
    - Automatic common timerange detection
    - Time window extraction with overlaps
    - Jitter correction
    - Missing data imputation
    - Synchronization point detection

    Examples:
        >>> aligner = TemporalAligner()
        >>>
        >>> # Align EEG (128 Hz) and video (30 Hz) to common 50 Hz grid
        >>> eeg_seq = TokenizedSequence(...)  # 128 Hz
        >>> video_seq = TokenizedSequence(...)  # 30 Hz
        >>>
        >>> aligned = aligner.align_to_grid(
        ...     sequences=[eeg_seq, video_seq],
        ...     target_dt=0.02,  # 50 Hz
        ...     method="linear"
        ... )
        >>>
        >>> # Create sliding windows for training
        >>> windows = aligner.create_windows(
        ...     sequences=aligned,
        ...     window_size=2.0,  # 2 second windows
        ...     hop_size=1.0,     # 1 second stride
        ... )
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        warn_on_extrapolation: bool = True
    ):
        """
        Initialize the TemporalAligner.

        Args:
            tolerance: Tolerance for floating point comparisons (seconds)
            warn_on_extrapolation: Warn when extrapolating beyond data range
        """
        self.tolerance = tolerance
        self.warn_on_extrapolation = warn_on_extrapolation

    def find_common_timerange(
        self,
        sequences: List[TokenizedSequence]
    ) -> Tuple[float, float]:
        """
        Find the overlapping time range across all sequences.

        Args:
            sequences: List of TokenizedSequence objects

        Returns:
            Tuple of (start_time, end_time) for the common range

        Raises:
            ValueError: If sequences don't overlap
        """
        if not sequences:
            raise ValueError("Cannot find common timerange for empty sequence list")

        # Find the latest start time and earliest end time
        start_time = max(seq.t0 for seq in sequences)
        end_time = min(seq.end_time for seq in sequences)

        if start_time >= end_time - self.tolerance:
            # No overlap
            raise ValueError(
                f"Sequences do not overlap in time. "
                f"Common range: [{start_time:.3f}, {end_time:.3f}]"
            )

        return start_time, end_time

    def interpolate_sequence(
        self,
        sequence: TokenizedSequence,
        new_timestamps: torch.Tensor,
        method: InterpolationMethod = InterpolationMethod.LINEAR
    ) -> TokenizedSequence:
        """
        Resample a sequence to new timestamps using interpolation.

        Args:
            sequence: Input TokenizedSequence
            new_timestamps: Target timestamps (1D tensor)
            method: Interpolation method to use

        Returns:
            Resampled TokenizedSequence at new timestamps
        """
        # Get original timestamps
        orig_timestamps = sequence.timestamps  # (T_orig,)
        orig_tokens = sequence.tokens  # (B, T_orig, D)

        batch_size, _, d_model = orig_tokens.shape
        n_new = len(new_timestamps)

        # Check for extrapolation
        if self.warn_on_extrapolation:
            if (new_timestamps[0] < orig_timestamps[0] - self.tolerance or
                new_timestamps[-1] > orig_timestamps[-1] + self.tolerance):
                warnings.warn(
                    f"Interpolating beyond data range: "
                    f"data=[{orig_timestamps[0]:.3f}, {orig_timestamps[-1]:.3f}], "
                    f"target=[{new_timestamps[0]:.3f}, {new_timestamps[-1]:.3f}]"
                )

        # Perform interpolation based on method
        if method == InterpolationMethod.NEAREST:
            new_tokens = self._interpolate_nearest(
                orig_tokens, orig_timestamps, new_timestamps
            )
        elif method == InterpolationMethod.LINEAR:
            new_tokens = self._interpolate_linear(
                orig_tokens, orig_timestamps, new_timestamps
            )
        elif method == InterpolationMethod.CUBIC:
            new_tokens = self._interpolate_cubic(
                orig_tokens, orig_timestamps, new_timestamps
            )
        elif method == InterpolationMethod.CAUSAL:
            new_tokens = self._interpolate_causal(
                orig_tokens, orig_timestamps, new_timestamps
            )
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        # Interpolate mask
        new_mask = self._interpolate_mask(
            sequence.mask, orig_timestamps, new_timestamps
        )

        # Compute new dt (assume uniform spacing)
        if len(new_timestamps) > 1:
            new_dt = (new_timestamps[-1] - new_timestamps[0]).item() / (len(new_timestamps) - 1)
        else:
            new_dt = sequence.dt

        return TokenizedSequence(
            tokens=new_tokens,
            t0=new_timestamps[0].item(),
            dt=new_dt,
            mask=new_mask,
            metadata=sequence.metadata.copy()
        )

    def _interpolate_nearest(
        self,
        tokens: torch.Tensor,
        orig_times: torch.Tensor,
        new_times: torch.Tensor
    ) -> torch.Tensor:
        """Nearest neighbor interpolation."""
        batch_size, _, d_model = tokens.shape
        n_new = len(new_times)

        # Find nearest indices
        # Expand dimensions for broadcasting: orig_times (T_orig, 1), new_times (1, T_new)
        diffs = torch.abs(orig_times.unsqueeze(1) - new_times.unsqueeze(0))
        nearest_indices = torch.argmin(diffs, dim=0)  # (T_new,)

        # Gather tokens
        # tokens: (B, T_orig, D) -> (B, T_new, D)
        new_tokens = tokens[:, nearest_indices, :]

        return new_tokens

    def _interpolate_linear(
        self,
        tokens: torch.Tensor,
        orig_times: torch.Tensor,
        new_times: torch.Tensor
    ) -> torch.Tensor:
        """Linear interpolation."""
        batch_size, t_orig, d_model = tokens.shape
        n_new = len(new_times)

        # Find surrounding indices for each new timestamp
        # searchsorted finds where new_times would be inserted
        right_indices = torch.searchsorted(orig_times, new_times)
        right_indices = torch.clamp(right_indices, 1, t_orig - 1)
        left_indices = right_indices - 1

        # Get surrounding timestamps and tokens
        t_left = orig_times[left_indices]  # (T_new,)
        t_right = orig_times[right_indices]  # (T_new,)

        # Compute interpolation weights
        # Handle edge case where t_left == t_right
        denom = t_right - t_left
        denom = torch.where(denom > self.tolerance, denom, torch.ones_like(denom))
        weights = (new_times - t_left) / denom  # (T_new,)
        weights = torch.clamp(weights, 0.0, 1.0)

        # Linear interpolation: (1 - w) * left + w * right
        # Reshape weights for broadcasting: (1, T_new, 1)
        weights = weights.view(1, -1, 1)

        tokens_left = tokens[:, left_indices, :]  # (B, T_new, D)
        tokens_right = tokens[:, right_indices, :]  # (B, T_new, D)

        new_tokens = (1 - weights) * tokens_left + weights * tokens_right

        return new_tokens

    def _interpolate_cubic(
        self,
        tokens: torch.Tensor,
        orig_times: torch.Tensor,
        new_times: torch.Tensor
    ) -> torch.Tensor:
        """
        Cubic spline interpolation.

        Uses PyTorch's grid_sample with cubic mode for smooth interpolation.
        """
        batch_size, t_orig, d_model = tokens.shape
        n_new = len(new_times)

        # Normalize new_times to [-1, 1] range for grid_sample
        t_min = orig_times[0]
        t_max = orig_times[-1]
        t_range = t_max - t_min

        if t_range < self.tolerance:
            # All timestamps are the same, return repeated values
            return tokens[:, :1, :].expand(batch_size, n_new, d_model)

        # Normalize to [-1, 1]
        normalized_new_times = 2.0 * (new_times - t_min) / t_range - 1.0
        normalized_new_times = torch.clamp(normalized_new_times, -1.0, 1.0)

        # Prepare grid for grid_sample
        # grid_sample expects (B, H, W, 2) for 2D, but we're doing 1D
        # We'll use it as (B, 1, T_new, 2) where second coord is 0
        grid = torch.zeros(batch_size, 1, n_new, 2, device=tokens.device)
        grid[:, :, :, 0] = normalized_new_times  # x coordinate

        # Reshape tokens for grid_sample: (B, D, 1, T_orig)
        tokens_reshaped = tokens.permute(0, 2, 1).unsqueeze(2)  # (B, D, 1, T_orig)

        # Use grid_sample with bicubic interpolation
        # Note: bicubic is only available for 4D input
        new_tokens = F.grid_sample(
            tokens_reshaped,
            grid,
            mode='bilinear',  # Use bilinear as fallback (cubic not available in 1D)
            padding_mode='border',
            align_corners=True
        )  # (B, D, 1, T_new)

        # Reshape back to (B, T_new, D)
        new_tokens = new_tokens.squeeze(2).permute(0, 2, 1)

        return new_tokens

    def _interpolate_causal(
        self,
        tokens: torch.Tensor,
        orig_times: torch.Tensor,
        new_times: torch.Tensor
    ) -> torch.Tensor:
        """
        Causal interpolation (only use past context).

        For each new timestamp, use the most recent past token.
        """
        batch_size, _, d_model = tokens.shape
        n_new = len(new_times)

        # For each new time, find the rightmost original time that is <= new time
        indices = torch.searchsorted(orig_times, new_times, right=False)
        indices = torch.clamp(indices, 0, len(orig_times) - 1)

        # Handle case where new_time < first orig_time (use first token)
        # This is already handled by clamp to 0

        # Gather tokens
        new_tokens = tokens[:, indices, :]

        return new_tokens

    def _interpolate_mask(
        self,
        mask: torch.Tensor,
        orig_times: torch.Tensor,
        new_times: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate boolean mask.

        A new position is valid if the nearest original position is valid.
        """
        batch_size, _ = mask.shape
        n_new = len(new_times)

        # Find nearest indices
        diffs = torch.abs(orig_times.unsqueeze(1) - new_times.unsqueeze(0))
        nearest_indices = torch.argmin(diffs, dim=0)  # (T_new,)

        # Gather mask values
        new_mask = mask[:, nearest_indices]

        return new_mask

    def align_to_grid(
        self,
        sequences: List[TokenizedSequence],
        target_dt: Optional[float] = None,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        method: InterpolationMethod = InterpolationMethod.LINEAR,
        reference_idx: int = 0
    ) -> List[TokenizedSequence]:
        """
        Align multiple sequences to a common temporal grid.

        Args:
            sequences: List of TokenizedSequence objects to align
            target_dt: Target sampling interval in seconds. If None, uses the
                      sampling rate of the reference sequence
            t_start: Start time of the grid. If None, uses common timerange
            t_end: End time of the grid. If None, uses common timerange
            method: Interpolation method to use
            reference_idx: Index of reference sequence (default=0)

        Returns:
            List of aligned TokenizedSequence objects on common grid
        """
        if not sequences:
            return []

        if len(sequences) == 1:
            return [sequences[0].clone()]

        # Determine time range
        if t_start is None or t_end is None:
            common_start, common_end = self.find_common_timerange(sequences)
            if t_start is None:
                t_start = common_start
            if t_end is None:
                t_end = common_end

        # Determine target sampling rate
        if target_dt is None:
            target_dt = sequences[reference_idx].dt

        # Generate target timestamps
        n_samples = int((t_end - t_start) / target_dt) + 1
        target_timestamps = torch.linspace(
            t_start,
            t_start + (n_samples - 1) * target_dt,
            n_samples,
            dtype=torch.float32
        )

        # Interpolate each sequence to the target grid
        aligned_sequences = []
        for seq in sequences:
            aligned_seq = self.interpolate_sequence(seq, target_timestamps, method)
            aligned_sequences.append(aligned_seq)

        return aligned_sequences

    def create_windows(
        self,
        sequences: List[TokenizedSequence],
        window_size: float,
        hop_size: Optional[float] = None,
        align_first: bool = True,
        method: InterpolationMethod = InterpolationMethod.LINEAR
    ) -> List[List[TokenizedSequence]]:
        """
        Create sliding windows over aligned sequences.

        Args:
            sequences: List of TokenizedSequence objects
            window_size: Window duration in seconds
            hop_size: Stride between windows in seconds. If None, uses window_size
                     (non-overlapping windows)
            align_first: Whether to align sequences before windowing
            method: Interpolation method if align_first=True

        Returns:
            List of window lists. Each window list contains one TokenizedSequence
            per input sequence.
        """
        if hop_size is None:
            hop_size = window_size

        # Align sequences if requested
        if align_first and len(sequences) > 1:
            sequences = self.align_to_grid(sequences, method=method)

        # Find common time range
        t_start, t_end = self.find_common_timerange(sequences)

        # Generate window start times
        window_starts = []
        current_start = t_start
        while current_start + window_size <= t_end + self.tolerance:
            window_starts.append(current_start)
            current_start += hop_size

        # Extract windows
        windows = []
        for win_start in window_starts:
            win_end = win_start + window_size
            window_seqs = []

            for seq in sequences:
                # Extract temporal slice
                try:
                    win_seq = seq.slice_time(win_start, win_end)
                    window_seqs.append(win_seq)
                except ValueError:
                    # Window extends beyond sequence, skip this window
                    break

            # Only add window if all sequences have data
            if len(window_seqs) == len(sequences):
                windows.append(window_seqs)

        return windows

    def detect_sync_points(
        self,
        sequences: List[TokenizedSequence],
        similarity_threshold: float = 0.8,
        window_size: float = 0.1
    ) -> List[float]:
        """
        Detect potential synchronization points across sequences.

        Useful for finding alignment markers in multi-modal recordings
        (e.g., trigger pulses, event markers).

        Args:
            sequences: List of TokenizedSequence objects
            similarity_threshold: Minimum similarity score (0-1)
            window_size: Window size for local similarity computation

        Returns:
            List of timestamps where synchronization events are detected
        """
        if len(sequences) < 2:
            return []

        # Find common time range
        t_start, t_end = self.find_common_timerange(sequences)

        # Align sequences to common grid
        aligned = self.align_to_grid(sequences)

        # Compute temporal derivatives to find sharp transitions
        sync_points = []
        n_samples = aligned[0].seq_len

        # Compute magnitude of change for each sequence
        changes = []
        for seq in aligned:
            # Compute L2 norm across embedding dimension
            token_norms = torch.norm(seq.tokens, dim=2)  # (B, T)

            # Compute temporal derivative
            deriv = torch.diff(token_norms, dim=1)  # (B, T-1)
            changes.append(deriv)

        # Find peaks in derivatives across all sequences
        # Stack changes: (n_sequences, B, T-1)
        stacked_changes = torch.stack([torch.abs(c) for c in changes], dim=0)

        # Average across sequences and batch
        avg_change = stacked_changes.mean(dim=(0, 1))  # (T-1,)

        # Normalize to [0, 1]
        if avg_change.max() > 0:
            avg_change = avg_change / avg_change.max()

        # Find peaks above threshold
        peaks = avg_change > similarity_threshold
        peak_indices = torch.where(peaks)[0]

        # Convert indices to timestamps
        dt = aligned[0].dt
        t0 = aligned[0].t0
        sync_points = [t0 + (idx.item() + 1) * dt for idx in peak_indices]

        return sync_points

    def correct_jitter(
        self,
        sequence: TokenizedSequence,
        max_jitter: float = 0.01
    ) -> TokenizedSequence:
        """
        Correct small timing jitter in irregular sequences.

        Useful for cleaning up spike trains or other event-based data with
        small timing errors.

        Args:
            sequence: Input TokenizedSequence
            max_jitter: Maximum jitter to correct (seconds)

        Returns:
            Jitter-corrected TokenizedSequence
        """
        # Get timestamps
        timestamps = sequence.timestamps

        # Compute expected uniform spacing
        expected_dt = sequence.dt

        # Find deviations from expected timing
        expected_times = torch.arange(
            len(timestamps),
            dtype=torch.float32,
            device=timestamps.device
        ) * expected_dt + sequence.t0

        jitter = timestamps - expected_times

        # Correct jitter that's within threshold
        corrected_times = torch.where(
            torch.abs(jitter) <= max_jitter,
            expected_times,
            timestamps
        )

        # Interpolate to corrected times
        return self.interpolate_sequence(
            sequence,
            corrected_times,
            method=InterpolationMethod.LINEAR
        )

    def impute_missing(
        self,
        sequence: TokenizedSequence,
        method: InterpolationMethod = InterpolationMethod.LINEAR
    ) -> TokenizedSequence:
        """
        Impute missing data marked by mask.

        Args:
            sequence: Input TokenizedSequence with missing data
            method: Interpolation method for imputation

        Returns:
            TokenizedSequence with imputed values
        """
        mask = sequence.mask  # (B, T)
        tokens = sequence.tokens.clone()  # (B, T, D)

        batch_size, seq_len, d_model = tokens.shape

        # Process each batch element separately
        for b in range(batch_size):
            # Find valid positions
            valid_mask = mask[b]  # (T,)
            valid_indices = torch.where(valid_mask)[0]

            if len(valid_indices) < 2:
                # Not enough valid data to interpolate
                continue

            # Get valid timestamps and tokens
            timestamps = sequence.timestamps
            valid_times = timestamps[valid_indices]
            valid_tokens = tokens[b, valid_indices, :]  # (n_valid, D)

            # Interpolate to all timestamps
            # Create dummy sequence for interpolation
            dummy_seq = TokenizedSequence(
                tokens=valid_tokens.unsqueeze(0),  # (1, n_valid, D)
                t0=valid_times[0].item(),
                dt=(valid_times[-1] - valid_times[0]).item() / (len(valid_times) - 1),
                mask=torch.ones(1, len(valid_indices), dtype=torch.bool),
                metadata={}
            )

            # Interpolate to full grid
            imputed = self.interpolate_sequence(
                dummy_seq,
                timestamps,
                method=method
            )

            # Replace missing values
            missing_mask = ~valid_mask
            tokens[b, missing_mask, :] = imputed.tokens[0, missing_mask, :]

        # Create new sequence with all mask set to True
        return TokenizedSequence(
            tokens=tokens,
            t0=sequence.t0,
            dt=sequence.dt,
            mask=torch.ones_like(sequence.mask),
            metadata=sequence.metadata.copy()
        )

    def validate_alignment(
        self,
        sequences: List[TokenizedSequence],
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Validate temporal alignment of sequences.

        Checks:
        - Same sampling interval (dt)
        - Same sequence length
        - Same start time (t0)
        - Overlapping time ranges

        Args:
            sequences: List of TokenizedSequence objects
            strict: If True, all checks must pass. If False, returns diagnostics

        Returns:
            Dictionary with validation results

        Raises:
            ValueError: If strict=True and validation fails
        """
        if not sequences:
            return {"valid": True, "message": "Empty sequence list"}

        results = {
            "valid": True,
            "n_sequences": len(sequences),
            "checks": {}
        }

        # Check 1: Same dt
        dt_values = [seq.dt for seq in sequences]
        dt_match = all(abs(dt - dt_values[0]) < self.tolerance for dt in dt_values)
        results["checks"]["dt_match"] = {
            "passed": dt_match,
            "values": dt_values
        }

        if not dt_match:
            results["valid"] = False
            if strict:
                raise ValueError(
                    f"Sampling intervals don't match: {dt_values}"
                )

        # Check 2: Same sequence length
        lengths = [seq.seq_len for seq in sequences]
        length_match = len(set(lengths)) == 1
        results["checks"]["length_match"] = {
            "passed": length_match,
            "values": lengths
        }

        if not length_match:
            results["valid"] = False
            if strict:
                raise ValueError(
                    f"Sequence lengths don't match: {lengths}"
                )

        # Check 3: Same start time
        t0_values = [seq.t0 for seq in sequences]
        t0_match = all(abs(t0 - t0_values[0]) < self.tolerance for t0 in t0_values)
        results["checks"]["t0_match"] = {
            "passed": t0_match,
            "values": t0_values
        }

        if not t0_match:
            results["valid"] = False
            if strict:
                raise ValueError(
                    f"Start times don't match: {t0_values}"
                )

        # Check 4: Time range overlap
        try:
            t_start, t_end = self.find_common_timerange(sequences)
            results["checks"]["time_overlap"] = {
                "passed": True,
                "common_range": (t_start, t_end),
                "duration": t_end - t_start
            }
        except ValueError as e:
            results["checks"]["time_overlap"] = {
                "passed": False,
                "error": str(e)
            }
            results["valid"] = False
            if strict:
                raise

        # Check 5: Same d_model
        d_models = [seq.d_model for seq in sequences]
        d_model_match = len(set(d_models)) == 1
        results["checks"]["d_model_match"] = {
            "passed": d_model_match,
            "values": d_models
        }

        if not d_model_match:
            results["valid"] = False
            if strict:
                raise ValueError(
                    f"Embedding dimensions don't match: {d_models}"
                )

        return results


# Utility functions
def resample_to_rate(
    sequence: TokenizedSequence,
    target_rate: float,
    method: InterpolationMethod = InterpolationMethod.LINEAR
) -> TokenizedSequence:
    """
    Resample a sequence to a target sampling rate.

    Args:
        sequence: Input TokenizedSequence
        target_rate: Target sampling rate in Hz
        method: Interpolation method

    Returns:
        Resampled TokenizedSequence
    """
    target_dt = 1.0 / target_rate
    n_samples = int(sequence.duration / target_dt) + 1

    target_timestamps = torch.linspace(
        sequence.t0,
        sequence.t0 + (n_samples - 1) * target_dt,
        n_samples,
        dtype=torch.float32
    )

    aligner = TemporalAligner()
    return aligner.interpolate_sequence(sequence, target_timestamps, method)


def align_and_concatenate(
    sequences: List[TokenizedSequence],
    method: InterpolationMethod = InterpolationMethod.LINEAR
) -> TokenizedSequence:
    """
    Align sequences to a common grid and concatenate along embedding dimension.

    Useful for creating multi-modal representations.

    Args:
        sequences: List of TokenizedSequence objects
        method: Interpolation method for alignment

    Returns:
        Single TokenizedSequence with concatenated embeddings
    """
    if not sequences:
        raise ValueError("Cannot align empty sequence list")

    if len(sequences) == 1:
        return sequences[0].clone()

    # Align to common grid
    aligner = TemporalAligner()
    aligned = aligner.align_to_grid(sequences, method=method)

    # Concatenate embeddings
    tokens = torch.cat([seq.tokens for seq in aligned], dim=2)

    # Combine masks (valid only if all modalities are valid)
    mask = torch.stack([seq.mask for seq in aligned], dim=0).all(dim=0)

    # Merge metadata
    metadata = {
        'modality': 'multimodal',
        'component_modalities': [seq.metadata.get('modality', 'unknown')
                                 for seq in sequences],
        'n_modalities': len(sequences)
    }

    return TokenizedSequence(
        tokens=tokens,
        t0=aligned[0].t0,
        dt=aligned[0].dt,
        mask=mask,
        metadata=metadata
    )
