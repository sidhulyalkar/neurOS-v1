"""
Comprehensive tests for temporal alignment utilities.

Tests cover:
- TokenizedSequence creation and validation
- Multi-modal alignment with different sampling rates
- Interpolation methods (nearest, linear, cubic, causal)
- Edge cases (missing data, jitter, extrapolation)
- Window creation with overlaps
- Synchronization point detection
"""

import pytest
import torch
import numpy as np

from neuros_neurofm.tokenizers.base_tokenizer import (
    TokenizedSequence,
    concatenate_sequences,
    batch_sequences,
)
from neuros_neurofm.tokenizers.temporal_alignment import (
    TemporalAligner,
    InterpolationMethod,
    resample_to_rate,
    align_and_concatenate,
)


class TestTokenizedSequence:
    """Tests for TokenizedSequence dataclass."""

    def test_creation_valid(self):
        """Test creating a valid TokenizedSequence."""
        tokens = torch.randn(4, 100, 512)
        mask = torch.ones(4, 100, dtype=torch.bool)

        seq = TokenizedSequence(
            tokens=tokens,
            t0=0.0,
            dt=0.01,
            mask=mask,
            metadata={'modality': 'eeg'}
        )

        assert seq.batch_size == 4
        assert seq.seq_len == 100
        assert seq.d_model == 512
        assert seq.duration == pytest.approx(1.0, rel=1e-5)
        assert seq.sampling_rate == pytest.approx(100.0, rel=1e-5)

    def test_creation_invalid_shape(self):
        """Test that invalid shapes raise errors."""
        # 2D tokens (missing batch dimension)
        with pytest.raises(ValueError, match="must be 3D"):
            TokenizedSequence(
                tokens=torch.randn(100, 512),
                t0=0.0,
                dt=0.01,
                mask=torch.ones(100, dtype=torch.bool),
                metadata={}
            )

    def test_creation_invalid_mask_shape(self):
        """Test that mismatched mask shape raises error."""
        with pytest.raises(ValueError, match="doesn't match"):
            TokenizedSequence(
                tokens=torch.randn(4, 100, 512),
                t0=0.0,
                dt=0.01,
                mask=torch.ones(4, 50, dtype=torch.bool),  # Wrong length
                metadata={}
            )

    def test_creation_invalid_dt(self):
        """Test that invalid dt raises error."""
        with pytest.raises(ValueError, match="dt must be positive"):
            TokenizedSequence(
                tokens=torch.randn(4, 100, 512),
                t0=0.0,
                dt=-0.01,  # Negative!
                mask=torch.ones(4, 100, dtype=torch.bool),
                metadata={}
            )

    def test_timestamps(self):
        """Test timestamp generation."""
        seq = TokenizedSequence(
            tokens=torch.randn(2, 10, 64),
            t0=1.0,
            dt=0.1,
            mask=torch.ones(2, 10, dtype=torch.bool),
            metadata={}
        )

        timestamps = seq.timestamps
        assert len(timestamps) == 10
        assert timestamps[0] == pytest.approx(1.0)
        assert timestamps[-1] == pytest.approx(1.9)

    def test_slice_time(self):
        """Test temporal slicing."""
        seq = TokenizedSequence(
            tokens=torch.randn(2, 100, 64),
            t0=0.0,
            dt=0.01,  # 100 Hz, 1 second total
            mask=torch.ones(2, 100, dtype=torch.bool),
            metadata={'modality': 'test'}
        )

        # Slice middle section
        sliced = seq.slice_time(0.2, 0.5)

        assert sliced.t0 >= 0.2 - seq.dt
        assert sliced.end_time <= 0.5 + seq.dt
        assert sliced.metadata['modality'] == 'test'

    def test_slice_time_invalid_range(self):
        """Test that invalid time range raises error."""
        seq = TokenizedSequence(
            tokens=torch.randn(2, 100, 64),
            t0=0.0,
            dt=0.01,
            mask=torch.ones(2, 100, dtype=torch.bool),
            metadata={}
        )

        with pytest.raises(ValueError, match="Invalid time range"):
            seq.slice_time(0.5, 0.3)  # End before start

    def test_to_device(self):
        """Test moving to device."""
        seq = TokenizedSequence(
            tokens=torch.randn(2, 10, 64),
            t0=0.0,
            dt=0.01,
            mask=torch.ones(2, 10, dtype=torch.bool),
            metadata={}
        )

        # Move to CPU (should work even if already on CPU)
        seq_cpu = seq.to('cpu')
        assert seq_cpu.tokens.device.type == 'cpu'
        assert seq_cpu.mask.device.type == 'cpu'

    def test_clone(self):
        """Test cloning."""
        seq = TokenizedSequence(
            tokens=torch.randn(2, 10, 64),
            t0=0.0,
            dt=0.01,
            mask=torch.ones(2, 10, dtype=torch.bool),
            metadata={'test': 'value'}
        )

        cloned = seq.clone()

        # Should be equal but not same object
        assert torch.allclose(cloned.tokens, seq.tokens)
        assert cloned.metadata == seq.metadata
        assert cloned.tokens is not seq.tokens  # Different tensor


class TestSequenceUtilities:
    """Tests for sequence utility functions."""

    def test_concatenate_sequences(self):
        """Test concatenating sequences."""
        seq1 = TokenizedSequence(
            tokens=torch.randn(2, 10, 64),
            t0=0.0,
            dt=0.01,
            mask=torch.ones(2, 10, dtype=torch.bool),
            metadata={}
        )

        seq2 = TokenizedSequence(
            tokens=torch.randn(2, 15, 64),
            t0=0.1,
            dt=0.01,
            mask=torch.ones(2, 15, dtype=torch.bool),
            metadata={}
        )

        concatenated = concatenate_sequences([seq1, seq2])

        assert concatenated.seq_len == 25
        assert concatenated.t0 == 0.0
        assert concatenated.dt == 0.01

    def test_concatenate_incompatible_batch_size(self):
        """Test that incompatible batch sizes raise error."""
        seq1 = TokenizedSequence(
            tokens=torch.randn(2, 10, 64),
            t0=0.0,
            dt=0.01,
            mask=torch.ones(2, 10, dtype=torch.bool),
            metadata={}
        )

        seq2 = TokenizedSequence(
            tokens=torch.randn(4, 10, 64),  # Different batch size
            t0=0.1,
            dt=0.01,
            mask=torch.ones(4, 10, dtype=torch.bool),
            metadata={}
        )

        with pytest.raises(ValueError, match="Batch size mismatch"):
            concatenate_sequences([seq1, seq2])

    def test_batch_sequences(self):
        """Test batching sequences with different lengths."""
        seq1 = TokenizedSequence(
            tokens=torch.randn(1, 10, 64),
            t0=0.0,
            dt=0.01,
            mask=torch.ones(1, 10, dtype=torch.bool),
            metadata={}
        )

        seq2 = TokenizedSequence(
            tokens=torch.randn(1, 15, 64),
            t0=0.0,
            dt=0.01,
            mask=torch.ones(1, 15, dtype=torch.bool),
            metadata={}
        )

        batched = batch_sequences([seq1, seq2])

        # Should be padded to length 15
        assert batched.batch_size == 2
        assert batched.seq_len == 15

        # First sequence should have padding mask
        assert batched.mask[0, :10].all()
        assert not batched.mask[0, 10:].any()


class TestTemporalAligner:
    """Tests for TemporalAligner class."""

    def test_find_common_timerange(self):
        """Test finding common time range."""
        seq1 = TokenizedSequence(
            tokens=torch.randn(2, 100, 64),
            t0=0.0,
            dt=0.01,
            mask=torch.ones(2, 100, dtype=torch.bool),
            metadata={}
        )

        seq2 = TokenizedSequence(
            tokens=torch.randn(2, 50, 64),
            t0=0.2,  # Starts later
            dt=0.01,
            mask=torch.ones(2, 50, dtype=torch.bool),
            metadata={}
        )

        aligner = TemporalAligner()
        t_start, t_end = aligner.find_common_timerange([seq1, seq2])

        # Common range should be [0.2, 0.7] (where they overlap)
        assert t_start == pytest.approx(0.2, abs=1e-6)
        assert t_end == pytest.approx(0.69, abs=1e-2)  # seq2 ends at 0.2 + 0.49

    def test_find_common_timerange_no_overlap(self):
        """Test that non-overlapping sequences raise error."""
        seq1 = TokenizedSequence(
            tokens=torch.randn(2, 10, 64),
            t0=0.0,
            dt=0.01,
            mask=torch.ones(2, 10, dtype=torch.bool),
            metadata={}
        )

        seq2 = TokenizedSequence(
            tokens=torch.randn(2, 10, 64),
            t0=1.0,  # No overlap
            dt=0.01,
            mask=torch.ones(2, 10, dtype=torch.bool),
            metadata={}
        )

        aligner = TemporalAligner()
        with pytest.raises(ValueError, match="do not overlap"):
            aligner.find_common_timerange([seq1, seq2])

    def test_interpolate_nearest(self):
        """Test nearest neighbor interpolation."""
        # Create sequence with known values
        tokens = torch.arange(10).view(1, 10, 1).float()  # Values 0-9
        seq = TokenizedSequence(
            tokens=tokens,
            t0=0.0,
            dt=0.1,  # 10 Hz
            mask=torch.ones(1, 10, dtype=torch.bool),
            metadata={}
        )

        # Resample to intermediate points
        new_times = torch.tensor([0.05, 0.15, 0.25])  # Between 0.0, 0.1, 0.2, 0.3

        aligner = TemporalAligner()
        resampled = aligner.interpolate_sequence(
            seq, new_times, InterpolationMethod.NEAREST
        )

        # Should get nearest values: 0, 1, 2
        assert resampled.tokens[0, 0, 0] == pytest.approx(0.0)
        assert resampled.tokens[0, 1, 0] == pytest.approx(1.0)
        assert resampled.tokens[0, 2, 0] == pytest.approx(2.0)

    def test_interpolate_linear(self):
        """Test linear interpolation."""
        # Create sequence with known values
        tokens = torch.arange(10).view(1, 10, 1).float()  # Values 0-9
        seq = TokenizedSequence(
            tokens=tokens,
            t0=0.0,
            dt=0.1,  # 10 Hz
            mask=torch.ones(1, 10, dtype=torch.bool),
            metadata={}
        )

        # Resample to midpoints
        new_times = torch.tensor([0.05, 0.15, 0.25])

        aligner = TemporalAligner()
        resampled = aligner.interpolate_sequence(
            seq, new_times, InterpolationMethod.LINEAR
        )

        # Should get interpolated values: 0.5, 1.5, 2.5
        assert resampled.tokens[0, 0, 0] == pytest.approx(0.5, abs=1e-5)
        assert resampled.tokens[0, 1, 0] == pytest.approx(1.5, abs=1e-5)
        assert resampled.tokens[0, 2, 0] == pytest.approx(2.5, abs=1e-5)

    def test_interpolate_causal(self):
        """Test causal interpolation (only past context)."""
        tokens = torch.arange(10).view(1, 10, 1).float()
        seq = TokenizedSequence(
            tokens=tokens,
            t0=0.0,
            dt=0.1,
            mask=torch.ones(1, 10, dtype=torch.bool),
            metadata={}
        )

        # Sample at 0.05 (between 0.0 and 0.1)
        # Causal should use 0.0 (past value)
        new_times = torch.tensor([0.05, 0.15])

        aligner = TemporalAligner()
        resampled = aligner.interpolate_sequence(
            seq, new_times, InterpolationMethod.CAUSAL
        )

        # Should use most recent past value
        assert resampled.tokens[0, 0, 0] == pytest.approx(0.0)
        assert resampled.tokens[0, 1, 0] == pytest.approx(1.0)

    def test_align_to_grid_different_rates(self):
        """Test aligning sequences with different sampling rates."""
        # EEG at 128 Hz
        eeg_seq = TokenizedSequence(
            tokens=torch.randn(2, 256, 64),  # 2 seconds
            t0=0.0,
            dt=1/128.0,
            mask=torch.ones(2, 256, dtype=torch.bool),
            metadata={'modality': 'eeg'}
        )

        # Video at 30 Hz
        video_seq = TokenizedSequence(
            tokens=torch.randn(2, 60, 64),  # 2 seconds
            t0=0.0,
            dt=1/30.0,
            mask=torch.ones(2, 60, dtype=torch.bool),
            metadata={'modality': 'video'}
        )

        # Align to 50 Hz
        aligner = TemporalAligner()
        aligned = aligner.align_to_grid(
            sequences=[eeg_seq, video_seq],
            target_dt=0.02,  # 50 Hz
            method=InterpolationMethod.LINEAR
        )

        assert len(aligned) == 2
        assert aligned[0].dt == pytest.approx(0.02)
        assert aligned[1].dt == pytest.approx(0.02)
        assert aligned[0].seq_len == aligned[1].seq_len

    def test_create_windows(self):
        """Test creating sliding windows."""
        seq = TokenizedSequence(
            tokens=torch.randn(2, 100, 64),
            t0=0.0,
            dt=0.01,  # 1 second total
            mask=torch.ones(2, 100, dtype=torch.bool),
            metadata={}
        )

        aligner = TemporalAligner()
        windows = aligner.create_windows(
            sequences=[seq],
            window_size=0.3,  # 300ms windows
            hop_size=0.1,      # 100ms stride (70% overlap)
            align_first=False
        )

        # Should create multiple windows
        assert len(windows) > 0

        # Each window should contain one sequence per input
        for win in windows:
            assert len(win) == 1
            assert win[0].duration <= 0.3 + seq.dt

    def test_create_windows_multimodal(self):
        """Test creating windows with multiple modalities."""
        seq1 = TokenizedSequence(
            tokens=torch.randn(2, 100, 64),
            t0=0.0,
            dt=0.01,
            mask=torch.ones(2, 100, dtype=torch.bool),
            metadata={'modality': 'eeg'}
        )

        seq2 = TokenizedSequence(
            tokens=torch.randn(2, 100, 64),
            t0=0.0,
            dt=0.01,
            mask=torch.ones(2, 100, dtype=torch.bool),
            metadata={'modality': 'lfp'}
        )

        aligner = TemporalAligner()
        windows = aligner.create_windows(
            sequences=[seq1, seq2],
            window_size=0.5,
            hop_size=0.25,
            align_first=False
        )

        # Each window should have both modalities
        for win in windows:
            assert len(win) == 2

    def test_validate_alignment_valid(self):
        """Test validation of properly aligned sequences."""
        seq1 = TokenizedSequence(
            tokens=torch.randn(2, 100, 64),
            t0=0.0,
            dt=0.01,
            mask=torch.ones(2, 100, dtype=torch.bool),
            metadata={}
        )

        seq2 = TokenizedSequence(
            tokens=torch.randn(2, 100, 64),
            t0=0.0,
            dt=0.01,
            mask=torch.ones(2, 100, dtype=torch.bool),
            metadata={}
        )

        aligner = TemporalAligner()
        result = aligner.validate_alignment([seq1, seq2], strict=True)

        assert result['valid']
        assert result['checks']['dt_match']['passed']
        assert result['checks']['length_match']['passed']
        assert result['checks']['t0_match']['passed']

    def test_validate_alignment_invalid(self):
        """Test validation of misaligned sequences."""
        seq1 = TokenizedSequence(
            tokens=torch.randn(2, 100, 64),
            t0=0.0,
            dt=0.01,
            mask=torch.ones(2, 100, dtype=torch.bool),
            metadata={}
        )

        seq2 = TokenizedSequence(
            tokens=torch.randn(2, 50, 64),  # Different length
            t0=0.0,
            dt=0.01,
            mask=torch.ones(2, 50, dtype=torch.bool),
            metadata={}
        )

        aligner = TemporalAligner()

        # Should raise with strict=True
        with pytest.raises(ValueError, match="don't match"):
            aligner.validate_alignment([seq1, seq2], strict=True)

        # Should return diagnostics with strict=False
        result = aligner.validate_alignment([seq1, seq2], strict=False)
        assert not result['valid']
        assert not result['checks']['length_match']['passed']

    def test_correct_jitter(self):
        """Test jitter correction."""
        # Create sequence with small timing errors
        tokens = torch.randn(1, 10, 64)
        seq = TokenizedSequence(
            tokens=tokens,
            t0=0.0,
            dt=0.01,
            mask=torch.ones(1, 10, dtype=torch.bool),
            metadata={}
        )

        aligner = TemporalAligner()
        corrected = aligner.correct_jitter(seq, max_jitter=0.001)

        # Should have uniform spacing
        corrected_times = corrected.timestamps
        diffs = torch.diff(corrected_times)
        assert torch.allclose(diffs, torch.full_like(diffs, 0.01), atol=1e-6)

    def test_impute_missing(self):
        """Test missing data imputation."""
        # Create sequence with missing values
        tokens = torch.randn(1, 10, 64)
        mask = torch.ones(1, 10, dtype=torch.bool)
        mask[0, 3:6] = False  # Mark middle section as missing

        seq = TokenizedSequence(
            tokens=tokens,
            t0=0.0,
            dt=0.01,
            mask=mask,
            metadata={}
        )

        aligner = TemporalAligner()
        imputed = aligner.impute_missing(seq, method=InterpolationMethod.LINEAR)

        # All positions should now be valid
        assert imputed.mask.all()


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_resample_to_rate(self):
        """Test resampling to target rate."""
        seq = TokenizedSequence(
            tokens=torch.randn(2, 100, 64),
            t0=0.0,
            dt=0.01,  # 100 Hz
            mask=torch.ones(2, 100, dtype=torch.bool),
            metadata={}
        )

        # Resample to 50 Hz
        resampled = resample_to_rate(seq, target_rate=50.0)

        assert resampled.sampling_rate == pytest.approx(50.0)
        assert resampled.duration == pytest.approx(seq.duration, rel=0.1)

    def test_align_and_concatenate(self):
        """Test aligning and concatenating embeddings."""
        seq1 = TokenizedSequence(
            tokens=torch.randn(2, 100, 64),
            t0=0.0,
            dt=0.01,
            mask=torch.ones(2, 100, dtype=torch.bool),
            metadata={'modality': 'eeg'}
        )

        seq2 = TokenizedSequence(
            tokens=torch.randn(2, 50, 32),  # Different rate and dim
            t0=0.0,
            dt=0.02,
            mask=torch.ones(2, 50, dtype=torch.bool),
            metadata={'modality': 'video'}
        )

        combined = align_and_concatenate([seq1, seq2])

        # Should have concatenated embedding dimension
        assert combined.d_model == 64 + 32
        assert combined.metadata['modality'] == 'multimodal'
        assert combined.metadata['n_modalities'] == 2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_sequence_alignment(self):
        """Test aligning single sequence (should return clone)."""
        seq = TokenizedSequence(
            tokens=torch.randn(2, 10, 64),
            t0=0.0,
            dt=0.01,
            mask=torch.ones(2, 10, dtype=torch.bool),
            metadata={}
        )

        aligner = TemporalAligner()
        aligned = aligner.align_to_grid([seq])

        assert len(aligned) == 1
        assert torch.allclose(aligned[0].tokens, seq.tokens)

    def test_empty_sequence_list(self):
        """Test that empty sequence list raises appropriate errors."""
        aligner = TemporalAligner()

        with pytest.raises(ValueError):
            aligner.find_common_timerange([])

        with pytest.raises(ValueError):
            concatenate_sequences([])

        with pytest.raises(ValueError):
            batch_sequences([])

    def test_very_short_sequence(self):
        """Test handling of very short sequences."""
        seq = TokenizedSequence(
            tokens=torch.randn(1, 2, 64),  # Only 2 timesteps
            t0=0.0,
            dt=0.1,
            mask=torch.ones(1, 2, dtype=torch.bool),
            metadata={}
        )

        # Should still work
        assert seq.duration == pytest.approx(0.1)

    def test_extrapolation_warning(self):
        """Test that extrapolation triggers warning."""
        seq = TokenizedSequence(
            tokens=torch.randn(1, 10, 64),
            t0=0.0,
            dt=0.1,
            mask=torch.ones(1, 10, dtype=torch.bool),
            metadata={}
        )

        # Try to interpolate beyond data range
        new_times = torch.tensor([2.0, 3.0])  # Way beyond data

        aligner = TemporalAligner(warn_on_extrapolation=True)

        with pytest.warns(UserWarning, match="beyond data range"):
            aligner.interpolate_sequence(seq, new_times, InterpolationMethod.LINEAR)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
