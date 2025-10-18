"""
Unit tests for neural tokenizers.
"""

import pytest
import torch

from neuros_neurofm.tokenizers import (
    SpikeTokenizer,
    BinnedTokenizer,
    LFPTokenizer,
)


class TestSpikeTokenizer:
    """Test spike tokenizer."""

    def test_initialization(self):
        """Test tokenizer can be initialized."""
        tokenizer = SpikeTokenizer(
            n_units=96,
            d_model=768,
        )
        assert tokenizer.n_units == 96
        assert tokenizer.d_model == 768

    def test_forward_basic(self):
        """Test basic forward pass."""
        tokenizer = SpikeTokenizer(
            n_units=96,
            d_model=768,
        )

        # Create dummy spike data
        batch_size = 2
        n_spikes = 100

        spike_times = torch.randn(batch_size, n_spikes) * 100  # 0-100ms
        spike_units = torch.randint(0, 96, (batch_size, n_spikes))

        # Forward pass
        tokens, mask = tokenizer(spike_times, spike_units)

        # Check output shapes
        assert tokens.shape == (batch_size, n_spikes, 768)
        assert mask.shape == (batch_size, n_spikes)
        assert mask.all()  # All valid spikes

    def test_forward_with_waveforms(self):
        """Test forward pass with waveform features."""
        tokenizer = SpikeTokenizer(
            n_units=96,
            d_model=768,
            use_waveform_features=True,
            waveform_dim=32,
        )

        batch_size = 2
        n_spikes = 100

        spike_times = torch.randn(batch_size, n_spikes) * 100
        spike_units = torch.randint(0, 96, (batch_size, n_spikes))
        spike_waveforms = torch.randn(batch_size, n_spikes, 32)

        # Forward pass
        tokens, mask = tokenizer(
            spike_times,
            spike_units,
            spike_waveforms=spike_waveforms,
        )

        assert tokens.shape == (batch_size, n_spikes, 768)


class TestBinnedTokenizer:
    """Test binned tokenizer."""

    def test_initialization(self):
        """Test tokenizer can be initialized."""
        tokenizer = BinnedTokenizer(
            n_units=96,
            d_model=768,
        )
        assert tokenizer.n_units == 96
        assert tokenizer.d_model == 768

    def test_forward_basic(self):
        """Test basic forward pass."""
        tokenizer = BinnedTokenizer(
            n_units=96,
            d_model=768,
        )

        # Create dummy binned data
        batch_size = 2
        seq_length = 200  # 200 time bins

        binned_data = torch.randn(batch_size, seq_length, 96)

        # Forward pass
        tokens, mask = tokenizer(binned_data)

        # Check output shapes
        assert tokens.shape == (batch_size, seq_length, 768)
        assert mask.shape == (batch_size, seq_length)

    def test_sqrt_transform(self):
        """Test sqrt transform is applied."""
        tokenizer_with = BinnedTokenizer(
            n_units=96,
            d_model=768,
            use_sqrt_transform=True,
        )

        tokenizer_without = BinnedTokenizer(
            n_units=96,
            d_model=768,
            use_sqrt_transform=False,
        )

        # Positive spike counts
        binned_data = torch.abs(torch.randn(2, 100, 96))

        tokens_with, _ = tokenizer_with(binned_data)
        tokens_without, _ = tokenizer_without(binned_data)

        # Outputs should differ
        assert not torch.allclose(tokens_with, tokens_without)


class TestLFPTokenizer:
    """Test LFP tokenizer."""

    def test_initialization(self):
        """Test tokenizer can be initialized."""
        tokenizer = LFPTokenizer(
            n_channels=64,
            d_model=768,
        )
        assert tokenizer.n_channels == 64
        assert tokenizer.d_model == 768

    def test_forward_basic(self):
        """Test basic forward pass."""
        tokenizer = LFPTokenizer(
            n_channels=64,
            d_model=768,
            pool_size=4,
        )

        # Create dummy LFP data
        batch_size = 2
        time_points = 1000  # 1000 samples

        lfp = torch.randn(batch_size, 64, time_points)

        # Forward pass
        tokens, mask = tokenizer(lfp, fs=250.0)

        # After pooling by 4, should have time_points // 4 tokens
        expected_seq_len = time_points // 4
        assert tokens.shape == (batch_size, expected_seq_len, 768)
        assert mask.shape == (batch_size, expected_seq_len)

    def test_spectral_features(self):
        """Test spectral feature extraction."""
        tokenizer_with = LFPTokenizer(
            n_channels=64,
            d_model=768,
            use_spectral_features=True,
        )

        tokenizer_without = LFPTokenizer(
            n_channels=64,
            d_model=768,
            use_spectral_features=False,
        )

        lfp = torch.randn(2, 64, 1000)

        tokens_with, _ = tokenizer_with(lfp)
        tokens_without, _ = tokenizer_without(lfp)

        # Both should work
        assert tokens_with.shape == tokens_without.shape
        # But outputs should differ
        assert not torch.allclose(tokens_with, tokens_without, atol=1e-3)
