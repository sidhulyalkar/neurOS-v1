"""
Unit tests for NeuroFMx tokenizers
"""

import pytest
import torch
import numpy as np

from neuros_neurofm.tokenizers.spike_tokenizer import SpikeTokenizer
from neuros_neurofm.tokenizers.binned_tokenizer import BinnedSpikeTokenizer
from neuros_neurofm.tokenizers.lfp_tokenizer import LFPTokenizer
from neuros_neurofm.tokenizers.calcium_tokenizer import CalciumTokenizer
from neuros_neurofm.tokenizers.eeg_tokenizer import EEGTokenizer
from neuros_neurofm.tokenizers.fmri_tokenizer import fMRITokenizer


class TestSpikeTokenizer:
    """Test SpikeTokenizer"""

    def test_init(self):
        """Test tokenizer initialization"""
        tokenizer = SpikeTokenizer(n_units=384, d_model=512)
        assert tokenizer.n_units == 384
        assert tokenizer.d_model == 512

    def test_forward_shape(self):
        """Test forward pass produces correct output shape"""
        batch_size = 8
        seq_len = 100
        n_units = 384
        d_model = 512

        tokenizer = SpikeTokenizer(n_units=n_units, d_model=d_model)

        # Create dummy spike data
        spikes = torch.randn(batch_size, seq_len, n_units)

        # Forward pass
        output = tokenizer(spikes)

        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)

    def test_forward_valid_range(self):
        """Test output values are in reasonable range"""
        tokenizer = SpikeTokenizer(n_units=384, d_model=512)
        spikes = torch.randn(4, 100, 384)

        output = tokenizer(spikes)

        # Check no NaNs or Infs
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_use_sqrt(self):
        """Test sqrt transform option"""
        tokenizer_sqrt = SpikeTokenizer(n_units=384, d_model=512, use_sqrt=True)
        tokenizer_no_sqrt = SpikeTokenizer(n_units=384, d_model=512, use_sqrt=False)

        spikes = torch.randn(4, 100, 384).abs()  # Positive for sqrt

        out_sqrt = tokenizer_sqrt(spikes)
        out_no_sqrt = tokenizer_no_sqrt(spikes)

        # Outputs should be different
        assert not torch.allclose(out_sqrt, out_no_sqrt)


class TestBinnedSpikeTokenizer:
    """Test BinnedSpikeTokenizer"""

    def test_init(self):
        """Test tokenizer initialization"""
        tokenizer = BinnedSpikeTokenizer(n_units=384, d_model=512, bin_size=0.01)
        assert tokenizer.n_units == 384
        assert tokenizer.bin_size == 0.01

    def test_forward_shape(self):
        """Test forward pass produces correct output shape"""
        batch_size = 8
        seq_len = 100
        n_units = 384
        d_model = 512

        tokenizer = BinnedSpikeTokenizer(n_units=n_units, d_model=d_model)

        # Create dummy binned spike data
        binned_spikes = torch.randint(0, 10, (batch_size, seq_len, n_units)).float()

        # Forward pass
        output = tokenizer(binned_spikes)

        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)

    def test_different_bin_sizes(self):
        """Test tokenizers with different bin sizes"""
        tokenizer_10ms = BinnedSpikeTokenizer(n_units=384, d_model=512, bin_size=0.01)
        tokenizer_50ms = BinnedSpikeTokenizer(n_units=384, d_model=512, bin_size=0.05)

        spikes = torch.randint(0, 5, (4, 100, 384)).float()

        out_10ms = tokenizer_10ms(spikes)
        out_50ms = tokenizer_50ms(spikes)

        # Both should produce same shape
        assert out_10ms.shape == out_50ms.shape


class TestLFPTokenizer:
    """Test LFPTokenizer"""

    def test_init(self):
        """Test tokenizer initialization"""
        tokenizer = LFPTokenizer(n_channels=128, d_model=512)
        assert tokenizer.n_channels == 128
        assert tokenizer.d_model == 512

    def test_forward_shape(self):
        """Test forward pass produces correct output shape"""
        batch_size = 8
        seq_len = 100
        n_channels = 128
        d_model = 512

        tokenizer = LFPTokenizer(n_channels=n_channels, d_model=d_model, target_seq_len=100)

        # Create dummy LFP data
        lfp = torch.randn(batch_size, seq_len, n_channels)

        # Forward pass
        output = tokenizer(lfp)

        # Check output shape
        assert output.shape == (batch_size, 100, d_model)  # target_seq_len=100

    def test_spectral_encoding(self):
        """Test spectral encoding option"""
        tokenizer_spectral = LFPTokenizer(n_channels=128, d_model=512, use_spectral=True)
        tokenizer_no_spectral = LFPTokenizer(n_channels=128, d_model=512, use_spectral=False)

        lfp = torch.randn(4, 100, 128)

        out_spectral = tokenizer_spectral(lfp)
        out_no_spectral = tokenizer_no_spectral(lfp)

        # Both should produce same shape
        assert out_spectral.shape == out_no_spectral.shape

    def test_variable_length_input(self):
        """Test handling of variable length inputs"""
        tokenizer = LFPTokenizer(n_channels=128, d_model=512, target_seq_len=100)

        # Test different input lengths
        lfp_short = torch.randn(4, 50, 128)
        lfp_long = torch.randn(4, 200, 128)

        out_short = tokenizer(lfp_short)
        out_long = tokenizer(lfp_long)

        # Both should produce target length
        assert out_short.shape[1] == 100
        assert out_long.shape[1] == 100


class TestCalciumTokenizer:
    """Test CalciumTokenizer"""

    def test_init(self):
        """Test tokenizer initialization"""
        tokenizer = CalciumTokenizer(n_cells=512, d_model=512)
        assert tokenizer.n_cells == 512
        assert tokenizer.d_model == 512

    def test_forward_shape(self):
        """Test forward pass produces correct output shape"""
        batch_size = 8
        seq_len = 100
        n_cells = 512
        d_model = 512

        tokenizer = CalciumTokenizer(n_cells=n_cells, d_model=d_model)

        # Create dummy calcium imaging data (dF/F)
        calcium = torch.randn(batch_size, seq_len, n_cells)

        # Forward pass
        output = tokenizer(calcium)

        # Check output shape
        assert output.shape == (batch_size, seq_len, d_model)

    def test_temporal_downsampling(self):
        """Test temporal downsampling"""
        tokenizer = CalciumTokenizer(n_cells=512, d_model=512, target_fps=10)

        # Simulate 30 Hz data
        calcium = torch.randn(4, 300, 512)  # 10 seconds at 30 Hz

        output = tokenizer(calcium)

        # Should downsample to ~10 Hz (100 frames for 10 seconds)
        assert output.shape[1] < 300


class TestEEGTokenizer:
    """Test EEGTokenizer"""

    def test_init(self):
        """Test tokenizer initialization"""
        tokenizer = EEGTokenizer(n_channels=64, d_model=512, sfreq=128)
        assert tokenizer.n_channels == 64
        assert tokenizer.d_model == 512
        assert tokenizer.sfreq == 128

    def test_forward_shape(self):
        """Test forward pass produces correct output shape"""
        batch_size = 8
        seq_len = 256
        n_channels = 64
        d_model = 512

        tokenizer = EEGTokenizer(n_channels=n_channels, d_model=d_model, sfreq=128)

        # Create dummy EEG data
        eeg = torch.randn(batch_size, seq_len, n_channels)

        # Forward pass
        output = tokenizer(eeg)

        # Check output shape
        assert output.shape[0] == batch_size
        assert output.shape[2] == d_model

    def test_spectral_encoder(self):
        """Test spectral encoding with frequency bands"""
        tokenizer = EEGTokenizer(
            n_channels=64,
            d_model=512,
            sfreq=128,
            extract_bands=True
        )

        eeg = torch.randn(4, 256, 64)
        output = tokenizer(eeg)

        # Check no NaNs
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_multiscale_temporal(self):
        """Test multi-scale temporal convolutions"""
        tokenizer = EEGTokenizer(
            n_channels=64,
            d_model=512,
            sfreq=128,
            temporal_kernels=[3, 7, 15, 31]
        )

        eeg = torch.randn(4, 256, 64)
        output = tokenizer(eeg)

        # Check valid output
        assert output.shape[0] == 4
        assert output.shape[2] == 512


class TestfMRITokenizer:
    """Test fMRITokenizer"""

    def test_init(self):
        """Test tokenizer initialization"""
        tokenizer = fMRITokenizer(n_rois=400, d_model=512)
        assert tokenizer.n_rois == 400
        assert tokenizer.d_model == 512

    def test_forward_shape(self):
        """Test forward pass produces correct output shape"""
        batch_size = 8
        seq_len = 150
        n_rois = 400
        d_model = 512

        tokenizer = fMRITokenizer(n_rois=n_rois, d_model=d_model, target_seq_len=100)

        # Create dummy fMRI BOLD data
        fmri = torch.randn(batch_size, seq_len, n_rois)

        # Forward pass
        output = tokenizer(fmri)

        # Check output shape
        assert output.shape == (batch_size, 100, d_model)

    def test_dilated_convolutions(self):
        """Test dilated convolutions for slow fMRI dynamics"""
        tokenizer = fMRITokenizer(
            n_rois=400,
            d_model=512,
            dilation_rates=[1, 2, 4, 8]
        )

        fmri = torch.randn(4, 150, 400)
        output = tokenizer(fmri)

        # Check valid output
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_variable_roi_numbers(self):
        """Test handling different ROI parcellations"""
        tokenizer_400 = fMRITokenizer(n_rois=400, d_model=512)
        tokenizer_1000 = fMRITokenizer(n_rois=1000, d_model=512)

        fmri_400 = torch.randn(4, 150, 400)
        fmri_1000 = torch.randn(4, 150, 1000)

        out_400 = tokenizer_400(fmri_400)
        out_1000 = tokenizer_1000(fmri_1000)

        # Both should produce same d_model
        assert out_400.shape[2] == out_1000.shape[2] == 512


# Integration test
class TestTokenizerIntegration:
    """Test tokenizers work together"""

    def test_all_tokenizers_same_output_dim(self):
        """Test all tokenizers produce same d_model output"""
        d_model = 512
        batch_size = 4
        seq_len = 100

        tokenizers_and_data = [
            (SpikeTokenizer(n_units=384, d_model=d_model), torch.randn(batch_size, seq_len, 384)),
            (BinnedSpikeTokenizer(n_units=384, d_model=d_model), torch.randint(0, 5, (batch_size, seq_len, 384)).float()),
            (LFPTokenizer(n_channels=128, d_model=d_model, target_seq_len=seq_len), torch.randn(batch_size, seq_len, 128)),
            (CalciumTokenizer(n_cells=512, d_model=d_model), torch.randn(batch_size, seq_len, 512)),
            (EEGTokenizer(n_channels=64, d_model=d_model, sfreq=128), torch.randn(batch_size, seq_len, 64)),
            (fMRITokenizer(n_rois=400, d_model=d_model, target_seq_len=seq_len), torch.randn(batch_size, seq_len, 400)),
        ]

        outputs = []
        for tokenizer, data in tokenizers_and_data:
            output = tokenizer(data)
            outputs.append(output)

            # Check shape
            assert output.shape[0] == batch_size
            assert output.shape[2] == d_model

        # All should have same feature dimension
        for output in outputs:
            assert output.shape[2] == d_model


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
