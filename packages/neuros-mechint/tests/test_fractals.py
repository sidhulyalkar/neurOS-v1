"""
Tests for Fractal Geometry Suite

Run with: pytest tests/test_fractals.py
"""

import pytest
import torch
from neuros_mechint.fractals import (
    HiguchiFractalDimension,
    DetrendedFluctuationAnalysis,
    HurstExponent,
    SpectralSlope,
    MultifractalSpectrum,
    SpectralPrior,
    FractionalBrownianMotion,
)


class TestHiguchiFD:
    """Tests for Higuchi Fractal Dimension"""

    def test_basic_computation(self):
        """Test basic FD computation"""
        fd = HiguchiFractalDimension(k_max=10)
        signal = torch.randn(32, 1000)
        result = fd.compute(signal)

        assert result.shape == (32,)
        assert torch.all(result > 1.0)  # FD should be > 1
        assert torch.all(result < 2.0)  # FD should be < 2

    def test_white_noise_fd(self):
        """Test that white noise gives FD ≈ 1.5"""
        fd = HiguchiFractalDimension(k_max=10)
        signal = torch.randn(100, 10000)  # Large for stability
        result = fd.compute(signal)

        # White noise should have FD ≈ 1.5
        assert torch.abs(result.mean() - 1.5) < 0.2

    def test_gpu_computation(self):
        """Test GPU computation if available"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        fd = HiguchiFractalDimension(k_max=10, device='cuda')
        signal = torch.randn(32, 1000)
        result = fd.compute(signal)

        assert result.device.type == 'cuda'
        assert result.shape == (32,)


class TestDFA:
    """Tests for Detrended Fluctuation Analysis"""

    def test_basic_computation(self):
        """Test basic DFA computation"""
        dfa = DetrendedFluctuationAnalysis(min_win=10, max_win=100)
        signal = torch.randn(32, 1000)
        alpha, fluctuations = dfa.compute(signal)

        assert alpha.shape == (32,)
        assert torch.all(alpha > 0)  # Alpha should be positive

    def test_white_noise_alpha(self):
        """Test that white noise gives alpha ≈ 0.5"""
        dfa = DetrendedFluctuationAnalysis(min_win=10, max_win=100)
        signal = torch.randn(10, 5000)  # Large for stability
        alpha, _ = dfa.compute(signal)

        # White noise should have alpha ≈ 0.5
        assert torch.abs(alpha.mean() - 0.5) < 0.15


class TestHurstExponent:
    """Tests for Hurst Exponent"""

    def test_basic_computation(self):
        """Test basic Hurst exponent computation"""
        hurst = HurstExponent(min_lag=10, max_lag=100)
        signal = torch.randn(32, 1000)
        h = hurst.compute(signal)

        assert h.shape == (32,)
        assert torch.all(h >= 0)  # H should be >= 0
        assert torch.all(h <= 1)  # H should be <= 1

    def test_white_noise_hurst(self):
        """Test that white noise gives H ≈ 0.5"""
        hurst = HurstExponent(min_lag=10, max_lag=100)
        signal = torch.randn(10, 5000)
        h = hurst.compute(signal)

        # White noise should have H ≈ 0.5
        assert torch.abs(h.mean() - 0.5) < 0.15


class TestSpectralSlope:
    """Tests for Spectral Slope"""

    def test_basic_computation(self):
        """Test basic spectral slope computation"""
        slope = SpectralSlope(freq_range=(1, 100))
        signal = torch.randn(32, 1000)
        beta, psd, freqs = slope.compute(signal)

        assert beta.shape == (32,)
        assert psd.shape[0] == 32
        assert freqs.shape[0] > 0

    def test_white_noise_slope(self):
        """Test that white noise gives β ≈ 0"""
        slope = SpectralSlope(freq_range=(1, 100))
        signal = torch.randn(10, 10000)  # Large for stability
        beta, _, _ = slope.compute(signal)

        # White noise should have β ≈ 0 (flat spectrum)
        assert torch.abs(beta.mean()) < 0.5


class TestMultifractalSpectrum:
    """Tests for Multifractal Spectrum"""

    def test_basic_computation(self):
        """Test basic multifractal spectrum computation"""
        mf = MultifractalSpectrum(q_range=(-5, 5), n_q=11)
        signal = torch.randn(32, 1000)
        spectrum = mf.compute(signal)

        assert 'singularity_spectrum' in spectrum
        assert 'generalized_dimensions' in spectrum
        assert spectrum['singularity_spectrum'].shape == (32, 11)


class TestSpectralPrior:
    """Tests for Spectral Prior Regularizer"""

    def test_basic_regularization(self):
        """Test basic regularization"""
        reg = SpectralPrior(target_beta=1.0, weight=0.01)
        activations = torch.randn(32, 256)
        loss = reg(activations)

        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0  # Loss should be non-negative

    def test_gradient_flow(self):
        """Test that gradients flow through regularizer"""
        reg = SpectralPrior(target_beta=1.0, weight=0.01)
        activations = torch.randn(32, 256, requires_grad=True)
        loss = reg(activations)
        loss.backward()

        assert activations.grad is not None
        assert not torch.all(activations.grad == 0)


class TestFractionalBrownianMotion:
    """Tests for Fractional Brownian Motion Generator"""

    def test_basic_generation(self):
        """Test basic fBm generation"""
        fbm = FractionalBrownianMotion(n_samples=1000, hurst=0.7)
        signal = fbm.generate(batch_size=32)

        assert signal.shape == (32, 1000)
        assert signal.dtype == torch.float32

    def test_different_hurst_values(self):
        """Test generation with different Hurst values"""
        for hurst in [0.3, 0.5, 0.7, 0.9]:
            fbm = FractionalBrownianMotion(n_samples=1000, hurst=hurst)
            signal = fbm.generate(batch_size=10)

            assert signal.shape == (10, 1000)

    def test_hurst_recovery(self):
        """Test that generated fBm has correct Hurst exponent"""
        target_hurst = 0.7
        fbm_gen = FractionalBrownianMotion(n_samples=5000, hurst=target_hurst)
        signal = fbm_gen.generate(batch_size=20)

        # Estimate Hurst from generated signal
        hurst_est = HurstExponent(min_lag=10, max_lag=100)
        h_estimated = hurst_est.compute(signal)

        # Should be close to target (within tolerance)
        assert torch.abs(h_estimated.mean() - target_hurst) < 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
