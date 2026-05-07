"""
Tests for ENGRAM spectral operator dynamics.

Verifies that FFT-based operator produces correct output shapes
and handles edge cases.
"""

import pytest
import torch

from neuros_neurofm.backbones.engram_fmx.modules.operator_dynamics import SpectralOperatorDynamics


class TestSpectralOperatorDynamics:
    """Test SpectralOperatorDynamics module."""

    @pytest.fixture
    def operator(self):
        """Create operator for testing."""
        return SpectralOperatorDynamics(
            hidden_dim=128,
            num_latents=64,
            operator_modes=16,
        )

    @pytest.fixture
    def latents(self):
        """Create sample latents."""
        B, K, D = 2, 64, 128
        return torch.randn(B, K, D)

    def test_output_shape(self, operator, latents):
        """Test output shape matches input shape."""
        output, diagnostics = operator(latents)
        assert output.shape == latents.shape

    def test_no_nan_output(self, operator, latents):
        """Test output does not contain NaN."""
        output, _ = operator(latents)
        assert not torch.isnan(output).any(), "Output contains NaN"

    def test_no_inf_output(self, operator, latents):
        """Test output does not contain Inf."""
        output, _ = operator(latents)
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_diagnostics_present(self, operator, latents):
        """Test diagnostics are returned."""
        _, diagnostics = operator(latents)

        assert "operator_total_spectral_energy" in diagnostics
        assert "operator_low_mode_energy_ratio" in diagnostics
        assert "operator_output_norm" in diagnostics

    def test_spectral_energy_positive(self, operator, latents):
        """Test that spectral energy is positive."""
        _, diagnostics = operator(latents)

        assert diagnostics["operator_total_spectral_energy"] >= 0
        assert 0 <= diagnostics["operator_low_mode_energy_ratio"] <= 1

    def test_small_latent_count(self):
        """Test operator with small latent count (K < operator_modes)."""
        operator = SpectralOperatorDynamics(
            hidden_dim=128,
            num_latents=8,  # Less than operator_modes
            operator_modes=16,
        )

        latents = torch.randn(2, 8, 128)
        output, _ = operator(latents)

        assert output.shape == latents.shape
        assert not torch.isnan(output).any()

    def test_odd_latent_count(self):
        """Test operator with odd number of latents."""
        operator = SpectralOperatorDynamics(
            hidden_dim=128,
            num_latents=63,  # Odd number
            operator_modes=16,
        )

        latents = torch.randn(2, 63, 128)
        output, _ = operator(latents)

        assert output.shape == latents.shape
        assert not torch.isnan(output).any()

    def test_very_small_latent_count(self):
        """Test operator with very small latent count."""
        operator = SpectralOperatorDynamics(
            hidden_dim=128,
            num_latents=4,
            operator_modes=16,
        )

        latents = torch.randn(2, 4, 128)
        output, _ = operator(latents)

        assert output.shape == latents.shape

    def test_gradients_flow(self, operator, latents):
        """Test that gradients flow through operator."""
        latents = latents.clone().requires_grad_(True)
        output, _ = operator(latents)

        loss = output.mean()
        loss.backward()

        assert latents.grad is not None
        assert not torch.isnan(latents.grad).any()

    def test_different_batch_sizes(self, operator):
        """Test with different batch sizes."""
        K, D = 64, 128

        for B in [1, 2, 4, 8]:
            latents = torch.randn(B, K, D)
            output, _ = operator(latents)
            assert output.shape == (B, K, D)


class TestSpectralOperatorModes:
    """Test SpectralOperatorDynamics mode handling."""

    def test_mode_count_capping(self):
        """Test that mode count is capped to valid range."""
        # Create operator with more modes than possible
        operator = SpectralOperatorDynamics(
            hidden_dim=128,
            num_latents=8,
            operator_modes=100,  # More than K//2 + 1 = 5
        )

        # Should cap to 5 modes
        assert operator.operator_modes <= 5

        latents = torch.randn(2, 8, 128)
        output, _ = operator(latents)
        assert output.shape == latents.shape

    def test_different_mode_counts(self):
        """Test with different numbers of modes."""
        D, K = 128, 64

        for modes in [4, 8, 16, 32]:
            operator = SpectralOperatorDynamics(
                hidden_dim=D,
                num_latents=K,
                operator_modes=modes,
            )

            latents = torch.randn(2, K, D)
            output, _ = operator(latents)

            assert output.shape == latents.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
