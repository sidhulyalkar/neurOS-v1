"""
Tests for ENGRAM attractor memory.

Verifies that memory weights sum to one, entropy is finite,
and output shapes are correct.
"""

import pytest
import torch

from neuros_neurofm.backbones.engram_fmx.modules.attractor_memory import AttractorMemory


class TestAttractorMemory:
    """Test AttractorMemory module."""

    @pytest.fixture
    def memory(self):
        """Create attractor memory for testing."""
        return AttractorMemory(
            hidden_dim=128,
            memory_slots=64,
            beta=8.0,
            alpha=0.5,
        )

    @pytest.fixture
    def queries(self):
        """Create sample queries."""
        B, K, D = 2, 32, 128
        return torch.randn(B, K, D)

    def test_output_shape(self, memory, queries):
        """Test output shape matches input shape."""
        output, diagnostics = memory(queries)
        assert output.shape == queries.shape

    def test_memory_weights_sum_to_one(self, memory, queries):
        """Test that memory attention weights sum to 1."""
        _, diagnostics = memory(queries)

        weights = diagnostics["memory_weights"]  # [B, K, M]
        weight_sums = weights.sum(dim=-1)  # [B, K]

        # Should sum to 1 (within numerical tolerance)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), \
            f"Weights do not sum to 1: min={weight_sums.min()}, max={weight_sums.max()}"

    def test_memory_entropy_is_finite(self, memory, queries):
        """Test that memory entropy is finite."""
        _, diagnostics = memory(queries)

        entropy = diagnostics["memory_entropy"]
        assert not torch.isnan(torch.tensor(entropy)), "Entropy is NaN"
        assert not torch.isinf(torch.tensor(entropy)), "Entropy is Inf"
        assert entropy >= 0, f"Entropy should be non-negative, got {entropy}"

    def test_memory_weights_are_probabilities(self, memory, queries):
        """Test that memory weights are valid probabilities."""
        _, diagnostics = memory(queries)

        weights = diagnostics["memory_weights"]

        # All weights should be non-negative
        assert (weights >= 0).all(), "Weights contain negative values"

        # All weights should be <= 1
        assert (weights <= 1).all(), "Weights contain values > 1"

    def test_no_nan_output(self, memory, queries):
        """Test that output does not contain NaN."""
        output, _ = memory(queries)
        assert not torch.isnan(output).any(), "Output contains NaN"

    def test_energy_computation(self, memory, queries):
        """Test energy computation."""
        energy = memory.compute_energy(queries)

        # Energy should have shape [B, K]
        assert energy.shape == queries.shape[:2]

        # Energy should be finite
        assert not torch.isnan(energy).any(), "Energy contains NaN"
        assert not torch.isinf(energy).any(), "Energy contains Inf"

    def test_memory_usage_statistics(self, memory, queries):
        """Test that memory usage stats are computed."""
        _, diagnostics = memory(queries)

        assert "memory_usage_max" in diagnostics
        assert "memory_usage_min" in diagnostics
        assert "memory_top_indices" in diagnostics

        # Usage should be between 0 and 1
        assert 0 <= diagnostics["memory_usage_max"] <= 1
        assert 0 <= diagnostics["memory_usage_min"] <= 1

    def test_different_beta_values(self, queries):
        """Test memory with different temperature values."""
        for beta in [1.0, 4.0, 8.0, 16.0]:
            memory = AttractorMemory(
                hidden_dim=128,
                memory_slots=64,
                beta=beta,
            )
            output, diagnostics = memory(queries)

            # Higher beta should lead to sharper attention (lower entropy)
            assert output.shape == queries.shape
            assert diagnostics["memory_entropy"] >= 0

    def test_different_alpha_values(self, queries):
        """Test memory with different residual mixing values."""
        for alpha in [0.0, 0.5, 1.0]:
            memory = AttractorMemory(
                hidden_dim=128,
                memory_slots=64,
                alpha=alpha,
            )
            output, _ = memory(queries)
            assert output.shape == queries.shape

    def test_gradients_flow(self, memory, queries):
        """Test that gradients flow through memory."""
        queries = queries.clone().requires_grad_(True)
        output, _ = memory(queries)

        loss = output.mean()
        loss.backward()

        assert queries.grad is not None
        assert not torch.isnan(queries.grad).any()


class TestAttractorMemoryScaling:
    """Test AttractorMemory scaling behavior."""

    def test_different_memory_sizes(self):
        """Test with different numbers of memory slots."""
        D = 128
        B, K = 2, 32

        for M in [32, 64, 128, 256]:
            memory = AttractorMemory(hidden_dim=D, memory_slots=M)
            queries = torch.randn(B, K, D)

            output, diagnostics = memory(queries)

            assert output.shape == queries.shape
            assert diagnostics["memory_weights"].shape[-1] == M

    def test_different_query_sizes(self):
        """Test with different numbers of query slots."""
        D, M = 128, 64
        B = 2

        memory = AttractorMemory(hidden_dim=D, memory_slots=M)

        for K in [16, 32, 64, 128]:
            queries = torch.randn(B, K, D)
            output, diagnostics = memory(queries)

            assert output.shape == (B, K, D)
            assert diagnostics["memory_weights"].shape == (B, K, M)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
