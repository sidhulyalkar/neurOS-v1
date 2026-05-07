"""
Tests for ENGRAM sparse anchor attention.

Verifies that sparse anchor selection produces valid indices
and handles edge cases.
"""

import pytest
import torch

from neuros_neurofm.backbones.engram_fmx.modules.sparse_anchor_attention import SparseAnchorAttention


class TestSparseAnchorAttention:
    """Test SparseAnchorAttention module."""

    @pytest.fixture
    def sparse_attn(self):
        """Create sparse attention for testing."""
        return SparseAnchorAttention(
            hidden_dim=128,
            num_heads=4,
            sparse_top_k=32,
        )

    @pytest.fixture
    def inputs(self):
        """Create sample inputs."""
        B, T, K, D = 2, 64, 16, 128
        latents = torch.randn(B, K, D)
        tokens = torch.randn(B, T, D)
        return latents, tokens

    def test_output_shape(self, sparse_attn, inputs):
        """Test output shape matches latent input shape."""
        latents, tokens = inputs
        output, _ = sparse_attn(latents, tokens)
        assert output.shape == latents.shape

    def test_selected_indices_valid(self, sparse_attn, inputs):
        """Test that selected indices are valid."""
        latents, tokens = inputs
        B, T, _ = tokens.shape

        _, diagnostics = sparse_attn(latents, tokens)
        indices = diagnostics["sparse_selected_indices"]  # [B, k]

        # All indices should be valid
        assert (indices >= 0).all()
        assert (indices < T).all()

    def test_num_selected_correct(self, sparse_attn, inputs):
        """Test that correct number of tokens are selected."""
        latents, tokens = inputs
        _, diagnostics = sparse_attn(latents, tokens)

        num_selected = diagnostics["sparse_num_selected"]
        assert num_selected == min(sparse_attn.sparse_top_k, tokens.shape[1])

    def test_top_k_greater_than_sequence(self):
        """Test handling when top_k > sequence length."""
        sparse_attn = SparseAnchorAttention(
            hidden_dim=128,
            num_heads=4,
            sparse_top_k=100,  # More than sequence length
        )

        B, T, K, D = 2, 32, 16, 128  # T < top_k
        latents = torch.randn(B, K, D)
        tokens = torch.randn(B, T, D)

        output, diagnostics = sparse_attn(latents, tokens)

        # Should select all tokens
        assert diagnostics["sparse_num_selected"] == T
        assert output.shape == latents.shape

    def test_with_attention_mask(self, sparse_attn, inputs):
        """Test sparse attention with attention mask."""
        latents, tokens = inputs
        B, T, _ = tokens.shape

        # Mask out second half
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, :T//2] = True

        output, diagnostics = sparse_attn(latents, tokens, attention_mask=mask)

        # Output should still have correct shape
        assert output.shape == latents.shape

        # Selected indices should only be from valid region
        indices = diagnostics["sparse_selected_indices"]
        assert (indices < T//2).all(), "Selected masked tokens"

    def test_fully_masked(self):
        """Test handling when all tokens are masked."""
        sparse_attn = SparseAnchorAttention(
            hidden_dim=128,
            num_heads=4,
            sparse_top_k=32,
        )

        B, T, K, D = 2, 64, 16, 128
        latents = torch.randn(B, K, D)
        tokens = torch.randn(B, T, D)
        mask = torch.zeros(B, T, dtype=torch.bool)  # All masked

        # Should handle gracefully (may select from -inf scores)
        output, _ = sparse_attn(latents, tokens, attention_mask=mask)
        assert output.shape == latents.shape

    def test_no_nan_output(self, sparse_attn, inputs):
        """Test output does not contain NaN."""
        latents, tokens = inputs
        output, _ = sparse_attn(latents, tokens)
        assert not torch.isnan(output).any()

    def test_diagnostics_present(self, sparse_attn, inputs):
        """Test diagnostics are returned."""
        latents, tokens = inputs
        _, diagnostics = sparse_attn(latents, tokens)

        assert "sparse_selected_indices" in diagnostics
        assert "sparse_router_scores_mean" in diagnostics
        assert "sparse_router_scores_std" in diagnostics
        assert "sparse_attn_entropy" in diagnostics
        assert "sparse_num_selected" in diagnostics

    def test_gradients_flow(self, sparse_attn, inputs):
        """Test gradients flow through sparse attention."""
        latents, tokens = inputs
        latents = latents.clone().requires_grad_(True)
        tokens = tokens.clone().requires_grad_(True)

        output, _ = sparse_attn(latents, tokens)
        loss = output.mean()
        loss.backward()

        assert latents.grad is not None
        assert tokens.grad is not None


class TestSparseAnchorScaling:
    """Test SparseAnchorAttention scaling behavior."""

    def test_different_top_k_values(self):
        """Test with different top_k values."""
        B, T, K, D = 2, 128, 16, 128

        for top_k in [8, 16, 32, 64]:
            sparse_attn = SparseAnchorAttention(
                hidden_dim=D,
                num_heads=4,
                sparse_top_k=top_k,
            )

            latents = torch.randn(B, K, D)
            tokens = torch.randn(B, T, D)

            output, diagnostics = sparse_attn(latents, tokens)

            assert output.shape == (B, K, D)
            assert diagnostics["sparse_num_selected"] == min(top_k, T)

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        B, K, D = 2, 16, 128
        sparse_attn = SparseAnchorAttention(
            hidden_dim=D,
            num_heads=4,
            sparse_top_k=32,
        )

        for T in [32, 64, 128, 256]:
            latents = torch.randn(B, K, D)
            tokens = torch.randn(B, T, D)

            output, diagnostics = sparse_attn(latents, tokens)

            assert output.shape == (B, K, D)
            assert diagnostics["sparse_num_selected"] == min(32, T)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
