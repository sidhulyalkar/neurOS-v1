"""
Tests for ENGRAM block shapes.

Verifies that ENGRAMBlock preserves expected tensor shapes
across all configurations.
"""

import pytest
import torch

from neuros_neurofm.backbones.engram_fmx.config import ENGRAMFMxConfig
from neuros_neurofm.backbones.engram_fmx.block import ENGRAMBlock


class TestENGRAMBlockShapes:
    """Test ENGRAMBlock shape preservation."""

    @pytest.fixture
    def tiny_config(self):
        """Create a tiny config for testing."""
        return ENGRAMFMxConfig.tiny()

    @pytest.fixture
    def sample_inputs(self, tiny_config):
        """Create sample inputs for testing."""
        B, T, D = 2, 64, tiny_config.hidden_dim
        K = tiny_config.num_latents

        tokens = torch.randn(B, T, D)
        latents = torch.randn(B, K, D)
        attention_mask = torch.ones(B, T, dtype=torch.bool)

        return tokens, latents, attention_mask

    def test_block_output_shapes(self, tiny_config, sample_inputs):
        """Test that block preserves input/output shapes."""
        tokens, latents, attention_mask = sample_inputs
        B, T, D = tokens.shape
        K = latents.shape[1]

        block = ENGRAMBlock(tiny_config)
        output = block(tokens, latents, attention_mask=attention_mask)

        assert output.sequence_output.shape == (B, T, D), \
            f"Expected sequence shape {(B, T, D)}, got {output.sequence_output.shape}"
        assert output.latent_output.shape == (B, K, D), \
            f"Expected latent shape {(B, K, D)}, got {output.latent_output.shape}"

    def test_block_requires_latents(self, tiny_config, sample_inputs):
        """Test block requires latents to be provided (from backbone)."""
        tokens, latents, attention_mask = sample_inputs
        B, T, D = tokens.shape
        K = tiny_config.num_latents

        block = ENGRAMBlock(tiny_config)

        # Should work with latents provided
        output = block(tokens, latents=latents, attention_mask=attention_mask)
        assert output.sequence_output.shape == (B, T, D)
        assert output.latent_output.shape == (B, K, D)

    def test_block_without_mask(self, tiny_config, sample_inputs):
        """Test block works without attention mask."""
        tokens, latents, _ = sample_inputs

        block = ENGRAMBlock(tiny_config)
        output = block(tokens, latents, attention_mask=None)

        assert output.sequence_output is not None
        assert output.latent_output is not None

    def test_block_variable_sequence_length(self, tiny_config):
        """Test block handles variable sequence lengths."""
        B, D = 2, tiny_config.hidden_dim
        K = tiny_config.num_latents

        block = ENGRAMBlock(tiny_config)

        for T in [32, 64, 128, 256]:
            tokens = torch.randn(B, T, D)
            latents = torch.randn(B, K, D)

            output = block(tokens, latents)

            assert output.sequence_output.shape == (B, T, D), \
                f"Failed for T={T}"
            assert output.latent_output.shape == (B, K, D)

    def test_block_diagnostics_present(self, tiny_config, sample_inputs):
        """Test that diagnostics are returned."""
        tokens, latents, attention_mask = sample_inputs

        block = ENGRAMBlock(tiny_config)
        output = block(tokens, latents, attention_mask=attention_mask)

        assert isinstance(output.diagnostics, dict)
        assert len(output.diagnostics) > 0, "Expected non-empty diagnostics"


class TestENGRAMBlockAblations:
    """Test ENGRAMBlock ablation configurations."""

    def test_no_local_processing(self):
        """Test block works without local processing."""
        config = ENGRAMFMxConfig.tiny()
        config.use_local_processing = False

        block = ENGRAMBlock(config)
        tokens = torch.randn(2, 64, config.hidden_dim)
        latents = torch.randn(2, config.num_latents, config.hidden_dim)

        output = block(tokens, latents)
        assert output.sequence_output.shape == tokens.shape

    def test_no_ssm(self):
        """Test block works without SSM."""
        config = ENGRAMFMxConfig.tiny()
        config.use_ssm = False

        block = ENGRAMBlock(config)
        tokens = torch.randn(2, 64, config.hidden_dim)
        latents = torch.randn(2, config.num_latents, config.hidden_dim)

        output = block(tokens, latents)
        assert output.sequence_output.shape == tokens.shape

    def test_no_memory(self):
        """Test block works without attractor memory."""
        config = ENGRAMFMxConfig.tiny()
        config.use_attractor_memory = False

        block = ENGRAMBlock(config)
        tokens = torch.randn(2, 64, config.hidden_dim)
        latents = torch.randn(2, config.num_latents, config.hidden_dim)

        output = block(tokens, latents)
        assert output.latent_output.shape == latents.shape

    def test_no_operator_dynamics(self):
        """Test block works without operator dynamics."""
        config = ENGRAMFMxConfig.tiny()
        config.use_operator_dynamics = False

        block = ENGRAMBlock(config)
        tokens = torch.randn(2, 64, config.hidden_dim)
        latents = torch.randn(2, config.num_latents, config.hidden_dim)

        output = block(tokens, latents)
        assert output.latent_output.shape == latents.shape

    def test_no_sparse_attention(self):
        """Test block works without sparse anchor attention."""
        config = ENGRAMFMxConfig.tiny()
        config.use_sparse_anchor_attention = False

        block = ENGRAMBlock(config)
        tokens = torch.randn(2, 64, config.hidden_dim)
        latents = torch.randn(2, config.num_latents, config.hidden_dim)

        output = block(tokens, latents)
        assert output.latent_output.shape == latents.shape

    def test_ssm_only(self):
        """Test block with SSM only (minimal config)."""
        config = ENGRAMFMxConfig.tiny()
        config.use_local_processing = False
        config.use_latent_workspace = False
        config.use_attractor_memory = False
        config.use_operator_dynamics = False
        config.use_sparse_anchor_attention = False
        config.use_controller = False

        block = ENGRAMBlock(config)
        tokens = torch.randn(2, 64, config.hidden_dim)

        output = block(tokens, latents=None)
        assert output.sequence_output.shape == tokens.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
