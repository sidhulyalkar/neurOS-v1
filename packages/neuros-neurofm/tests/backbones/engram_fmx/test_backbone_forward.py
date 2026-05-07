"""
Tests for ENGRAM backbone forward pass.

Verifies that ENGRAMBackbone runs correctly end-to-end.
"""

import pytest
import torch

from neuros_neurofm.backbones.engram_fmx.config import ENGRAMFMxConfig
from neuros_neurofm.backbones.engram_fmx.backbone import ENGRAMBackbone, ENGRAMBackboneOutput


class TestENGRAMBackboneForward:
    """Test ENGRAMBackbone forward pass."""

    @pytest.fixture
    def tiny_config(self):
        """Create a tiny config for testing."""
        return ENGRAMFMxConfig.tiny()

    @pytest.fixture
    def backbone(self, tiny_config):
        """Create backbone for testing."""
        return ENGRAMBackbone(tiny_config)

    def test_forward_basic(self, backbone, tiny_config):
        """Test basic forward pass."""
        B, T = 2, 64
        D_in = tiny_config.input_dim

        tokens = torch.randn(B, T, D_in)
        output = backbone(tokens)

        assert isinstance(output, ENGRAMBackboneOutput)
        assert output.sequence_output.shape == (B, T, tiny_config.output_dim)
        assert output.latent_output.shape == (B, tiny_config.num_latents, tiny_config.hidden_dim)

    def test_forward_with_attention_mask(self, backbone, tiny_config):
        """Test forward pass with attention mask."""
        B, T = 2, 64
        D_in = tiny_config.input_dim

        tokens = torch.randn(B, T, D_in)
        # Create mask where first half is valid
        attention_mask = torch.zeros(B, T, dtype=torch.bool)
        attention_mask[:, :T//2] = True

        output = backbone(tokens, attention_mask=attention_mask)

        assert output.sequence_output.shape == (B, T, tiny_config.output_dim)

    def test_forward_no_diagnostics(self, backbone, tiny_config):
        """Test forward pass without diagnostics."""
        B, T = 2, 64
        tokens = torch.randn(B, T, tiny_config.input_dim)

        output = backbone(tokens, return_diagnostics=False)

        assert len(output.diagnostics) == 0

    def test_forward_variable_batch_size(self, backbone, tiny_config):
        """Test forward pass with variable batch sizes."""
        T = 64
        D_in = tiny_config.input_dim

        for B in [1, 2, 4, 8]:
            tokens = torch.randn(B, T, D_in)
            output = backbone(tokens)

            assert output.sequence_output.shape[0] == B

    def test_forward_variable_sequence_length(self, backbone, tiny_config):
        """Test forward pass with variable sequence lengths."""
        B = 2
        D_in = tiny_config.input_dim

        for T in [32, 64, 128, 256]:
            tokens = torch.randn(B, T, D_in)
            output = backbone(tokens)

            assert output.sequence_output.shape == (B, T, tiny_config.output_dim)

    def test_output_shapes_consistency(self, backbone, tiny_config):
        """Test that output shapes are consistent with config."""
        B, T = 2, 64
        tokens = torch.randn(B, T, tiny_config.input_dim)

        output = backbone(tokens)

        # Sequence output should match output_dim
        assert output.sequence_output.shape[-1] == tiny_config.output_dim

        # Latent output should have num_latents slots
        assert output.latent_output.shape[1] == tiny_config.num_latents

        # Latent output should have hidden_dim (before output projection)
        assert output.latent_output.shape[-1] == tiny_config.hidden_dim

    def test_diagnostics_have_layer_prefix(self, backbone, tiny_config):
        """Test that diagnostics include layer prefixes."""
        B, T = 2, 64
        tokens = torch.randn(B, T, tiny_config.input_dim)

        output = backbone(tokens, return_diagnostics=True)

        # Check for layer-prefixed diagnostics
        layer_keys = [k for k in output.diagnostics.keys() if k.startswith("layer")]
        assert len(layer_keys) > 0, "Expected layer-prefixed diagnostics"

    def test_memory_states_returned(self, backbone, tiny_config):
        """Test that memory states are returned."""
        B, T = 2, 64
        tokens = torch.randn(B, T, tiny_config.input_dim)

        output = backbone(tokens)

        # Should have states for each layer
        assert output.memory_state is not None
        assert len(output.memory_state) == tiny_config.num_layers


class TestENGRAMBackboneConfigs:
    """Test ENGRAMBackbone with different config sizes."""

    def test_tiny_config(self):
        """Test tiny config runs."""
        backbone = ENGRAMBackbone.tiny()
        tokens = torch.randn(2, 64, backbone.config.input_dim)

        output = backbone(tokens)
        assert output.sequence_output is not None

    def test_small_config(self):
        """Test small config runs."""
        backbone = ENGRAMBackbone.small()
        tokens = torch.randn(2, 64, backbone.config.input_dim)

        output = backbone(tokens)
        assert output.sequence_output is not None

    def test_custom_input_output_dims(self):
        """Test backbone with different input/output dimensions."""
        config = ENGRAMFMxConfig.tiny()
        config.input_dim = 128
        config.hidden_dim = 256
        config.output_dim = 64

        backbone = ENGRAMBackbone(config)
        tokens = torch.randn(2, 64, 128)

        output = backbone(tokens)

        assert output.sequence_output.shape[-1] == 64

    def test_get_num_params(self):
        """Test parameter counting."""
        backbone = ENGRAMBackbone.tiny()

        n_params = backbone.get_num_params()
        assert n_params > 0

        n_params_no_emb = backbone.get_num_params(non_embedding=True)
        assert n_params_no_emb <= n_params


class TestENGRAMBackboneGradients:
    """Test ENGRAMBackbone gradients flow correctly."""

    def test_backward_pass(self):
        """Test that gradients flow through the backbone."""
        config = ENGRAMFMxConfig.tiny()
        backbone = ENGRAMBackbone(config)

        tokens = torch.randn(2, 64, config.input_dim, requires_grad=True)
        output = backbone(tokens)

        # Compute loss and backward
        loss = output.sequence_output.mean() + output.latent_output.mean()
        loss.backward()

        # Check gradients exist
        assert tokens.grad is not None
        assert tokens.grad.shape == tokens.shape

        # Check model parameters have gradients (except sparse router which uses non-diff topk)
        for name, param in backbone.named_parameters():
            if param.requires_grad:
                # Skip router parameters - topk indices are non-differentiable
                if "router" in name:
                    continue
                assert param.grad is not None, f"No gradient for {name}"

    def test_no_nan_gradients(self):
        """Test that no NaN gradients occur."""
        config = ENGRAMFMxConfig.tiny()
        backbone = ENGRAMBackbone(config)

        tokens = torch.randn(2, 64, config.input_dim, requires_grad=True)
        output = backbone(tokens)

        loss = output.sequence_output.mean()
        loss.backward()

        # Check for NaN gradients
        for name, param in backbone.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
