"""
Tests for ENGRAM ablation flags.

Verifies that config flags correctly enable/disable modules.
"""

import pytest
import torch

from neuros_neurofm.backbones.engram_fmx.config import ENGRAMFMxConfig
from neuros_neurofm.backbones.engram_fmx.block import ENGRAMBlock
from neuros_neurofm.backbones.engram_fmx.backbone import ENGRAMBackbone


class TestAblationFlags:
    """Test ablation configuration flags."""

    @pytest.fixture
    def base_inputs(self):
        """Create base inputs for testing."""
        B, T, K, D = 2, 64, 32, 128
        tokens = torch.randn(B, T, D)
        latents = torch.randn(B, K, D)
        return tokens, latents

    def test_full_model_runs(self, base_inputs):
        """Test full model with all components enabled."""
        config = ENGRAMFMxConfig.tiny()
        block = ENGRAMBlock(config)

        tokens, latents = base_inputs
        output = block(tokens, latents)

        assert output.sequence_output is not None
        assert output.latent_output is not None

    def test_no_local_processing_flag(self, base_inputs):
        """Test disabling local processing."""
        config = ENGRAMFMxConfig.tiny()
        config.use_local_processing = False

        block = ENGRAMBlock(config)
        assert block.local_processing is None

        tokens, latents = base_inputs
        output = block(tokens, latents)
        assert output.sequence_output.shape == tokens.shape

    def test_no_ssm_flag(self, base_inputs):
        """Test disabling SSM."""
        config = ENGRAMFMxConfig.tiny()
        config.use_ssm = False

        block = ENGRAMBlock(config)
        assert block.selective_ssm is None

        tokens, latents = base_inputs
        output = block(tokens, latents)
        assert output.sequence_output.shape == tokens.shape

    def test_no_latent_workspace_flag(self, base_inputs):
        """Test disabling latent workspace."""
        config = ENGRAMFMxConfig.tiny()
        config.use_latent_workspace = False

        block = ENGRAMBlock(config)
        assert block.latent_workspace is None

        tokens, latents = base_inputs
        output = block(tokens, latents)
        # Latents should be unchanged (passed through)
        assert output.latent_output is not None

    def test_no_attractor_memory_flag(self, base_inputs):
        """Test disabling attractor memory."""
        config = ENGRAMFMxConfig.tiny()
        config.use_attractor_memory = False

        block = ENGRAMBlock(config)
        assert block.attractor_memory is None

        tokens, latents = base_inputs
        output = block(tokens, latents)
        assert output.latent_output.shape == latents.shape

    def test_no_operator_dynamics_flag(self, base_inputs):
        """Test disabling operator dynamics."""
        config = ENGRAMFMxConfig.tiny()
        config.use_operator_dynamics = False

        block = ENGRAMBlock(config)
        assert block.operator_dynamics is None

        tokens, latents = base_inputs
        output = block(tokens, latents)
        assert output.latent_output.shape == latents.shape

    def test_no_sparse_attention_flag(self, base_inputs):
        """Test disabling sparse anchor attention."""
        config = ENGRAMFMxConfig.tiny()
        config.use_sparse_anchor_attention = False

        block = ENGRAMBlock(config)
        assert block.sparse_anchor_attention is None

        tokens, latents = base_inputs
        output = block(tokens, latents)
        assert output.latent_output.shape == latents.shape

    def test_no_controller_flag(self, base_inputs):
        """Test disabling controller/gated fusion."""
        config = ENGRAMFMxConfig.tiny()
        config.use_controller = False

        block = ENGRAMBlock(config)
        assert block.gated_fusion is None

        tokens, latents = base_inputs
        output = block(tokens, latents)
        assert output.latent_output is not None


class TestAblationConfigurations:
    """Test common ablation configurations."""

    def test_ssm_only_config(self):
        """Test SSM-only configuration (minimal model)."""
        config = ENGRAMFMxConfig.tiny()
        config.use_local_processing = False
        config.use_latent_workspace = False
        config.use_attractor_memory = False
        config.use_operator_dynamics = False
        config.use_sparse_anchor_attention = False
        config.use_controller = False

        backbone = ENGRAMBackbone(config)
        tokens = torch.randn(2, 64, config.input_dim)

        output = backbone(tokens)
        assert output.sequence_output.shape == (2, 64, config.output_dim)

    def test_latent_only_config(self):
        """Test latent-only configuration (no SSM)."""
        config = ENGRAMFMxConfig.tiny()
        config.use_local_processing = False
        config.use_ssm = False

        backbone = ENGRAMBackbone(config)
        tokens = torch.randn(2, 64, config.input_dim)

        output = backbone(tokens)
        assert output.sequence_output.shape == (2, 64, config.output_dim)
        assert output.latent_output.shape == (2, config.num_latents, config.hidden_dim)

    def test_no_memory_config(self):
        """Test configuration without memory (test operator/attention only)."""
        config = ENGRAMFMxConfig.tiny()
        config.use_attractor_memory = False

        backbone = ENGRAMBackbone(config)
        tokens = torch.randn(2, 64, config.input_dim)

        output = backbone(tokens)
        assert output.sequence_output is not None

    def test_no_operator_config(self):
        """Test configuration without operator dynamics."""
        config = ENGRAMFMxConfig.tiny()
        config.use_operator_dynamics = False

        backbone = ENGRAMBackbone(config)
        tokens = torch.randn(2, 64, config.input_dim)

        output = backbone(tokens)
        assert output.sequence_output is not None


class TestAblationDiagnostics:
    """Test that ablations affect diagnostics correctly."""

    def test_no_memory_removes_memory_diagnostics(self):
        """Test that disabling memory removes memory diagnostics."""
        config = ENGRAMFMxConfig.tiny()
        config.use_attractor_memory = False

        backbone = ENGRAMBackbone(config)
        tokens = torch.randn(2, 64, config.input_dim)

        output = backbone(tokens, return_diagnostics=True)

        # Should not have memory-related diagnostics
        memory_keys = [k for k in output.diagnostics.keys() if "memory" in k]
        assert len(memory_keys) == 0, f"Found unexpected memory diagnostics: {memory_keys}"

    def test_no_operator_removes_operator_diagnostics(self):
        """Test that disabling operator removes operator diagnostics."""
        config = ENGRAMFMxConfig.tiny()
        config.use_operator_dynamics = False

        backbone = ENGRAMBackbone(config)
        tokens = torch.randn(2, 64, config.input_dim)

        output = backbone(tokens, return_diagnostics=True)

        # Should not have operator-related diagnostics
        operator_keys = [k for k in output.diagnostics.keys() if "operator" in k]
        assert len(operator_keys) == 0, f"Found unexpected operator diagnostics: {operator_keys}"

    def test_full_model_has_all_diagnostics(self):
        """Test that full model has all diagnostic categories."""
        config = ENGRAMFMxConfig.tiny()
        backbone = ENGRAMBackbone(config)
        tokens = torch.randn(2, 64, config.input_dim)

        output = backbone(tokens, return_diagnostics=True)

        # Should have diagnostics from each enabled module
        diagnostic_categories = ["local", "ssm", "workspace", "memory", "operator", "sparse"]

        for category in diagnostic_categories:
            matching_keys = [k for k in output.diagnostics.keys() if category in k]
            assert len(matching_keys) > 0, f"Missing diagnostics for {category}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
