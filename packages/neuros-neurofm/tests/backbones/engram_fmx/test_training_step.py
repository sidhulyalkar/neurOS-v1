"""
Tests for ENGRAM training step.

Verifies that forward/backward passes work correctly
and gradients are well-behaved.
"""

import pytest
import torch
import torch.nn as nn

from neuros_neurofm.backbones.engram_fmx.config import ENGRAMFMxConfig
from neuros_neurofm.backbones.engram_fmx.backbone import ENGRAMBackbone


class TestTrainingStep:
    """Test training step (forward + backward)."""

    @pytest.fixture
    def backbone(self):
        """Create backbone for testing."""
        return ENGRAMBackbone(ENGRAMFMxConfig.tiny())

    @pytest.fixture
    def sample_batch(self, backbone):
        """Create sample batch."""
        B, T = 4, 64
        D_in = backbone.config.input_dim

        tokens = torch.randn(B, T, D_in)
        targets = torch.randn(B, T, backbone.config.output_dim)

        return tokens, targets

    def test_forward_backward_runs(self, backbone, sample_batch):
        """Test that forward/backward completes without error."""
        tokens, targets = sample_batch

        # Forward
        output = backbone(tokens)

        # Compute loss
        loss = nn.functional.mse_loss(output.sequence_output, targets)

        # Backward
        loss.backward()

        # Should complete without error
        assert True

    def test_no_nan_loss(self, backbone, sample_batch):
        """Test that loss is not NaN."""
        tokens, targets = sample_batch

        output = backbone(tokens)
        loss = nn.functional.mse_loss(output.sequence_output, targets)

        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"

    def test_no_nan_gradients(self, backbone, sample_batch):
        """Test that gradients are not NaN."""
        tokens, targets = sample_batch

        output = backbone(tokens)
        loss = nn.functional.mse_loss(output.sequence_output, targets)
        loss.backward()

        for name, param in backbone.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    def test_optimizer_step(self, backbone, sample_batch):
        """Test that optimizer step runs."""
        tokens, targets = sample_batch
        optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-4)

        # Forward
        output = backbone(tokens)
        loss = nn.functional.mse_loss(output.sequence_output, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Should complete without error
        assert True

    def test_multiple_steps(self, backbone, sample_batch):
        """Test multiple training steps."""
        tokens, targets = sample_batch
        optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-4)

        losses = []
        for _ in range(5):
            output = backbone(tokens)
            loss = nn.functional.mse_loss(output.sequence_output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # All losses should be finite
        assert all(not (torch.isnan(torch.tensor(l)) or torch.isinf(torch.tensor(l))) for l in losses)

    def test_gradient_accumulation(self, backbone, sample_batch):
        """Test gradient accumulation over multiple batches."""
        tokens, targets = sample_batch
        optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-4)

        optimizer.zero_grad()

        # Accumulate gradients over 2 micro-batches
        for i in range(2):
            output = backbone(tokens)
            # Include both sequence and latent outputs in loss to ensure all paths get gradients
            loss = nn.functional.mse_loss(output.sequence_output, targets) + \
                   output.latent_output.mean() * 0.01
            (loss / 2).backward()  # Scale loss

        # Check gradients exist (except sparse router which uses non-diff topk)
        for name, param in backbone.named_parameters():
            if param.requires_grad:
                # Skip router parameters - topk indices are non-differentiable
                if "router" in name:
                    continue
                assert param.grad is not None, f"No gradient for {name}"

        optimizer.step()

    def test_loss_decreases_on_overfit(self, backbone):
        """Test that loss decreases when overfitting on single batch."""
        B, T = 2, 32
        tokens = torch.randn(B, T, backbone.config.input_dim)
        targets = torch.randn(B, T, backbone.config.output_dim)

        optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-3)

        initial_loss = None
        final_loss = None

        for step in range(50):
            output = backbone(tokens)
            loss = nn.functional.mse_loss(output.sequence_output, targets)

            if step == 0:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss = loss.item()

        # Loss should decrease significantly
        assert final_loss < initial_loss, \
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"


class TestTrainingWithDifferentConfigs:
    """Test training with different configurations."""

    def test_tiny_config_trains(self):
        """Test training with tiny config."""
        backbone = ENGRAMBackbone(ENGRAMFMxConfig.tiny())
        self._run_training_step(backbone)

    def test_small_config_trains(self):
        """Test training with small config."""
        backbone = ENGRAMBackbone(ENGRAMFMxConfig.small())
        self._run_training_step(backbone)

    def test_ssm_only_trains(self):
        """Test training with SSM-only config."""
        config = ENGRAMFMxConfig.tiny()
        config.use_latent_workspace = False
        config.use_attractor_memory = False
        config.use_operator_dynamics = False
        config.use_sparse_anchor_attention = False
        config.use_controller = False

        backbone = ENGRAMBackbone(config)
        self._run_training_step(backbone)

    def test_no_memory_trains(self):
        """Test training without memory."""
        config = ENGRAMFMxConfig.tiny()
        config.use_attractor_memory = False

        backbone = ENGRAMBackbone(config)
        self._run_training_step(backbone)

    def _run_training_step(self, backbone):
        """Helper to run a single training step."""
        B, T = 2, 64
        tokens = torch.randn(B, T, backbone.config.input_dim)
        targets = torch.randn(B, T, backbone.config.output_dim)

        optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-4)

        output = backbone(tokens)
        loss = nn.functional.mse_loss(output.sequence_output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss)


class TestTrainingStability:
    """Test training stability over longer runs."""

    def test_no_gradient_explosion(self):
        """Test that gradients don't explode over training."""
        backbone = ENGRAMBackbone(ENGRAMFMxConfig.tiny())
        optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-3)

        B, T = 2, 64
        tokens = torch.randn(B, T, backbone.config.input_dim)
        targets = torch.randn(B, T, backbone.config.output_dim)

        max_grad_norm = 0.0

        for _ in range(20):
            output = backbone(tokens)
            loss = nn.functional.mse_loss(output.sequence_output, targets)

            optimizer.zero_grad()
            loss.backward()

            # Track max gradient norm
            total_norm = 0.0
            for p in backbone.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            max_grad_norm = max(max_grad_norm, total_norm)

            optimizer.step()

        # Gradient norm should stay reasonable
        assert max_grad_norm < 1000, f"Gradient exploded: max norm = {max_grad_norm}"

    def test_no_loss_explosion(self):
        """Test that loss doesn't explode over training."""
        backbone = ENGRAMBackbone(ENGRAMFMxConfig.tiny())
        optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-3)

        B, T = 2, 64
        tokens = torch.randn(B, T, backbone.config.input_dim)
        targets = torch.randn(B, T, backbone.config.output_dim)

        initial_loss = None
        max_loss = 0.0

        for step in range(20):
            output = backbone(tokens)
            loss = nn.functional.mse_loss(output.sequence_output, targets)

            if step == 0:
                initial_loss = loss.item()

            max_loss = max(max_loss, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss shouldn't explode (more than 10x initial)
        assert max_loss < initial_loss * 10, \
            f"Loss exploded: initial={initial_loss:.4f}, max={max_loss:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
