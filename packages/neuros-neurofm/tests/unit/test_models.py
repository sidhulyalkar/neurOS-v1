"""
Unit tests for NeuroFM-X models.
"""

import pytest
import torch

# Skip Mamba tests if mamba-ssm is not available
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

from neuros_neurofm.models import NeuroFMX
from neuros_neurofm.fusion import PerceiverIO


class TestPerceiverIO:
    """Test Perceiver-IO fusion module."""

    def test_initialization(self):
        """Test Perceiver can be initialized."""
        perceiver = PerceiverIO(
            n_latents=128,
            latent_dim=512,
            input_dim=768,
        )
        assert perceiver.n_latents == 128
        assert perceiver.latent_dim == 512

    def test_forward_basic(self):
        """Test basic forward pass."""
        perceiver = PerceiverIO(
            n_latents=128,
            latent_dim=512,
            input_dim=768,
            n_layers=3,
        )

        # Create dummy input
        batch_size = 2
        seq_len = 200
        input_dim = 768

        inputs = torch.randn(batch_size, seq_len, input_dim)

        # Forward pass
        latents = perceiver(inputs)

        # Check output shape
        assert latents.shape == (batch_size, 128, 512)

    def test_with_attention_mask(self):
        """Test forward pass with attention mask."""
        perceiver = PerceiverIO(
            n_latents=128,
            latent_dim=512,
            input_dim=768,
        )

        batch_size = 2
        seq_len = 200

        inputs = torch.randn(batch_size, seq_len, 768)
        # Mask out second half
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[:, seq_len // 2:] = False

        # Forward pass
        latents = perceiver(inputs, attention_mask=mask)

        assert latents.shape == (batch_size, 128, 512)


@pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba-ssm not installed")
class TestMambaBackbone:
    """Test Mamba backbone."""

    def test_initialization(self):
        """Test backbone can be initialized."""
        from neuros_neurofm.models.mamba_backbone import MambaBackbone

        backbone = MambaBackbone(
            d_model=768,
            n_blocks=4,  # Use fewer blocks for testing
        )
        assert backbone.d_model == 768
        assert backbone.n_blocks == 4

    def test_forward_basic(self):
        """Test basic forward pass."""
        from neuros_neurofm.models.mamba_backbone import MambaBackbone

        backbone = MambaBackbone(
            d_model=768,
            n_blocks=4,
            use_multi_rate=False,
        )

        batch_size = 2
        seq_len = 200

        inputs = torch.randn(batch_size, seq_len, 768)

        # Forward pass
        outputs = backbone(inputs)

        assert outputs.shape == (batch_size, seq_len, 768)

    def test_multi_rate(self):
        """Test multi-rate backbone."""
        from neuros_neurofm.models.mamba_backbone import MambaBackbone

        backbone = MambaBackbone(
            d_model=768,
            n_blocks=2,  # Fewer blocks for speed
            use_multi_rate=True,
            downsample_rates=[1, 4],
            fusion_method="concat",
        )

        batch_size = 2
        seq_len = 200

        inputs = torch.randn(batch_size, seq_len, 768)
        outputs = backbone(inputs)

        assert outputs.shape == (batch_size, seq_len, 768)


@pytest.mark.skipif(not MAMBA_AVAILABLE, reason="mamba-ssm not installed")
class TestNeuroFMX:
    """Test complete NeuroFMX model."""

    def test_initialization(self):
        """Test model can be initialized."""
        model = NeuroFMX(
            d_model=768,
            n_mamba_blocks=4,  # Small model for testing
            n_latents=64,
            latent_dim=256,
        )

        assert model.d_model == 768
        assert model.n_latents == 64

    def test_forward_basic(self):
        """Test basic forward pass."""
        model = NeuroFMX(
            d_model=768,
            n_mamba_blocks=4,
            n_latents=64,
            latent_dim=256,
            use_multi_rate=False,  # Disable for speed
        )

        batch_size = 2
        seq_len = 200

        tokens = torch.randn(batch_size, seq_len, 768)

        # Forward pass
        outputs = model(tokens)

        # Should output latents
        assert outputs.shape == (batch_size, 64, 256)

    def test_encode(self):
        """Test encode method."""
        model = NeuroFMX(
            d_model=768,
            n_mamba_blocks=4,
            n_latents=64,
            latent_dim=256,
            use_multi_rate=False,
        )

        tokens = torch.randn(2, 200, 768)
        latents = model.encode(tokens)

        assert latents.shape == (2, 64, 256)

    def test_parameter_counts(self):
        """Test parameter counting."""
        model = NeuroFMX(
            d_model=768,
            n_mamba_blocks=4,
            n_latents=64,
            latent_dim=256,
        )

        param_counts = model.get_num_params()

        assert "backbone" in param_counts
        assert "fusion" in param_counts
        assert "total" in param_counts
        assert param_counts["total"] > 0

    def test_from_config(self):
        """Test creating model from config."""
        config = {
            "d_model": 512,
            "n_blocks": 4,
            "n_latents": 64,
            "latent_dim": 256,
        }

        model = NeuroFMX.from_config(config)

        assert model.d_model == 512
        assert model.n_latents == 64

    def test_save_load(self, tmp_path):
        """Test saving and loading model."""
        model = NeuroFMX(
            d_model=512,
            n_mamba_blocks=2,
            n_latents=32,
            latent_dim=128,
            use_multi_rate=False,
        )

        # Save
        save_path = tmp_path / "model.pt"
        config = {"d_model": 512, "n_blocks": 2}
        model.save_pretrained(str(save_path), config)

        # Load
        loaded_model = NeuroFMX.from_pretrained(str(save_path))

        # Test that loaded model works
        tokens = torch.randn(1, 100, 512)
        output = loaded_model(tokens)

        assert output.shape == (1, 32, 128)
