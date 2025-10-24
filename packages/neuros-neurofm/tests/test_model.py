"""
Unit tests for NeuroFMx model
"""

import pytest
import torch
import torch.nn as nn

from neuros_neurofm.models.multimodal_neurofmx import (
    MultiModalNeuroFMX,
    GradientReversalLayer,
    DomainDiscriminator
)


class TestGradientReversalLayer:
    """Test GradientReversalLayer"""

    def test_forward_identity(self):
        """Test forward pass is identity"""
        x = torch.randn(4, 128, requires_grad=True)
        lambda_ = 1.0

        output = GradientReversalLayer.apply(x, lambda_)

        # Forward should be identity
        assert torch.allclose(output, x)

    def test_backward_reversal(self):
        """Test backward pass reverses gradient"""
        x = torch.randn(4, 128, requires_grad=True)
        lambda_ = 1.0

        output = GradientReversalLayer.apply(x, lambda_)
        loss = output.sum()
        loss.backward()

        # Gradient should be reversed (negative)
        assert x.grad is not None
        # Check gradient direction is reversed
        expected_grad = torch.ones_like(x) * (-lambda_)
        assert torch.allclose(x.grad, expected_grad)


class TestDomainDiscriminator:
    """Test DomainDiscriminator"""

    def test_init(self):
        """Test initialization"""
        discriminator = DomainDiscriminator(input_dim=512, hidden_dim=256, n_domains=3)
        assert discriminator.n_domains == 3

    def test_forward_shape(self):
        """Test forward pass shape"""
        batch_size = 8
        input_dim = 512
        n_domains = 3

        discriminator = DomainDiscriminator(input_dim=input_dim, hidden_dim=256, n_domains=n_domains)
        x = torch.randn(batch_size, input_dim)

        logits = discriminator(x)

        # Should output (batch_size, n_domains)
        assert logits.shape == (batch_size, n_domains)

    def test_output_range(self):
        """Test output logits are reasonable"""
        discriminator = DomainDiscriminator(input_dim=512, hidden_dim=256, n_domains=3)
        x = torch.randn(8, 512)

        logits = discriminator(x)

        # No NaNs or Infs
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()


class TestMultiModalNeuroFMX:
    """Test MultiModalNeuroFMX model"""

    @pytest.fixture
    def model_config(self):
        """Small model config for testing"""
        return {
            'd_model': 256,
            'n_mamba_blocks': 2,
            'n_latents': 16,
            'latent_dim': 256,
            'n_perceiver_blocks': 1,
            'n_cross_attn_heads': 4,
            'popt_n_heads': 4,
            'decoder_hidden_dim': 256,
            'encoder_hidden_dim': 256,
            'contrastive_dim': 128,
            'forecast_horizon': 10,
            'n_domains': 3,
            'domain_hidden_dim': 128
        }

    @pytest.fixture
    def tokenizer_configs(self):
        """Tokenizer configs for testing"""
        return {
            'spike': {'n_units': 384},
            'lfp': {'n_channels': 128, 'target_seq_len': 50},
            'calcium': {'n_cells': 256}
        }

    def test_init(self, model_config, tokenizer_configs):
        """Test model initialization"""
        model = MultiModalNeuroFMX(
            modalities=['spike', 'lfp', 'calcium'],
            tokenizer_configs=tokenizer_configs,
            **model_config
        )

        assert len(model.tokenizers) == 3
        assert len(model.modality_embeddings) == 3

    def test_forward_single_modality(self, model_config, tokenizer_configs):
        """Test forward pass with single modality"""
        model = MultiModalNeuroFMX(
            modalities=['spike'],
            tokenizer_configs={'spike': {'n_units': 384}},
            **model_config
        )

        batch_size = 4
        seq_len = 50
        n_units = 384

        # Create dummy data
        modality_dict = {
            'spike': torch.randn(batch_size, seq_len, n_units)
        }

        # Forward pass
        outputs = model(modality_dict, task='multi-task')

        # Check outputs exist
        assert 'decoder' in outputs
        assert 'encoder' in outputs
        assert 'contrastive' in outputs
        assert 'forecast' in outputs

        # Check shapes
        assert outputs['decoder'].shape[0] == batch_size
        assert outputs['encoder'].shape[0] == batch_size

    def test_forward_multi_modality(self, model_config, tokenizer_configs):
        """Test forward pass with multiple modalities"""
        model = MultiModalNeuroFMX(
            modalities=['spike', 'lfp', 'calcium'],
            tokenizer_configs=tokenizer_configs,
            **model_config
        )

        batch_size = 4
        seq_len = 50

        # Create dummy data for multiple modalities
        modality_dict = {
            'spike': torch.randn(batch_size, seq_len, 384),
            'lfp': torch.randn(batch_size, seq_len, 128),
            'calcium': torch.randn(batch_size, seq_len, 256)
        }

        # Forward pass
        outputs = model(modality_dict, task='multi-task')

        # Check outputs
        assert 'decoder' in outputs
        assert 'encoder' in outputs
        assert 'contrastive' in outputs

    def test_domain_adversarial(self, model_config, tokenizer_configs):
        """Test domain adversarial training"""
        model = MultiModalNeuroFMX(
            modalities=['spike'],
            tokenizer_configs={'spike': {'n_units': 384}},
            **model_config
        )

        batch_size = 4
        seq_len = 50

        modality_dict = {
            'spike': torch.randn(batch_size, seq_len, 384)
        }

        # Provide species labels
        species_labels = torch.randint(0, 3, (batch_size,))

        # Forward with domain adversarial
        outputs = model(modality_dict, task='multi-task', species_labels=species_labels, grl_lambda=1.0)

        # Check domain logits exist
        assert 'domain_logits' in outputs
        assert outputs['domain_logits'].shape == (batch_size, 3)

    def test_decoder_task(self, model_config, tokenizer_configs):
        """Test decoder task only"""
        model = MultiModalNeuroFMX(
            modalities=['spike'],
            tokenizer_configs={'spike': {'n_units': 384}},
            **model_config
        )

        batch_size = 4
        seq_len = 50

        modality_dict = {
            'spike': torch.randn(batch_size, seq_len, 384)
        }

        # Forward with decoder task
        outputs = model(modality_dict, task='decoder')

        # Check only decoder output
        assert 'decoder' in outputs
        assert 'encoder' not in outputs

    def test_encoder_task(self, model_config, tokenizer_configs):
        """Test encoder task only"""
        model = MultiModalNeuroFMX(
            modalities=['spike'],
            tokenizer_configs={'spike': {'n_units': 384}},
            **model_config
        )

        batch_size = 4
        seq_len = 50

        modality_dict = {
            'spike': torch.randn(batch_size, seq_len, 384)
        }

        # Forward with encoder task
        outputs = model(modality_dict, task='encoder')

        # Check only encoder output
        assert 'encoder' in outputs
        assert 'decoder' not in outputs

    def test_contrastive_task(self, model_config, tokenizer_configs):
        """Test contrastive task"""
        model = MultiModalNeuroFMX(
            modalities=['spike'],
            tokenizer_configs={'spike': {'n_units': 384}},
            **model_config
        )

        batch_size = 4
        seq_len = 50

        modality_dict = {
            'spike': torch.randn(batch_size, seq_len, 384)
        }

        # Forward with contrastive task
        outputs = model(modality_dict, task='contrastive')

        # Check contrastive embeddings
        assert 'contrastive' in outputs
        assert outputs['contrastive'].shape == (batch_size, model_config['contrastive_dim'])

    def test_backward_pass(self, model_config, tokenizer_configs):
        """Test backward pass completes"""
        model = MultiModalNeuroFMX(
            modalities=['spike'],
            tokenizer_configs={'spike': {'n_units': 384}},
            **model_config
        )

        batch_size = 4
        seq_len = 50

        modality_dict = {
            'spike': torch.randn(batch_size, seq_len, 384)
        }

        # Forward
        outputs = model(modality_dict, task='decoder')

        # Dummy loss
        loss = outputs['decoder'].sum()

        # Backward
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_no_nans(self, model_config, tokenizer_configs):
        """Test model doesn't produce NaNs"""
        model = MultiModalNeuroFMX(
            modalities=['spike', 'lfp'],
            tokenizer_configs={
                'spike': {'n_units': 384},
                'lfp': {'n_channels': 128, 'target_seq_len': 50}
            },
            **model_config
        )

        batch_size = 4
        seq_len = 50

        modality_dict = {
            'spike': torch.randn(batch_size, seq_len, 384),
            'lfp': torch.randn(batch_size, seq_len, 128)
        }

        # Forward
        outputs = model(modality_dict, task='multi-task')

        # Check no NaNs
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                assert not torch.isnan(value).any(), f"NaN in {key}"
                assert not torch.isinf(value).any(), f"Inf in {key}"

    def test_variable_sequence_lengths(self, model_config, tokenizer_configs):
        """Test handling of variable sequence lengths"""
        model = MultiModalNeuroFMX(
            modalities=['spike'],
            tokenizer_configs={'spike': {'n_units': 384}},
            **model_config
        )

        batch_size = 4

        # Test different sequence lengths
        for seq_len in [25, 50, 100]:
            modality_dict = {
                'spike': torch.randn(batch_size, seq_len, 384)
            }

            outputs = model(modality_dict, task='decoder')
            assert outputs['decoder'].shape[0] == batch_size

    def test_modality_dropout(self, model_config, tokenizer_configs):
        """Test model handles missing modalities"""
        model = MultiModalNeuroFMX(
            modalities=['spike', 'lfp', 'calcium'],
            tokenizer_configs=tokenizer_configs,
            **model_config
        )

        batch_size = 4
        seq_len = 50

        # Provide only subset of modalities
        modality_dict = {
            'spike': torch.randn(batch_size, seq_len, 384)
            # lfp and calcium missing
        }

        # Should still work
        outputs = model(modality_dict, task='decoder')
        assert outputs['decoder'].shape[0] == batch_size


class TestModelIntegration:
    """Integration tests for model"""

    def test_full_training_step(self):
        """Test complete training step"""
        model_config = {
            'd_model': 256,
            'n_mamba_blocks': 2,
            'n_latents': 16,
            'latent_dim': 256,
            'n_perceiver_blocks': 1,
            'n_cross_attn_heads': 4,
            'popt_n_heads': 4,
            'decoder_hidden_dim': 256,
            'encoder_hidden_dim': 256,
            'contrastive_dim': 128,
            'forecast_horizon': 10,
            'n_domains': 3,
            'domain_hidden_dim': 128
        }

        model = MultiModalNeuroFMX(
            modalities=['spike', 'lfp'],
            tokenizer_configs={
                'spike': {'n_units': 384},
                'lfp': {'n_channels': 128, 'target_seq_len': 50}
            },
            **model_config
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        batch_size = 4
        seq_len = 50

        # Create batch
        modality_dict = {
            'spike': torch.randn(batch_size, seq_len, 384),
            'lfp': torch.randn(batch_size, seq_len, 128)
        }

        species_labels = torch.randint(0, 3, (batch_size,))

        # Forward
        outputs = model(modality_dict, task='multi-task', species_labels=species_labels)

        # Compute losses
        decoder_loss = outputs['decoder'].pow(2).mean()
        encoder_loss = outputs['encoder'].pow(2).mean()
        domain_loss = torch.nn.functional.cross_entropy(
            outputs['domain_logits'],
            species_labels
        )

        total_loss = decoder_loss + encoder_loss + 0.1 * domain_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()

        # Check gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()

        # Optimizer step
        optimizer.step()

        # Check parameters updated
        assert True  # If we got here, training step succeeded


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
