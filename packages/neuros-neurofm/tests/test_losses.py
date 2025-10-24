"""
Unit tests for NeuroFMx loss functions
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuros_neurofm.losses.contrastive_loss import (
    InfoNCELoss,
    TriModalContrastiveLoss,
    TemporalContrastiveLoss
)
from neuros_neurofm.losses.domain_adversarial import (
    DomainAdversarialLoss,
    DomainConfusionLoss,
    MMDLoss
)
from neuros_neurofm.losses.multitask_loss import (
    UncertaintyWeightedLoss,
    GradNormLoss,
    MultiTaskLossManager
)


class TestInfoNCELoss:
    """Test InfoNCELoss"""

    def test_init(self):
        """Test initialization"""
        loss_fn = InfoNCELoss(temperature=0.07)
        assert loss_fn.temperature == 0.07

    def test_forward_shape(self):
        """Test forward pass produces scalar loss"""
        loss_fn = InfoNCELoss(temperature=0.07)

        batch_size = 8
        dim = 128

        anchor = torch.randn(batch_size, dim)
        positive = torch.randn(batch_size, dim)

        loss = loss_fn(anchor, positive)

        # Should be scalar
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_with_explicit_negatives(self):
        """Test with explicit negative samples"""
        loss_fn = InfoNCELoss(temperature=0.07)

        batch_size = 8
        n_negatives = 16
        dim = 128

        anchor = torch.randn(batch_size, dim)
        positive = torch.randn(batch_size, dim)
        negatives = torch.randn(batch_size, n_negatives, dim)

        loss = loss_fn(anchor, positive, negatives)

        assert loss.item() >= 0

    def test_perfect_alignment(self):
        """Test loss is low when anchor and positive are identical"""
        loss_fn = InfoNCELoss(temperature=0.07)

        batch_size = 8
        dim = 128

        anchor = torch.randn(batch_size, dim)
        positive = anchor.clone()  # Perfect alignment

        loss = loss_fn(anchor, positive)

        # Loss should be close to 0
        assert loss.item() < 1.0

    def test_temperature_effect(self):
        """Test temperature affects loss magnitude"""
        batch_size = 8
        dim = 128

        anchor = torch.randn(batch_size, dim)
        positive = torch.randn(batch_size, dim)

        loss_low_temp = InfoNCELoss(temperature=0.01)(anchor, positive)
        loss_high_temp = InfoNCELoss(temperature=1.0)(anchor, positive)

        # Different temperatures should give different losses
        assert not torch.isclose(loss_low_temp, loss_high_temp)


class TestTriModalContrastiveLoss:
    """Test TriModalContrastiveLoss"""

    def test_init(self):
        """Test initialization"""
        loss_fn = TriModalContrastiveLoss(
            temperature=0.07,
            neural_weight=1.0,
            behavior_weight=1.0,
            stimulus_weight=1.0
        )
        assert loss_fn.temperature == 0.07

    def test_forward_shape(self):
        """Test forward pass produces loss dict"""
        loss_fn = TriModalContrastiveLoss()

        batch_size = 8
        dim = 128

        neural = torch.randn(batch_size, dim)
        behavior = torch.randn(batch_size, dim)
        stimulus = torch.randn(batch_size, dim)
        timestamps = torch.arange(batch_size).float()

        loss, components = loss_fn(neural, behavior, stimulus, timestamps)

        # Check total loss
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

        # Check components
        assert 'neural_behavior' in components
        assert 'neural_stimulus' in components
        assert 'behavior_stimulus' in components

    def test_with_stimulus_ids(self):
        """Test with stimulus IDs for grouping"""
        loss_fn = TriModalContrastiveLoss()

        batch_size = 8
        dim = 128

        neural = torch.randn(batch_size, dim)
        behavior = torch.randn(batch_size, dim)
        stimulus = torch.randn(batch_size, dim)
        timestamps = torch.arange(batch_size).float()
        stimulus_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        loss, components = loss_fn(neural, behavior, stimulus, timestamps, stimulus_ids)

        assert loss.item() >= 0


class TestTemporalContrastiveLoss:
    """Test TemporalContrastiveLoss"""

    def test_init(self):
        """Test initialization"""
        loss_fn = TemporalContrastiveLoss(temperature=0.07, temporal_window=5)
        assert loss_fn.temporal_window == 5

    def test_forward_shape(self):
        """Test forward pass produces scalar loss"""
        loss_fn = TemporalContrastiveLoss(temporal_window=5)

        batch_size = 8
        seq_len = 50
        dim = 128

        embeddings = torch.randn(batch_size, seq_len, dim)

        loss = loss_fn(embeddings)

        # Should be scalar
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_temporal_window_effect(self):
        """Test different temporal windows"""
        batch_size = 4
        seq_len = 50
        dim = 128

        embeddings = torch.randn(batch_size, seq_len, dim)

        loss_small_window = TemporalContrastiveLoss(temporal_window=3)(embeddings)
        loss_large_window = TemporalContrastiveLoss(temporal_window=10)(embeddings)

        # Should produce different losses
        assert not torch.isclose(loss_small_window, loss_large_window, rtol=0.1)


class TestDomainAdversarialLoss:
    """Test DomainAdversarialLoss"""

    def test_init(self):
        """Test initialization"""
        loss_fn = DomainAdversarialLoss()
        assert loss_fn is not None

    def test_forward_shape(self):
        """Test forward pass produces scalar loss"""
        loss_fn = DomainAdversarialLoss()

        batch_size = 8
        n_domains = 3

        domain_logits = torch.randn(batch_size, n_domains)
        domain_labels = torch.randint(0, n_domains, (batch_size,))

        loss = loss_fn(domain_logits, domain_labels)

        # Should be scalar
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_perfect_prediction(self):
        """Test loss is low with perfect predictions"""
        loss_fn = DomainAdversarialLoss()

        batch_size = 8
        n_domains = 3

        domain_labels = torch.randint(0, n_domains, (batch_size,))

        # Create perfect logits
        domain_logits = torch.zeros(batch_size, n_domains)
        for i, label in enumerate(domain_labels):
            domain_logits[i, label] = 10.0  # High logit for correct class

        loss = loss_fn(domain_logits, domain_labels)

        # Loss should be very low
        assert loss.item() < 0.1


class TestDomainConfusionLoss:
    """Test DomainConfusionLoss"""

    def test_init(self):
        """Test initialization"""
        loss_fn = DomainConfusionLoss()
        assert loss_fn is not None

    def test_forward_shape(self):
        """Test forward pass produces scalar loss"""
        loss_fn = DomainConfusionLoss()

        batch_size = 8
        n_domains = 3

        domain_logits = torch.randn(batch_size, n_domains)

        loss = loss_fn(domain_logits)

        # Should be scalar
        assert loss.shape == torch.Size([])

    def test_uniform_distribution(self):
        """Test loss is minimized with uniform distribution"""
        loss_fn = DomainConfusionLoss()

        batch_size = 8
        n_domains = 3

        # Uniform logits
        uniform_logits = torch.zeros(batch_size, n_domains)

        # Random logits
        random_logits = torch.randn(batch_size, n_domains) * 5

        loss_uniform = loss_fn(uniform_logits)
        loss_random = loss_fn(random_logits)

        # Uniform should have lower (more negative) loss
        assert loss_uniform.item() < loss_random.item()


class TestMMDLoss:
    """Test MMDLoss (Maximum Mean Discrepancy)"""

    def test_init(self):
        """Test initialization"""
        loss_fn = MMDLoss(kernel_mul=2.0, kernel_num=5)
        assert loss_fn.kernel_mul == 2.0
        assert loss_fn.kernel_num == 5

    def test_forward_shape(self):
        """Test forward pass produces scalar loss"""
        loss_fn = MMDLoss()

        batch_size = 8
        dim = 128

        source_features = torch.randn(batch_size, dim)
        target_features = torch.randn(batch_size, dim)

        loss = loss_fn(source_features, target_features)

        # Should be scalar
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0

    def test_identical_distributions(self):
        """Test loss is near zero for identical distributions"""
        loss_fn = MMDLoss()

        batch_size = 16
        dim = 128

        features = torch.randn(batch_size, dim)

        # Same features as source and target
        loss = loss_fn(features, features)

        # Loss should be very close to 0
        assert loss.item() < 1e-4

    def test_different_distributions(self):
        """Test loss is positive for different distributions"""
        loss_fn = MMDLoss()

        batch_size = 16
        dim = 128

        source_features = torch.randn(batch_size, dim)
        target_features = torch.randn(batch_size, dim) + 5.0  # Shifted distribution

        loss = loss_fn(source_features, target_features)

        # Loss should be positive
        assert loss.item() > 0


class TestUncertaintyWeightedLoss:
    """Test UncertaintyWeightedLoss"""

    def test_init(self):
        """Test initialization"""
        task_names = ['task1', 'task2', 'task3']
        loss_fn = UncertaintyWeightedLoss(task_names=task_names)

        assert len(loss_fn.log_vars) == 3

    def test_forward_shape(self):
        """Test forward pass produces scalar loss"""
        task_names = ['task1', 'task2', 'task3']
        loss_fn = UncertaintyWeightedLoss(task_names=task_names)

        losses = {
            'task1': torch.tensor(1.0),
            'task2': torch.tensor(2.0),
            'task3': torch.tensor(0.5)
        }

        total_loss, weighted_losses = loss_fn(losses)

        # Should be scalar
        assert total_loss.shape == torch.Size([])
        assert total_loss.item() > 0

        # Check weighted losses returned
        assert len(weighted_losses) == 3

    def test_learnable_weights(self):
        """Test weights are learnable"""
        task_names = ['task1', 'task2']
        loss_fn = UncertaintyWeightedLoss(task_names=task_names)

        # Check log_vars require grad
        for log_var in loss_fn.log_vars:
            assert log_var.requires_grad

    def test_backward_pass(self):
        """Test backward pass updates log_vars"""
        task_names = ['task1', 'task2']
        loss_fn = UncertaintyWeightedLoss(task_names=task_names)

        losses = {
            'task1': torch.tensor(1.0, requires_grad=True),
            'task2': torch.tensor(2.0, requires_grad=True)
        }

        total_loss, _ = loss_fn(losses)
        total_loss.backward()

        # Check log_vars have gradients
        for log_var in loss_fn.log_vars:
            assert log_var.grad is not None


class TestGradNormLoss:
    """Test GradNormLoss"""

    def test_init(self):
        """Test initialization"""
        task_names = ['task1', 'task2', 'task3']
        loss_fn = GradNormLoss(task_names=task_names, alpha=1.5)

        assert len(loss_fn.weights) == 3
        assert loss_fn.alpha == 1.5

    def test_forward_shape(self):
        """Test forward pass produces scalar loss"""
        task_names = ['task1', 'task2']
        loss_fn = GradNormLoss(task_names=task_names)

        losses = {
            'task1': torch.tensor(1.0),
            'task2': torch.tensor(2.0)
        }

        total_loss, weighted_losses = loss_fn(losses)

        # Should be scalar
        assert total_loss.shape == torch.Size([])
        assert total_loss.item() > 0

    def test_learnable_weights(self):
        """Test weights are learnable"""
        task_names = ['task1', 'task2']
        loss_fn = GradNormLoss(task_names=task_names)

        # Check weights require grad
        for weight in loss_fn.weights:
            assert weight.requires_grad


class TestMultiTaskLossManager:
    """Test MultiTaskLossManager"""

    def test_init_uncertainty(self):
        """Test initialization with uncertainty weighting"""
        task_names = ['decoder', 'encoder', 'contrastive']
        manager = MultiTaskLossManager(
            task_names=task_names,
            method='uncertainty'
        )

        assert manager.method == 'uncertainty'
        assert manager.loss_fn is not None

    def test_init_gradnorm(self):
        """Test initialization with gradnorm"""
        task_names = ['decoder', 'encoder', 'contrastive']
        manager = MultiTaskLossManager(
            task_names=task_names,
            method='gradnorm'
        )

        assert manager.method == 'gradnorm'

    def test_init_manual(self):
        """Test initialization with manual weights"""
        task_names = ['decoder', 'encoder', 'contrastive']
        manual_weights = {'decoder': 1.0, 'encoder': 1.0, 'contrastive': 0.5}

        manager = MultiTaskLossManager(
            task_names=task_names,
            method='manual',
            manual_weights=manual_weights
        )

        assert manager.method == 'manual'
        assert manager.manual_weights == manual_weights

    def test_compute_loss_manual(self):
        """Test compute_loss with manual weights"""
        task_names = ['task1', 'task2']
        manual_weights = {'task1': 1.0, 'task2': 0.5}

        manager = MultiTaskLossManager(
            task_names=task_names,
            method='manual',
            manual_weights=manual_weights
        )

        losses = {
            'task1': torch.tensor(2.0),
            'task2': torch.tensor(4.0)
        }

        total_loss, loss_dict = manager.compute_loss(losses)

        # Check weighted sum: 1.0 * 2.0 + 0.5 * 4.0 = 4.0
        assert torch.isclose(total_loss, torch.tensor(4.0))

    def test_compute_loss_uncertainty(self):
        """Test compute_loss with uncertainty weighting"""
        task_names = ['task1', 'task2']

        manager = MultiTaskLossManager(
            task_names=task_names,
            method='uncertainty'
        )

        losses = {
            'task1': torch.tensor(1.0),
            'task2': torch.tensor(2.0)
        }

        total_loss, loss_dict = manager.compute_loss(losses)

        # Should produce scalar loss
        assert total_loss.shape == torch.Size([])
        assert total_loss.item() > 0

    def test_get_task_weights(self):
        """Test get_task_weights method"""
        task_names = ['task1', 'task2']
        manual_weights = {'task1': 1.0, 'task2': 0.5}

        manager = MultiTaskLossManager(
            task_names=task_names,
            method='manual',
            manual_weights=manual_weights
        )

        weights = manager.get_task_weights()

        assert len(weights) == 2
        assert weights['task1'] == 1.0
        assert weights['task2'] == 0.5


class TestLossIntegration:
    """Integration tests for loss functions"""

    def test_full_loss_computation(self):
        """Test computing all losses together"""
        batch_size = 8
        dim = 128
        n_domains = 3

        # Create dummy outputs
        neural_emb = torch.randn(batch_size, dim)
        behavior_emb = torch.randn(batch_size, dim)
        stimulus_emb = torch.randn(batch_size, dim)
        timestamps = torch.arange(batch_size).float()

        decoder_output = torch.randn(batch_size, 10)
        decoder_target = torch.randn(batch_size, 10)

        encoder_output = torch.randn(batch_size, 384)
        encoder_target = torch.randn(batch_size, 384)

        domain_logits = torch.randn(batch_size, n_domains)
        domain_labels = torch.randint(0, n_domains, (batch_size,))

        # Compute losses
        contrastive_loss_fn = TriModalContrastiveLoss()
        contrastive_loss, _ = contrastive_loss_fn(
            neural_emb, behavior_emb, stimulus_emb, timestamps
        )

        decoder_loss = F.mse_loss(decoder_output, decoder_target)
        encoder_loss = F.mse_loss(encoder_output, encoder_target)

        domain_loss_fn = DomainAdversarialLoss()
        domain_loss = domain_loss_fn(domain_logits, domain_labels)

        # Combine with MultiTaskLossManager
        task_names = ['decoder', 'encoder', 'contrastive', 'domain']
        manager = MultiTaskLossManager(
            task_names=task_names,
            method='uncertainty'
        )

        losses = {
            'decoder': decoder_loss,
            'encoder': encoder_loss,
            'contrastive': contrastive_loss,
            'domain': domain_loss
        }

        total_loss, loss_dict = manager.compute_loss(losses)

        # Check total loss is valid
        assert total_loss.shape == torch.Size([])
        assert total_loss.item() > 0
        assert not torch.isnan(total_loss)

        # Test backward
        total_loss.backward()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
