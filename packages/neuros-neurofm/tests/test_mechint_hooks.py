"""
Tests for Mechanistic Interpretability Hooks

Tests the training/evaluation hook system for automatic mech-int integration.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import json

from neuros_mechint.hooks import (
    MechIntConfig,
    ActivationSampler,
    MechIntHooks,
    EvalMechIntRunner,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(64, 128)
            self.layer2 = nn.Linear(128, 64)
            self.layer3 = nn.Linear(64, 32)

        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            x = self.layer3(x)
            return x

    return SimpleModel()


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Test MechIntConfig
# ============================================================================

def test_mechint_config_defaults():
    """Test default configuration."""
    config = MechIntConfig()

    assert config.save_hidden_every_n_steps == 200
    assert config.storage_backend == 'local'
    assert config.max_activations_per_shard == 10000
    assert config.verbose is True
    assert len(config.sample_layers) > 0
    assert len(config.analyses_to_run) > 0


def test_mechint_config_custom():
    """Test custom configuration."""
    config = MechIntConfig(
        sample_layers=['layer1', 'layer2'],
        save_hidden_every_n_steps=100,
        analyses_to_run=['sae'],
        storage_backend='s3',
        s3_bucket='test-bucket'
    )

    assert config.sample_layers == ['layer1', 'layer2']
    assert config.save_hidden_every_n_steps == 100
    assert config.analyses_to_run == ['sae']
    assert config.storage_backend == 's3'
    assert config.s3_bucket == 'test-bucket'


def test_mechint_config_from_dict():
    """Test creating config from dictionary."""
    config_dict = {
        'sample_layers': ['layer1'],
        'save_hidden_every_n_steps': 50,
        'verbose': False
    }

    config = MechIntConfig(**config_dict)

    assert config.sample_layers == ['layer1']
    assert config.save_hidden_every_n_steps == 50
    assert config.verbose is False


# ============================================================================
# Test ActivationSampler
# ============================================================================

def test_activation_sampler_init(temp_dir):
    """Test ActivationSampler initialization."""
    sampler = ActivationSampler(
        layers=['layer1', 'layer2'],
        save_dir=str(temp_dir)
    )

    assert sampler.layers == ['layer1', 'layer2']
    assert sampler.save_dir == temp_dir
    assert len(sampler.activations) == 0
    assert len(sampler.hooks) == 0
    assert sampler.sample_count == 0


def test_activation_sampler_register_hooks(simple_model, temp_dir):
    """Test registering hooks on model."""
    sampler = ActivationSampler(
        layers=['layer1', 'layer2'],
        save_dir=str(temp_dir)
    )

    registered = sampler.register_hooks(simple_model)

    assert len(registered) == 2
    assert 'layer1' in registered
    assert 'layer2' in registered
    assert len(sampler.hooks) == 2


def test_activation_sampler_capture(simple_model, temp_dir):
    """Test capturing activations during forward pass."""
    sampler = ActivationSampler(
        layers=['layer1'],
        save_dir=str(temp_dir)
    )

    sampler.register_hooks(simple_model)

    # Run forward pass
    x = torch.randn(4, 64)
    _ = simple_model(x)

    # Check activations were captured
    assert 'layer1' in sampler.activations
    assert len(sampler.activations['layer1']) == 1
    assert sampler.activations['layer1'][0].shape == (4, 128)


def test_activation_sampler_save(simple_model, temp_dir):
    """Test saving activations to disk."""
    sampler = ActivationSampler(
        layers=['layer1', 'layer2'],
        save_dir=str(temp_dir)
    )

    sampler.register_hooks(simple_model)

    # Run forward passes
    for _ in range(3):
        x = torch.randn(4, 64)
        _ = simple_model(x)

    # Save activations
    save_path = sampler.save_activations(global_step=100)

    assert save_path is not None
    assert Path(save_path).exists()

    # Load and verify
    data = torch.load(save_path)
    assert data['global_step'] == 100
    assert data['shard_id'] == 0
    assert 'layer1' in data['activations']
    assert 'layer2' in data['activations']
    assert data['activations']['layer1'].shape[0] == 12  # 3 batches * 4 samples


def test_activation_sampler_clear_cache(simple_model, temp_dir):
    """Test clearing activation cache."""
    sampler = ActivationSampler(
        layers=['layer1'],
        save_dir=str(temp_dir)
    )

    sampler.register_hooks(simple_model)

    # Capture activations
    x = torch.randn(4, 64)
    _ = simple_model(x)

    assert len(sampler.activations['layer1']) == 1

    # Clear cache
    sampler.clear_cache()

    assert len(sampler.activations) == 0


def test_activation_sampler_statistics(simple_model, temp_dir):
    """Test getting sampler statistics."""
    sampler = ActivationSampler(
        layers=['layer1'],
        save_dir=str(temp_dir)
    )

    sampler.register_hooks(simple_model)

    # Run forward passes
    for _ in range(2):
        x = torch.randn(4, 64)
        _ = simple_model(x)

    stats = sampler.get_statistics()

    assert stats['current_cache_samples'] == 8  # 2 batches * 4 samples
    assert stats['total_samples_saved'] == 0  # Nothing saved yet
    assert stats['total_shards'] == 0
    assert 'layer1' in stats['tracked_layers']


def test_activation_sampler_clear_hooks(simple_model, temp_dir):
    """Test removing hooks."""
    sampler = ActivationSampler(
        layers=['layer1'],
        save_dir=str(temp_dir)
    )

    sampler.register_hooks(simple_model)
    assert len(sampler.hooks) == 1

    sampler.clear_hooks()
    assert len(sampler.hooks) == 0

    # Forward pass should not capture activations
    x = torch.randn(4, 64)
    _ = simple_model(x)

    assert len(sampler.activations) == 0


# ============================================================================
# Test MechIntHooks
# ============================================================================

def test_mechint_hooks_init():
    """Test MechIntHooks initialization."""
    config = MechIntConfig(
        sample_layers=['layer1'],
        storage_backend='local'
    )

    hooks = MechIntHooks(config)

    assert hooks.config == config
    assert hooks.sampler is None
    assert len(hooks.saved_shards) == 0


def test_mechint_hooks_init_from_dict():
    """Test initialization from dictionary."""
    config_dict = {
        'sample_layers': ['layer1'],
        'save_hidden_every_n_steps': 100
    }

    hooks = MechIntHooks(config_dict)

    assert hooks.config.sample_layers == ['layer1']
    assert hooks.config.save_hidden_every_n_steps == 100


def test_mechint_hooks_register(simple_model, temp_dir):
    """Test registering hooks on model."""
    config = MechIntConfig(
        sample_layers=['layer1', 'layer2'],
        storage_path=str(temp_dir)
    )

    hooks = MechIntHooks(config)
    hooks.register_hooks(simple_model)

    assert hooks.sampler is not None
    assert len(hooks.sampler.hooks) == 2


def test_mechint_hooks_training_step(simple_model, temp_dir):
    """Test on_training_step callback."""
    config = MechIntConfig(
        sample_layers=['layer1'],
        save_hidden_every_n_steps=10,
        storage_path=str(temp_dir),
        verbose=False
    )

    hooks = MechIntHooks(config)
    hooks.register_hooks(simple_model)

    # Mock trainer
    class MockTrainer:
        current_epoch = 0

    trainer = MockTrainer()

    # Run forward passes
    for step in range(15):
        x = torch.randn(4, 64)
        outputs = simple_model(x)

        hooks.on_training_step(
            trainer=trainer,
            pl_module=simple_model,
            outputs=outputs,
            batch={'x': x},
            batch_idx=step,
            global_step=step
        )

    # Should have saved at steps 0 and 10
    assert len(hooks.saved_shards) >= 1


def test_mechint_hooks_manifest(simple_model, temp_dir):
    """Test manifest generation at training end."""
    config = MechIntConfig(
        sample_layers=['layer1'],
        storage_path=str(temp_dir),
        verbose=False
    )

    hooks = MechIntHooks(config)
    hooks.register_hooks(simple_model)

    # Mock trainer
    class MockTrainer:
        current_epoch = 0

    trainer = MockTrainer()

    # Trigger training end
    hooks.on_train_end(trainer, simple_model)

    # Check manifest was created
    manifest_path = temp_dir / 'manifest.json'
    assert manifest_path.exists()

    # Load and verify
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    assert 'config' in manifest
    assert 'shards' in manifest
    assert 'total_shards' in manifest


# ============================================================================
# Test EvalMechIntRunner
# ============================================================================

def test_eval_runner_init(simple_model):
    """Test EvalMechIntRunner initialization."""
    config = MechIntConfig(
        sample_layers=['layer1'],
        analyses_to_run=['neuron']
    )

    runner = EvalMechIntRunner(
        model=simple_model,
        config=config,
        device='cpu'
    )

    assert runner.model is simple_model
    assert runner.config == config
    assert runner.device == 'cpu'
    assert len(runner.results) == 0


def test_eval_runner_load_activations(simple_model, temp_dir):
    """Test loading activation shards."""
    # Create some dummy shards
    for i in range(3):
        shard_data = {
            'global_step': i * 100,
            'shard_id': i,
            'activations': {
                'layer1': torch.randn(10, 128),
                'layer2': torch.randn(10, 64)
            },
            'metadata': {}
        }

        shard_path = temp_dir / f'activations_shard_{i:06d}_step_{i*100}.pt'
        torch.save(shard_data, shard_path)

    # Load activations
    config = MechIntConfig(storage_path=str(temp_dir))
    runner = EvalMechIntRunner(simple_model, config, device='cpu')

    activations = runner._load_activations(str(temp_dir))

    assert 'layer1' in activations
    assert 'layer2' in activations
    assert activations['layer1'].shape[0] == 30  # 3 shards * 10 samples
    assert activations['layer2'].shape[0] == 30


def test_eval_runner_neuron_analysis(simple_model, temp_dir):
    """Test running neuron analysis."""
    # Create dummy activations
    activations = {
        'layer1': torch.randn(100, 128)
    }

    config = MechIntConfig(analyses_to_run=['neuron'])
    runner = EvalMechIntRunner(simple_model, config, device='cpu')

    results = runner._run_neuron_analysis(activations)

    assert 'layer1' in results
    assert 'mean_activation' in results['layer1']
    assert 'max_activation' in results['layer1']
    assert 'sparsity' in results['layer1']
    assert len(results['layer1']['mean_activation']) == 128


def test_eval_runner_feature_analysis(simple_model, temp_dir):
    """Test running feature analysis."""
    # Create dummy activations
    activations = {
        'layer1': torch.randn(100, 128)
    }

    config = MechIntConfig(analyses_to_run=['feature'])
    runner = EvalMechIntRunner(simple_model, config, device='cpu')

    results = runner._run_feature_analysis(activations)

    assert 'layer1' in results
    assert 'explained_variance' in results['layer1']
    assert 'n_components' in results['layer1']


def test_eval_runner_export_results(simple_model, temp_dir):
    """Test exporting results."""
    config = MechIntConfig(
        storage_path=str(temp_dir),
        analyses_to_run=['neuron']
    )
    runner = EvalMechIntRunner(simple_model, config, device='cpu')

    # Set dummy results
    runner.results = {
        'config': config.__dict__,
        'analyses': {
            'neuron': {'layer1': {'mean': [1.0, 2.0, 3.0]}}
        }
    }

    output_dir = temp_dir / 'results'
    runner.export_results(str(output_dir))

    # Check files were created
    assert (output_dir / 'mechint_results.json').exists()
    assert (output_dir / 'mechint_report.md').exists()


# ============================================================================
# Test PyTorch Lightning Integration
# ============================================================================

@pytest.mark.skipif(
    not __import__('importlib').util.find_spec('pytorch_lightning'),
    reason="pytorch-lightning not installed"
)
def test_lightning_callback_import():
    """Test importing Lightning callback."""
    from neuros_mechint.hooks import MechIntCallback

    config = MechIntConfig(sample_layers=['layer1'])
    callback = MechIntCallback(config)

    assert callback.hooks is not None
    assert callback.hooks.config == config


# ============================================================================
# Test FastAPI Integration
# ============================================================================

@pytest.mark.skipif(
    not __import__('importlib').util.find_spec('fastapi'),
    reason="fastapi not installed"
)
def test_fastapi_integration_import():
    """Test importing FastAPI integration."""
    from neuros_mechint.hooks import FastAPIIntegrationMixin

    config = MechIntConfig(sample_layers=['layer1'])

    class DummyModel(nn.Module):
        def forward(self, x):
            return x

    model = DummyModel()
    mixin = FastAPIIntegrationMixin(model, config, device='cpu')

    assert mixin.model is model
    assert mixin.config == config
    assert mixin.runner is not None


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_workflow(simple_model, temp_dir):
    """Test complete workflow from training to evaluation."""
    # 1. Setup hooks
    config = MechIntConfig(
        sample_layers=['layer1', 'layer2'],
        save_hidden_every_n_steps=5,
        storage_path=str(temp_dir),
        analyses_to_run=['neuron'],
        verbose=False
    )

    hooks = MechIntHooks(config)
    hooks.register_hooks(simple_model)

    # 2. Mock training
    class MockTrainer:
        current_epoch = 0

    trainer = MockTrainer()

    for step in range(10):
        x = torch.randn(4, 64)
        outputs = simple_model(x)

        hooks.on_training_step(
            trainer=trainer,
            pl_module=simple_model,
            outputs=outputs,
            batch={'x': x},
            batch_idx=step,
            global_step=step
        )

    hooks.on_train_end(trainer, simple_model)

    # 3. Run evaluation
    runner = EvalMechIntRunner(simple_model, config, device='cpu')

    results = runner.run_mechint_eval(
        hidden_shards_path=str(temp_dir)
    )

    # 4. Verify results
    assert 'config' in results
    assert 'analyses' in results
    assert 'neuron' in results['analyses']

    # 5. Export results
    output_dir = temp_dir / 'results'
    runner.export_results(str(output_dir))

    assert (output_dir / 'mechint_results.json').exists()
    assert (output_dir / 'mechint_report.md').exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
