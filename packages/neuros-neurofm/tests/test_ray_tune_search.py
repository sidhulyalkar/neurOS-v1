"""
Tests for Ray Tune hyperparameter search module.

Tests the NeuroFMXRayTuner and related components.
"""

import pytest
import tempfile
from pathlib import Path
import yaml
import json

try:
    import ray
    from ray import tune
    from ray.air import session
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from neuros_neurofm.optimization import (
    NeuroFMXRayTuner,
    NeuroFMXSearchSpace,
    RAY_TUNE_AVAILABLE,
)


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")
class TestNeuroFMXSearchSpace:
    """Test search space definitions."""

    def test_default_search_space(self):
        """Test default search space creation."""
        space = NeuroFMXSearchSpace.get_default_search_space()

        assert 'model' in space
        assert 'training' in space
        assert 'losses' in space
        assert 'multi_rate' in space
        assert 'optimizer' in space

        # Check model parameters
        assert 'd_model' in space['model']
        assert 'n_layers' in space['model']
        assert 'dropout' in space['model']

        # Check training parameters
        assert 'lr' in space['training']
        assert 'batch_size' in space['training']

    def test_small_search_space(self):
        """Test small search space."""
        space = NeuroFMXSearchSpace.get_small_search_space()

        assert 'model' in space
        assert 'training' in space
        assert 'losses' in space

        # Should have fewer options than default
        assert len(space) <= len(NeuroFMXSearchSpace.get_default_search_space())

    def test_architecture_search_space(self):
        """Test architecture search space."""
        space = NeuroFMXSearchSpace.get_architecture_search_space()

        assert 'model' in space
        assert 'multi_rate' in space

        # Focus on architectural parameters
        assert 'd_model' in space['model']
        assert 'n_layers' in space['model']
        assert 'latent_dim' in space['model']


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")
class TestNeuroFMXRayTuner:
    """Test Ray Tune hyperparameter tuner."""

    @pytest.fixture
    def simple_train_fn(self):
        """Create a simple training function for testing."""
        def train_fn(config):
            """Simple training function that reports dummy metrics."""
            for epoch in range(5):
                # Simulate training
                import time
                time.sleep(0.1)

                # Report metrics
                session.report({
                    'training_iteration': epoch + 1,
                    'val_loss': 1.0 - (epoch / 10.0),  # Decreasing loss
                    'val_r2': epoch / 10.0,  # Increasing R²
                })

        return train_fn

    @pytest.fixture
    def small_search_space(self):
        """Create a minimal search space for testing."""
        return {
            'model': {
                'd_model': tune.choice([256, 512]),
                'dropout': tune.uniform(0.1, 0.2),
            },
            'training': {
                'lr': tune.loguniform(1e-4, 1e-3),
                'batch_size': tune.choice([16, 32]),
            },
        }

    def test_tuner_initialization(self, simple_train_fn):
        """Test tuner initialization."""
        tuner = NeuroFMXRayTuner(
            train_fn=simple_train_fn,
            search_algorithm='asha',
            num_samples=2,
            max_concurrent_trials=1,
            resources_per_trial={'cpu': 1, 'gpu': 0},
            verbose=0,
        )

        assert tuner.train_fn is simple_train_fn
        assert tuner.search_algorithm_name == 'asha'
        assert tuner.num_samples == 2
        assert tuner.metric == 'val_loss'
        assert tuner.mode == 'min'

    def test_flatten_search_space(self, simple_train_fn, small_search_space):
        """Test search space flattening."""
        tuner = NeuroFMXRayTuner(
            train_fn=simple_train_fn,
            search_space=small_search_space,
            verbose=0,
        )

        flat = tuner.flattened_search_space

        assert 'model.d_model' in flat
        assert 'model.dropout' in flat
        assert 'training.lr' in flat
        assert 'training.batch_size' in flat

    def test_unflatten_config(self, simple_train_fn):
        """Test config unflattening."""
        tuner = NeuroFMXRayTuner(
            train_fn=simple_train_fn,
            verbose=0,
        )

        flat_config = {
            'model.d_model': 512,
            'model.dropout': 0.1,
            'training.lr': 1e-4,
        }

        nested = tuner._unflatten_config(flat_config)

        assert nested['model']['d_model'] == 512
        assert nested['model']['dropout'] == 0.1
        assert nested['training']['lr'] == 1e-4

    def test_scheduler_creation_asha(self, simple_train_fn):
        """Test ASHA scheduler creation."""
        tuner = NeuroFMXRayTuner(
            train_fn=simple_train_fn,
            search_algorithm='asha',
            grace_period=5,
            reduction_factor=3,
            verbose=0,
        )

        from ray.tune.schedulers import ASHAScheduler
        assert isinstance(tuner.scheduler, ASHAScheduler)

    def test_scheduler_creation_pbt(self, simple_train_fn):
        """Test PBT scheduler creation."""
        tuner = NeuroFMXRayTuner(
            train_fn=simple_train_fn,
            search_algorithm='pbt',
            verbose=0,
        )

        from ray.tune.schedulers import PopulationBasedTraining
        assert isinstance(tuner.scheduler, PopulationBasedTraining)

    @pytest.mark.slow
    def test_run_tuning(self, simple_train_fn, small_search_space, tmp_path):
        """Test running hyperparameter tuning."""
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(num_cpus=2, ignore_reinit_error=True)

        tuner = NeuroFMXRayTuner(
            train_fn=simple_train_fn,
            search_algorithm='random',
            search_space=small_search_space,
            num_samples=2,
            max_concurrent_trials=1,
            resources_per_trial={'cpu': 1, 'gpu': 0},
            max_epochs=5,
            local_dir=str(tmp_path),
            experiment_name='test_experiment',
            use_mlflow=False,
            use_wandb=False,
            verbose=0,
        )

        results = tuner.run()

        assert results is not None
        assert tuner.best_config is not None
        assert tuner.best_result is not None

        # Shutdown Ray
        ray.shutdown()

    @pytest.mark.slow
    def test_export_best_config(self, simple_train_fn, small_search_space, tmp_path):
        """Test exporting best configuration."""
        if not ray.is_initialized():
            ray.init(num_cpus=2, ignore_reinit_error=True)

        tuner = NeuroFMXRayTuner(
            train_fn=simple_train_fn,
            search_space=small_search_space,
            num_samples=2,
            max_concurrent_trials=1,
            resources_per_trial={'cpu': 1, 'gpu': 0},
            local_dir=str(tmp_path),
            use_mlflow=False,
            use_wandb=False,
            verbose=0,
        )

        results = tuner.run()

        # Export YAML
        yaml_path = tmp_path / 'best_config.yaml'
        tuner.export_best_config(str(yaml_path), format='yaml')
        assert yaml_path.exists()

        # Load and verify
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        assert 'metadata' in config
        assert 'hyperparameters' in config
        assert 'best_metric' in config['metadata']
        assert 'best_value' in config['metadata']

        # Export JSON
        json_path = tmp_path / 'best_config.json'
        tuner.export_best_config(str(json_path), format='json')
        assert json_path.exists()

        with open(json_path, 'r') as f:
            config = json.load(f)

        assert 'metadata' in config
        assert 'hyperparameters' in config

        ray.shutdown()

    def test_invalid_search_algorithm(self, simple_train_fn):
        """Test handling of invalid search algorithm."""
        # Should not raise, just fall back to FIFO
        tuner = NeuroFMXRayTuner(
            train_fn=simple_train_fn,
            search_algorithm='invalid_algorithm',
            verbose=0,
        )

        from ray.tune.schedulers import FIFOScheduler
        assert isinstance(tuner.scheduler, FIFOScheduler)

    def test_custom_metric_mode(self, simple_train_fn):
        """Test custom metric and mode."""
        tuner = NeuroFMXRayTuner(
            train_fn=simple_train_fn,
            metric='val_r2',
            mode='max',
            verbose=0,
        )

        assert tuner.metric == 'val_r2'
        assert tuner.mode == 'max'


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")
class TestTrainFunctionCreation:
    """Test training function creation utilities."""

    def test_create_neurofmx_train_fn(self):
        """Test creating NeuroFMX training function."""
        from neuros_neurofm.optimization.ray_tune_search import create_neurofmx_train_fn

        data_config = {'data_path': './data', 'batch_size': 32}
        base_config = {'checkpoint_dir': './checkpoints'}

        train_fn = create_neurofmx_train_fn(
            data_config=data_config,
            base_config=base_config,
            num_epochs=10,
        )

        assert callable(train_fn)


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")
class TestIntegration:
    """Integration tests."""

    @pytest.mark.slow
    def test_full_pipeline(self, tmp_path):
        """Test complete hyperparameter tuning pipeline."""
        if not ray.is_initialized():
            ray.init(num_cpus=2, ignore_reinit_error=True)

        # Define simple training function
        def train_fn(config):
            for epoch in range(3):
                import time
                time.sleep(0.05)

                # Simulate decreasing loss based on hyperparameters
                lr = config['training']['lr']
                loss = 1.0 / (1.0 + epoch * lr * 10)

                session.report({
                    'training_iteration': epoch + 1,
                    'val_loss': loss,
                    'val_r2': 1.0 - loss,
                })

        # Small search space
        search_space = {
            'model': {
                'd_model': tune.choice([256, 512]),
            },
            'training': {
                'lr': tune.uniform(0.001, 0.01),
                'batch_size': tune.choice([16, 32]),
            },
        }

        # Create tuner
        tuner = NeuroFMXRayTuner(
            train_fn=train_fn,
            search_algorithm='asha',
            search_space=search_space,
            num_samples=4,
            max_concurrent_trials=2,
            resources_per_trial={'cpu': 1, 'gpu': 0},
            max_epochs=3,
            grace_period=2,
            local_dir=str(tmp_path),
            experiment_name='integration_test',
            use_mlflow=False,
            use_wandb=False,
            verbose=0,
        )

        # Run tuning
        results = tuner.run()

        # Verify results
        assert tuner.best_config is not None
        assert tuner.best_result is not None
        assert 'model' in tuner.best_config
        assert 'training' in tuner.best_config

        # Export config
        config_path = tmp_path / 'best_config.yaml'
        tuner.export_best_config(str(config_path))
        assert config_path.exists()

        # Verify best checkpoint path
        checkpoint_path = tuner.get_best_checkpoint_path()
        # May be None if checkpointing not enabled in simple test

        ray.shutdown()


# Utility function for running tests
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
