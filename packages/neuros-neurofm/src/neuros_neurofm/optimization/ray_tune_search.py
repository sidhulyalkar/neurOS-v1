"""
Ray Tune Hyperparameter Search for NeuroFM-X.

This module provides a production-ready Ray Tune-based hyperparameter optimization
system for NeuroFMX, supporting multiple search strategies, distributed training,
and experiment tracking integration.

Features:
- Multiple search algorithms: ASHA, PBT, Bayesian Optimization (OptunaSearch)
- Comprehensive search spaces for all NeuroFMX hyperparameters
- Integration with MLflow and Weights & Biases
- Ray Train integration for distributed training
- Checkpoint management and resumption
- Export best hyperparameters to YAML config
- Population-Based Training with custom perturbation strategies
- Early stopping and resource-aware scheduling

Example:
    >>> from neuros_neurofm.optimization.ray_tune_search import NeuroFMXRayTuner
    >>> tuner = NeuroFMXRayTuner(
    ...     train_fn=my_train_function,
    ...     search_algorithm='asha',
    ...     num_samples=100,
    ...     resources_per_trial={'cpu': 4, 'gpu': 1}
    ... )
    >>> results = tuner.run()
    >>> tuner.export_best_config('best_config.yaml')
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining, FIFOScheduler
    from ray.tune.search import ConcurrencyLimiter
    from ray.tune.search.optuna import OptunaSearch
    from ray.air import session
    from ray.air.config import ScalingConfig, RunConfig, CheckpointConfig
    from ray.train.torch import TorchTrainer
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


logger = logging.getLogger(__name__)


class NeuroFMXSearchSpace:
    """
    Comprehensive search space definitions for NeuroFMX hyperparameters.

    Provides pre-defined search spaces for:
    - Model architecture (d_model, n_layers, etc.)
    - Training parameters (lr, batch_size, etc.)
    - Loss weights and regularization
    - Fusion and aggregation parameters
    """

    @staticmethod
    def get_default_search_space() -> Dict[str, Any]:
        """
        Get the default comprehensive search space for NeuroFMX.

        Returns
        -------
        dict
            Ray Tune search space configuration with all hyperparameters.
        """
        return {
            # Model Architecture
            'model': {
                'd_model': tune.choice([256, 512, 768, 1024, 2048]),
                'n_layers': tune.choice([4, 8, 12, 16, 24]),
                'mamba_d_state': tune.choice([16, 32, 64]),
                'mamba_d_conv': tune.choice([2, 4, 8]),
                'mamba_expand': tune.choice([1, 2, 4]),
                'latent_dim': tune.choice([128, 256, 512, 1024]),
                'n_latents': tune.choice([32, 64, 128, 256]),
                'n_perceiver_layers': tune.choice([2, 3, 4, 6]),
                'n_popt_layers': tune.choice([2, 3, 4, 6]),
                'dropout': tune.uniform(0.0, 0.3),
            },

            # Training Parameters
            'training': {
                'lr': tune.loguniform(1e-5, 1e-3),
                'batch_size': tune.choice([16, 32, 64, 128]),
                'weight_decay': tune.loguniform(1e-6, 1e-2),
                'warmup_epochs': tune.choice([5, 10, 15, 20]),
                'gradient_clip_val': tune.uniform(0.5, 2.0),
                'fusion_freq': tune.choice([1, 2, 4, 8]),
            },

            # Multi-rate Configuration
            'multi_rate': {
                'enabled': tune.choice([True, False]),
                'rates': tune.choice([[1, 4, 16], [1, 2, 4], [1, 4, 8, 16]]),
                'fusion_method': tune.choice(['concat', 'add', 'attention']),
            },

            # Loss Weights
            'losses': {
                'mask_ratio': tune.uniform(0.15, 0.75),
                'contrastive_weight': tune.uniform(0.1, 2.0),
                'contrastive_temperature': tune.loguniform(0.01, 0.2),
                'decoder_weight': tune.uniform(0.5, 2.0),
                'encoder_weight': tune.uniform(0.1, 1.0),
                'diffusion_weight': tune.uniform(0.1, 0.5),
            },

            # Optimizer Configuration
            'optimizer': {
                'betas': tune.choice([
                    (0.9, 0.95),
                    (0.9, 0.98),
                    (0.9, 0.999),
                ]),
                'eps': tune.loguniform(1e-9, 1e-7),
            },
        }

    @staticmethod
    def get_small_search_space() -> Dict[str, Any]:
        """
        Get a smaller search space for quick experimentation.

        Focuses on the most impactful hyperparameters.
        """
        return {
            'model': {
                'd_model': tune.choice([256, 512, 768]),
                'n_layers': tune.choice([8, 12, 16]),
                'latent_dim': tune.choice([256, 512]),
                'dropout': tune.uniform(0.05, 0.2),
            },
            'training': {
                'lr': tune.loguniform(1e-4, 5e-4),
                'batch_size': tune.choice([32, 64]),
                'fusion_freq': tune.choice([1, 2, 4]),
            },
            'losses': {
                'mask_ratio': tune.uniform(0.3, 0.6),
                'contrastive_weight': tune.uniform(0.2, 1.0),
            },
        }

    @staticmethod
    def get_architecture_search_space() -> Dict[str, Any]:
        """
        Search space focused on architectural hyperparameters.

        Useful for neural architecture search (NAS) style experiments.
        """
        return {
            'model': {
                'd_model': tune.choice([512, 768, 1024, 2048]),
                'n_layers': tune.choice([4, 8, 12, 16, 24, 32]),
                'mamba_d_state': tune.choice([16, 32, 64, 128]),
                'latent_dim': tune.choice([256, 512, 1024]),
                'n_latents': tune.choice([64, 128, 256, 512]),
                'n_perceiver_layers': tune.choice([2, 3, 4, 6, 8]),
                'n_popt_layers': tune.choice([2, 3, 4, 6, 8]),
            },
            'multi_rate': {
                'enabled': tune.choice([True, False]),
                'fusion_method': tune.choice(['concat', 'add', 'attention']),
            },
        }


class NeuroFMXRayTuner:
    """
    Ray Tune-based hyperparameter optimization for NeuroFMX.

    Supports multiple search algorithms and schedulers, with seamless integration
    with experiment tracking tools (MLflow, W&B) and distributed training.

    Parameters
    ----------
    train_fn : callable
        Training function that accepts a config dict and reports metrics.
        Should call session.report(metrics) to report intermediate results.
    search_algorithm : str, optional
        Search algorithm to use: 'asha', 'pbt', 'bayesian', 'random', 'grid'.
        Default: 'asha'.
    num_samples : int, optional
        Number of hyperparameter configurations to try.
        Default: 100.
    max_concurrent_trials : int, optional
        Maximum number of concurrent trials.
        Default: 4.
    resources_per_trial : dict, optional
        Resources to allocate per trial (e.g., {'cpu': 4, 'gpu': 1}).
        Default: {'cpu': 4, 'gpu': 0}.
    metric : str, optional
        Metric to optimize.
        Default: 'val_loss'.
    mode : str, optional
        Optimization mode: 'min' or 'max'.
        Default: 'min'.
    search_space : dict, optional
        Custom search space. If None, uses default.
    max_epochs : int, optional
        Maximum training epochs per trial.
        Default: 100.
    grace_period : int, optional
        Minimum number of epochs before early stopping (for ASHA).
        Default: 10.
    reduction_factor : int, optional
        Reduction factor for ASHA scheduler.
        Default: 3.
    experiment_name : str, optional
        Name for the experiment.
        Default: 'neurofmx_tuning'.
    local_dir : str, optional
        Directory to save results.
        Default: './ray_results'.
    use_mlflow : bool, optional
        Enable MLflow tracking.
        Default: True.
    use_wandb : bool, optional
        Enable Weights & Biases tracking.
        Default: False.
    wandb_project : str, optional
        W&B project name.
        Default: 'neurofmx'.
    checkpoint_freq : int, optional
        Checkpoint frequency (in epochs).
        Default: 10.
    keep_checkpoints_num : int, optional
        Number of checkpoints to keep.
        Default: 3.
    resume : bool, optional
        Resume from previous run.
        Default: False.
    """

    def __init__(
        self,
        train_fn: Callable,
        search_algorithm: str = 'asha',
        num_samples: int = 100,
        max_concurrent_trials: int = 4,
        resources_per_trial: Optional[Dict[str, float]] = None,
        metric: str = 'val_loss',
        mode: str = 'min',
        search_space: Optional[Dict[str, Any]] = None,
        max_epochs: int = 100,
        grace_period: int = 10,
        reduction_factor: int = 3,
        experiment_name: str = 'neurofmx_tuning',
        local_dir: str = './ray_results',
        use_mlflow: bool = True,
        use_wandb: bool = False,
        wandb_project: str = 'neurofmx',
        checkpoint_freq: int = 10,
        keep_checkpoints_num: int = 3,
        resume: bool = False,
        verbose: int = 1,
    ):
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray Tune is required for NeuroFMXRayTuner. "
                "Install with: pip install ray[tune] ray[train]"
            )

        self.train_fn = train_fn
        self.search_algorithm_name = search_algorithm.lower()
        self.num_samples = num_samples
        self.max_concurrent_trials = max_concurrent_trials
        self.resources_per_trial = resources_per_trial or {'cpu': 4, 'gpu': 0}
        self.metric = metric
        self.mode = mode
        self.max_epochs = max_epochs
        self.grace_period = grace_period
        self.reduction_factor = reduction_factor
        self.experiment_name = experiment_name
        self.local_dir = Path(local_dir)
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.checkpoint_freq = checkpoint_freq
        self.keep_checkpoints_num = keep_checkpoints_num
        self.resume = resume
        self.verbose = verbose

        # Initialize search space
        if search_space is None:
            self.search_space = NeuroFMXSearchSpace.get_default_search_space()
        else:
            self.search_space = search_space

        # Flatten search space for Ray Tune
        self.flattened_search_space = self._flatten_search_space(self.search_space)

        # Initialize scheduler and search algorithm
        self.scheduler = self._create_scheduler()
        self.search_alg = self._create_search_algorithm()

        # Results storage
        self.results = None
        self.best_config = None
        self.best_result = None

        logger.info(f"Initialized NeuroFMXRayTuner with {search_algorithm} algorithm")
        logger.info(f"Search space: {len(self.flattened_search_space)} hyperparameters")
        logger.info(f"Num samples: {num_samples}, Max concurrent: {max_concurrent_trials}")

    def _flatten_search_space(self, space: Dict, prefix: str = '') -> Dict:
        """Flatten nested search space dictionary for Ray Tune."""
        flat = {}
        for key, value in space.items():
            new_key = f"{prefix}{key}" if prefix else key
            if isinstance(value, dict) and not hasattr(value, 'sample'):
                # Nested dict, recursively flatten
                flat.update(self._flatten_search_space(value, f"{new_key}."))
            else:
                # Tune search space or primitive value
                flat[new_key] = value
        return flat

    def _unflatten_config(self, flat_config: Dict) -> Dict:
        """Unflatten config back to nested structure."""
        nested = {}
        for key, value in flat_config.items():
            parts = key.split('.')
            current = nested
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        return nested

    def _create_scheduler(self):
        """Create the trial scheduler based on search algorithm."""
        if self.search_algorithm_name == 'asha':
            scheduler = ASHAScheduler(
                time_attr='training_iteration',
                metric=self.metric,
                mode=self.mode,
                max_t=self.max_epochs,
                grace_period=self.grace_period,
                reduction_factor=self.reduction_factor,
            )
            logger.info(
                f"Created ASHA scheduler: grace_period={self.grace_period}, "
                f"reduction_factor={self.reduction_factor}"
            )

        elif self.search_algorithm_name == 'pbt':
            # Population-Based Training scheduler
            scheduler = PopulationBasedTraining(
                time_attr='training_iteration',
                metric=self.metric,
                mode=self.mode,
                perturbation_interval=10,  # Perturb every 10 epochs
                hyperparam_mutations={
                    'training.lr': tune.loguniform(1e-5, 1e-3),
                    'training.batch_size': [16, 32, 64],
                    'losses.contrastive_weight': tune.uniform(0.1, 2.0),
                    'model.dropout': tune.uniform(0.0, 0.3),
                },
            )
            logger.info("Created PBT scheduler with custom perturbation strategy")

        else:
            # Default FIFO scheduler
            scheduler = FIFOScheduler()
            logger.info("Using FIFO scheduler (no early stopping)")

        return scheduler

    def _create_search_algorithm(self):
        """Create the search algorithm."""
        search_alg = None

        if self.search_algorithm_name == 'bayesian':
            # Bayesian Optimization using Optuna
            try:
                from ray.tune.search.optuna import OptunaSearch
                search_alg = OptunaSearch(
                    metric=self.metric,
                    mode=self.mode,
                )
                logger.info("Created Bayesian optimization (OptunaSearch)")
            except ImportError:
                logger.warning(
                    "OptunaSearch not available. Install with: pip install optuna"
                )
                search_alg = None

        elif self.search_algorithm_name == 'random':
            # Random search (default in Ray Tune)
            search_alg = None
            logger.info("Using random search")

        elif self.search_algorithm_name == 'grid':
            # Grid search - convert continuous to discrete
            logger.warning(
                "Grid search selected. Consider reducing search space size or "
                "converting continuous parameters to discrete choices."
            )
            search_alg = None

        # Apply concurrency limiter
        if search_alg is not None:
            search_alg = ConcurrencyLimiter(
                search_alg,
                max_concurrent=self.max_concurrent_trials
            )

        return search_alg

    def run(self) -> Any:
        """
        Run hyperparameter optimization.

        Returns
        -------
        ray.tune.ExperimentAnalysis
            Results of the tuning run.
        """
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            logger.info("Initialized Ray")

        # Create trainable wrapper
        trainable = self._create_trainable()

        # Setup experiment tracking callbacks
        callbacks = self._setup_callbacks()

        # Configure checkpointing
        checkpoint_config = CheckpointConfig(
            num_to_keep=self.keep_checkpoints_num,
            checkpoint_frequency=self.checkpoint_freq,
            checkpoint_at_end=True,
        )

        # Configure run
        run_config = RunConfig(
            name=self.experiment_name,
            local_dir=str(self.local_dir),
            checkpoint_config=checkpoint_config,
            callbacks=callbacks,
            verbose=self.verbose,
        )

        # Create reporter for progress display
        reporter = CLIReporter(
            metric_columns=[self.metric, 'training_iteration'],
            max_report_frequency=30,  # seconds
        )

        # Create tuner
        tuner = tune.Tuner(
            trainable,
            param_space=self.flattened_search_space,
            tune_config=tune.TuneConfig(
                metric=self.metric,
                mode=self.mode,
                scheduler=self.scheduler,
                search_alg=self.search_alg,
                num_samples=self.num_samples,
            ),
            run_config=run_config,
        )

        # Run tuning
        logger.info(f"Starting hyperparameter search with {self.num_samples} trials...")
        start_time = datetime.now()

        try:
            self.results = tuner.fit()

            elapsed = datetime.now() - start_time
            logger.info(f"Tuning completed in {elapsed}")

            # Extract best results
            self._extract_best_results()

            # Log summary
            self._log_summary()

            return self.results

        except Exception as e:
            logger.error(f"Tuning failed with error: {e}", exc_info=True)
            raise

    def _create_trainable(self) -> Callable:
        """
        Create a trainable function that wraps the user's training function.

        This wrapper handles:
        - Config unflattening
        - Metric reporting
        - Checkpointing
        """
        def trainable(config):
            # Unflatten config
            nested_config = self._unflatten_config(config)

            # Call user's training function
            # The train_fn should call session.report(metrics) to report results
            self.train_fn(nested_config)

        # Wrap with resources
        return tune.with_resources(
            trainable,
            resources=self.resources_per_trial
        )

    def _setup_callbacks(self) -> List:
        """Setup experiment tracking callbacks."""
        callbacks = []

        # MLflow callback
        if self.use_mlflow:
            try:
                from ray.tune.integration.mlflow import MLflowLoggerCallback
                callbacks.append(
                    MLflowLoggerCallback(
                        tracking_uri=os.getenv('MLFLOW_TRACKING_URI', 'mlruns'),
                        experiment_name=self.experiment_name,
                        save_artifact=True,
                    )
                )
                logger.info("Enabled MLflow tracking")
            except ImportError:
                logger.warning("MLflow callback not available")

        # Weights & Biases callback
        if self.use_wandb:
            try:
                from ray.tune.integration.wandb import WandbLoggerCallback
                callbacks.append(
                    WandbLoggerCallback(
                        project=self.wandb_project,
                        group=self.experiment_name,
                        save_checkpoints=True,
                    )
                )
                logger.info(f"Enabled W&B tracking (project: {self.wandb_project})")
            except ImportError:
                logger.warning("W&B callback not available")

        return callbacks

    def _extract_best_results(self):
        """Extract best trial results."""
        if self.results is None:
            return

        try:
            best_result = self.results.get_best_result(
                metric=self.metric,
                mode=self.mode
            )

            self.best_result = best_result
            self.best_config = self._unflatten_config(best_result.config)

            logger.info(f"Best trial: {best_result.path}")
            logger.info(f"Best {self.metric}: {best_result.metrics[self.metric]:.6f}")

        except Exception as e:
            logger.error(f"Failed to extract best results: {e}")

    def _log_summary(self):
        """Log summary of tuning results."""
        if self.results is None:
            return

        logger.info("=" * 80)
        logger.info("HYPERPARAMETER TUNING SUMMARY")
        logger.info("=" * 80)

        # Best result
        if self.best_result:
            logger.info(f"\nBest {self.metric}: {self.best_result.metrics[self.metric]:.6f}")
            logger.info("\nBest hyperparameters:")
            self._log_config(self.best_config, indent=2)

        # Trial statistics
        df = self.results.get_dataframe()
        logger.info(f"\nTotal trials: {len(df)}")
        logger.info(f"Completed trials: {len(df[df['done'] == True])}")
        logger.info(f"Failed trials: {len(df[df['error'] == True])}")

        # Metric statistics
        metric_values = df[self.metric].dropna()
        logger.info(f"\n{self.metric} statistics:")
        logger.info(f"  Mean: {metric_values.mean():.6f}")
        logger.info(f"  Std: {metric_values.std():.6f}")
        logger.info(f"  Min: {metric_values.min():.6f}")
        logger.info(f"  Max: {metric_values.max():.6f}")

        logger.info("=" * 80)

    def _log_config(self, config: Dict, indent: int = 0):
        """Recursively log config dict."""
        prefix = " " * indent
        for key, value in config.items():
            if isinstance(value, dict):
                logger.info(f"{prefix}{key}:")
                self._log_config(value, indent + 2)
            else:
                logger.info(f"{prefix}{key}: {value}")

    def export_best_config(
        self,
        output_path: str,
        format: str = 'yaml'
    ) -> None:
        """
        Export best hyperparameters to file.

        Parameters
        ----------
        output_path : str
            Path to save config file.
        format : str, optional
            Output format: 'yaml' or 'json'.
            Default: 'yaml'.
        """
        if self.best_config is None:
            logger.error("No best config available. Run tuning first.")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata
        export_config = {
            'metadata': {
                'experiment_name': self.experiment_name,
                'best_metric': self.metric,
                'best_value': float(self.best_result.metrics[self.metric]),
                'timestamp': datetime.now().isoformat(),
                'search_algorithm': self.search_algorithm_name,
            },
            'hyperparameters': self.best_config,
        }

        # Export
        if format == 'yaml':
            with open(output_path, 'w') as f:
                yaml.dump(export_config, f, default_flow_style=False, indent=2)
            logger.info(f"Exported best config to {output_path} (YAML)")

        elif format == 'json':
            with open(output_path, 'w') as f:
                json.dump(export_config, f, indent=2)
            logger.info(f"Exported best config to {output_path} (JSON)")

        else:
            raise ValueError(f"Unknown format: {format}")

    def get_best_checkpoint_path(self) -> Optional[str]:
        """
        Get path to best checkpoint.

        Returns
        -------
        str or None
            Path to best checkpoint, or None if not available.
        """
        if self.best_result is None:
            return None

        return self.best_result.checkpoint.path if self.best_result.checkpoint else None

    def plot_optimization_history(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """
        Plot optimization history.

        Parameters
        ----------
        save_path : str, optional
            Path to save plot.
        show : bool, optional
            Whether to display plot.
            Default: True.
        """
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
        except ImportError:
            logger.error("matplotlib and pandas required for plotting")
            return

        if self.results is None:
            logger.error("No results available. Run tuning first.")
            return

        # Get results dataframe
        df = self.results.get_dataframe()

        # Plot learning curves
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Metric over time
        ax = axes[0, 0]
        for trial_id in df['trial_id'].unique()[:20]:  # Plot first 20 trials
            trial_df = df[df['trial_id'] == trial_id]
            ax.plot(
                trial_df['training_iteration'],
                trial_df[self.metric],
                alpha=0.3
            )
        ax.set_xlabel('Training Iteration')
        ax.set_ylabel(self.metric)
        ax.set_title(f'{self.metric} over Training')
        ax.grid(True, alpha=0.3)

        # 2. Metric distribution
        ax = axes[0, 1]
        metric_values = df[self.metric].dropna()
        ax.hist(metric_values, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(
            self.best_result.metrics[self.metric],
            color='red',
            linestyle='--',
            label='Best'
        )
        ax.set_xlabel(self.metric)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{self.metric} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Hyperparameter importance (learning rate)
        ax = axes[1, 0]
        if 'training.lr' in df.columns:
            ax.scatter(df['training.lr'], df[self.metric], alpha=0.5)
            ax.set_xscale('log')
            ax.set_xlabel('Learning Rate')
            ax.set_ylabel(self.metric)
            ax.set_title('Learning Rate vs Performance')
            ax.grid(True, alpha=0.3)

        # 4. Hyperparameter importance (d_model)
        ax = axes[1, 1]
        if 'model.d_model' in df.columns:
            # Group by d_model and plot boxplot
            grouped = df.groupby('model.d_model')[self.metric].apply(list)
            ax.boxplot(grouped.values, labels=grouped.index)
            ax.set_xlabel('d_model')
            ax.set_ylabel(self.metric)
            ax.set_title('Model Dimension vs Performance')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved optimization history plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def create_neurofmx_train_fn(
    data_config: Dict[str, Any],
    base_config: Dict[str, Any],
    num_epochs: Optional[int] = None,
) -> Callable:
    """
    Create a training function for NeuroFMX hyperparameter tuning.

    This function creates a trainable that can be used with NeuroFMXRayTuner.

    Parameters
    ----------
    data_config : dict
        Data configuration (paths, preprocessing, etc.).
    base_config : dict
        Base model configuration (will be updated with tuned hyperparameters).
    num_epochs : int, optional
        Number of epochs per trial. If None, uses max_epochs from search.

    Returns
    -------
    callable
        Training function compatible with Ray Tune.

    Example
    -------
    >>> data_config = {'data_path': 'path/to/data', 'batch_size': 32}
    >>> base_config = {'checkpoint_dir': './checkpoints'}
    >>> train_fn = create_neurofmx_train_fn(data_config, base_config)
    >>> tuner = NeuroFMXRayTuner(train_fn=train_fn)
    >>> results = tuner.run()
    """
    def train_fn(config: Dict[str, Any]):
        """Training function for a single trial."""
        import torch
        from torch.utils.data import DataLoader
        from ray.air import session

        # Merge base config with hyperparameters
        merged_config = {**base_config, **config}

        # TODO: User should implement actual training logic here
        # This is a template showing the required structure

        # 1. Create model
        # from neuros_neurofm.models import NeuroFMXComplete
        # model = NeuroFMXComplete.from_config(merged_config)

        # 2. Setup data
        # train_loader = DataLoader(...)
        # val_loader = DataLoader(...)

        # 3. Setup optimizer
        # optimizer = torch.optim.AdamW(
        #     model.parameters(),
        #     lr=config['training']['lr'],
        #     weight_decay=config['training']['weight_decay'],
        # )

        # 4. Training loop
        epochs = num_epochs or merged_config.get('max_epochs', 100)

        for epoch in range(epochs):
            # Training
            # train_loss = train_epoch(model, train_loader, optimizer)

            # Validation
            # val_loss, val_metrics = validate(model, val_loader)

            # Report metrics to Ray Tune
            # session.report({
            #     'training_iteration': epoch + 1,
            #     'train_loss': train_loss,
            #     'val_loss': val_loss,
            #     **val_metrics,
            # })

            # Dummy metrics for template
            import numpy as np
            session.report({
                'training_iteration': epoch + 1,
                'val_loss': np.random.random(),
                'val_r2': np.random.random(),
            })

    return train_fn


# Example usage
if __name__ == '__main__':
    # Example: Run hyperparameter search

    # 1. Create training function
    data_config = {
        'data_path': './data',
        'batch_size': 32,
    }
    base_config = {
        'checkpoint_dir': './checkpoints',
    }
    train_fn = create_neurofmx_train_fn(data_config, base_config, num_epochs=10)

    # 2. Create tuner with ASHA scheduler
    tuner = NeuroFMXRayTuner(
        train_fn=train_fn,
        search_algorithm='asha',
        num_samples=20,
        max_concurrent_trials=4,
        resources_per_trial={'cpu': 4, 'gpu': 1},
        metric='val_loss',
        mode='min',
        max_epochs=10,
        grace_period=3,
        experiment_name='neurofmx_asha_test',
        use_mlflow=True,
        use_wandb=False,
    )

    # 3. Run tuning
    results = tuner.run()

    # 4. Export best config
    tuner.export_best_config('best_config.yaml')

    # 5. Plot results
    tuner.plot_optimization_history('optimization_history.png')

    print("Hyperparameter search completed!")
    print(f"Best config saved to: best_config.yaml")
    print(f"Best checkpoint: {tuner.get_best_checkpoint_path()}")
