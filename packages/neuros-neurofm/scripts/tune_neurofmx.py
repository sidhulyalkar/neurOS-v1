#!/usr/bin/env python
"""
Quick-start script for Ray Tune hyperparameter optimization of NeuroFMX.

This script provides a simple command-line interface for tuning NeuroFMX
models on your data with minimal configuration.

Usage:
    # Quick test (2 trials, 5 epochs each)
    python scripts/tune_neurofmx.py --quick_test

    # Small-scale search (20 trials)
    python scripts/tune_neurofmx.py --data_path ./data --num_samples 20

    # Full-scale search with GPUs
    python scripts/tune_neurofmx.py \
        --data_path ./data \
        --search_algorithm asha \
        --num_samples 100 \
        --gpus_per_trial 1 \
        --max_concurrent 4

    # Architecture search
    python scripts/tune_neurofmx.py \
        --data_path ./data \
        --search_space architecture \
        --num_samples 50

    # With experiment tracking
    python scripts/tune_neurofmx.py \
        --data_path ./data \
        --use_mlflow \
        --use_wandb \
        --wandb_project my_project
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check Ray Tune availability
try:
    from neuros_neurofm.optimization import (
        NeuroFMXRayTuner,
        NeuroFMXSearchSpace,
        RAY_TUNE_AVAILABLE,
    )
    if not RAY_TUNE_AVAILABLE:
        raise ImportError("Ray Tune not available")
except ImportError:
    logger.error(
        "Ray Tune is not installed. Install with:\n"
        "  pip install 'ray[tune]' 'ray[train]' optuna"
    )
    sys.exit(1)

from ray.air import session
import torch
from torch.utils.data import DataLoader


def create_train_function(data_path: str, config: dict):
    """
    Create training function for Ray Tune.

    Parameters
    ----------
    data_path : str
        Path to training data.
    config : dict
        Base configuration.

    Returns
    -------
    callable
        Training function.
    """
    def train_fn(tune_config: dict):
        """Training function for a single trial."""
        import torch
        from neuros_neurofm.models import NeuroFMXComplete
        from neuros_neurofm.datasets import SyntheticNeuroDataset
        from torch.utils.data import DataLoader

        # Merge configurations
        merged = {**config, **tune_config}

        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create model
        try:
            model = NeuroFMXComplete(
                d_model=tune_config['model']['d_model'],
                n_mamba_blocks=tune_config['model']['n_layers'],
                latent_dim=tune_config['model']['latent_dim'],
                n_latents=tune_config['model']['n_latents'],
                n_perceiver_layers=tune_config['model']['n_perceiver_layers'],
                n_popt_layers=tune_config['model']['n_popt_layers'],
                dropout=tune_config['model']['dropout'],
                use_multi_rate=tune_config['multi_rate']['enabled'],
                downsample_rates=tune_config['multi_rate']['rates'],
                enable_decoder=True,
                decoder_output_dim=merged.get('behavior_dim', 2),
                encoder_output_dim=merged.get('n_units', 384),
            ).to(device)
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            # Report failure
            for epoch in range(merged['epochs_per_trial']):
                session.report({
                    'training_iteration': epoch + 1,
                    'val_loss': float('inf'),
                    'val_r2': -1.0,
                })
            return

        # Create datasets
        # TODO: Replace with actual data loading
        train_dataset = SyntheticNeuroDataset(
            n_samples=1000,
            n_units=merged.get('n_units', 384),
            seq_length=merged.get('seq_length', 100),
            behavior_dim=merged.get('behavior_dim', 2),
        )
        val_dataset = SyntheticNeuroDataset(
            n_samples=200,
            n_units=merged.get('n_units', 384),
            seq_length=merged.get('seq_length', 100),
            behavior_dim=merged.get('behavior_dim', 2),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=tune_config['training']['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=tune_config['training']['batch_size'],
            shuffle=False,
            num_workers=2,
        )

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tune_config['training']['lr'],
            weight_decay=tune_config['training']['weight_decay'],
            betas=tune_config['optimizer']['betas'],
            eps=tune_config['optimizer']['eps'],
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=merged['epochs_per_trial'],
            eta_min=1e-6,
        )

        # Training loop
        best_val_loss = float('inf')

        for epoch in range(merged['epochs_per_trial']):
            # Training
            model.train()
            train_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                spikes = batch['spikes'].to(device)
                behavior = batch['behavior'].to(device)

                optimizer.zero_grad()

                # Forward pass
                predictions = model(spikes, task='decoder')

                # Loss
                loss = torch.nn.functional.mse_loss(predictions, behavior)

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    tune_config['training']['gradient_clip_val']
                )
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= n_batches

            # Validation
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    spikes = batch['spikes'].to(device)
                    behavior = batch['behavior'].to(device)

                    predictions = model(spikes, task='decoder')
                    loss = torch.nn.functional.mse_loss(predictions, behavior)

                    val_loss += loss.item()
                    all_preds.append(predictions.cpu().numpy())
                    all_targets.append(behavior.cpu().numpy())

            val_loss /= len(val_loader)

            # Compute R²
            import numpy as np
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            ss_res = np.sum((all_targets - all_preds) ** 2)
            ss_tot = np.sum((all_targets - all_targets.mean(axis=0, keepdims=True)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-8)

            # Update scheduler
            scheduler.step()

            # Track best
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Report to Ray Tune
            session.report({
                'training_iteration': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_r2': r2,
                'best_val_loss': best_val_loss,
                'lr': optimizer.param_groups[0]['lr'],
            })

    return train_fn


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Ray Tune Hyperparameter Optimization for NeuroFMX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Data
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to training data')
    parser.add_argument('--n_units', type=int, default=384,
                        help='Number of neural units')
    parser.add_argument('--seq_length', type=int, default=100,
                        help='Sequence length')
    parser.add_argument('--behavior_dim', type=int, default=2,
                        help='Behavior dimension')

    # Search configuration
    parser.add_argument('--search_algorithm', type=str, default='asha',
                        choices=['asha', 'pbt', 'bayesian', 'random'],
                        help='Search algorithm')
    parser.add_argument('--search_space', type=str, default='small',
                        choices=['default', 'small', 'architecture'],
                        help='Search space')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of trials')
    parser.add_argument('--max_concurrent', type=int, default=4,
                        help='Max concurrent trials')

    # Resources
    parser.add_argument('--cpus_per_trial', type=int, default=4,
                        help='CPUs per trial')
    parser.add_argument('--gpus_per_trial', type=float, default=0,
                        help='GPUs per trial')

    # Training
    parser.add_argument('--epochs_per_trial', type=int, default=20,
                        help='Training epochs per trial')
    parser.add_argument('--grace_period', type=int, default=5,
                        help='Grace period for early stopping')
    parser.add_argument('--reduction_factor', type=int, default=3,
                        help='Reduction factor for ASHA')

    # Output
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--results_dir', type=str, default='./ray_results',
                        help='Results directory')

    # Tracking
    parser.add_argument('--use_mlflow', action='store_true',
                        help='Enable MLflow tracking')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable W&B tracking')
    parser.add_argument('--wandb_project', type=str, default='neurofmx',
                        help='W&B project name')

    # Quick test
    parser.add_argument('--quick_test', action='store_true',
                        help='Quick test mode (2 trials, 5 epochs)')

    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        logger.info("Running in quick test mode")
        args.num_samples = 2
        args.epochs_per_trial = 5
        args.max_concurrent = 1
        args.grace_period = 2
        args.search_space = 'small'

    # Generate experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f'neurofmx_{args.search_algorithm}_{timestamp}'

    # Base configuration
    base_config = {
        'n_units': args.n_units,
        'seq_length': args.seq_length,
        'behavior_dim': args.behavior_dim,
        'epochs_per_trial': args.epochs_per_trial,
    }

    # Create training function
    logger.info("Creating training function...")
    train_fn = create_train_function(args.data_path, base_config)

    # Select search space
    if args.search_space == 'default':
        search_space = NeuroFMXSearchSpace.get_default_search_space()
    elif args.search_space == 'small':
        search_space = NeuroFMXSearchSpace.get_small_search_space()
    elif args.search_space == 'architecture':
        search_space = NeuroFMXSearchSpace.get_architecture_search_space()
    else:
        raise ValueError(f"Unknown search space: {args.search_space}")

    # Create tuner
    logger.info("Creating Ray Tune tuner...")
    tuner = NeuroFMXRayTuner(
        train_fn=train_fn,
        search_algorithm=args.search_algorithm,
        num_samples=args.num_samples,
        max_concurrent_trials=args.max_concurrent,
        resources_per_trial={
            'cpu': args.cpus_per_trial,
            'gpu': args.gpus_per_trial,
        },
        metric='val_loss',
        mode='min',
        search_space=search_space,
        max_epochs=args.epochs_per_trial,
        grace_period=args.grace_period,
        reduction_factor=args.reduction_factor,
        experiment_name=args.experiment_name,
        local_dir=args.results_dir,
        use_mlflow=args.use_mlflow,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        checkpoint_freq=10,
        keep_checkpoints_num=3,
        verbose=2,
    )

    # Print configuration
    logger.info("=" * 80)
    logger.info("HYPERPARAMETER SEARCH CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Experiment name: {args.experiment_name}")
    logger.info(f"Search algorithm: {args.search_algorithm}")
    logger.info(f"Search space: {args.search_space}")
    logger.info(f"Number of trials: {args.num_samples}")
    logger.info(f"Max concurrent: {args.max_concurrent}")
    logger.info(f"Epochs per trial: {args.epochs_per_trial}")
    logger.info(f"Resources per trial: {args.cpus_per_trial} CPUs, {args.gpus_per_trial} GPUs")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info("=" * 80)

    # Run tuning
    try:
        results = tuner.run()

        # Export results
        results_dir = Path(args.results_dir) / args.experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)

        # Export best config
        config_path = results_dir / 'best_config.yaml'
        tuner.export_best_config(str(config_path))
        logger.info(f"Best config saved to: {config_path}")

        # Plot optimization history
        plot_path = results_dir / 'optimization_history.png'
        try:
            tuner.plot_optimization_history(str(plot_path), show=False)
            logger.info(f"Optimization plot saved to: {plot_path}")
        except Exception as e:
            logger.warning(f"Failed to create plot: {e}")

        # Print summary
        logger.info("=" * 80)
        logger.info("OPTIMIZATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Best checkpoint: {tuner.get_best_checkpoint_path()}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
