"""
Example: Hyperparameter Tuning with Ray Tune for NeuroFMX

This script demonstrates how to use the Ray Tune hyperparameter search
for optimizing NeuroFMX models.

Usage:
    python ray_tune_example.py --search_algorithm asha --num_samples 50
"""

import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

# Import NeuroFMX components
from neuros_neurofm.optimization import (
    NeuroFMXRayTuner,
    NeuroFMXSearchSpace,
    RAY_TUNE_AVAILABLE,
)

if not RAY_TUNE_AVAILABLE:
    raise ImportError(
        "Ray Tune is not available. Install with: pip install 'ray[tune]' 'ray[train]'"
    )

from ray.air import session
from neuros_neurofm.models import NeuroFMXComplete
from neuros_neurofm.datasets import SyntheticNeuroDataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dataloaders(config: dict) -> tuple:
    """
    Create train and validation dataloaders.

    Parameters
    ----------
    config : dict
        Configuration with data parameters.

    Returns
    -------
    tuple
        (train_loader, val_loader)
    """
    # Create synthetic dataset for demonstration
    # In practice, replace with actual data loading
    train_dataset = SyntheticNeuroDataset(
        n_samples=1000,
        n_units=config.get('n_units', 384),
        seq_length=config.get('seq_length', 100),
        behavior_dim=config.get('behavior_dim', 2),
    )

    val_dataset = SyntheticNeuroDataset(
        n_samples=200,
        n_units=config.get('n_units', 384),
        seq_length=config.get('seq_length', 100),
        behavior_dim=config.get('behavior_dim', 2),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        # Move to device
        spikes = batch['spikes'].to(device)
        behavior = batch['behavior'].to(device)

        # Forward pass
        optimizer.zero_grad()

        # Tokenize inputs
        tokens = spikes  # Assuming spikes are already tokenized

        # Get predictions
        predictions = model(tokens, task='decoder')

        # Compute loss
        loss = torch.nn.functional.mse_loss(predictions, behavior)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(model, val_loader, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            spikes = batch['spikes'].to(device)
            behavior = batch['behavior'].to(device)

            tokens = spikes
            predictions = model(tokens, task='decoder')

            loss = torch.nn.functional.mse_loss(predictions, behavior)
            total_loss += loss.item()

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(behavior.cpu().numpy())

    # Compute R²
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    ss_res = np.sum((all_targets - all_predictions) ** 2)
    ss_tot = np.sum((all_targets - all_targets.mean(axis=0, keepdims=True)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    val_loss = total_loss / len(val_loader)

    return val_loss, {'r2': r2}


def create_train_fn(base_config: dict, num_epochs: int = 20):
    """
    Create training function for Ray Tune.

    Parameters
    ----------
    base_config : dict
        Base configuration with data paths, etc.
    num_epochs : int
        Number of epochs per trial.

    Returns
    -------
    callable
        Training function compatible with Ray Tune.
    """
    def train_fn(config: dict):
        """Training function for a single trial."""
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Merge base config with hyperparameters
        merged_config = {**base_config, **config}

        # Create model
        model_config = {
            'd_model': config['model']['d_model'],
            'n_mamba_blocks': config['model']['n_layers'],
            'latent_dim': config['model']['latent_dim'],
            'n_latents': config['model']['n_latents'],
            'n_perceiver_layers': config['model']['n_perceiver_layers'],
            'n_popt_layers': config['model']['n_popt_layers'],
            'dropout': config['model']['dropout'],
            'use_multi_rate': config['multi_rate']['enabled'],
            'downsample_rates': config['multi_rate']['rates'],
            'enable_decoder': True,
            'decoder_output_dim': merged_config.get('behavior_dim', 2),
            'encoder_output_dim': merged_config.get('n_units', 384),
        }

        model = NeuroFMXComplete(**model_config).to(device)

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(merged_config)

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay'],
            betas=config['optimizer']['betas'],
            eps=config['optimizer']['eps'],
        )

        # Setup learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=1e-6,
        )

        # Training loop
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, device)

            # Validate
            val_loss, val_metrics = validate(model, val_loader, device)

            # Update learning rate
            scheduler.step()

            # Track best
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Report to Ray Tune
            session.report({
                'training_iteration': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_r2': val_metrics['r2'],
                'best_val_loss': best_val_loss,
                'lr': optimizer.param_groups[0]['lr'],
            })

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_r2={val_metrics['r2']:.4f}"
            )

    return train_fn


def main(args):
    """Main training function."""
    # Base configuration
    base_config = {
        'n_units': 384,
        'seq_length': 100,
        'behavior_dim': 2,
        'checkpoint_dir': args.checkpoint_dir,
    }

    # Create training function
    train_fn = create_train_fn(base_config, num_epochs=args.epochs_per_trial)

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
        checkpoint_freq=args.checkpoint_freq,
        keep_checkpoints_num=args.keep_checkpoints,
        verbose=2,
    )

    # Run tuning
    logger.info("=" * 80)
    logger.info("Starting Ray Tune Hyperparameter Search")
    logger.info("=" * 80)
    logger.info(f"Search algorithm: {args.search_algorithm}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Max concurrent trials: {args.max_concurrent}")
    logger.info(f"Epochs per trial: {args.epochs_per_trial}")
    logger.info("=" * 80)

    results = tuner.run()

    # Export best configuration
    output_path = Path(args.results_dir) / args.experiment_name / 'best_config.yaml'
    tuner.export_best_config(str(output_path), format='yaml')

    # Plot optimization history
    plot_path = Path(args.results_dir) / args.experiment_name / 'optimization_history.png'
    tuner.plot_optimization_history(str(plot_path), show=False)

    # Print summary
    logger.info("=" * 80)
    logger.info("HYPERPARAMETER SEARCH COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Best config saved to: {output_path}")
    logger.info(f"Best checkpoint: {tuner.get_best_checkpoint_path()}")
    logger.info(f"Optimization plot: {plot_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Ray Tune Hyperparameter Search for NeuroFMX'
    )

    # Search configuration
    parser.add_argument(
        '--search_algorithm',
        type=str,
        default='asha',
        choices=['asha', 'pbt', 'bayesian', 'random', 'grid'],
        help='Search algorithm to use',
    )
    parser.add_argument(
        '--search_space',
        type=str,
        default='small',
        choices=['default', 'small', 'architecture'],
        help='Search space to use',
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=50,
        help='Number of hyperparameter configurations to try',
    )
    parser.add_argument(
        '--max_concurrent',
        type=int,
        default=4,
        help='Maximum concurrent trials',
    )

    # Resources
    parser.add_argument(
        '--cpus_per_trial',
        type=int,
        default=4,
        help='CPUs per trial',
    )
    parser.add_argument(
        '--gpus_per_trial',
        type=float,
        default=0,
        help='GPUs per trial (can be fractional)',
    )

    # Training configuration
    parser.add_argument(
        '--epochs_per_trial',
        type=int,
        default=20,
        help='Training epochs per trial',
    )
    parser.add_argument(
        '--grace_period',
        type=int,
        default=5,
        help='Minimum epochs before early stopping (ASHA)',
    )
    parser.add_argument(
        '--reduction_factor',
        type=int,
        default=3,
        help='Reduction factor for ASHA',
    )

    # Experiment tracking
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='neurofmx_tuning',
        help='Experiment name',
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./ray_results',
        help='Results directory',
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints',
        help='Checkpoint directory',
    )
    parser.add_argument(
        '--checkpoint_freq',
        type=int,
        default=10,
        help='Checkpoint frequency (epochs)',
    )
    parser.add_argument(
        '--keep_checkpoints',
        type=int,
        default=3,
        help='Number of checkpoints to keep',
    )

    # Logging
    parser.add_argument(
        '--use_mlflow',
        action='store_true',
        help='Enable MLflow tracking',
    )
    parser.add_argument(
        '--use_wandb',
        action='store_true',
        help='Enable Weights & Biases tracking',
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='neurofmx',
        help='W&B project name',
    )

    args = parser.parse_args()

    main(args)
