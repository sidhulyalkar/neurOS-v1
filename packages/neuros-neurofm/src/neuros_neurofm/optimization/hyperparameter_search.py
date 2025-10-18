"""
Hyperparameter search and optimization for NeuroFM-X.

Provides utilities for tuning model hyperparameters using various strategies.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import json
from pathlib import Path

import numpy as np
import torch

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class HyperparameterSearch:
    """Hyperparameter search using Optuna.

    Parameters
    ----------
    objective_fn : callable
        Function to optimize. Should accept a trial object and return a metric.
    direction : str, optional
        Optimization direction ('minimize' or 'maximize').
        Default: 'maximize'.
    n_trials : int, optional
        Number of trials.
        Default: 100.
    study_name : str, optional
        Name for the study.
        Default: 'neurofmx_hparam_search'.
    """

    def __init__(
        self,
        objective_fn: Callable,
        direction: str = 'maximize',
        n_trials: int = 100,
        study_name: str = 'neurofmx_hparam_search',
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "optuna is required for hyperparameter search. "
                "Install with: pip install optuna"
            )

        self.objective_fn = objective_fn
        self.direction = direction
        self.n_trials = n_trials
        self.study_name = study_name

    def search(
        self,
        param_space: Optional[Dict[str, Any]] = None,
        pruner: Optional[str] = 'median',
    ) -> Dict[str, Any]:
        """Run hyperparameter search.

        Parameters
        ----------
        param_space : dict, optional
            Parameter space specification.
        pruner : str, optional
            Pruning strategy ('median', 'percentile', None).
            Default: 'median'.

        Returns
        -------
        dict
            Best hyperparameters found.
        """
        # Create pruner
        if pruner == 'median':
            pruner_obj = optuna.pruners.MedianPruner()
        elif pruner == 'percentile':
            pruner_obj = optuna.pruners.PercentilePruner(percentile=25.0)
        else:
            pruner_obj = None

        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            pruner=pruner_obj,
        )

        # Run optimization
        study.optimize(self.objective_fn, n_trials=self.n_trials)

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value

        print(f"Best {self.direction} value: {best_value:.4f}")
        print(f"Best parameters: {best_params}")

        return {
            'best_params': best_params,
            'best_value': best_value,
            'study': study,
        }


class GridSearch:
    """Grid search over hyperparameters.

    Simpler alternative to Bayesian optimization.

    Parameters
    ----------
    param_grid : dict
        Dictionary mapping parameter names to lists of values.
    objective_fn : callable
        Function to evaluate. Takes params dict, returns metric.
    direction : str, optional
        'minimize' or 'maximize'.
        Default: 'maximize'.
    """

    def __init__(
        self,
        param_grid: Dict[str, List],
        objective_fn: Callable,
        direction: str = 'maximize',
    ):
        self.param_grid = param_grid
        self.objective_fn = objective_fn
        self.direction = direction

    def search(self) -> Dict[str, Any]:
        """Run grid search.

        Returns
        -------
        dict
            Best parameters and results.
        """
        from itertools import product

        # Generate all combinations
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = list(product(*values))

        best_score = float('-inf') if self.direction == 'maximize' else float('inf')
        best_params = None
        all_results = []

        print(f"Running grid search: {len(combinations)} combinations")

        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))

            try:
                score = self.objective_fn(params)

                all_results.append({
                    'params': params,
                    'score': score,
                })

                # Update best
                if self.direction == 'maximize':
                    if score > best_score:
                        best_score = score
                        best_params = params
                else:
                    if score < best_score:
                        best_score = score
                        best_params = params

                print(f"Trial {i+1}/{len(combinations)}: {params} -> {score:.4f}")

            except Exception as e:
                print(f"Trial {i+1} failed: {e}")
                continue

        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': all_results,
        }


def create_neurofmx_objective(
    train_loader,
    val_loader,
    n_epochs: int = 5,
    device: str = 'cpu',
) -> Callable:
    """Create objective function for NeuroFM-X hyperparameter tuning.

    Parameters
    ----------
    train_loader : DataLoader
        Training data.
    val_loader : DataLoader
        Validation data.
    n_epochs : int, optional
        Number of training epochs per trial.
    device : str, optional
        Device to train on.

    Returns
    -------
    callable
        Objective function for hyperparameter search.
    """

    def objective(trial_or_params):
        """Objective function."""
        # Extract parameters
        if OPTUNA_AVAILABLE and isinstance(trial_or_params, optuna.Trial):
            # Optuna trial
            trial = trial_or_params
            params = {
                'd_model': trial.suggest_categorical('d_model', [128, 256, 512]),
                'n_latents': trial.suggest_categorical('n_latents', [16, 32, 64]),
                'latent_dim': trial.suggest_categorical('latent_dim', [64, 128, 256]),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
                'dropout': trial.suggest_uniform('dropout', 0.0, 0.3),
            }
        else:
            # Grid search params
            params = trial_or_params

        # Build model
        from neuros_neurofm.tokenizers import BinnedTokenizer
        from neuros_neurofm.fusion import PerceiverIO
        from neuros_neurofm.models import PopT, MultiTaskHeads

        tokenizer = BinnedTokenizer(
            n_units=96,
            d_model=params['d_model'],
        ).to(device)

        fusion = PerceiverIO(
            n_latents=params['n_latents'],
            latent_dim=params['latent_dim'],
            input_dim=params['d_model'],
            dropout=params['dropout'],
        ).to(device)

        popt = PopT(
            d_model=params['latent_dim'],
            dropout=params['dropout'],
        ).to(device)

        heads = MultiTaskHeads(
            input_dim=params['latent_dim'],
            decoder_output_dim=2,
            dropout=params['dropout'],
        ).to(device)

        # Optimizer
        all_params = (
            list(tokenizer.parameters()) +
            list(fusion.parameters()) +
            list(popt.parameters()) +
            list(heads.parameters())
        )
        optimizer = torch.optim.AdamW(all_params, lr=params['learning_rate'])

        # Train
        for epoch in range(n_epochs):
            # Training
            tokenizer.train()
            fusion.train()
            popt.train()
            heads.train()

            for batch in train_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                optimizer.zero_grad()
                tokens, _ = tokenizer(batch['spikes'])
                latents = fusion(tokens)
                aggregated = popt(latents)
                predictions = heads(aggregated, task='decoder')

                loss = torch.nn.functional.mse_loss(
                    predictions,
                    batch['behavior_target']
                )
                loss.backward()
                optimizer.step()

        # Evaluate
        tokenizer.eval()
        fusion.eval()
        popt.eval()
        heads.eval()

        val_r2_scores = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                tokens, _ = tokenizer(batch['spikes'])
                latents = fusion(tokens)
                aggregated = popt(latents)
                predictions = heads(aggregated, task='decoder')

                # R² score
                targets = batch['behavior_target']
                ss_res = ((targets - predictions) ** 2).sum()
                ss_tot = ((targets - targets.mean()) ** 2).sum()
                r2 = 1 - ss_res / (ss_tot + 1e-8)
                val_r2_scores.append(r2.item())

        mean_r2 = np.mean(val_r2_scores)
        return mean_r2

    return objective


def save_best_hyperparameters(
    results: Dict[str, Any],
    save_path: str,
):
    """Save best hyperparameters to JSON file.

    Parameters
    ----------
    results : dict
        Results from hyperparameter search.
    save_path : str
        Path to save JSON file.
    """
    save_dict = {
        'best_params': results['best_params'],
        'best_value': results.get('best_value') or results.get('best_score'),
    }

    with open(save_path, 'w') as f:
        json.dump(save_dict, f, indent=2)

    print(f"Best hyperparameters saved to {save_path}")


def load_hyperparameters(load_path: str) -> Dict[str, Any]:
    """Load hyperparameters from JSON file.

    Parameters
    ----------
    load_path : str
        Path to JSON file.

    Returns
    -------
    dict
        Loaded hyperparameters.
    """
    with open(load_path, 'r') as f:
        params = json.load(f)

    return params['best_params']
