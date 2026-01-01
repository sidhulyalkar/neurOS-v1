#!/usr/bin/env python3
"""
SAE Hyperparameter Search

Systematically explores SAE hyperparameters to find optimal architecture:
- Hidden dimension (64, 128, 256, 512, 1024)
- Sparsity penalty (0.001, 0.005, 0.01, 0.02, 0.05)
- Learning rate (0.0001, 0.0005, 0.001, 0.005)
- Activation function (ReLU, Leaky ReLU, GELU)

Evaluation metrics:
- Reconstruction loss
- Orientation selectivity
- Feature sparsity
- Feature diversity

Usage:
    python scripts/sae_hyperparameter_search.py \
        --session-id 754829445 \
        --allen-cache allen_validation_cache \
        --output-dir hyperparameter_search
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfigurableSAE(nn.Module):
    """SAE with configurable architecture."""

    def __init__(self, input_dim: int, hidden_dim: int, activation: str = 'relu'):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

        # Configurable activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        h = self.activation(self.encoder(x))
        x_recon = self.decoder(h)
        return x_recon, h

    def encode(self, x):
        with torch.no_grad():
            h = self.activation(self.encoder(x))
        return h


def train_sae_with_config(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: Dict,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Train SAE with specific hyperparameter configuration.

    Returns evaluation metrics.
    """
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    # Initialize SAE
    sae = ConfigurableSAE(
        input_dim=X_train.shape[1],
        hidden_dim=config['hidden_dim'],
        activation=config['activation']
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(sae.parameters(), lr=config['lr'])

    # Training loop
    losses = []
    epochs = config.get('epochs', 50)

    for epoch in range(epochs):
        sae.train()

        # Forward pass
        x_recon, h = sae(X_train_t)

        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(x_recon, X_train_t)

        # Sparsity loss
        sparsity_loss = config['sparsity'] * h.abs().mean()

        # Total loss
        loss = recon_loss + sparsity_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # Evaluate
    sae.eval()
    with torch.no_grad():
        x_test_recon, h_test = sae(X_test_t)
        test_recon_loss = nn.functional.mse_loss(x_test_recon, X_test_t).item()

    # Extract features for analysis
    h_test_np = h_test.cpu().numpy()

    # Evaluate orientation selectivity
    from scipy.stats import pearsonr

    ori_sin = np.sin(np.deg2rad(y_test * 2))
    ori_cos = np.cos(np.deg2rad(y_test * 2))

    correlations = []
    for feat_idx in range(h_test_np.shape[1]):
        feat_response = h_test_np[:, feat_idx]
        corr_sin, _ = pearsonr(feat_response, ori_sin)
        corr_cos, _ = pearsonr(feat_response, ori_cos)
        correlations.append(max(abs(corr_sin), abs(corr_cos)))

    correlations = np.array(correlations)
    n_significant = np.sum(correlations > 0.3)
    fraction_selective = n_significant / len(correlations)

    # Sparsity metrics
    l0_sparsity = np.mean(h_test_np == 0)
    lifetime_sparsity = np.mean(h_test_np > 0, axis=0).mean()

    # Feature diversity (pairwise correlation between features)
    feature_corr_matrix = np.corrcoef(h_test_np.T)
    mean_feature_corr = (feature_corr_matrix.sum() - h_test_np.shape[1]) / (h_test_np.shape[1] ** 2 - h_test_np.shape[1])

    results = {
        'config': config,
        'test_recon_loss': test_recon_loss,
        'orientation_selectivity': {
            'max_correlation': float(np.max(correlations)),
            'mean_correlation': float(np.mean(correlations)),
            'n_significant': int(n_significant),
            'fraction_selective': float(fraction_selective)
        },
        'sparsity': {
            'l0_sparsity': float(l0_sparsity),
            'lifetime_sparsity': float(lifetime_sparsity)
        },
        'diversity': {
            'mean_feature_correlation': float(mean_feature_corr)
        }
    }

    return results


def grid_search(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    search_space: Dict[str, List],
    top_k: int = 10
) -> List[Dict]:
    """
    Perform grid search over hyperparameter space.

    Returns top K configurations by composite score.
    """
    # Generate all combinations
    keys = list(search_space.keys())
    values = list(search_space.values())
    all_configs = [dict(zip(keys, combo)) for combo in product(*values)]

    logger.info(f"Grid search over {len(all_configs)} configurations")

    results = []

    for i, config in enumerate(all_configs, 1):
        logger.info(f"\n[{i}/{len(all_configs)}] Testing config: {config}")

        try:
            result = train_sae_with_config(X_train, X_test, y_train, y_test, config)
            results.append(result)

            logger.info(f"  Recon loss: {result['test_recon_loss']:.4f}")
            logger.info(f"  Selectivity: {result['orientation_selectivity']['fraction_selective']*100:.1f}%")
            logger.info(f"  Sparsity: {result['sparsity']['lifetime_sparsity']*100:.1f}%")

        except Exception as e:
            logger.error(f"  Failed: {e}")
            continue

    # Compute composite scores
    # Score = selectivity - 0.5*recon_loss - 0.2*abs(sparsity - 0.3) - 0.1*feature_corr
    # (We want high selectivity, low recon loss, ~30% sparsity, low feature correlation)

    for r in results:
        selectivity_score = r['orientation_selectivity']['fraction_selective']
        recon_penalty = r['test_recon_loss']
        sparsity_penalty = abs(r['sparsity']['lifetime_sparsity'] - 0.3)
        diversity_penalty = r['diversity']['mean_feature_correlation']

        r['composite_score'] = selectivity_score - 0.5*recon_penalty - 0.2*sparsity_penalty - 0.1*diversity_penalty

    # Sort by composite score
    results.sort(key=lambda x: x['composite_score'], reverse=True)

    return results[:top_k]


def main():
    parser = argparse.ArgumentParser(description='SAE hyperparameter search')
    parser.add_argument('--session-id', type=int, required=True, help='Session ID')
    parser.add_argument('--allen-cache', type=str, default='allen_validation_cache', help='Allen cache directory')
    parser.add_argument('--output-dir', type=str, default='hyperparameter_search', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick search with fewer configs')
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("SAE HYPERPARAMETER SEARCH")
    logger.info("="*80)

    # Load data
    logger.info(f"\nLoading session {args.session_id}...")
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "neuros-foundation" / "src"))
    from neuros.datasets.allen_datasets import AllenVisualCodingValidator

    validator = AllenVisualCodingValidator(
        session_id=args.session_id,
        cache_dir=args.allen_cache,
        brain_areas=['VISp'],
        use_all_units=True
    )

    windows = validator.get_neural_windows()

    # Extract data - filter out null orientations
    valid_data = []
    valid_orientations = []

    for w in windows:
        ori = w.metadata['orientation']

        # Skip null/NaN orientations
        if ori == 'null' or ori is None:
            continue

        # Try to convert to float
        try:
            ori_float = float(ori)
            ori_180 = ori_float % 180  # Convert to 0-180°

            valid_data.append(w.data.mean(axis=0))
            valid_orientations.append(ori_180)
        except (ValueError, TypeError):
            continue

    raw_data = np.array(valid_data)
    orientations = np.array(valid_orientations)

    n_skipped = len(windows) - len(valid_data)
    if n_skipped > 0:
        logger.info(f"  Skipped {n_skipped} windows with invalid orientations")

    # Normalize
    raw_mean = raw_data.mean(axis=0, keepdims=True)
    raw_std = raw_data.std(axis=0, keepdims=True) + 1e-8
    raw_norm = (raw_data - raw_mean) / raw_std

    # Split
    n_train = int(0.8 * len(raw_norm))
    X_train = raw_norm[:n_train]
    X_test = raw_norm[n_train:]
    y_train = orientations[:n_train]
    y_test = orientations[n_train:]

    logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Define search space
    if args.quick:
        search_space = {
            'hidden_dim': [128, 256],
            'sparsity': [0.01, 0.02],
            'lr': [0.001],
            'activation': ['relu'],
            'epochs': [30]
        }
    else:
        search_space = {
            'hidden_dim': [64, 128, 256, 512],
            'sparsity': [0.005, 0.01, 0.02, 0.05],
            'lr': [0.0005, 0.001, 0.002],
            'activation': ['relu', 'leaky_relu'],
            'epochs': [50]
        }

    logger.info(f"\nSearch space:")
    for k, v in search_space.items():
        logger.info(f"  {k}: {v}")

    # Run grid search
    logger.info("\n" + "="*80)
    logger.info("STARTING GRID SEARCH")
    logger.info("="*80)

    top_configs = grid_search(X_train, X_test, y_train, y_test, search_space, top_k=10)

    # Print results
    logger.info("\n" + "="*80)
    logger.info("TOP 10 CONFIGURATIONS")
    logger.info("="*80)

    for i, result in enumerate(top_configs, 1):
        logger.info(f"\n[{i}] Score: {result['composite_score']:.3f}")
        logger.info(f"  Config: {result['config']}")
        logger.info(f"  Selectivity: {result['orientation_selectivity']['fraction_selective']*100:.1f}% "
                   f"(max_corr: {result['orientation_selectivity']['max_correlation']:.3f})")
        logger.info(f"  Recon loss: {result['test_recon_loss']:.4f}")
        logger.info(f"  Sparsity: {result['sparsity']['lifetime_sparsity']*100:.1f}%")
        logger.info(f"  Feature diversity: {result['diversity']['mean_feature_correlation']:.3f}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    results_path = output_dir / f'hyperparameter_search_session_{args.session_id}.json'
    with open(results_path, 'w') as f:
        json.dump({
            'session_id': args.session_id,
            'search_space': search_space,
            'top_configs': top_configs
        }, f, indent=2)

    logger.info(f"\n✓ Results saved to: {results_path}")

    # Create summary table
    df = pd.DataFrame([
        {
            'rank': i,
            'hidden_dim': r['config']['hidden_dim'],
            'sparsity': r['config']['sparsity'],
            'lr': r['config']['lr'],
            'activation': r['config']['activation'],
            'selectivity_%': r['orientation_selectivity']['fraction_selective'] * 100,
            'recon_loss': r['test_recon_loss'],
            'sparsity_%': r['sparsity']['lifetime_sparsity'] * 100,
            'score': r['composite_score']
        }
        for i, r in enumerate(top_configs, 1)
    ])

    csv_path = output_dir / f'hyperparameter_search_session_{args.session_id}.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"✓ Summary table saved to: {csv_path}")

    logger.info("\n" + "="*80)
    logger.info("RECOMMENDED CONFIGURATION")
    logger.info("="*80)
    best = top_configs[0]
    logger.info(f"Hidden dim: {best['config']['hidden_dim']}")
    logger.info(f"Sparsity: {best['config']['sparsity']}")
    logger.info(f"Learning rate: {best['config']['lr']}")
    logger.info(f"Activation: {best['config']['activation']}")
    logger.info(f"\nExpected performance:")
    logger.info(f"  Selectivity: {best['orientation_selectivity']['fraction_selective']*100:.1f}%")
    logger.info(f"  Reconstruction: {best['test_recon_loss']:.4f}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
