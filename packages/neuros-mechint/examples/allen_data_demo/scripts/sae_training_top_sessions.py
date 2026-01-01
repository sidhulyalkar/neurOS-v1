#!/usr/bin/env python3
"""
SAE Training on Top Allen Sessions

This script demonstrates how to train SAEs on the best Allen sessions
identified by the multi-session validation analysis.

Usage:
    # Train on single best session
    python examples/sae_training_top_sessions.py --session-config session_analysis/recommended_sessions.json

    # Train on top 5 sessions
    python examples/sae_training_top_sessions.py --session-config session_analysis/recommended_sessions.json --use-top-n 5

    # Train with specific hyperparameters
    python examples/sae_training_top_sessions.py --session-config session_analysis/recommended_sessions.json \
        --sae-dim 256 --sparsity 0.01 --epochs 100
"""

import json
import argparse
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_session_config(config_path: str) -> dict:
    """Load recommended session configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_allen_session_data(
    session_id: int,
    cache_dir: str,
    use_all_units: bool = True
) -> tuple:
    """
    Load neural data from a specific Allen session.

    Returns
    -------
    windows : List[NeuralWindow]
        Neural activity windows
    labels : Dict[str, np.ndarray]
        Orientation labels
    properties : Dict
        Session properties
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "neuros-foundation" / "src"))

    from neuros.datasets.allen_datasets import AllenVisualCodingValidator

    logger.info(f"Loading Allen session {session_id}...")

    validator = AllenVisualCodingValidator(
        session_id=session_id,
        cache_dir=cache_dir,
        brain_areas=['VISp'],
        use_all_units=use_all_units
    )

    # Extract neural windows
    windows = validator.get_neural_windows(
        window_length=1.0,
        stride=0.5,
        bin_size=0.02
    )

    # Get labels
    labels = validator.get_task_labels()

    # Get properties
    properties = validator.get_neural_properties()

    logger.info(f"  ✓ Loaded {len(windows)} windows from {properties['n_units']} units")

    return windows, labels, properties


def prepare_sae_training_data(windows: List, labels: Dict) -> tuple:
    """
    Prepare data for SAE training.

    Returns
    -------
    X : np.ndarray
        Neural activity matrix [n_windows, n_features]
    y : np.ndarray
        Orientation labels (0-180°)
    """
    # Filter windows with valid orientations and extract data
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
            # Convert direction to orientation (0-180°)
            ori_180 = ori_float % 180

            # Extract mean firing rate for this window
            valid_data.append(w.data.mean(axis=0))
            valid_orientations.append(ori_180)
        except (ValueError, TypeError):
            # Skip windows with invalid orientation values
            continue

    # Convert to numpy arrays
    X = np.array(valid_data)
    orientations = np.array(valid_orientations)

    n_skipped = len(windows) - len(valid_data)
    if n_skipped > 0:
        logger.info(f"  Skipped {n_skipped} windows with invalid orientations")

    logger.info(f"  Training data shape: {X.shape}")
    logger.info(f"  Labels shape: {orientations.shape}")
    logger.info(f"  Unique orientations: {np.unique(orientations)}")

    return X, orientations


class SimpleSAE(nn.Module):
    """
    Simple Sparse Autoencoder for demonstration.

    For production, use a proper SAE implementation with:
    - L1 sparsity penalty
    - Tied weights option
    - TopK activation
    - Feature normalization
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encode
        h = self.relu(self.encoder(x))

        # Decode
        x_recon = self.decoder(h)

        return x_recon, h

    def encode(self, x):
        """Get SAE features."""
        with torch.no_grad():
            h = self.relu(self.encoder(x))
        return h


def train_sae_on_session(
    session_id: int,
    cache_dir: str,
    sae_dim: int = 128,
    sparsity: float = 0.01,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict[str, Any]:
    """
    Train SAE on a single session and evaluate orientation selectivity.

    Returns
    -------
    Dict with:
        - sae_model
        - training_loss
        - validation_results (orientation selectivity of SAE features)
    """
    # Load data
    windows, labels, properties = load_allen_session_data(
        session_id=session_id,
        cache_dir=cache_dir,
        use_all_units=True
    )

    # Prepare training data
    X, orientations = prepare_sae_training_data(windows, labels)

    # Normalize
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X_norm = (X - X_mean) / X_std

    # Split train/test
    n_train = int(0.8 * len(X_norm))
    X_train = X_norm[:n_train]
    X_test = X_norm[n_train:]
    y_train = orientations[:n_train]
    y_test = orientations[n_train:]

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)

    # Initialize SAE
    input_dim = X_train.shape[1]
    sae = SimpleSAE(input_dim=input_dim, hidden_dim=sae_dim).to(device)

    # Optimizer
    optimizer = optim.Adam(sae.parameters(), lr=lr)

    # Training loop
    logger.info(f"\nTraining SAE on session {session_id}...")
    logger.info(f"  Input dim: {input_dim}, Hidden dim: {sae_dim}")
    logger.info(f"  Device: {device}")

    losses = []

    for epoch in range(epochs):
        sae.train()

        # Forward pass
        x_recon, h = sae(X_train_t)

        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(x_recon, X_train_t)

        # L1 sparsity penalty
        sparsity_loss = sparsity * h.abs().mean()

        # Total loss
        loss = recon_loss + sparsity_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs}: Loss = {loss.item():.4f} (recon: {recon_loss.item():.4f}, sparsity: {sparsity_loss.item():.4f})")

    # Evaluate on test set
    sae.eval()
    with torch.no_grad():
        x_test_recon, h_test = sae(X_test_t)
        test_loss = nn.functional.mse_loss(x_test_recon, X_test_t)

    logger.info(f"  Test loss: {test_loss.item():.4f}")

    # Analyze SAE features for orientation selectivity
    logger.info("\nAnalyzing SAE feature orientation selectivity...")

    from scipy.stats import pearsonr

    h_test_np = h_test.cpu().numpy()

    # Compute orientation tuning using circular correlation
    orientation_sin = np.sin(np.deg2rad(y_test * 2))
    orientation_cos = np.cos(np.deg2rad(y_test * 2))

    correlations = []
    for feature_idx in range(h_test_np.shape[1]):
        feature_response = h_test_np[:, feature_idx]

        corr_sin, _ = pearsonr(feature_response, orientation_sin)
        corr_cos, _ = pearsonr(feature_response, orientation_cos)

        correlations.append(max(abs(corr_sin), abs(corr_cos)))

    correlations = np.array(correlations)

    max_corr = np.max(correlations)
    mean_corr = np.mean(correlations)
    n_significant = np.sum(correlations > 0.3)
    fraction_selective = n_significant / len(correlations)

    logger.info(f"  SAE Features:")
    logger.info(f"    Max correlation: {max_corr:.3f}")
    logger.info(f"    Mean correlation: {mean_corr:.3f}")
    logger.info(f"    Significant features (>0.3): {n_significant}/{len(correlations)} ({fraction_selective*100:.1f}%)")

    return {
        'session_id': session_id,
        'sae_model': sae,
        'training_losses': losses,
        'test_loss': test_loss.item(),
        'validation_results': {
            'max_correlation': float(max_corr),
            'mean_correlation': float(mean_corr),
            'n_significant': int(n_significant),
            'fraction_selective': float(fraction_selective),
            'feature_correlations': correlations.tolist()
        },
        'data_stats': {
            'n_windows': len(X),
            'n_units': input_dim,
            'n_sae_features': sae_dim
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Train SAE on top Allen sessions')
    parser.add_argument('--session-config', type=str, required=True, help='Path to recommended_sessions.json')
    parser.add_argument('--allen-cache', type=str, default='allen_validation_cache', help='Allen cache directory')
    parser.add_argument('--use-top-n', type=int, default=1, help='Train on top N sessions')
    parser.add_argument('--sae-dim', type=int, default=128, help='SAE hidden dimension')
    parser.add_argument('--sparsity', type=float, default=0.01, help='Sparsity penalty weight')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='sae_models', help='Output directory for models')
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("SAE TRAINING ON TOP ALLEN SESSIONS")
    logger.info("="*80)

    # Load session configuration
    logger.info(f"\nLoading session configuration from: {args.session_config}")
    config = load_session_config(args.session_config)

    # Get top N sessions
    top_sessions = config['recommended_sessions'][f'top_{args.use_top_n}'] if args.use_top_n > 1 else [config['recommended_sessions']['best_overall']]

    logger.info(f"\nTraining SAE on {len(top_sessions)} session(s):")
    for sid in top_sessions:
        logger.info(f"  - Session {sid}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Train SAE on each session
    all_results = []

    for session_id in top_sessions:
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING ON SESSION {session_id}")
        logger.info(f"{'='*80}")

        try:
            results = train_sae_on_session(
                session_id=session_id,
                cache_dir=args.allen_cache,
                sae_dim=args.sae_dim,
                sparsity=args.sparsity,
                epochs=args.epochs,
                lr=args.lr
            )

            all_results.append(results)

            # Save model
            model_path = output_dir / f"sae_session_{session_id}.pt"
            torch.save(results['sae_model'].state_dict(), model_path)
            logger.info(f"\n✓ Model saved to: {model_path}")

        except Exception as e:
            logger.error(f"Failed to train on session {session_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary report
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)

    for results in all_results:
        val = results['validation_results']
        logger.info(f"\nSession {results['session_id']}:")
        logger.info(f"  Test loss: {results['test_loss']:.4f}")
        logger.info(f"  SAE features: {val['n_significant']}/{results['data_stats']['n_sae_features']} selective ({val['fraction_selective']*100:.1f}%)")
        logger.info(f"  Max correlation: {val['max_correlation']:.3f}")

    # Save results
    results_path = output_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        # Remove model from results before saving
        save_results = []
        for r in all_results:
            r_copy = r.copy()
            r_copy.pop('sae_model')
            save_results.append(r_copy)
        json.dump(save_results, f, indent=2)

    logger.info(f"\n✓ Results saved to: {results_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
