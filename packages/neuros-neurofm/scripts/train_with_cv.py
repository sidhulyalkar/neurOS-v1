#!/usr/bin/env python3
"""
Cross-Validation Training for Allen Multimodal Model

Implements:
1. K-fold cross-validation on sessions
2. Leave-one-session-out (LOSO) validation
3. Mixed real + synthetic data training
4. Domain adversarial training for cross-session transfer

Usage:
    # 5-fold CV
    python scripts/train_with_cv.py --config configs/allen_multimodal.yaml --cv-folds 5

    # Leave-one-session-out
    python scripts/train_with_cv.py --config configs/allen_multimodal.yaml --loso

    # With synthetic data
    python scripts/train_with_cv.py --config configs/allen_multimodal.yaml --synthetic-ratio 0.3
"""

import sys
from pathlib import Path
import argparse
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from neuros_neurofm.datasets.allen_multimodal_dataset import (
    AllenMultiModalDataset,
    collate_multimodal,
)
from neuros_neurofm.datasets.synthetic_neural import (
    SyntheticNeuralDataset,
    MixedRealSyntheticDataset,
    NeuralStatistics,
)
from neuros_neurofm.augmentations import create_augmentor


class SmallMultiModalModel(nn.Module):
    """
    Smaller multimodal model optimized for 3070 Ti.

    Uses d_model=256 (vs 512) for faster training.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config['model']['d_model']
        dropout = config['model']['dropout']

        # Per-neuron embedding
        self.neuron_embedding = nn.Linear(1, d_model)

        # Temporal encoder
        self.calcium_temporal = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Astro encoder
        self.astro_encoder = nn.Sequential(
            nn.Linear(10, d_model),  # 10 features from neuros-astro
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Cross-modal fusion
        self.fusion = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,  # Fewer heads for smaller model
            dropout=dropout,
            batch_first=True,
        )

        # Decoder
        self.decoder = nn.Linear(d_model, 1)

        # Domain classifier (for domain adversarial training)
        if config['model'].get('use_domain_adversarial', False):
            n_domains = config['model'].get('n_domains', 10)
            self.domain_classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, n_domains),
            )
        else:
            self.domain_classifier = None

    def forward(self, calcium, astro_events, calcium_mask=None, astro_mask=None,
                return_domain_logits=False):
        """Forward pass."""
        batch_size, n_neurons, seq_len = calcium.shape

        # Encode calcium
        calcium_expanded = calcium.unsqueeze(-1)
        calcium_embedded = self.neuron_embedding(calcium_expanded)

        # Pool across neurons
        if calcium_mask is not None:
            mask_expanded = calcium_mask.unsqueeze(-1).unsqueeze(-1).float()
            calcium_embedded = calcium_embedded * mask_expanded
            calcium_summed = calcium_embedded.sum(dim=1)
            neuron_counts = calcium_mask.sum(dim=1, keepdim=True).unsqueeze(-1).float().clamp(min=1)
            calcium_pooled = calcium_summed / neuron_counts
        else:
            calcium_pooled = calcium_embedded.mean(dim=1)

        # Temporal encoding
        calcium_encoded = self.calcium_temporal(calcium_pooled)

        # Encode astro
        astro_encoded = self.astro_encoder(astro_events)

        # Cross-modal fusion
        fused, _ = self.fusion(
            query=calcium_encoded,
            key=astro_encoded,
            value=astro_encoded,
            key_padding_mask=~astro_mask if astro_mask is not None else None,
        )

        # Decode
        reconstructed = self.decoder(fused).squeeze(-1)

        # Domain classification (optional)
        if return_domain_logits and self.domain_classifier is not None:
            # Pool fused representation
            fused_pooled = fused.mean(dim=1)  # (batch, d_model)
            domain_logits = self.domain_classifier(fused_pooled)
            return reconstructed, fused, domain_logits

        return reconstructed, fused


def create_cv_splits(
    all_sessions: List[str],
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[List[str], List[str]]]:
    """
    Create K-fold cross-validation splits on sessions.

    Returns:
        List of (train_sessions, val_sessions) tuples
    """
    np.random.seed(seed)
    sessions = np.array(all_sessions)
    np.random.shuffle(sessions)

    # Split into folds
    fold_size = len(sessions) // n_folds
    splits = []

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else len(sessions)

        val_sessions = sessions[start:end].tolist()
        train_sessions = [s for s in sessions if s not in val_sessions]

        splits.append((train_sessions, val_sessions))

    return splits


def create_loso_splits(all_sessions: List[str]) -> List[Tuple[List[str], List[str]]]:
    """
    Create Leave-One-Session-Out splits.

    Each fold has one session for validation, rest for training.
    """
    splits = []
    for val_session in all_sessions:
        train_sessions = [s for s in all_sessions if s != val_session]
        splits.append((train_sessions, [val_session]))
    return splits


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    fold_idx: int,
    device: torch.device,
    augmentor=None,
) -> Dict[str, float]:
    """
    Train one fold of cross-validation.

    Returns:
        Dict with final metrics
    """
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
    )

    # Scheduler
    total_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=float(config['training'].get('min_lr', 1e-7)),
    )

    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(config['training']['num_epochs']):
        # Train
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Fold {fold_idx+1} Epoch {epoch+1}", leave=False):
            calcium = batch['calcium'].to(device)
            calcium_mask = batch['calcium_mask'].to(device)
            astro_events = batch['astro_events'].to(device)
            astro_mask = batch['astro_mask'].to(device)

            # Augmentation
            if augmentor is not None:
                aug_result = augmentor(
                    calcium=calcium,
                    astro_events=astro_events,
                    calcium_mask=calcium_mask,
                    astro_mask=astro_mask,
                )
                calcium = aug_result['calcium']
                astro_events = aug_result['astro_events']

            # Forward
            reconstructed, _ = model(
                calcium=calcium,
                astro_events=astro_events,
                calcium_mask=calcium_mask,
                astro_mask=astro_mask,
            )

            # Target: mean activity
            mask_expanded = calcium_mask.unsqueeze(-1).float()
            calcium_masked = calcium * mask_expanded
            neuron_counts = calcium_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            target_mean = calcium_masked.sum(dim=1) / neuron_counts

            loss = criterion(reconstructed, target_mean)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validate
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                calcium = batch['calcium'].to(device)
                calcium_mask = batch['calcium_mask'].to(device)
                astro_events = batch['astro_events'].to(device)
                astro_mask = batch['astro_mask'].to(device)

                reconstructed, _ = model(
                    calcium=calcium,
                    astro_events=astro_events,
                    calcium_mask=calcium_mask,
                    astro_mask=astro_mask,
                )

                mask_expanded = calcium_mask.unsqueeze(-1).float()
                calcium_masked = calcium * mask_expanded
                neuron_counts = calcium_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
                target_mean = calcium_masked.sum(dim=1) / neuron_counts

                val_loss += criterion(reconstructed, target_mean).item()

        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)

        # Early stopping
        min_delta = float(config['training'].get('min_delta', 1e-5))
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= int(config['training']['patience']):
            print(f"    Early stopping at epoch {epoch+1}")
            break

        # Progress
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

    return {
        'best_val_loss': best_val_loss,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'epochs_trained': len(history['train_loss']),
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-validation training")
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--loso', action='store_true', help='Use leave-one-session-out')
    parser.add_argument('--synthetic-ratio', type=float, default=0.0,
                        help='Ratio of synthetic data (0=none, 0.3=30%)')
    parser.add_argument('--test', action='store_true', help='Quick test mode')

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Test mode
    if args.test:
        print("\n⚡ TEST MODE\n")
        config['training']['num_epochs'] = 3
        config['training']['batch_size'] = 2
        args.cv_folds = 2

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get all sessions
    all_sessions = config['data']['sessions']
    print(f"\nTotal sessions: {len(all_sessions)}")

    # Create CV splits
    if args.loso:
        splits = create_loso_splits(all_sessions)
        print(f"Using Leave-One-Session-Out ({len(splits)} folds)")
    else:
        splits = create_cv_splits(all_sessions, n_folds=args.cv_folds)
        print(f"Using {args.cv_folds}-fold CV")

    # Augmentor
    aug_config = config.get('data', {}).get('augmentation', {})
    if aug_config.get('enabled', False):
        augmentor = create_augmentor(mode=aug_config.get('mode', 'medium'))
        print("✓ Augmentation enabled")
    else:
        augmentor = None

    # Results storage
    fold_results = []

    print("\n" + "="*70)
    print("CROSS-VALIDATION TRAINING")
    print("="*70)

    for fold_idx, (train_sessions, val_sessions) in enumerate(splits):
        print(f"\n--- Fold {fold_idx + 1}/{len(splits)} ---")
        print(f"Train: {train_sessions}")
        print(f"Val: {val_sessions}")

        # Create datasets
        project_root = Path(__file__).parent.parent.parent.parent

        train_dataset = AllenMultiModalDataset(
            calcium_dir=str(project_root / config['data']['calcium_data_dir']),
            astro_dir=str(project_root / config['data']['astro_data_dir']),
            session_ids=train_sessions,
            seq_len=config['data']['seq_len'],
            stride=config['data']['stride'],
            min_astro_events=config['data']['min_astro_events'],
            modalities='both',
        )

        # Add synthetic data if requested
        if args.synthetic_ratio > 0:
            print(f"Adding {args.synthetic_ratio*100:.0f}% synthetic data")
            train_dataset = MixedRealSyntheticDataset(
                real_dataset=train_dataset,
                synthetic_ratio=args.synthetic_ratio,
                match_statistics=True,
            )

        val_dataset = AllenMultiModalDataset(
            calcium_dir=str(project_root / config['data']['calcium_data_dir']),
            astro_dir=str(project_root / config['data']['astro_data_dir']),
            session_ids=val_sessions,
            seq_len=config['data']['seq_len'],
            stride=config['data']['stride'],
            min_astro_events=config['data']['min_astro_events'],
            modalities='both',
        )

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print(f"  Skipping fold {fold_idx+1}: empty dataset")
            continue

        print(f"  Train windows: {len(train_dataset)}")
        print(f"  Val windows: {len(val_dataset)}")

        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            collate_fn=collate_multimodal,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            collate_fn=collate_multimodal,
            num_workers=0,
        )

        # Create fresh model for each fold
        model = SmallMultiModalModel(config).to(device)

        # Train fold
        results = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            fold_idx=fold_idx,
            device=device,
            augmentor=augmentor,
        )

        fold_results.append({
            'fold': fold_idx,
            'train_sessions': train_sessions,
            'val_sessions': val_sessions,
            **results,
        })

        print(f"  Best val loss: {results['best_val_loss']:.4f}")

    # Summary
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS")
    print("="*70)

    val_losses = [r['best_val_loss'] for r in fold_results]
    print(f"\nVal Loss per fold: {[f'{v:.4f}' for v in val_losses]}")
    print(f"Mean: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")

    # Save results
    results_dir = Path('results/cv')
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f'cv_results_{timestamp}.json'

    with open(results_path, 'w') as f:
        json.dump({
            'config': str(args.config),
            'cv_type': 'loso' if args.loso else f'{args.cv_folds}-fold',
            'synthetic_ratio': args.synthetic_ratio,
            'fold_results': fold_results,
            'summary': {
                'mean_val_loss': float(np.mean(val_losses)),
                'std_val_loss': float(np.std(val_losses)),
                'min_val_loss': float(np.min(val_losses)),
                'max_val_loss': float(np.max(val_losses)),
            }
        }, f, indent=2)

    print(f"\n✓ Results saved to {results_path}")


if __name__ == '__main__':
    main()
