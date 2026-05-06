#!/usr/bin/env python3
"""
Training script for Allen Visual Coding multimodal model.

Integrates calcium imaging with astrocyte tokens from neuros-astro.

Usage:
    # Quick test (1 epoch, small batch)
    python scripts/train_allen_multimodal.py --config configs/allen_multimodal.yaml --test

    # Full training
    python scripts/train_allen_multimodal.py --config configs/allen_multimodal.yaml

    # Resume from checkpoint
    python scripts/train_allen_multimodal.py --config configs/allen_multimodal.yaml --resume checkpoints/last.pt
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add neuros-neurofm to path
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
from neuros_neurofm.augmentations import create_augmentor, AugmentationConfig


class SimpleMultiModalModel(nn.Module):
    """
    Optimized multimodal model for 3070 Ti.

    Uses smaller d_model (256) and stronger regularization.
    Handles variable neuron counts via learned embedding + pooling.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config['model']['d_model']
        dropout = config['model'].get('dropout', 0.2)

        # Calcium encoder - per-neuron embedding then temporal encoding
        self.neuron_embedding = nn.Linear(1, d_model)

        self.calcium_temporal = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Astro encoder (10 features from neuros-astro)
        self.astro_encoder = nn.Sequential(
            nn.Linear(10, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # Fusion layer (fewer heads for smaller d_model)
        n_heads = 4 if d_model <= 256 else 8
        self.fusion = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Decoder
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, calcium, astro_events, calcium_mask=None, astro_mask=None):
        """
        Forward pass with variable neuron counts.

        Args:
            calcium: (batch, n_neurons, seq_len)
            astro_events: (batch, n_events, n_features)
            calcium_mask: (batch, n_neurons)
            astro_mask: (batch, n_events)

        Returns:
            reconstructed: (batch, seq_len) - mean activity prediction
            latents: (batch, seq_len, d_model)
        """
        batch_size, n_neurons, seq_len = calcium.shape

        # Encode calcium: per-neuron embedding across time
        # Reshape: (batch, n_neurons, seq_len) -> (batch, n_neurons, seq_len, 1)
        calcium_expanded = calcium.unsqueeze(-1)

        # Embed each neuron-timepoint: (batch, n_neurons, seq_len, 1) -> (batch, n_neurons, seq_len, d_model)
        calcium_embedded = self.neuron_embedding(calcium_expanded)

        # Pool across neurons (mean of valid neurons only)
        if calcium_mask is not None:
            # Mask invalid neurons: (batch, n_neurons, 1, 1)
            mask_expanded = calcium_mask.unsqueeze(-1).unsqueeze(-1).float()
            calcium_embedded = calcium_embedded * mask_expanded
            # Sum across neurons: (batch, seq_len, d_model)
            calcium_summed = calcium_embedded.sum(dim=1)
            # Mean: divide by number of valid neurons (batch, 1, 1)
            neuron_counts = calcium_mask.sum(dim=1, keepdim=True).unsqueeze(-1).float().clamp(min=1)
            calcium_pooled = calcium_summed / neuron_counts  # (batch, seq_len, d_model)
        else:
            calcium_pooled = calcium_embedded.mean(dim=1)  # (batch, seq_len, d_model)

        # Temporal encoding
        calcium_encoded = self.calcium_temporal(calcium_pooled)  # (batch, seq_len, d_model)

        # Encode astro events
        astro_encoded = self.astro_encoder(astro_events)  # (batch, n_events, d_model)

        # Cross-modal attention
        # Query: calcium, Key/Value: astro
        fused, _ = self.fusion(
            query=calcium_encoded,
            key=astro_encoded,
            value=astro_encoded,
            key_padding_mask=~astro_mask if astro_mask is not None else None,
        )

        # Decode to mean activity prediction
        # (batch, seq_len, d_model) -> (batch, seq_len, 1) -> (batch, seq_len)
        reconstructed = self.decoder(fused).squeeze(-1)  # (batch, seq_len)

        return reconstructed, fused


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_datasets(config: dict):
    """Create train/val/test datasets."""

    data_config = config['data']
    modalities_config = config['modalities']

    # Use absolute paths
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    calcium_dir = project_root / data_config['calcium_data_dir']
    astro_dir = project_root / data_config['astro_data_dir']

    datasets = {}

    for split in ['train', 'val', 'test']:
        session_ids = data_config[f'{split}_sessions']

        if len(session_ids) == 0:
            print(f"⚠️  No sessions for {split} split")
            datasets[split] = None
            continue

        dataset = AllenMultiModalDataset(
            calcium_dir=str(calcium_dir),
            astro_dir=str(astro_dir),
            session_ids=session_ids,
            seq_len=data_config['seq_len'],
            modalities='both',
            stride=data_config['stride'],
            min_astro_events=data_config['min_astro_events'],
            temporal_alignment=data_config['alignment_method'],
        )

        # Check if dataset is empty
        if len(dataset) == 0:
            print(f"⚠️  {split.capitalize()} dataset has 0 windows! Check seq_len and min_astro_events.")
            datasets[split] = None
        else:
            # Add synthetic data for training split only
            synthetic_config = data_config.get('augmentation', {}).get('synthetic', {})
            if split == 'train' and synthetic_config.get('enabled', False):
                synthetic_ratio = synthetic_config.get('synthetic_ratio', 0.3)
                print(f"  Adding {synthetic_ratio*100:.0f}% synthetic data...")
                dataset = MixedRealSyntheticDataset(
                    real_dataset=dataset,
                    synthetic_ratio=synthetic_ratio,
                    match_statistics=synthetic_config.get('match_real_statistics', True),
                )

            datasets[split] = dataset
            print(f"✓ {split.capitalize()}: {len(dataset)} windows")

    return datasets


def create_dataloaders(datasets: dict, config: dict):
    """Create dataloaders."""

    train_config = config['training']
    hardware_config = config['hardware']

    loaders = {}

    for split, dataset in datasets.items():
        if dataset is None:
            loaders[split] = None
            continue

        loaders[split] = DataLoader(
            dataset,
            batch_size=train_config['batch_size'],
            shuffle=(split == 'train'),
            collate_fn=collate_multimodal,
            num_workers=hardware_config['num_workers'],
            pin_memory=hardware_config['pin_memory'],
        )

    return loaders


def train_epoch(model, dataloader, optimizer, criterion, device, config, augmentor=None):
    """Train for one epoch with optional augmentation."""

    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        # Move to device
        calcium = batch['calcium'].to(device)
        calcium_mask = batch['calcium_mask'].to(device)
        astro_events = batch['astro_events'].to(device)
        astro_mask = batch['astro_mask'].to(device)

        # Apply augmentation (only during training)
        if augmentor is not None:
            aug_result = augmentor(
                calcium=calcium,
                astro_events=astro_events,
                astro_timestamps=None,
                calcium_mask=calcium_mask,
                astro_mask=astro_mask,
            )
            calcium = aug_result['calcium']
            astro_events = aug_result['astro_events']
            # Masks may be modified by dropout augmentations
            if aug_result['astro_mask'] is not None:
                astro_mask = aug_result['astro_mask']

        # Forward pass
        reconstructed, _ = model(
            calcium=calcium,
            astro_events=astro_events,
            calcium_mask=calcium_mask,
            astro_mask=astro_mask,
        )

        # Compute target: mean activity across valid neurons
        # calcium: (batch, n_neurons, seq_len)
        # calcium_mask: (batch, n_neurons)
        mask_expanded = calcium_mask.unsqueeze(-1).float()  # (batch, n_neurons, 1)
        calcium_masked = calcium * mask_expanded
        neuron_counts = calcium_mask.sum(dim=1, keepdim=True).float().clamp(min=1)  # (batch, 1)
        target_mean = calcium_masked.sum(dim=1) / neuron_counts  # (batch, seq_len)

        # Loss: predict mean activity
        loss = criterion(reconstructed, target_mean)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config['training']['grad_clip']
        )

        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validate model."""

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
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

            # Compute target: mean activity across valid neurons
            mask_expanded = calcium_mask.unsqueeze(-1).float()
            calcium_masked = calcium * mask_expanded
            neuron_counts = calcium_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            target_mean = calcium_masked.sum(dim=1) / neuron_counts

            loss = criterion(reconstructed, target_mean)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train Allen multimodal model")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--test', action='store_true', help='Quick test mode (1 epoch)')

    args = parser.parse_args()

    # Load config
    config = load_config(Path(args.config))

    # Override for test mode
    if args.test:
        print("\n⚡ TEST MODE: 1 epoch, small batch\n")
        config['training']['num_epochs'] = 1
        config['training']['batch_size'] = 2
        config['hardware']['num_workers'] = 0

    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Setup device
    device = torch.device(config['hardware']['device'])
    print(f"Using device: {device}")

    # Create datasets
    print("\n" + "="*70)
    print("LOADING DATASETS")
    print("="*70 + "\n")

    datasets = create_datasets(config)

    if datasets['train'] is None or len(datasets['train']) == 0:
        print("\n❌ No training data found!")
        print("\nPossible issues:")
        print("  1. seq_len too large - your traces are only 15-17 seconds")
        print("     → Try seq_len: 100 (10 seconds) instead of 200")
        print("  2. min_astro_events too high")
        print("     → Already set to 1, should be fine")
        print("  3. Calcium/astro paths incorrect")
        print("     → Check paths in config")
        return

    # Create dataloaders
    loaders = create_dataloaders(datasets, config)

    # Create model
    print("\n" + "="*70)
    print("INITIALIZING MODEL")
    print("="*70 + "\n")

    model = SimpleMultiModalModel(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Optimizer
    learning_rate = float(config['training']['learning_rate'])  # Ensure float
    weight_decay = float(config['training']['weight_decay'])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Loss
    criterion = nn.MSELoss()

    # Create augmentor (crucial for small datasets!)
    augmentor = None
    aug_config = config.get('data', {}).get('augmentation', {})
    if aug_config.get('enabled', False):
        aug_mode = aug_config.get('mode', 'medium')
        print(f"\n✓ Data augmentation enabled (mode: {aug_mode})")

        # Create augmentor with config overrides
        augmentor = create_augmentor(
            mode=aug_mode,
            gaussian_noise_prob=aug_config.get('gaussian_noise_prob', 0.5),
            gaussian_noise_std=aug_config.get('gaussian_noise_std', 0.02),
            scaling_prob=aug_config.get('scaling_prob', 0.5),
            neuron_dropout_prob=aug_config.get('neuron_dropout_prob', 0.3),
            neuron_dropout_rate=aug_config.get('neuron_dropout_rate', 0.1),
            time_shift_prob=aug_config.get('time_shift_prob', 0.2),
            time_shift_max=aug_config.get('time_shift_max', 5),
            astro_jitter_prob=aug_config.get('astro_jitter_prob', 0.3),
            astro_jitter_sigma=aug_config.get('astro_jitter_sigma', 0.3),
        )
    else:
        print("\n⚠ Data augmentation DISABLED (consider enabling for small datasets)")

    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70 + "\n")

    best_val_loss = float('inf')

    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        print("-" * 50)

        # Train (with augmentation)
        train_loss = train_epoch(
            model=model,
            dataloader=loaders['train'],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=config,
            augmentor=augmentor,  # Apply augmentation during training
        )

        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        if loaders['val'] is not None:
            val_loss = validate(
                model=model,
                dataloader=loaders['val'],
                criterion=criterion,
                device=device,
            )

            print(f"Val Loss:   {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                checkpoint_dir = Path(config['checkpointing']['save_dir'])
                checkpoint_dir.mkdir(exist_ok=True, parents=True)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config,
                }, checkpoint_dir / 'best.pt')

                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"\nBest validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
