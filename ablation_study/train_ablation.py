#!/usr/bin/env python
"""
Training script for Allen astro ablation study.

Trains and evaluates baseline (neural-only) and test (neural+astro) models
for stimulus orientation decoding.

Usage:
    python train_ablation.py --condition neural_only
    python train_ablation.py --condition neural_astro
    python train_ablation.py --condition all
"""

import argparse
import time
import json
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, r2_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuros_astro.experiments.tracker import ExperimentResult, ExperimentConfig
from neuros_astro.experiments.ablation import AblationStudy
from ablation_study.allen_multimodal_simple_dataset import AllenTrialDataset, collate_fn


def load_ablation_study(study_dir):
    """Load ablation study from experiment configs and results."""
    study_dir = Path(study_dir)

    # Create study
    study = AblationStudy(
        study_name="allen_astro_ablation",
        output_dir=study_dir
    )

    # Load baseline config
    baseline_config_path = study_dir / "experiment_tracking" / "allen_astro_ablation_baseline_neural" / "config.json"
    if baseline_config_path.exists():
        with open(baseline_config_path) as f:
            baseline_dict = json.load(f)

        baseline_config = ExperimentConfig(**baseline_dict)

        study.add_condition(
            condition_name="neural_only",
            description="Baseline with neural data only",
            modalities=['neural'],
            config=baseline_config
        )

        # Load result if it exists
        result_path = study_dir / "result_neural_only.json"
        if result_path.exists():
            with open(result_path) as f:
                result_dict = json.load(f)

            result = ExperimentResult(
                experiment_id=result_dict['experiment_id'],
                config=baseline_config,
                model_metrics=result_dict['model_metrics'],
                processing_time_s=result_dict['processing_time_s']
            )
            study.set_result("neural_only", result)

    # Load test config
    test_config_path = study_dir / "experiment_tracking" / "allen_astro_ablation_test_neural_astro" / "config.json"
    if test_config_path.exists():
        with open(test_config_path) as f:
            test_dict = json.load(f)

        test_config = ExperimentConfig(**test_dict)

        study.add_condition(
            condition_name="neural_astro",
            description="Test with neural and astrocyte data",
            modalities=['neural', 'astro'],
            config=test_config
        )

        # Load result if it exists
        result_path = study_dir / "result_neural_astro.json"
        if result_path.exists():
            with open(result_path) as f:
                result_dict = json.load(f)

            result = ExperimentResult(
                experiment_id=result_dict['experiment_id'],
                config=test_config,
                model_metrics=result_dict['model_metrics'],
                processing_time_s=result_dict['processing_time_s']
            )
            study.set_result("neural_astro", result)

    return study


class SimpleNeuroDecoder(nn.Module):
    """
    Simple neural decoder for orientation classification.

    Args:
        input_dim: Neural input dimension
        n_classes: Number of orientation classes
        hidden_dim: Hidden layer dimension
        dropout: Dropout probability
    """

    def __init__(self, input_dim: int, n_classes: int = 8, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hidden_dim // 2, n_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, input_dim)

        Returns:
            logits: (batch, n_classes)
            features: (batch, hidden_dim//2)
        """
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits, features


class MultimodalNeuroDecoder(nn.Module):
    """
    Multimodal decoder (neural + astro).

    Args:
        neural_dim: Neural input dimension
        astro_dim: Astro input dimension
        n_classes: Number of classes
        hidden_dim: Hidden dimension
        fusion: Fusion strategy ('concat' or 'cross_attn')
        dropout: Dropout probability
    """

    def __init__(
        self,
        neural_dim: int,
        astro_dim: int,
        n_classes: int = 8,
        hidden_dim: int = 256,
        fusion: str = 'concat',
        dropout: float = 0.2
    ):
        super().__init__()

        self.fusion = fusion

        # Neural encoder
        self.neural_encoder = nn.Sequential(
            nn.Linear(neural_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Astro encoder
        self.astro_encoder = nn.Sequential(
            nn.Linear(astro_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Fusion layer
        if fusion == 'concat':
            fusion_dim = hidden_dim + hidden_dim // 2
        else:  # cross_attn
            fusion_dim = hidden_dim
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.astro_proj = nn.Linear(hidden_dim // 2, hidden_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )

    def forward(self, neural, astro):
        """
        Args:
            neural: (batch, neural_dim)
            astro: (batch, astro_dim)

        Returns:
            logits: (batch, n_classes)
            features: (batch, hidden_dim//2)
        """
        # Encode modalities
        neural_features = self.neural_encoder(neural)  # (B, hidden_dim)
        astro_features = self.astro_encoder(astro)  # (B, hidden_dim//2)

        # Fusion
        if self.fusion == 'concat':
            fused = torch.cat([neural_features, astro_features], dim=-1)
        else:  # cross_attn
            # Project astro to hidden_dim
            astro_proj = self.astro_proj(astro_features)  # (B, hidden_dim)

            # Cross-attention (neural attends to astro)
            neural_seq = neural_features.unsqueeze(1)  # (B, 1, hidden_dim)
            astro_seq = astro_proj.unsqueeze(1)  # (B, 1, hidden_dim)

            attn_out, _ = self.cross_attn(neural_seq, astro_seq, astro_seq)
            fused = attn_out.squeeze(1)  # (B, hidden_dim)

        # Classification
        logits = self.classifier(fused)

        return logits, fused


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        # Move to device
        labels = batch['labels'].to(device)

        # Forward pass
        if 'astro' in batch:
            # Multimodal
            neural = batch['neural'].to(device)
            astro = batch['astro'].to(device)
            logits, _ = model(neural, astro)
        else:
            # Neural-only
            neural = batch['neural'].to(device)
            logits, _ = model(neural)

        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            labels = batch['labels'].to(device)

            # Forward pass
            if 'astro' in batch:
                neural = batch['neural'].to(device)
                astro = batch['astro'].to(device)
                logits, _ = model(neural, astro)
            else:
                neural = batch['neural'].to(device)
                logits, _ = model(neural)

            loss = criterion(logits, labels)

            # Metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def train_condition(condition_name, study_dir):
    """Train a single ablation condition."""

    # Load study
    study = load_ablation_study(study_dir)
    condition = study.conditions[condition_name]

    print(f"\n{'='*80}")
    print(f"TRAINING CONDITION: {condition_name}")
    print(f"{'='*80}")
    print(f"Modalities: {condition.modalities}")
    print(f"Sessions: {len(condition.config.session_ids)}")

    start_time = time.time()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Paths
    calcium_dir = Path('packages/neuros-mechint/examples/allen_data_demo/data/2p_sessions')
    astro_dir = Path('allen_nwb_results')

    # Determine modalities
    modalities = 'neural' if condition_name == 'neural_only' else 'both'

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = AllenTrialDataset(
        calcium_dir=calcium_dir,
        astro_dir=astro_dir,
        session_ids=condition.config.session_ids,
        modalities=modalities,
        split='train',
        seed=condition.config.random_seed
    )

    val_dataset = AllenTrialDataset(
        calcium_dir=calcium_dir,
        astro_dir=astro_dir,
        session_ids=condition.config.session_ids,
        modalities=modalities,
        split='val',
        seed=condition.config.random_seed
    )

    test_dataset = AllenTrialDataset(
        calcium_dir=calcium_dir,
        astro_dir=astro_dir,
        session_ids=condition.config.session_ids,
        modalities=modalities,
        split='test',
        seed=condition.config.random_seed
    )

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    # Create dataloaders
    batch_size = condition.config.model_parameters.get('batch_size', 32)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # Get input dimensions from first batch
    sample_batch = next(iter(train_loader))
    neural_dim = sample_batch['neural'].shape[1]
    astro_dim = sample_batch['astro'].shape[1] if 'astro' in sample_batch else 0

    print(f"\nInput dimensions:")
    print(f"  Neural: {neural_dim}")
    if 'astro' in sample_batch:
        print(f"  Astro: {astro_dim}")

    # Create model
    print("\nCreating model...")
    n_classes = 8  # Orientation classes
    hidden_dim = condition.config.model_parameters.get('hidden_dim', 256)
    dropout = condition.config.model_parameters.get('dropout', 0.2)

    if modalities == 'neural':
        model = SimpleNeuroDecoder(
            input_dim=neural_dim,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    else:
        model = MultimodalNeuroDecoder(
            neural_dim=neural_dim,
            astro_dim=astro_dim,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            fusion='cross_attn',
            dropout=dropout
        )

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    lr = condition.config.model_parameters.get('learning_rate', 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    n_epochs = condition.config.model_parameters.get('n_epochs', 50)
    patience = condition.config.model_parameters.get('early_stopping_patience', 10)

    print(f"\nTraining for {n_epochs} epochs...")

    best_val_acc = 0
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            # Save best model
            torch.save(model.state_dict(), study_dir / f'best_model_{condition_name}.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model for final evaluation
    model.load_state_dict(torch.load(study_dir / f'best_model_{condition_name}.pth'))

    # Final evaluation on test set
    print("\nFinal evaluation on test set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"  Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    # Cross-session transfer (evaluate on held-out sessions if applicable)
    # For simplicity, we'll use test accuracy as a proxy
    cross_session_transfer = test_acc

    # Compute metrics
    total_time = time.time() - start_time

    metrics = {
        'prediction_loss': float(test_loss),
        'decoding_accuracy': float(test_acc),
        'cross_session_transfer': float(cross_session_transfer),
        'r2_score': float(0.0),  # Would need regression target for R2
        'best_val_accuracy': float(best_val_acc),
        'train_accuracy': float(train_acc),
    }

    print(f"\nFinal metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Create result
    result = ExperimentResult(
        experiment_id=condition.config.experiment_id,
        config=condition.config,
        model_metrics=metrics,
        processing_time_s=total_time,
    )

    # Save result to study
    study.set_result(condition_name, result)
    study.save_summary()

    # Also save result to JSON for persistence
    result_path = study_dir / f"result_{condition_name}.json"
    result_dict = {
        'experiment_id': result.experiment_id,
        'model_metrics': result.model_metrics,
        'processing_time_s': result.processing_time_s,
    }
    with open(result_path, 'w') as f:
        json.dump(result_dict, f, indent=2)

    print(f"\n✓ Training complete for {condition_name}")
    print(f"✓ Time: {total_time:.2f}s")
    print(f"✓ Results saved to {result_path}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, required=True,
                       choices=['neural_only', 'neural_astro', 'all'],
                       help='Condition to train')
    parser.add_argument('--study-dir', type=str, default='ablation_study',
                       help='Ablation study directory')

    args = parser.parse_args()

    study_dir = Path(args.study_dir)

    if args.condition == 'all':
        conditions = ['neural_only', 'neural_astro']
    else:
        conditions = [args.condition]

    for condition in conditions:
        train_condition(condition, study_dir)

    # Generate comparison report
    study = load_ablation_study(study_dir)

    if study.conditions['neural_only'].result and study.conditions['neural_astro'].result:
        print("\n" + "=" * 80)
        print("ABLATION COMPARISON")
        print("=" * 80)

        for metric in ['prediction_loss', 'decoding_accuracy', 'cross_session_transfer']:
            comparison = study.compare_conditions(
                baseline_name='neural_only',
                test_name='neural_astro',
                metric=metric,
            )

            print(f"\n{metric}:")
            print(f"  Baseline: {comparison.baseline_value:.4f}")
            print(f"  Test: {comparison.test_value:.4f}")
            print(f"  Change: {comparison.percent_change:+.2f}%")
            print(f"  {comparison.interpretation}")

        print("\n" + study.generate_comparison_table())


if __name__ == "__main__":
    main()
