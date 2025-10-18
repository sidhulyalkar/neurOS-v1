"""
NeuroFM-X Quick Start Demo

This script demonstrates the complete NeuroFM-X pipeline:
1. Generate synthetic neural data
2. Create and configure model
3. Train with PyTorch (simple version, no Lightning)
4. Evaluate performance
5. Test transfer learning with adapters

Run: python examples/quickstart_demo.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

print("="*70)
print("NeuroFM-X Quick Start Demo")
print("="*70)

# Create synthetic dataset
print("\n[1/6] Generating synthetic neural data...")
from neuros_neurofm.datasets.synthetic import (
    SyntheticNeuralDataset,
    collate_neurofmx,
    create_dataloaders,
)

dataset = SyntheticNeuralDataset(
    n_samples=1000,
    n_units=96,
    seq_length=100,
    behavior_dim=2,
    seed=42,
)

train_loader, val_loader = create_dataloaders(
    dataset,
    batch_size=32,
    train_split=0.8,
    num_workers=0,  # Use 0 for compatibility
)

print(f"  ✓ Created dataset: {len(dataset)} samples")
print(f"  ✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Build model without Mamba (since mamba-ssm may not be installed)
print("\n[2/6] Building NeuroFM-X model...")
from neuros_neurofm.tokenizers import BinnedTokenizer
from neuros_neurofm.fusion import PerceiverIO
from neuros_neurofm.models.popt import PopT
from neuros_neurofm.models.heads import MultiTaskHeads


class SimpleNeuroFMX(nn.Module):
    """Simplified NeuroFM-X without Mamba (for CPU demo)."""

    def __init__(self):
        super().__init__()

        # Tokenizer
        self.tokenizer = BinnedTokenizer(
            n_units=96,
            d_model=256,
            use_sqrt_transform=True,
        )

        # Simple temporal encoder (replaces Mamba)
        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=2,
        )

        # Perceiver fusion
        self.fusion = PerceiverIO(
            n_latents=32,
            latent_dim=128,
            input_dim=256,
            n_layers=2,
        )

        # PopT aggregator
        self.popt = PopT(
            d_model=128,
            n_layers=2,
            n_output_seeds=1,
        )

        # Multi-task heads
        self.heads = MultiTaskHeads(
            input_dim=128,
            decoder_output_dim=2,  # 2D position
            encoder_output_dim=96,  # 96 neurons
            enable_decoder=True,
            enable_encoder=True,
            enable_contrastive=False,  # Disable for simplicity
        )

    def forward(self, spikes):
        # Tokenize
        tokens, mask = self.tokenizer(spikes)

        # Temporal encoding
        tokens = self.temporal_encoder(tokens)

        # Fusion
        latents = self.fusion(tokens, mask)

        # PopT aggregation
        aggregated = self.popt(latents)

        return aggregated


model = SimpleNeuroFMX()
num_params = sum(p.numel() for p in model.parameters())
print(f"  ✓ Model created: {num_params:,} parameters")

# Training loop
print("\n[3/6] Training model...")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
num_epochs = 5

for epoch in range(num_epochs):
    # Train
    model.train()
    train_losses = []

    for batch in train_loader:
        optimizer.zero_grad()

        # Forward
        latents = model(batch["spikes"])

        # Decoder loss (predict last position from spikes)
        pred_behavior = model.heads(latents, task="decoder")
        loss = F.mse_loss(pred_behavior, batch["behavior_target"])

        # Backward
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    # Validate
    model.eval()
    val_losses = []
    val_r2_scores = []

    with torch.no_grad():
        for batch in val_loader:
            latents = model(batch["spikes"])
            pred_behavior = model.heads(latents, task="decoder")

            # Loss
            loss = F.mse_loss(pred_behavior, batch["behavior_target"])
            val_losses.append(loss.item())

            # R² score
            ss_res = ((batch["behavior_target"] - pred_behavior) ** 2).sum()
            ss_tot = ((batch["behavior_target"] - batch["behavior_target"].mean()) ** 2).sum()
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            val_r2_scores.append(r2.item())

    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)
    avg_r2 = np.mean(val_r2_scores)

    print(f"  Epoch {epoch+1}/{num_epochs}: "
          f"train_loss={avg_train_loss:.4f}, "
          f"val_loss={avg_val_loss:.4f}, "
          f"R²={avg_r2:.4f}")

print(f"  ✓ Training complete!")

# Evaluate
print("\n[4/6] Evaluating final performance...")
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch in val_loader:
        latents = model(batch["spikes"])
        pred_behavior = model.heads(latents, task="decoder")
        all_preds.append(pred_behavior)
        all_targets.append(batch["behavior_target"])

all_preds = torch.cat(all_preds, dim=0)
all_targets = torch.cat(all_targets, dim=0)

# Compute final R²
ss_res = ((all_targets - all_preds) ** 2).sum()
ss_tot = ((all_targets - all_targets.mean()) ** 2).sum()
final_r2 = 1 - ss_res / (ss_tot + 1e-8)

print(f"  ✓ Final R² score: {final_r2:.4f}")

# Test transfer learning
print("\n[5/6] Testing transfer learning with Unit-ID adapter...")
from neuros_neurofm.adapters import UnitIDAdapter

# Create adapter
adapter = UnitIDAdapter(
    backbone_dim=128,
    n_units=96,
    bottleneck_dim=64,
    freeze_backbone=True,
)

print(f"  ✓ Adapter created with {sum(p.numel() for p in adapter.parameters()):,} parameters")

# Freeze base model
for param in model.parameters():
    param.requires_grad = False

# Test adapter forward pass
test_batch = next(iter(val_loader))
latents = model(test_batch["spikes"])

# Reshape for adapter (needs batch, units, dim)
latents_expanded = latents.unsqueeze(1).expand(-1, 96, -1)
unit_indices = torch.arange(96).unsqueeze(0).expand(latents_expanded.shape[0], -1)

adapted_latents = adapter(latents_expanded, unit_indices)
print(f"  ✓ Adapter forward pass: {latents_expanded.shape} → {adapted_latents.shape}")

# Model saving/loading
print("\n[6/6] Testing model save/load...")
save_path = Path("neurofmx_demo_checkpoint.pt")

# Save
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": num_epochs,
    "r2_score": final_r2.item(),
}
torch.save(checkpoint, save_path)
print(f"  ✓ Model saved to {save_path}")

# Load
loaded_checkpoint = torch.load(save_path)
new_model = SimpleNeuroFMX()
new_model.load_state_dict(loaded_checkpoint["model_state_dict"], strict=False)
print(f"  ✓ Model loaded successfully")
print(f"  ✓ Checkpoint info: Epoch {loaded_checkpoint['epoch']}, R²={loaded_checkpoint['r2_score']:.4f}")

# Clean up
save_path.unlink()

# Summary
print("\n" + "="*70)
print("✅ DEMO COMPLETE!")
print("="*70)
print("\nWhat we demonstrated:")
print("  1. ✅ Generated synthetic neural data with behavioral correlates")
print("  2. ✅ Built NeuroFM-X model (Tokenizer→Encoder→Fusion→PopT→Heads)")
print(f"  3. ✅ Trained for {num_epochs} epochs (R²: {final_r2:.4f})")
print("  4. ✅ Evaluated behavioral decoding performance")
print("  5. ✅ Tested transfer learning with Unit-ID adapter")
print("  6. ✅ Saved and loaded model checkpoint")
print("\nNext steps:")
print("  - Try with real neural data (IBL, Allen, DANDI)")
print("  - Add more modalities (LFP, calcium imaging)")
print("  - Fine-tune with LoRA adapters")
print("  - Scale up with PyTorch Lightning + distributed training")
print("="*70)
