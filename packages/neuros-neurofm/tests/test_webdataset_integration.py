"""
Integration test for WebDataset pipeline.

Tests the complete workflow:
1. Create synthetic data
2. Convert to shards
3. Load and train with PyTorch
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from neuros_neurofm.datasets import (
    create_shards_from_arrays,
    create_webdataset_dataloader,
    SyntheticNeuralDataset,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=96, output_dim=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: (batch, time, input_dim)
        # Take mean over time
        x_mean = x.mean(dim=1)
        return self.fc(x_mean)


def test_end_to_end_workflow():
    """Test complete workflow from data creation to training."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Create synthetic data
        print("\n1. Creating synthetic data...")
        dataset = SyntheticNeuralDataset(
            n_samples=500,
            n_units=96,
            seq_length=100,
            behavior_dim=2,
        )

        # Extract data
        spikes_list = []
        behavior_list = []
        for i in range(len(dataset)):
            sample = dataset[i]
            spikes_list.append(sample["spikes"].numpy())
            behavior_list.append(sample["behavior"].numpy())

        data_dict = {
            "spikes": np.stack(spikes_list),
            "behavior": np.stack(behavior_list),
        }

        print(f"   Created {len(spikes_list)} samples")
        print(f"   Spikes shape: {data_dict['spikes'].shape}")
        print(f"   Behavior shape: {data_dict['behavior'].shape}")

        # Step 2: Convert to shards
        print("\n2. Converting to shards...")
        shard_dir = tmpdir / "shards"
        summary = create_shards_from_arrays(
            output_dir=shard_dir,
            data_dict=data_dict,
            shard_size=100,
        )

        print(f"   Created {summary['total_shards']} shards")
        print(f"   Total samples: {summary['total_samples']}")

        # Step 3: Create DataLoader
        print("\n3. Creating DataLoader...")
        train_loader = create_webdataset_dataloader(
            shard_dir=shard_dir,
            batch_size=32,
            num_workers=0,  # 0 for testing
            shuffle=True,
        )

        # Step 4: Create model
        print("\n4. Creating model...")
        model = SimpleModel(input_dim=96, output_dim=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Step 5: Training loop
        print("\n5. Running training loop...")
        model.train()

        n_batches = 0
        total_loss = 0.0

        for batch in train_loader:
            # Get data
            spikes = batch["spikes"]  # (batch, time, units)
            behavior = batch["behavior"]  # (batch, time, dims)

            # Get target (last time point)
            target = behavior[:, -1, :]  # (batch, dims)

            # Forward pass
            output = model(spikes)
            loss = criterion(output, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if n_batches >= 10:  # Limit for test
                break

        avg_loss = total_loss / n_batches
        print(f"   Processed {n_batches} batches")
        print(f"   Average loss: {avg_loss:.4f}")

        # Verify
        assert n_batches > 0
        assert avg_loss > 0
        assert not np.isnan(avg_loss)

        print("\n✓ Integration test passed!")


def test_multimodal_training():
    """Test training with multiple modalities."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create multi-modal data
        n_samples = 200
        data_dict = {
            "spikes": np.random.randn(n_samples, 100, 96).astype(np.float32),
            "lfp": np.random.randn(n_samples, 64, 1000).astype(np.float32),
            "behavior": np.random.randn(n_samples, 100, 2).astype(np.float32),
        }

        # Convert to shards
        shard_dir = tmpdir / "shards"
        create_shards_from_arrays(shard_dir, data_dict, shard_size=50)

        # Create DataLoader
        loader = create_webdataset_dataloader(
            shard_dir=shard_dir,
            batch_size=16,
            num_workers=0,
            shuffle=True,
        )

        # Verify we can load multi-modal batches
        batch = next(iter(loader))

        assert "spikes" in batch
        assert "lfp" in batch
        assert "behavior" in batch

        assert batch["spikes"].shape == (16, 100, 96)
        assert batch["lfp"].shape == (16, 64, 1000)
        assert batch["behavior"].shape == (16, 100, 2)

        print("✓ Multi-modal training test passed!")


def test_resumption_workflow():
    """Test training with resumption."""
    from neuros_neurofm.datasets import ResumableWebDatasetLoader

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create data
        n_samples = 300
        data_dict = {
            "spikes": np.random.randn(n_samples, 50, 32).astype(np.float32),
            "behavior": np.random.randn(n_samples, 50, 2).astype(np.float32),
        }

        # Convert to shards
        shard_dir = tmpdir / "shards"
        create_shards_from_arrays(shard_dir, data_dict, shard_size=100)

        # Create resumable loader
        checkpoint_path = tmpdir / "checkpoint.json"
        loader = ResumableWebDatasetLoader(
            shard_dir=shard_dir,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=50,
            shuffle=False,
        )

        # Process some samples
        samples_processed = 0
        for sample in loader:
            samples_processed += 1
            if samples_processed >= 150:
                break

        # Save final state
        loader.save_checkpoint()

        # Create new loader - should resume
        loader2 = ResumableWebDatasetLoader(
            shard_dir=shard_dir,
            checkpoint_path=checkpoint_path,
            shuffle=False,
        )

        # Get state
        state = loader2.get_state()

        # Should have resumed from checkpoint
        assert state["sample_offset"] > 0 or state["shard_idx"] > 0

        print(f"✓ Resumption test passed! Resumed from: {state}")


def test_large_batch_training():
    """Test with larger batches and multiple epochs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create data
        n_samples = 1000
        data_dict = {
            "spikes": np.random.randn(n_samples, 100, 96).astype(np.float32),
            "behavior": np.random.randn(n_samples, 100, 2).astype(np.float32),
        }

        # Convert to shards
        shard_dir = tmpdir / "shards"
        create_shards_from_arrays(shard_dir, data_dict, shard_size=200)

        # Create model
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        # Train for multiple epochs
        n_epochs = 2
        for epoch in range(n_epochs):
            # Create fresh loader for each epoch
            train_loader = create_webdataset_dataloader(
                shard_dir=shard_dir,
                batch_size=64,
                num_workers=0,
                shuffle=True,
            )

            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                spikes = batch["spikes"]
                behavior = batch["behavior"]
                target = behavior[:, -1, :]

                output = model(spikes)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}, Batches: {n_batches}")

        print("✓ Multi-epoch training test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
