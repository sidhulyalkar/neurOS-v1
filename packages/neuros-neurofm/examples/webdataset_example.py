#!/usr/bin/env python
"""
Example usage of WebDataset sharding pipeline for NeuroFMX.

This script demonstrates:
1. Creating shards from synthetic data
2. Loading shards with resumable iteration
3. Using shards with PyTorch DataLoader
4. Inspecting shard metadata
"""

import numpy as np
import torch
from pathlib import Path

from neuros_neurofm.datasets import (
    # Writer components
    WebDatasetWriter,
    create_shards_from_arrays,
    # Loader components
    WebDatasetLoader,
    ResumableWebDatasetLoader,
    create_webdataset_dataloader,
    ShardedDatasetInfo,
    # Synthetic data for demo
    SyntheticNeuralDataset,
)


def example_1_basic_writing():
    """Example 1: Basic shard writing from numpy arrays."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Shard Writing")
    print("=" * 60)

    # Create synthetic data
    n_samples = 5000
    data = {
        "spikes": np.random.randn(n_samples, 100, 96),  # (samples, time, units)
        "behavior": np.random.randn(n_samples, 100, 2),  # (samples, time, dims)
        "lfp": np.random.randn(n_samples, 64, 1000),    # (samples, channels, time)
    }

    print(f"Created synthetic data with {n_samples} samples")
    print(f"Modalities: {list(data.keys())}")

    # Create shards
    output_dir = Path("./example_shards_basic")
    summary = create_shards_from_arrays(
        output_dir=output_dir,
        data_dict=data,
        shard_size=1000,  # 1000 samples per shard
    )

    print(f"\nCreated {summary['total_shards']} shards")
    print(f"Total samples: {summary['total_samples']}")


def example_2_advanced_writing():
    """Example 2: Advanced shard writing with custom metadata."""
    print("\n" + "=" * 60)
    print("Example 2: Advanced Shard Writing")
    print("=" * 60)

    # Create writer
    output_dir = Path("./example_shards_advanced")
    writer = WebDatasetWriter(
        output_dir=output_dir,
        shard_size=500,
        compression="none",
    )

    # Write samples with custom metadata
    n_samples = 2000
    for i in range(n_samples):
        # Create sample with multiple modalities
        sample = {
            "spikes": np.random.poisson(5, size=(100, 96)),
            "behavior": np.sin(np.linspace(0, 2*np.pi, 100))[:, None] * np.random.randn(100, 2),
            "stimulus": np.random.randn(100, 32),
        }

        # Custom metadata
        metadata = {
            "trial_id": i,
            "condition": "left" if i % 2 == 0 else "right",
            "timestamp": f"2024-01-{(i % 30) + 1:02d}",
        }

        writer.write_sample(sample, metadata=metadata)

        if (i + 1) % 500 == 0:
            print(f"Wrote {i + 1} samples...")

    # Finalize
    summary = writer.finalize()
    print(f"\nCreated {summary['total_shards']} shards")


def example_3_basic_loading():
    """Example 3: Basic shard loading."""
    print("\n" + "=" * 60)
    print("Example 3: Basic Shard Loading")
    print("=" * 60)

    # Create loader
    shard_dir = Path("./example_shards_basic")

    if not shard_dir.exists():
        print(f"Shard directory not found. Run example_1_basic_writing() first.")
        return

    loader = WebDatasetLoader(
        shard_dir=shard_dir,
        shuffle=True,
        buffer_size=100,
    )

    # Load first few samples
    print("Loading first 5 samples...")
    for i, sample in enumerate(loader):
        if i >= 5:
            break

        print(f"\nSample {i}:")
        for modality, data in sample.items():
            if modality == "metadata":
                print(f"  {modality}: {data}")
            elif hasattr(data, "shape"):
                print(f"  {modality}: shape={data.shape}, dtype={data.dtype}")


def example_4_resumable_loading():
    """Example 4: Resumable loading with checkpointing."""
    print("\n" + "=" * 60)
    print("Example 4: Resumable Loading")
    print("=" * 60)

    shard_dir = Path("./example_shards_basic")
    checkpoint_path = Path("./example_checkpoint.json")

    if not shard_dir.exists():
        print(f"Shard directory not found. Run example_1_basic_writing() first.")
        return

    # Create resumable loader
    loader = ResumableWebDatasetLoader(
        shard_dir=shard_dir,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=100,
        shuffle=False,
    )

    # Simulate interruption after 250 samples
    print("Processing samples (will interrupt at 250)...")
    for i, sample in enumerate(loader):
        if i == 250:
            print(f"\nInterrupting at sample {i}")
            state = loader.get_state()
            print(f"State: {state}")
            break

    # Resume from checkpoint
    print("\nResuming from checkpoint...")
    loader2 = ResumableWebDatasetLoader(
        shard_dir=shard_dir,
        checkpoint_path=checkpoint_path,
        shuffle=False,
    )

    # Continue processing
    count = 0
    for i, sample in enumerate(loader2):
        count += 1
        if count == 5:
            print(f"Successfully resumed! Processed {count} more samples.")
            break


def example_5_dataloader():
    """Example 5: Using with PyTorch DataLoader."""
    print("\n" + "=" * 60)
    print("Example 5: PyTorch DataLoader Integration")
    print("=" * 60)

    shard_dir = Path("./example_shards_basic")

    if not shard_dir.exists():
        print(f"Shard directory not found. Run example_1_basic_writing() first.")
        return

    # Create DataLoader
    dataloader = create_webdataset_dataloader(
        shard_dir=shard_dir,
        batch_size=32,
        num_workers=2,
        shuffle=True,
        buffer_size=500,
    )

    # Process batches
    print("Processing batches...")
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break

        print(f"\nBatch {i}:")
        for modality, data in batch.items():
            if modality == "metadata":
                print(f"  {modality}: {len(data)} items")
            elif isinstance(data, torch.Tensor):
                print(f"  {modality}: shape={data.shape}, dtype={data.dtype}")
            elif isinstance(data, list):
                print(f"  {modality}: list of {len(data)} items")


def example_6_multiworker():
    """Example 6: Multi-worker data loading."""
    print("\n" + "=" * 60)
    print("Example 6: Multi-Worker Data Loading")
    print("=" * 60)

    shard_dir = Path("./example_shards_basic")

    if not shard_dir.exists():
        print(f"Shard directory not found. Run example_1_basic_writing() first.")
        return

    # Create DataLoader with multiple workers
    print("Creating DataLoader with 4 workers...")
    dataloader = create_webdataset_dataloader(
        shard_dir=shard_dir,
        batch_size=64,
        num_workers=4,
        shuffle=True,
        prefetch_factor=2,
    )

    # Benchmark throughput
    import time
    print("Benchmarking throughput...")

    start_time = time.time()
    total_samples = 0
    n_batches = 20

    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        batch_size = len(batch["spikes"])
        total_samples += batch_size

    elapsed = time.time() - start_time
    throughput = total_samples / elapsed

    print(f"Processed {total_samples} samples in {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} samples/sec")


def example_7_inspect_shards():
    """Example 7: Inspecting shard metadata."""
    print("\n" + "=" * 60)
    print("Example 7: Inspecting Shards")
    print("=" * 60)

    shard_dir = Path("./example_shards_basic")

    if not shard_dir.exists():
        print(f"Shard directory not found. Run example_1_basic_writing() first.")
        return

    # Create info object
    info = ShardedDatasetInfo(shard_dir)

    # Print summary
    info.print_summary()

    # Get detailed summary
    summary = info.get_summary()
    print(f"\nDetailed info:")
    print(f"  Samples per shard: {summary['shard_size']}")
    print(f"  Modalities: {summary.get('modalities', 'N/A')}")


def example_8_training_loop():
    """Example 8: Realistic training loop with WebDataset."""
    print("\n" + "=" * 60)
    print("Example 8: Training Loop Example")
    print("=" * 60)

    shard_dir = Path("./example_shards_basic")

    if not shard_dir.exists():
        print(f"Shard directory not found. Run example_1_basic_writing() first.")
        return

    # Create DataLoader
    train_loader = create_webdataset_dataloader(
        shard_dir=shard_dir,
        batch_size=32,
        num_workers=2,
        shuffle=True,
    )

    # Simulate training
    print("Running training loop...")

    for epoch in range(2):
        print(f"\nEpoch {epoch + 1}/2")

        epoch_loss = 0.0
        n_batches = 0

        for i, batch in enumerate(train_loader):
            if i >= 10:  # Limit for demo
                break

            # Get data
            spikes = batch["spikes"]
            behavior = batch["behavior"]

            # Simulate forward pass and loss
            loss = torch.randn(1).item()  # Dummy loss
            epoch_loss += loss
            n_batches += 1

            if (i + 1) % 5 == 0:
                avg_loss = epoch_loss / n_batches
                print(f"  Batch {i + 1}, Avg Loss: {avg_loss:.4f}")

        avg_epoch_loss = epoch_loss / n_batches
        print(f"Epoch {epoch + 1} complete. Avg Loss: {avg_epoch_loss:.4f}")


def main():
    """Run all examples."""
    print("WebDataset Pipeline Examples for NeuroFMX")
    print("=" * 60)

    # Run examples
    try:
        example_1_basic_writing()
        example_2_advanced_writing()
        example_3_basic_loading()
        example_4_resumable_loading()
        example_5_dataloader()
        example_6_multiworker()
        example_7_inspect_shards()
        example_8_training_loop()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
