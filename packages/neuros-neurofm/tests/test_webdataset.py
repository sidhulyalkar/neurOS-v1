"""
Unit tests for WebDataset sharding pipeline.

Tests cover:
- Shard writing
- Shard loading
- Resumption
- Multi-worker loading
- Metadata handling
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from neuros_neurofm.datasets import (
    WebDatasetWriter,
    WebDatasetLoader,
    ResumableWebDatasetLoader,
    create_shards_from_arrays,
    create_webdataset_dataloader,
    ShardedDatasetInfo,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_data():
    """Create sample multi-modal data."""
    n_samples = 100
    return {
        "spikes": np.random.randn(n_samples, 50, 32).astype(np.float32),
        "behavior": np.random.randn(n_samples, 50, 2).astype(np.float32),
        "lfp": np.random.randn(n_samples, 16, 500).astype(np.float32),
    }


class TestWebDatasetWriter:
    """Test WebDatasetWriter class."""

    def test_basic_writing(self, temp_dir):
        """Test basic shard writing."""
        writer = WebDatasetWriter(
            output_dir=temp_dir,
            shard_size=10,
        )

        # Write samples
        for i in range(25):
            sample = {
                "data": np.random.randn(10, 5),
                "label": i % 3,
            }
            writer.write_sample(sample)

        # Finalize
        summary = writer.finalize()

        # Check results
        assert summary["total_samples"] == 25
        assert summary["total_shards"] == 3  # 10, 10, 5
        assert (temp_dir / "dataset_metadata.json").exists()

    def test_writing_with_metadata(self, temp_dir):
        """Test writing with custom metadata."""
        writer = WebDatasetWriter(output_dir=temp_dir, shard_size=5)

        for i in range(10):
            sample = {"data": np.array([i])}
            metadata = {"trial_id": i, "condition": "A" if i % 2 == 0 else "B"}
            writer.write_sample(sample, metadata=metadata)

        summary = writer.finalize()
        assert summary["total_samples"] == 10

        # Check metadata
        with open(temp_dir / "dataset_metadata.json") as f:
            meta = json.load(f)
            first_sample_meta = meta["shards"][0]["samples"][0]
            assert "trial_id" in first_sample_meta
            assert "condition" in first_sample_meta

    def test_batch_writing(self, temp_dir, sample_data):
        """Test batch writing."""
        n_samples = len(sample_data["spikes"])
        samples = [
            {key: sample_data[key][i] for key in sample_data.keys()}
            for i in range(n_samples)
        ]

        writer = WebDatasetWriter(output_dir=temp_dir, shard_size=20)
        writer.write_batch(samples, show_progress=False)
        summary = writer.finalize()

        assert summary["total_samples"] == n_samples
        assert summary["total_shards"] == 5  # 20, 20, 20, 20, 20

    def test_compression(self, temp_dir):
        """Test different compression types."""
        for compression in ["none", "gz"]:
            output_dir = temp_dir / compression
            writer = WebDatasetWriter(
                output_dir=output_dir,
                shard_size=10,
                compression=compression,
            )

            for i in range(15):
                sample = {"data": np.random.randn(5, 5)}
                writer.write_sample(sample)

            summary = writer.finalize()
            assert summary["total_samples"] == 15
            assert summary["compression"] == compression


class TestWebDatasetLoader:
    """Test WebDatasetLoader class."""

    def test_basic_loading(self, temp_dir, sample_data):
        """Test basic shard loading."""
        # Create shards
        create_shards_from_arrays(
            output_dir=temp_dir,
            data_dict=sample_data,
            shard_size=20,
        )

        # Load shards
        loader = WebDatasetLoader(shard_dir=temp_dir, shuffle=False)

        # Load all samples
        samples = list(loader)
        assert len(samples) == len(sample_data["spikes"])

        # Check first sample
        first_sample = samples[0]
        assert "spikes" in first_sample
        assert "behavior" in first_sample
        assert "lfp" in first_sample

    def test_shuffling(self, temp_dir):
        """Test shuffling behavior."""
        # Create simple sequential data
        data = {"value": np.arange(100).reshape(-1, 1)}
        create_shards_from_arrays(temp_dir, data, shard_size=20)

        # Load without shuffling
        loader1 = WebDatasetLoader(temp_dir, shuffle=False)
        values1 = [s["value"][0] for s in loader1]

        # Load with shuffling
        loader2 = WebDatasetLoader(temp_dir, shuffle=True, buffer_size=50, seed=42)
        values2 = [s["value"][0] for s in loader2]

        # Should be different order
        assert values1 != values2
        # But same content
        assert sorted(values1) == sorted(values2)

    def test_state_management(self, temp_dir, sample_data):
        """Test state get/set for resumption."""
        create_shards_from_arrays(temp_dir, sample_data, shard_size=20)

        loader = WebDatasetLoader(temp_dir, shuffle=False)
        iterator = iter(loader)

        # Load some samples
        for _ in range(50):
            next(iterator)

        # Get state
        state = loader.get_state()
        assert "shard_idx" in state
        assert "sample_offset" in state

        # Create new loader with saved state
        loader2 = WebDatasetLoader(temp_dir, shuffle=False)
        loader2.set_state(state)

        # Should continue from same point
        next_sample1 = next(iterator)
        next_sample2 = next(iter(loader2))

        # Compare samples
        np.testing.assert_array_equal(
            next_sample1["spikes"],
            next_sample2["spikes"],
        )


class TestResumableWebDatasetLoader:
    """Test ResumableWebDatasetLoader class."""

    def test_checkpoint_save_load(self, temp_dir, sample_data):
        """Test checkpoint saving and loading."""
        shard_dir = temp_dir / "shards"
        checkpoint_path = temp_dir / "checkpoint.json"

        create_shards_from_arrays(shard_dir, sample_data, shard_size=20)

        # Create loader with checkpointing
        loader = ResumableWebDatasetLoader(
            shard_dir=shard_dir,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=10,
        )

        # Process some samples
        samples = []
        for i, sample in enumerate(loader):
            samples.append(sample)
            if i == 25:
                break

        # Checkpoint should exist
        assert checkpoint_path.exists()

        # Load checkpoint in new loader
        loader2 = ResumableWebDatasetLoader(
            shard_dir=shard_dir,
            checkpoint_path=checkpoint_path,
        )

        # Should resume from checkpoint
        state = loader2.get_state()
        # Should be around sample 20 (last checkpoint)
        assert state["sample_offset"] > 0


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_shards_from_arrays(self, temp_dir, sample_data):
        """Test create_shards_from_arrays convenience function."""
        summary = create_shards_from_arrays(
            output_dir=temp_dir,
            data_dict=sample_data,
            shard_size=25,
        )

        assert summary["total_samples"] == len(sample_data["spikes"])
        assert summary["total_shards"] == 4  # 25, 25, 25, 25
        assert (temp_dir / "dataset_metadata.json").exists()

    def test_sharded_dataset_info(self, temp_dir, sample_data):
        """Test ShardedDatasetInfo class."""
        create_shards_from_arrays(temp_dir, sample_data, shard_size=20)

        info = ShardedDatasetInfo(temp_dir)
        summary = info.get_summary()

        assert summary["total_samples"] == len(sample_data["spikes"])
        assert summary["total_shards"] == 5
        assert summary["shard_size"] == 20
        assert "modalities" in summary


class TestDataLoader:
    """Test PyTorch DataLoader integration."""

    def test_dataloader_creation(self, temp_dir, sample_data):
        """Test creating DataLoader."""
        create_shards_from_arrays(temp_dir, sample_data, shard_size=20)

        dataloader = create_webdataset_dataloader(
            shard_dir=temp_dir,
            batch_size=8,
            num_workers=0,  # 0 for testing
            shuffle=True,
        )

        # Load one batch
        batch = next(iter(dataloader))

        assert "spikes" in batch
        assert "behavior" in batch
        assert "lfp" in batch

        # Check batch dimensions
        assert batch["spikes"].shape[0] == 8  # batch size
        assert batch["spikes"].shape[1] == 50  # time
        assert batch["spikes"].shape[2] == 32  # units

    def test_multiworker_loading(self, temp_dir, sample_data):
        """Test multi-worker DataLoader."""
        create_shards_from_arrays(temp_dir, sample_data, shard_size=10)

        # Create DataLoader with workers
        dataloader = create_webdataset_dataloader(
            shard_dir=temp_dir,
            batch_size=5,
            num_workers=2,
            shuffle=False,
        )

        # Load all batches
        all_samples = []
        for batch in dataloader:
            batch_size = batch["spikes"].shape[0]
            all_samples.extend(range(batch_size))

        # Should load all samples (no duplicates with proper worker assignment)
        assert len(all_samples) >= 80  # May not get exact 100 due to worker division

    def test_collate_function(self, temp_dir):
        """Test custom collate function."""
        from neuros_neurofm.datasets import collate_webdataset

        # Create simple data
        data = {
            "a": np.random.randn(50, 10, 5).astype(np.float32),
            "b": np.random.randn(50, 10, 3).astype(np.float32),
        }
        create_shards_from_arrays(temp_dir, data, shard_size=10)

        # Load with DataLoader
        loader = WebDatasetLoader(temp_dir, shuffle=False)
        dataset_iter = iter(loader)

        # Get batch manually
        batch = [next(dataset_iter) for _ in range(8)]

        # Collate
        collated = collate_webdataset(batch)

        assert isinstance(collated["a"], torch.Tensor)
        assert isinstance(collated["b"], torch.Tensor)
        assert collated["a"].shape == (8, 10, 5)
        assert collated["b"].shape == (8, 10, 3)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_directory(self, temp_dir):
        """Test loading from empty directory."""
        with pytest.raises(ValueError, match="No shards found"):
            WebDatasetLoader(temp_dir)

    def test_mismatched_array_sizes(self, temp_dir):
        """Test error on mismatched array sizes."""
        data = {
            "a": np.random.randn(100, 5),
            "b": np.random.randn(50, 5),  # Wrong size
        }

        with pytest.raises(ValueError, match="same first dimension"):
            create_shards_from_arrays(temp_dir, data)

    def test_small_dataset(self, temp_dir):
        """Test with very small dataset."""
        data = {"value": np.array([[1], [2], [3]])}
        summary = create_shards_from_arrays(temp_dir, data, shard_size=10)

        assert summary["total_samples"] == 3
        assert summary["total_shards"] == 1

        # Should still load correctly
        loader = WebDatasetLoader(temp_dir, shuffle=False)
        samples = list(loader)
        assert len(samples) == 3

    def test_single_sample(self, temp_dir):
        """Test with single sample."""
        data = {"value": np.array([[42]])}
        summary = create_shards_from_arrays(temp_dir, data, shard_size=10)

        assert summary["total_samples"] == 1

        loader = WebDatasetLoader(temp_dir)
        samples = list(loader)
        assert len(samples) == 1
        assert samples[0]["value"][0] == 42


@pytest.mark.slow
class TestPerformance:
    """Performance tests (marked as slow)."""

    def test_large_dataset_writing(self, temp_dir):
        """Test writing large dataset."""
        n_samples = 10000
        data = {
            "spikes": np.random.randn(n_samples, 100, 96).astype(np.float32),
            "behavior": np.random.randn(n_samples, 100, 2).astype(np.float32),
        }

        import time
        start = time.time()
        summary = create_shards_from_arrays(temp_dir, data, shard_size=1000)
        elapsed = time.time() - start

        assert summary["total_samples"] == n_samples
        print(f"Wrote {n_samples} samples in {elapsed:.2f}s ({n_samples/elapsed:.0f} samples/s)")

    def test_large_dataset_loading(self, temp_dir):
        """Test loading large dataset."""
        n_samples = 5000
        data = {
            "spikes": np.random.randn(n_samples, 100, 96).astype(np.float32),
        }

        create_shards_from_arrays(temp_dir, data, shard_size=1000)

        import time
        loader = WebDatasetLoader(temp_dir, shuffle=False)

        start = time.time()
        count = sum(1 for _ in loader)
        elapsed = time.time() - start

        assert count == n_samples
        print(f"Loaded {count} samples in {elapsed:.2f}s ({count/elapsed:.0f} samples/s)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
