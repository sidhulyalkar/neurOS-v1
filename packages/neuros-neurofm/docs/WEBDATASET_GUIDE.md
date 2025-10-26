# WebDataset Sharding Pipeline for NeuroFMX

This guide explains how to use the WebDataset sharding pipeline for scalable multi-modal neural data loading in NeuroFMX.

## Overview

The WebDataset pipeline enables efficient training on large-scale neural datasets by:

- **Sharding**: Splitting large datasets into manageable tar files
- **Lazy Loading**: Loading data on-demand from disk
- **Resumable Iteration**: Checkpointing progress for fault tolerance
- **Multi-Worker Support**: Parallel data loading across workers
- **Shuffling**: Efficient shuffling within and across shards
- **Multi-Modal**: Supporting multiple data modalities (EEG, spikes, LFP, fMRI, video, etc.)

## Quick Start

### 1. Convert Data to Shards

```python
from neuros_neurofm.datasets import create_shards_from_arrays
import numpy as np

# Create sample data
data = {
    "spikes": np.random.randn(10000, 100, 96),
    "behavior": np.random.randn(10000, 100, 2),
    "lfp": np.random.randn(10000, 64, 1000),
}

# Convert to shards
create_shards_from_arrays(
    output_dir="./shards",
    data_dict=data,
    shard_size=1000,  # 1000 samples per shard
)
```

### 2. Load Shards in Training

```python
from neuros_neurofm.datasets import create_webdataset_dataloader

# Create DataLoader
train_loader = create_webdataset_dataloader(
    shard_dir="./shards",
    batch_size=32,
    num_workers=4,
    shuffle=True,
)

# Use in training loop
for batch in train_loader:
    spikes = batch["spikes"]      # (batch, time, units)
    behavior = batch["behavior"]  # (batch, time, dims)
    # ... your training code ...
```

## Detailed Usage

### Writing Shards

#### From Numpy Arrays

```python
from neuros_neurofm.datasets import WebDatasetWriter
import numpy as np

writer = WebDatasetWriter(
    output_dir="./shards",
    shard_size=1000,
    compression="none",  # or "gz", "bz2", "xz"
)

# Write samples one at a time
for i in range(10000):
    sample = {
        "spikes": np.random.randn(100, 96),
        "behavior": np.random.randn(100, 2),
    }

    metadata = {
        "trial_id": i,
        "condition": "left" if i % 2 == 0 else "right",
    }

    writer.write_sample(sample, metadata=metadata)

summary = writer.finalize()
```

#### From NWB Files

```python
from neuros_neurofm.datasets import NWBToWebDatasetConverter

converter = NWBToWebDatasetConverter(
    output_dir="./shards",
    shard_size=1000,
    modalities=["spikes", "lfp", "behavior"],
    bin_size_ms=10.0,
    sequence_length=100,
    overlap=0.5,
)

# Convert single file
converter.convert_nwb_file("data.nwb")

# Or convert multiple files
converter.convert_nwb_files(["data1.nwb", "data2.nwb", "data3.nwb"])

summary = converter.finalize()
```

#### Using CLI Tool

```bash
# Convert NWB files
python scripts/convert_to_shards.py \
    --input data/*.nwb \
    --output ./shards \
    --format nwb \
    --shard-size 1000 \
    --workers 8 \
    --validate

# Convert NPZ file
python scripts/convert_to_shards.py \
    --input data.npz \
    --output ./shards \
    --format npz \
    --shard-size 1000

# With compression
python scripts/convert_to_shards.py \
    --input data/*.nwb \
    --output ./shards \
    --compression gz \
    --shard-size 1000
```

### Loading Shards

#### Basic Loading

```python
from neuros_neurofm.datasets import WebDatasetLoader

loader = WebDatasetLoader(
    shard_dir="./shards",
    shuffle=True,
    buffer_size=1000,  # Shuffle buffer size
)

for sample in loader:
    print(sample["spikes"].shape)
    print(sample["behavior"].shape)
    # ... process sample ...
```

#### Resumable Loading

```python
from neuros_neurofm.datasets import ResumableWebDatasetLoader

loader = ResumableWebDatasetLoader(
    shard_dir="./shards",
    checkpoint_path="./checkpoint.json",
    checkpoint_interval=1000,  # Save every 1000 samples
)

# Training loop
for sample in loader:
    # ... training code ...
    pass
    # Checkpoint is automatically saved every 1000 samples

# If interrupted, resume from checkpoint
loader2 = ResumableWebDatasetLoader(
    shard_dir="./shards",
    checkpoint_path="./checkpoint.json",
)
# Automatically resumes from last checkpoint
```

#### With PyTorch DataLoader

```python
from neuros_neurofm.datasets import create_webdataset_dataloader

# Create DataLoader with all best practices
train_loader = create_webdataset_dataloader(
    shard_dir="./shards",
    batch_size=32,
    num_workers=4,
    shuffle=True,
    buffer_size=1000,
    prefetch_factor=2,
    pin_memory=True,
)

# Use in training
for epoch in range(num_epochs):
    for batch in train_loader:
        # batch is dict of tensors
        spikes = batch["spikes"]      # (batch, time, units)
        behavior = batch["behavior"]  # (batch, time, dims)

        # ... forward pass, loss, backward ...
```

### Inspecting Shards

```python
from neuros_neurofm.datasets import ShardedDatasetInfo

info = ShardedDatasetInfo("./shards")

# Print summary
info.print_summary()

# Get summary dict
summary = info.get_summary()
print(f"Total samples: {summary['total_samples']}")
print(f"Total shards: {summary['total_shards']}")
print(f"Modalities: {summary['modalities']}")
```

## Shard Format

WebDataset shards are tar archives with the following structure:

```
shard_000000.tar:
├── sample_000000001.spikes.pyd        # Pickled numpy/torch tensor
├── sample_000000001.behavior.pyd      # Pickled numpy/torch tensor
├── sample_000000001.lfp.pyd          # Pickled numpy/torch tensor
├── sample_000000001.metadata.json    # JSON metadata
├── sample_000000002.spikes.pyd
├── sample_000000002.behavior.pyd
├── sample_000000002.lfp.pyd
├── sample_000000002.metadata.json
...

shard_000001.tar:
├── sample_000001001.spikes.pyd
...

dataset_metadata.json  # Global metadata
```

Each sample consists of:
- **Modality files**: `.pyd` for tensors/arrays, `.json` for structured data
- **Metadata file**: JSON with sample information

## Best Practices

### For Writing

1. **Choose appropriate shard size**:
   - Too small: More overhead, slower loading
   - Too large: Less parallel efficiency, harder to shuffle
   - Recommended: 500-2000 samples per shard

2. **Use compression for storage**:
   - `none`: Fastest read, largest size
   - `gz`: Good compression, moderate speed
   - `bz2`: Better compression, slower
   - `xz`: Best compression, slowest

3. **Preserve metadata**:
   - Include trial IDs, conditions, timestamps
   - Helps with debugging and data provenance

### For Loading

1. **Enable shuffling**:
   - Use `shuffle=True` for training
   - Set appropriate `buffer_size` (1000-5000)
   - Disable for validation/testing

2. **Use multiple workers**:
   - Set `num_workers` based on CPU cores
   - Typical: 4-8 workers
   - Enable `prefetch_factor` for faster loading

3. **Enable resumption for long training**:
   - Use `ResumableWebDatasetLoader`
   - Set `checkpoint_interval` appropriately
   - Save checkpoints to persistent storage

4. **Pin memory for GPU training**:
   - Set `pin_memory=True`
   - Faster CPU to GPU transfers
   - Requires extra RAM

## Multi-Worker Behavior

When using `num_workers > 1` in DataLoader:

1. Shards are automatically divided among workers
2. Each worker loads its assigned shards independently
3. Shuffling happens within each worker
4. No sample duplication or overlap
5. Deterministic with fixed seed

Example with 4 workers and 8 shards:
- Worker 0: shards 0, 4
- Worker 1: shards 1, 5
- Worker 2: shards 2, 6
- Worker 3: shards 3, 7

## Advanced Features

### Custom Collate Function

```python
from neuros_neurofm.datasets import WebDatasetLoader
import torch

def custom_collate(batch):
    # Custom batch processing
    spikes = torch.stack([b["spikes"] for b in batch])
    behavior = torch.stack([b["behavior"] for b in batch])

    # Add custom preprocessing
    spikes = spikes / spikes.max()  # Normalize

    return {
        "spikes": spikes,
        "behavior": behavior,
    }

loader = torch.utils.data.DataLoader(
    WebDatasetLoader("./shards"),
    batch_size=32,
    collate_fn=custom_collate,
)
```

### State Management

```python
# Save state manually
loader = WebDatasetLoader("./shards")
iterator = iter(loader)

for i in range(100):
    sample = next(iterator)

state = loader.get_state()
# state = {"shard_idx": 0, "sample_offset": 100}

# Restore state
loader.set_state(state)
# Continue from exactly where we left off
```

### Parallel Conversion

```bash
# Use multiple workers for faster conversion
python scripts/convert_to_shards.py \
    --input data/*.nwb \
    --output ./shards \
    --workers 16 \
    --shard-size 1000
```

## Performance Tips

1. **SSD vs HDD**: Use SSD for shards - 10x faster random access
2. **Prefetching**: Increase `prefetch_factor` to 3-4 for pipelined loading
3. **Buffer Size**: Larger buffers = better shuffling but more RAM
4. **Batch Size**: Larger batches = better GPU utilization but more RAM
5. **Compression**: Only use if storage is limited, adds CPU overhead

## Troubleshooting

### Slow Loading

- Increase `num_workers`
- Increase `prefetch_factor`
- Reduce `buffer_size` if RAM limited
- Use SSD instead of HDD
- Disable compression

### Out of Memory

- Reduce `batch_size`
- Reduce `buffer_size`
- Reduce `prefetch_factor`
- Disable `pin_memory`
- Use fewer workers

### Worker Crashes

- Reduce `num_workers`
- Check for corrupted shards
- Ensure enough RAM per worker
- Check for pickle-incompatible objects

### Inconsistent Results

- Set `seed` parameter
- Disable shuffling for debugging
- Check for non-deterministic operations
- Verify data preprocessing

## Examples

See `examples/webdataset_example.py` for comprehensive examples:

```bash
cd packages/neuros-neurofm
python examples/webdataset_example.py
```

Examples include:
1. Basic shard writing
2. Advanced shard writing with metadata
3. Basic loading
4. Resumable loading
5. PyTorch DataLoader integration
6. Multi-worker loading
7. Inspecting shards
8. Training loop

## API Reference

### Writer Classes

- `WebDatasetWriter`: Low-level shard writer
- `NWBToWebDatasetConverter`: NWB to shard converter
- `create_shards_from_arrays()`: Convenience function

### Loader Classes

- `WebDatasetLoader`: Basic iterable dataset
- `ResumableWebDatasetLoader`: With checkpoint support
- `create_webdataset_dataloader()`: Convenience function
- `ShardedDatasetInfo`: Metadata inspection

### Utility Functions

- `collate_webdataset()`: Default collate function
- `get_state()`: Get iteration state
- `set_state()`: Restore iteration state

## Integration with NeuroFMX

### Training Script

```python
from neuros_neurofm.datasets import create_webdataset_dataloader
from neuros_neurofm.models import NeuroFMX
import torch

# Load data
train_loader = create_webdataset_dataloader(
    shard_dir="./train_shards",
    batch_size=32,
    num_workers=4,
    shuffle=True,
)

val_loader = create_webdataset_dataloader(
    shard_dir="./val_shards",
    batch_size=32,
    num_workers=4,
    shuffle=False,
)

# Create model
model = NeuroFMX(
    input_dim=96,
    hidden_dim=256,
    n_layers=8,
)

# Training loop
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        spikes = batch["spikes"]
        behavior = batch["behavior"]

        # Forward pass
        output = model(spikes)
        loss = criterion(output, behavior)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            # ... validation code ...
            pass
```

## License

MIT License - see LICENSE file for details.
