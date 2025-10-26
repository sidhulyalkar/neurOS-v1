# Temporal Alignment Utilities for NeuroFMX

## Overview

The temporal alignment utilities provide robust tools for synchronizing multi-modal neural recordings with different sampling rates and temporal resolutions. This is essential for training foundation models on heterogeneous neural data.

## Key Components

### 1. TokenizedSequence

A unified data structure representing tokenized neural data with temporal information.

```python
from neuros_neurofm.tokenizers import TokenizedSequence

# Create a sequence
seq = TokenizedSequence(
    tokens=torch.randn(4, 100, 512),  # (batch, time, embedding_dim)
    t0=0.0,                           # Start time (seconds)
    dt=0.01,                          # Sampling interval (seconds)
    mask=torch.ones(4, 100, dtype=torch.bool),  # Valid positions
    metadata={'modality': 'eeg', 'channels': 64}
)

# Access properties
print(f"Duration: {seq.duration}s")
print(f"Sampling rate: {seq.sampling_rate}Hz")
print(f"Timestamps: {seq.timestamps}")
```

**Key Features:**
- Automatic validation of shapes and parameters
- Properties for duration, sampling rate, timestamps
- Methods for temporal slicing, device movement, cloning
- Metadata storage for modality-specific information

### 2. TemporalAligner

Main class for aligning multiple sequences to a common temporal grid.

```python
from neuros_neurofm.tokenizers import TemporalAligner, InterpolationMethod

aligner = TemporalAligner()

# Align EEG (128 Hz) and video (30 Hz) to common 50 Hz grid
aligned_eeg, aligned_video = aligner.align_to_grid(
    sequences=[eeg_seq, video_seq],
    target_dt=0.02,  # 50 Hz
    method=InterpolationMethod.LINEAR
)
```

**Supported Operations:**

#### Alignment to Common Grid
```python
aligned = aligner.align_to_grid(
    sequences=[seq1, seq2, seq3],
    target_dt=0.02,      # Target sampling interval
    t_start=0.0,         # Start time (optional)
    t_end=5.0,           # End time (optional)
    method=InterpolationMethod.LINEAR
)
```

#### Interpolation Methods
- **NEAREST**: Nearest neighbor (fast, preserves discrete values)
- **LINEAR**: Linear interpolation (smooth, good for continuous signals)
- **CUBIC**: Cubic spline (smoothest, best for high-quality resampling)
- **CAUSAL**: Only uses past context (for online/causal processing)

#### Sliding Window Creation
```python
windows = aligner.create_windows(
    sequences=[eeg_seq, video_seq],
    window_size=2.0,    # 2 second windows
    hop_size=1.0,       # 1 second stride (50% overlap)
    align_first=True    # Align before windowing
)

# Each window contains aligned sequences
for eeg_win, video_win in windows:
    # Process window
    pass
```

#### Missing Data Imputation
```python
# Impute missing data marked by mask
imputed_seq = aligner.impute_missing(
    sequence=seq_with_gaps,
    method=InterpolationMethod.LINEAR
)
```

#### Jitter Correction
```python
# Correct small timing irregularities
corrected_seq = aligner.correct_jitter(
    sequence=irregular_seq,
    max_jitter=0.01  # 10ms tolerance
)
```

#### Synchronization Detection
```python
# Find sync points across modalities
sync_times = aligner.detect_sync_points(
    sequences=[seq1, seq2],
    similarity_threshold=0.8,
    window_size=0.1
)
```

#### Validation
```python
# Validate alignment
result = aligner.validate_alignment(
    sequences=[aligned1, aligned2],
    strict=True  # Raise on mismatch
)

print(f"Valid: {result['valid']}")
print(f"Checks: {result['checks']}")
```

### 3. Utility Functions

#### Resample to Rate
```python
from neuros_neurofm.tokenizers import resample_to_rate

# Resample sequence to specific rate
resampled = resample_to_rate(
    sequence=eeg_seq,
    target_rate=100.0,  # 100 Hz
    method=InterpolationMethod.LINEAR
)
```

#### Align and Concatenate
```python
from neuros_neurofm.tokenizers import align_and_concatenate

# Align multiple modalities and concatenate embeddings
fused = align_and_concatenate([eeg_seq, video_seq, audio_seq])
# Result has combined embedding dimension
```

#### Concatenate Sequences
```python
from neuros_neurofm.tokenizers import concatenate_sequences

# Concatenate along time dimension
combined = concatenate_sequences([seq1, seq2, seq3])
```

#### Batch Sequences
```python
from neuros_neurofm.tokenizers import batch_sequences

# Batch sequences with different lengths (pads shorter ones)
batched = batch_sequences([short_seq, long_seq])
```

## Common Use Cases

### 1. EEG + Video Alignment

```python
from neuros_neurofm.tokenizers import (
    EEGTokenizer,
    TokenizedSequence,
    TemporalAligner,
    InterpolationMethod
)

# Tokenize EEG (128 Hz)
eeg_tokenizer = EEGTokenizer(n_channels=64, d_model=512)
_, eeg_seq = eeg_tokenizer(eeg_data, t0=0.0, return_sequence=True)

# Tokenize video (30 fps)
video_seq = TokenizedSequence(
    tokens=video_embeddings,
    t0=0.0,
    dt=1/30.0,
    mask=video_mask,
    metadata={'modality': 'video'}
)

# Align to common 50 Hz grid
aligner = TemporalAligner()
aligned_eeg, aligned_video = aligner.align_to_grid(
    sequences=[eeg_seq, video_seq],
    target_dt=0.02,
    method=InterpolationMethod.LINEAR
)

# Create training windows
windows = aligner.create_windows(
    sequences=[aligned_eeg, aligned_video],
    window_size=2.0,
    hop_size=1.0
)
```

### 2. Spike Train + LFP Alignment

```python
from neuros_neurofm.tokenizers import SpikeTokenizer

# Tokenize spikes (1ms resolution)
spike_tokenizer = SpikeTokenizer(n_units=96, d_model=512)
_, _, spike_seq = spike_tokenizer(
    spike_times, spike_units,
    t0=0.0,
    return_sequence=True
)

# Align with LFP (1000 Hz downsampled to 100 Hz)
aligner = TemporalAligner()
aligned_spikes, aligned_lfp = aligner.align_to_grid(
    sequences=[spike_seq, lfp_seq],
    target_dt=0.01,  # 100 Hz
    method=InterpolationMethod.LINEAR
)
```

### 3. Multi-Modal Fusion

```python
from neuros_neurofm.tokenizers import align_and_concatenate

# Different modalities with different sampling rates
eeg_seq = ...      # 128 Hz, 256-dim embeddings
video_seq = ...    # 30 Hz, 384-dim embeddings
audio_seq = ...    # 200 Hz, 128-dim embeddings

# Align and fuse into single representation
fused_seq = align_and_concatenate([eeg_seq, video_seq, audio_seq])
# Result: aligned temporal grid with 768-dim embeddings (256+384+128)

print(f"Fused: {fused_seq.d_model}-dim embeddings at {fused_seq.sampling_rate}Hz")
```

### 4. Handling Recording Artifacts

```python
# Sequence with missing data (artifacts, dropouts)
seq_with_gaps = TokenizedSequence(
    tokens=noisy_tokens,
    t0=0.0,
    dt=0.01,
    mask=valid_mask,  # False where data is invalid
    metadata={}
)

# Impute missing segments
aligner = TemporalAligner()
clean_seq = aligner.impute_missing(
    seq_with_gaps,
    method=InterpolationMethod.LINEAR
)

# Correct timing jitter
clean_seq = aligner.correct_jitter(
    clean_seq,
    max_jitter=0.005  # 5ms tolerance
)
```

## Integration with Tokenizers

All tokenizers now support the TokenizedSequence format:

### Updated Tokenizer Interface

```python
# Old way (still supported for backward compatibility)
tokens = tokenizer(data)  # Returns tensor

# New way (recommended)
tokens, sequence = tokenizer(
    data,
    t0=0.0,              # Start time
    return_sequence=True # Return TokenizedSequence
)

# Access temporal information
print(f"Duration: {sequence.duration}s")
print(f"Rate: {sequence.sampling_rate}Hz")
```

### Example: EEG Tokenizer

```python
from neuros_neurofm.tokenizers import EEGTokenizer

tokenizer = EEGTokenizer(
    n_channels=64,
    d_model=512,
    sfreq=128.0
)

# Tokenize with temporal info
tokens, eeg_seq = tokenizer(
    eeg_data,           # (batch, time, channels)
    mask=valid_mask,    # Optional mask
    t0=0.0,            # Start time
    return_sequence=True
)

# Use with alignment
aligner = TemporalAligner()
aligned = aligner.align_to_grid([eeg_seq], target_dt=0.01)
```

### Example: Spike Tokenizer

```python
from neuros_neurofm.tokenizers import SpikeTokenizer

tokenizer = SpikeTokenizer(
    n_units=96,
    d_model=512,
    bin_size_ms=1.0
)

tokens, mask, spike_seq = tokenizer(
    spike_times,        # (batch, n_spikes) in ms
    spike_units,        # (batch, n_spikes) unit IDs
    t0=0.0,
    return_sequence=True
)
```

## Best Practices

### 1. Choosing Sampling Rates

```python
# Find common time range first
aligner = TemporalAligner()
t_start, t_end = aligner.find_common_timerange([seq1, seq2, seq3])

# Choose target rate based on:
# - Nyquist criterion for highest frequency content
# - Computational efficiency
# - Typical: highest original rate or median rate

# Example: EEG (128 Hz) + video (30 Hz) -> use 50-100 Hz
target_dt = 0.01  # 100 Hz
```

### 2. Window Size Selection

```python
# Consider:
# - Receptive field of model
# - Temporal dependencies in task
# - GPU memory constraints

# For event-related potentials: 1-2 seconds
# For behavior: 2-5 seconds
# For long-term dynamics: 5-10 seconds

windows = aligner.create_windows(
    sequences=aligned_seqs,
    window_size=2.0,    # Task-dependent
    hop_size=1.0,       # 50% overlap is common
)
```

### 3. Handling Different Modalities

```python
# Different modalities may need different preprocessing
# Apply before alignment:

# 1. Normalize/standardize each modality
eeg_seq.tokens = (eeg_seq.tokens - eeg_mean) / eeg_std
video_seq.tokens = (video_seq.tokens - video_mean) / video_std

# 2. Align
aligned = aligner.align_to_grid([eeg_seq, video_seq])

# 3. Fuse or process separately
fused = align_and_concatenate(aligned)
```

### 4. Validation and Debugging

```python
# Always validate alignment
validation = aligner.validate_alignment(
    sequences=aligned_seqs,
    strict=False  # Get diagnostics even if invalid
)

if not validation['valid']:
    print("Alignment issues detected:")
    for check_name, check_result in validation['checks'].items():
        if not check_result['passed']:
            print(f"  {check_name}: {check_result}")
```

## Performance Considerations

### 1. Memory Efficiency

```python
# For large datasets, use windowing to process in chunks
for window_seqs in aligner.create_windows(...):
    # Process window
    output = model(window_seqs)
    # Free memory
    del window_seqs
```

### 2. Interpolation Speed

```python
# Method speed (fastest to slowest):
# NEAREST > CAUSAL > LINEAR > CUBIC

# For real-time processing, use NEAREST or CAUSAL
# For offline analysis, LINEAR or CUBIC are preferred
```

### 3. Batch Processing

```python
# Batch sequences before alignment when possible
from neuros_neurofm.tokenizers import batch_sequences

# Instead of aligning individually:
seqs = [seq1, seq2, seq3, ...]
aligned = [aligner.align_to_grid([s]) for s in seqs]

# Batch first, then align:
batched = batch_sequences(seqs)
aligned = aligner.align_to_grid([batched])
```

## Testing

Comprehensive tests are available in `tests/test_temporal_alignment.py`:

```bash
# Run all tests
pytest tests/test_temporal_alignment.py -v

# Run specific test class
pytest tests/test_temporal_alignment.py::TestTemporalAligner -v

# Run with coverage
pytest tests/test_temporal_alignment.py --cov=neuros_neurofm.tokenizers
```

## Example Demo

A comprehensive demonstration is available:

```bash
cd packages/neuros-neurofm
python examples/temporal_alignment_demo.py
```

This demo shows:
1. Basic multi-modal alignment
2. Spike train alignment
3. Sliding window creation
4. Multi-modal fusion
5. Missing data handling
6. Interpolation method comparison
7. Synchronization detection

## API Reference

### TokenizedSequence

```python
@dataclass
class TokenizedSequence:
    tokens: torch.Tensor        # (B, T, D)
    t0: float                   # Start time (s)
    dt: float                   # Sampling interval (s)
    mask: torch.Tensor          # (B, T) valid positions
    metadata: Dict[str, Any]    # Modality info

    # Properties
    @property
    def batch_size(self) -> int
    @property
    def seq_len(self) -> int
    @property
    def d_model(self) -> int
    @property
    def duration(self) -> float
    @property
    def end_time(self) -> float
    @property
    def sampling_rate(self) -> float
    @property
    def timestamps(self) -> torch.Tensor
    @property
    def valid_tokens(self) -> torch.Tensor

    # Methods
    def to(device) -> TokenizedSequence
    def clone() -> TokenizedSequence
    def slice_time(start, end) -> TokenizedSequence
```

### TemporalAligner

```python
class TemporalAligner:
    def __init__(tolerance=1e-6, warn_on_extrapolation=True)

    def find_common_timerange(sequences) -> Tuple[float, float]

    def interpolate_sequence(
        sequence,
        new_timestamps,
        method=InterpolationMethod.LINEAR
    ) -> TokenizedSequence

    def align_to_grid(
        sequences,
        target_dt=None,
        t_start=None,
        t_end=None,
        method=InterpolationMethod.LINEAR,
        reference_idx=0
    ) -> List[TokenizedSequence]

    def create_windows(
        sequences,
        window_size,
        hop_size=None,
        align_first=True,
        method=InterpolationMethod.LINEAR
    ) -> List[List[TokenizedSequence]]

    def detect_sync_points(
        sequences,
        similarity_threshold=0.8,
        window_size=0.1
    ) -> List[float]

    def correct_jitter(
        sequence,
        max_jitter=0.01
    ) -> TokenizedSequence

    def impute_missing(
        sequence,
        method=InterpolationMethod.LINEAR
    ) -> TokenizedSequence

    def validate_alignment(
        sequences,
        strict=True
    ) -> Dict[str, Any]
```

### InterpolationMethod

```python
class InterpolationMethod(str, Enum):
    NEAREST = "nearest"
    LINEAR = "linear"
    CUBIC = "cubic"
    CAUSAL = "causal"
```

## Troubleshooting

### Issue: Sequences don't overlap

```python
# Check time ranges
for seq in sequences:
    print(f"{seq.metadata.get('modality')}: [{seq.t0}, {seq.end_time}]")

# Solution: Trim to common range
t_start, t_end = aligner.find_common_timerange(sequences)
trimmed = [seq.slice_time(t_start, t_end) for seq in sequences]
```

### Issue: Different sampling rates cause aliasing

```python
# Check Nyquist criterion
for seq in sequences:
    print(f"{seq.metadata.get('modality')}: {seq.sampling_rate} Hz")

# Solution: Use target rate at least 2x highest frequency
# For EEG with 50 Hz content, use >= 100 Hz target rate
```

### Issue: Memory errors with large windows

```python
# Solution: Process windows in batches
window_batch_size = 32
for i in range(0, len(windows), window_batch_size):
    batch = windows[i:i+window_batch_size]
    # Process batch
```

## Contributing

To add support for a new tokenizer:

1. Inherit from `BaseTokenizer`
2. Implement `forward()` to return `TokenizedSequence`
3. Set `self.default_sampling_rate`
4. Add tests and examples

See `eeg_tokenizer.py` and `spike_tokenizer.py` for reference implementations.
