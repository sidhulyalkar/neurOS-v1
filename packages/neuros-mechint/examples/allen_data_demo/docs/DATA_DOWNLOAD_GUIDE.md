# Data Download Guide for SAE Validation

## Quick Start

### Option 1: Use the Validation-Specific Downloader (Recommended)

I've created a focused script specifically for the validation framework:

```bash
# Install AllenSDK first
pip install allensdk numpy pandas

# Download 5 sessions (good for testing)
python scripts/download_validation_data.py --num-sessions 5

# Or download more for thorough validation
python scripts/download_validation_data.py --num-sessions 10
```

**What this downloads:**
- Sessions with **drifting gratings** stimulus (needed for orientation analysis)
- **V1 (VISp)** recordings from mouse visual cortex
- Sessions with **100+ good quality units** (can adjust with `--min-units`)
- Data is cached in `./allen_validation_cache/` by default

**Download size:** ~2-5 GB per session (so 5 sessions ≈ 10-25 GB total)

### Option 2: Use the Existing NeuroFM Downloader

The repo already has a more general downloader:

```bash
# From the repo root
cd packages/neuros-neurofm/scripts

# Download with drifting gratings filter
python download_allen_data.py \
    --num-sessions 5 \
    --stimulus-type drifting_gratings \
    --brain-area VISp \
    --data-dir ../../../data/allen_validation
```

## Detailed Usage

### Validation Script Options

```bash
python scripts/download_validation_data.py --help
```

**Key options:**
- `--cache-dir DIR` - Where to cache data (default: `./allen_validation_cache`)
- `--num-sessions N` - How many sessions to download (default: 5, recommend 3-10)
- `--min-units N` - Minimum good units per session (default: 100)
- `--brain-areas AREA [AREA ...]` - Brain regions (default: VISp for V1)

### Examples

**Quick test (3 sessions, ~6-15 GB):**
```bash
python scripts/download_validation_data.py --num-sessions 3
```

**Thorough validation (10 sessions, ~20-50 GB):**
```bash
python scripts/download_validation_data.py --num-sessions 10
```

**Custom cache location:**
```bash
python scripts/download_validation_data.py \
    --cache-dir /mnt/data/allen_cache \
    --num-sessions 5
```

**Multiple visual areas:**
```bash
python scripts/download_validation_data.py \
    --brain-areas VISp VISl VISal \
    --min-units 50
```

## What Gets Downloaded

The script downloads Allen Institute **Neuropixels recordings** with:

1. **Stimulus type:** Drifting gratings (oriented bars)
   - Essential for orientation selectivity validation
   - Multiple orientations (0°, 45°, 90°, 135°, etc.)

2. **Brain regions:** Primary visual cortex (V1/VISp)
   - Known to have orientation-selective neurons
   - Ground truth for validating SAE features

3. **Data quality:** Only "good" quality units
   - Filtered for isolation quality
   - Suitable for interpretability analysis

4. **Session metadata:**
   - Spike times for each unit
   - Stimulus timing and parameters
   - Brain region annotations

## Using Downloaded Data

### In the Validation Example

The downloaded data works seamlessly with the validation framework:

```python
from neuros.datasets import AllenVisualCodingValidator

# Use a specific downloaded session
validator = AllenVisualCodingValidator(
    session_id=756029989,  # From download output
    cache_dir='./allen_validation_cache',
    brain_areas=['VISp']
)

# Extract windows for analysis
windows = validator.get_neural_windows(
    window_length=1.0,
    stride=0.5,
    bin_size=0.02
)
```

### Auto-Selection Mode

Or let the validator auto-select a good session:

```python
# No session_id specified - auto-selects best session
validator = AllenVisualCodingValidator(
    cache_dir='./allen_validation_cache',
    brain_areas=['VISp'],
    min_units=100
)
```

## BCI Data (No Download Needed!)

The BCI motor imagery validator uses **mock data** by default - no download required:

```python
from neuros.datasets import BCIMotorImageryValidator

# Mock data generated automatically
bci = BCIMotorImageryValidator(n_trials=200)
windows = bci.get_neural_windows()
```

To use real BCI data, install:
```bash
pip install mne moabb
```

Then use the MOABB loaders in [`packages/neuros-foundation/src/neuros/datasets/bci_datasets.py`](packages/neuros-foundation/src/neuros/datasets/bci_datasets.py).

## Troubleshooting

### "No suitable sessions found"

Try:
```bash
# Lower the unit requirement
python scripts/download_validation_data.py --min-units 50

# Or try different brain areas
python scripts/download_validation_data.py --brain-areas VISl VISal
```

### "ImportError: allensdk not found"

Install AllenSDK:
```bash
pip install allensdk numpy pandas
```

### Slow download

- Each session is 2-5 GB and downloads can be slow
- Consider starting with `--num-sessions 3` for testing
- Download continues from where it left off if interrupted

### Disk space

Check available space:
```bash
df -h .
```

Each session needs ~2-5 GB. For 10 sessions, ensure you have at least **50 GB free**.

## Where Files Are Stored

After downloading, your directory structure will look like:

```
allen_validation_cache/
├── manifest.json                    # AllenSDK manifest
├── validation_sessions.txt          # Session summary
└── session_*/                       # Cached session data
    ├── spike_times/
    ├── stimulus_presentations.nwb
    └── ...
```

## Using Both Downloaders

Both scripts can coexist:

```bash
# Validation-focused (drifting gratings, V1, high quality)
python scripts/download_validation_data.py --num-sessions 5

# General NeuroFM training data
python packages/neuros-neurofm/scripts/download_allen_data.py \
    --num-sessions 10 \
    --stimulus-type natural_images
```

## Next Steps After Download

1. ✅ **Verify download:**
   - Check that `validation_sessions.txt` exists
   - Review session IDs and statistics

2. ✅ **Run validation:**
   ```bash
   python examples/sae_validation_example.py
   ```

3. ✅ **Integrate with your SAE:**
   - Train SAE on NeuroFM embeddings
   - Use validation framework to test features
   - Generate validation report

## Quick Reference

| Task | Command |
|------|---------|
| Quick test (3 sessions) | `python scripts/download_validation_data.py --num-sessions 3` |
| Recommended (5 sessions) | `python scripts/download_validation_data.py --num-sessions 5` |
| Thorough (10 sessions) | `python scripts/download_validation_data.py --num-sessions 10` |
| Check what's downloaded | `cat allen_validation_cache/validation_sessions.txt` |
| Custom cache location | `--cache-dir /path/to/cache` |
| Lower quality threshold | `--min-units 50` |

## Summary

✅ **Validation script:** [`scripts/download_validation_data.py`](scripts/download_validation_data.py)
✅ **Existing script:** [`packages/neuros-neurofm/scripts/download_allen_data.py`](packages/neuros-neurofm/scripts/download_allen_data.py)
✅ **No BCI download needed** - uses mock data by default
✅ **Cache directory:** `./allen_validation_cache/` (customizable)
✅ **Typical size:** 2-5 GB per session
✅ **Recommended:** 5 sessions for good validation
