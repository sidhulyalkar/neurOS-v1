# neuros-astro: Immediate Action Plan

## 🎯 Goal: Get to publication-ready results in 3 weeks

---

## TODAY: Getting Started (2-3 hours)

### Step 1: Verify Current State (15 min)

```bash
cd /mnt/c/Users/sidso/Documents/neurOS-v1/packages/neuros-astro

# Verify all tests pass
pytest tests/ -v

# Run the end-to-end example
python examples/00_end_to_end_pipeline.py

# Check outputs
ls -lh output/
```

**Expected outputs:**
- `synthetic_traces.npy`
- `events.parquet`
- `astro_tokens.npz`
- `neurofm_manifest.json`

### Step 2: Implement Basic Visualization (1 hour)

Create `neuros_astro/visualization/event_plots.py`:

```python
"""Event visualization utilities for publication figures."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def plot_event_raster(events, frame_rate_hz, figsize=(12, 6), save_path=None):
    """
    Plot event raster showing when/where events occur.

    Args:
        events: List of AstroEvent objects
        frame_rate_hz: Frame rate for time conversion
        figsize: Figure size tuple
        save_path: Optional path to save figure

    Returns:
        fig, ax objects
    """
    if len(events) == 0:
        print("No events to plot")
        return None, None

    fig, ax = plt.subplots(figsize=figsize)

    # Extract data
    region_ids = [e.region_id for e in events]
    unique_regions = sorted(set(region_ids))
    region_to_y = {r: i for i, r in enumerate(unique_regions)}

    # Plot each event as a horizontal line
    for event in events:
        y = region_to_y[event.region_id]
        onset_s = event.onset_frame / frame_rate_hz
        offset_s = event.offset_frame / frame_rate_hz
        peak_s = event.peak_frame / frame_rate_hz

        # Event duration as horizontal line
        ax.plot([onset_s, offset_s], [y, y], 'b-', linewidth=2, alpha=0.6)
        # Peak marker
        ax.plot(peak_s, y, 'ro', markersize=4)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Astrocyte Region', fontsize=12)
    ax.set_yticks(range(len(unique_regions)))
    ax.set_yticklabels(unique_regions)
    ax.set_title('Astrocyte Calcium Event Raster', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig, ax


def plot_event_distributions(events, figsize=(12, 8), save_path=None):
    """
    Plot distributions of event features.

    Args:
        events: List of AstroEvent objects
        figsize: Figure size tuple
        save_path: Optional path to save figure

    Returns:
        fig object
    """
    if len(events) == 0:
        print("No events to plot")
        return None

    # Extract features
    durations = [e.duration_s for e in events]
    amplitudes = [e.peak_dff for e in events]
    confidences = [e.confidence for e in events]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Duration distribution
    axes[0, 0].hist(durations, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Duration (s)', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title('Event Duration Distribution', fontweight='bold')
    axes[0, 0].axvline(np.median(durations), color='red', linestyle='--',
                       label=f'Median: {np.median(durations):.2f}s')
    axes[0, 0].legend()

    # Amplitude distribution
    axes[0, 1].hist(amplitudes, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Peak ΔF/F', fontsize=11)
    axes[0, 1].set_ylabel('Count', fontsize=11)
    axes[0, 1].set_title('Event Amplitude Distribution', fontweight='bold')
    axes[0, 1].axvline(np.median(amplitudes), color='red', linestyle='--',
                       label=f'Median: {np.median(amplitudes):.3f}')
    axes[0, 1].legend()

    # Confidence distribution
    axes[1, 0].hist(confidences, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Confidence', fontsize=11)
    axes[1, 0].set_ylabel('Count', fontsize=11)
    axes[1, 0].set_title('Event Confidence Distribution', fontweight='bold')

    # Duration vs Amplitude scatter
    axes[1, 1].scatter(durations, amplitudes, alpha=0.5, c=confidences,
                       cmap='viridis', s=50, edgecolors='black', linewidth=0.5)
    axes[1, 1].set_xlabel('Duration (s)', fontsize=11)
    axes[1, 1].set_ylabel('Peak ΔF/F', fontsize=11)
    axes[1, 1].set_title('Duration vs Amplitude', fontweight='bold')
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('Confidence', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_event_statistics_summary(events):
    """Print statistical summary of events."""
    if len(events) == 0:
        print("No events to summarize")
        return

    durations = [e.duration_s for e in events]
    amplitudes = [e.peak_dff for e in events]
    confidences = [e.confidence for e in events]

    print("=" * 60)
    print("EVENT STATISTICS SUMMARY")
    print("=" * 60)
    print(f"Total events: {len(events)}")
    print()
    print("Duration (s):")
    print(f"  Mean ± SD: {np.mean(durations):.2f} ± {np.std(durations):.2f}")
    print(f"  Median [IQR]: {np.median(durations):.2f} [{np.percentile(durations, 25):.2f}-{np.percentile(durations, 75):.2f}]")
    print(f"  Range: {np.min(durations):.2f} - {np.max(durations):.2f}")
    print()
    print("Amplitude (ΔF/F):")
    print(f"  Mean ± SD: {np.mean(amplitudes):.3f} ± {np.std(amplitudes):.3f}")
    print(f"  Median [IQR]: {np.median(amplitudes):.3f} [{np.percentile(amplitudes, 25):.3f}-{np.percentile(amplitudes, 75):.3f}]")
    print(f"  Range: {np.min(amplitudes):.3f} - {np.max(amplitudes):.3f}")
    print()
    print("Confidence:")
    print(f"  Mean ± SD: {np.mean(confidences):.3f} ± {np.std(confidences):.3f}")
    print("=" * 60)
```

**Test it:**

```bash
# Create test script
cat > test_viz.py << 'EOF'
from neuros_astro.io.synthetic import generate_synthetic_astro_traces
from neuros_astro.events.event_detection import detect_events_from_traces
from neuros_astro.visualization.event_plots import (
    plot_event_raster,
    plot_event_distributions,
    plot_event_statistics_summary
)
from pathlib import Path

# Generate data
traces, _ = generate_synthetic_astro_traces(n_regions=10, duration_s=60.0,
                                            frame_rate_hz=10.0, seed=42)

# Detect events
events = detect_events_from_traces(traces, frame_rate_hz=10.0, session_id="test")

# Visualize
output_dir = Path("./figures")
output_dir.mkdir(exist_ok=True)

plot_event_raster(events, frame_rate_hz=10.0, save_path=output_dir / "event_raster.png")
plot_event_distributions(events, save_path=output_dir / "event_distributions.png")
plot_event_statistics_summary(events)

print(f"\nFigures saved to {output_dir}/")
EOF

python test_viz.py
```

### Step 3: Connect to Your Allen Data (30 min)

You already have Allen data! Let's use it:

```bash
cd /mnt/c/Users/sidso/Documents/neurOS-v1/packages/neuros-mechint/examples/allen_data_demo

# Check what sessions you have
ls -lh *.nwb 2>/dev/null || echo "NWB files might be elsewhere"

# Check for processed data
ls -lh data/ 2>/dev/null || ls -lh
```

Create `examples/04_allen_data_validation.py`:

```python
"""Validate neuros-astro on real Allen Visual Coding data."""

import numpy as np
from pathlib import Path
from neuros_astro.events.event_detection import detect_events_from_traces
from neuros_astro.networks.functional_connectivity import build_event_coactivation_graph
from neuros_astro.tokenization.event_tokenizer import AstroEventTokenizer
from neuros_astro.visualization.event_plots import (
    plot_event_raster,
    plot_event_distributions,
    plot_event_statistics_summary
)

def main():
    print("="*70)
    print("neuros-astro: Allen Visual Coding Dataset Validation")
    print("="*70)

    # TODO: Load your actual Allen traces
    # For now, point to where your Allen data lives
    allen_data_path = Path("../../neuros-mechint/examples/allen_data_demo")

    print(f"\nLooking for Allen data in: {allen_data_path}")
    print(f"Exists: {allen_data_path.exists()}")

    # Example: Load traces from your Allen cache
    # traces = load_allen_traces(session_id=...)  # You'll need to implement this

    # For demonstration, show what the pipeline would be:
    print("\n[Pipeline Steps]")
    print("1. Load Allen ROI fluorescence traces")
    print("2. Run neuros-astro event detection")
    print("3. Build coactivation networks")
    print("4. Generate validation figures")
    print("5. Export results")

    print("\n[Next Steps]")
    print("- Identify Allen session with astrocyte data")
    print("- Extract fluorescence traces")
    print("- Run full neuros-astro pipeline")
    print("- Compare to expected astrocyte dynamics (slow, 1-10s events)")

if __name__ == "__main__":
    main()
```

### Step 4: Check neuroFMx Integration Points (30 min)

```bash
cd /mnt/c/Users/sidso/Documents/neurOS-v1/packages/neuros-neurofm

# Find how modalities are currently defined
find . -name "*.py" -exec grep -l "modality" {} \; | head -10

# Look for config examples
find . -name "*.yaml" -o -name "*.yml" | head -10

# Check for loader patterns
find . -name "*loader*.py" | head -5
```

Document what you find - this will guide Phase 3 implementation.

---

## THIS WEEK: Core Validation (Days 1-7)

### Day 1-2: Visualization Complete
- [ ] Implement all visualization functions
- [ ] Test on synthetic data
- [ ] Generate example figures for paper

### Day 3-4: Allen Data Integration
- [ ] Load Allen calcium traces
- [ ] Run event detection
- [ ] Validate results biologically
- [ ] Generate validation report

### Day 5-6: Network Analysis
- [ ] Build networks from Allen events
- [ ] Analyze network properties
- [ ] Compare across sessions/conditions
- [ ] Document findings

### Day 7: Week 1 Review
- [ ] Figures look publication-quality
- [ ] Allen validation convincing
- [ ] Draft validation report
- [ ] Plan Week 2 experiments

---

## Key Decisions Needed Today

### Decision 1: Which Allen Dataset?
Options:
1. Use existing Allen Visual Coding session from your mechint work
2. Download new Allen astrocyte-labeled dataset (if available)
3. Start with any Allen 2p calcium imaging

**Recommended**: Start with what you have, validate pipeline, then expand

### Decision 2: Visualization Style?
- Publication venue determines figure style
- Nature journals: clean, minimal
- PLOS/eLife: more detailed, colorful

**Recommended**: Start with matplotlib defaults, refine later

### Decision 3: Timeline Commitment?
- 3-week sprint to preprint? (aggressive)
- 6-week timeline to journal submission? (comfortable)
- Flexible based on results? (realistic)

**Recommended**: Aim for 3 weeks, be ready to extend if experiments need iteration

---

## Quick Commands Cheatsheet

```bash
# Run tests
cd packages/neuros-astro && pytest tests/ -v

# Run examples
python examples/00_end_to_end_pipeline.py
python examples/01_dataset_triage_example.py
python examples/02_python_api_example.py

# Check test coverage
pytest tests/ --cov=neuros_astro --cov-report=html
# View: open htmlcov/index.html

# Code quality
ruff check neuros_astro/
ruff format neuros_astro/

# Install with all features
pip install -e ".[all]"

# Generate figures
python test_viz.py

# Check package is importable from anywhere
python -c "from neuros_astro import __version__; print(__version__)"
```

---

## Success Criteria for Today

By end of today, you should have:
- ✅ Basic visualization working
- ✅ Example figures generated
- ✅ Path to Allen data identified
- ✅ Plan for Week 1 solidified

---

## Getting Help

If you get stuck:
1. Check existing tests for usage examples
2. Read docstrings in the source code
3. Review the whitepaper for scientific context
4. Ask specific questions about implementation

---

## Motivation

You're closer than you think! The hard infrastructure work is done:
- ✅ Package works
- ✅ Tests pass
- ✅ Core algorithms implemented
- ✅ CLI functional

What's left is the "fun" part:
- 🎨 Making publication figures
- 🔬 Running real experiments
- 📊 Analyzing results
- 📝 Writing it up

**You could have a preprint in 3 weeks.** Let's do this! 🚀
