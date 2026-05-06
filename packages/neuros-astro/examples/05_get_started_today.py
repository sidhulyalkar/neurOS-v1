#!/usr/bin/env python3
"""
Quick Start: Run neuros-astro pipeline TODAY!

This script demonstrates the complete workflow with synthetic data,
then shows you how to connect to real Allen data.

Run this first to verify everything works, then move to real data.
"""

import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Check if visualization is available
try:
    import matplotlib.pyplot as plt
    HAS_VIZ = True
except ImportError:
    print("⚠️  matplotlib not installed. Run: pip install matplotlib")
    HAS_VIZ = False

from neuros_astro.io.synthetic import generate_synthetic_astro_traces, generate_synthetic_astro_movie
from neuros_astro.events.event_detection import detect_events_from_traces, detect_candidate_events_from_movie
from neuros_astro.networks.functional_connectivity import build_event_coactivation_graph
from neuros_astro.networks.graph_features import compute_graph_summary_features
from neuros_astro.tokenization.event_tokenizer import AstroEventTokenizer
from neuros_astro.tokenization.astro_tokenizer import BinnedAstroTokenizer
from neuros_astro.export.to_parquet import save_events_parquet, load_events_parquet
from neuros_astro.export.to_neurofm import (
    save_tokenized_sequence_npz,
    load_tokenized_sequence_npz,
    build_neurofm_manifest,
    save_neurofm_manifest,
)


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_synthetic_validation():
    """Run complete pipeline on synthetic data."""
    print_section("PART 1: Synthetic Data Validation")

    output_dir = Path("./starter_output")
    output_dir.mkdir(exist_ok=True)

    # Configuration
    session_id = "synthetic_demo"
    n_regions = 15
    duration_s = 120.0  # 2 minutes
    frame_rate_hz = 10.0

    # Step 1: Generate synthetic traces
    print("\n[1/7] Generating synthetic astrocyte calcium traces...")
    traces, gt_events = generate_synthetic_astro_traces(
        n_regions=n_regions,
        duration_s=duration_s,
        frame_rate_hz=frame_rate_hz,
        n_events_per_region=8,
        seed=42,
    )
    print(f"  ✓ Generated traces: {traces.shape} [regions, timepoints]")
    print(f"  ✓ Ground truth: {len(gt_events)} events")

    # Save traces
    traces_path = output_dir / "synthetic_traces.npy"
    np.save(traces_path, traces)

    # Step 2: Detect events from traces
    print("\n[2/7] Detecting events from traces...")
    events = detect_events_from_traces(
        traces=traces,
        frame_rate_hz=frame_rate_hz,
        session_id=session_id,
        z_threshold=2.0,
        min_duration_s=0.8,
        merge_gap_s=0.5,
    )
    print(f"  ✓ Detected: {len(events)} events")

    if len(events) > 0:
        durations = [e.duration_s for e in events]
        amplitudes = [e.peak_dff for e in events]
        print(f"  ✓ Duration: {np.mean(durations):.2f} ± {np.std(durations):.2f}s")
        print(f"  ✓ Amplitude: {np.mean(amplitudes):.3f} ± {np.std(amplitudes):.3f} ΔF/F")

        # Detection performance
        print(f"  ✓ Detection rate: {len(events)/len(gt_events)*100:.1f}% of ground truth")

    # Save events
    events_path = output_dir / "events.parquet"
    save_events_parquet(events, events_path)
    print(f"  ✓ Saved: {events_path}")

    # Step 3: Build coactivation networks
    print("\n[3/7] Building astrocyte coactivation networks...")
    graphs = build_event_coactivation_graph(
        events=events,
        session_id=session_id,
        frame_rate_hz=frame_rate_hz,
        window_size_s=30.0,
        stride_s=10.0,
        min_edge_weight=0.15,
    )
    print(f"  ✓ Generated: {len(graphs)} network windows")

    if len(graphs) > 0:
        # Analyze first graph
        features = compute_graph_summary_features(graphs[0])
        print(f"  ✓ Example network: {features['n_nodes']} nodes, {features['n_edges']} edges")
        print(f"  ✓ Density: {features['density']:.3f}")
        print(f"  ✓ Mean degree: {features['degree_mean']:.2f}")

    # Step 4: Tokenize events
    print("\n[4/7] Tokenizing events (irregular sampling)...")
    event_tokenizer = AstroEventTokenizer(normalize=True)
    event_tokens = event_tokenizer.tokenize(events, session_id=session_id)
    print(f"  ✓ Event tokens: ({len(event_tokens.tokens)}, {len(event_tokens.feature_names)}) [n_events, n_features]")
    print(f"  ✓ Features: {', '.join(event_tokens.feature_names[:5])}...")

    # Save event tokens
    event_tokens_path = output_dir / "event_tokens.npz"
    save_tokenized_sequence_npz(event_tokens, event_tokens_path)

    # Step 5: Tokenize with binning
    print("\n[5/7] Tokenizing with regular binning...")
    binned_tokenizer = BinnedAstroTokenizer(bin_size_s=5.0, normalize=True)
    binned_tokens = binned_tokenizer.tokenize(
        events=events,
        duration_s=duration_s,
        session_id=session_id,
        graphs=graphs
    )
    print(f"  ✓ Binned tokens: ({len(binned_tokens.tokens)}, {len(binned_tokens.feature_names)}) [n_bins, n_features]")
    print(f"  ✓ Features: {', '.join(binned_tokens.feature_names[:5])}...")

    # Save binned tokens
    binned_tokens_path = output_dir / "binned_tokens.npz"
    save_tokenized_sequence_npz(binned_tokens, binned_tokens_path)

    # Step 6: Generate neuroFMx manifest
    print("\n[6/7] Creating neuroFMx integration manifest...")
    manifest = build_neurofm_manifest(
        session_id=session_id,
        modalities={
            "astro_events": {
                "type": "event_tokens",
                "path": event_tokens_path.name,
                "sampling": "irregular",
                "timestamp_key": "timestamps_s",
            },
            "astro_binned": {
                "type": "binned_tokens",
                "path": binned_tokens_path.name,
                "sampling": "regular",
                "timestamp_key": "timestamps_s",
                "bin_size_s": 5.0,
            }
        },
        metadata={
            "n_events": len(events),
            "n_regions": n_regions,
            "n_graphs": len(graphs),
            "duration_s": duration_s,
            "frame_rate_hz": frame_rate_hz,
        }
    )

    manifest_path = output_dir / "neurofm_manifest.json"
    save_neurofm_manifest(manifest, manifest_path)
    print(f"  ✓ Manifest: {manifest_path}")

    # Step 7: Verify roundtrip
    print("\n[7/7] Verifying data roundtrip...")
    loaded_events = load_events_parquet(events_path)
    loaded_tokens = load_tokenized_sequence_npz(event_tokens_path)
    print(f"  ✓ Events: {len(loaded_events)} loaded correctly")
    print(f"  ✓ Tokens: ({len(loaded_tokens.tokens)}, {len(loaded_tokens.feature_names)}) loaded correctly")

    # Summary
    print_section("SYNTHETIC VALIDATION COMPLETE")
    print(f"\nAll outputs saved to: {output_dir.absolute()}/")
    print("\nFiles created:")
    for file in output_dir.iterdir():
        size = file.stat().st_size / 1024
        print(f"  - {file.name:30s} ({size:.1f} KB)")

    return events, graphs, event_tokens, binned_tokens


def visualize_results(events, output_dir="./starter_output"):
    """Create visualization if matplotlib available."""
    if not HAS_VIZ:
        print("\n⚠️  Skipping visualization (matplotlib not installed)")
        return

    print_section("PART 2: Generate Visualizations")

    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Event statistics
    print("\n[Visualization] Event Statistics:")
    durations = [e.duration_s for e in events]
    amplitudes = [e.peak_dff for e in events]
    confidences = [e.confidence for e in events]

    print(f"  Duration: {np.mean(durations):.2f} ± {np.std(durations):.2f}s")
    print(f"  Amplitude: {np.mean(amplitudes):.3f} ± {np.std(amplitudes):.3f} ΔF/F")
    print(f"  Confidence: {np.mean(confidences):.3f} ± {np.std(confidences):.3f}")

    # Simple raster plot
    print("\n[Visualization] Creating event raster plot...")
    fig, ax = plt.subplots(figsize=(12, 6))

    region_ids = sorted(set([e.region_id for e in events]))
    region_to_y = {r: i for i, r in enumerate(region_ids)}

    for event in events:
        y = region_to_y[event.region_id]
        onset_s = event.onset_frame / 10.0  # Assuming 10Hz
        offset_s = event.offset_frame / 10.0
        ax.plot([onset_s, offset_s], [y, y], 'b-', linewidth=2, alpha=0.7)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Astrocyte Region', fontsize=12)
    ax.set_title('Astrocyte Calcium Event Raster', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    raster_path = fig_dir / "event_raster.png"
    plt.savefig(raster_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {raster_path}")
    plt.close()

    # Duration histogram
    print("\n[Visualization] Creating duration histogram...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(durations, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.median(durations), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(durations):.2f}s')
    ax.set_xlabel('Event Duration (s)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Event Duration Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    duration_path = fig_dir / "duration_distribution.png"
    plt.savefig(duration_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {duration_path}")
    plt.close()

    print(f"\nFigures saved to: {fig_dir.absolute()}/")


def show_next_steps():
    """Show what to do next."""
    print_section("NEXT STEPS: Moving to Real Data")

    print("\n📊 Option 1: Download Allen Astrocyte Data")
    print("    If Allen has astrocyte-labeled datasets:")
    print("    - Use AllenSDK to query for GFAP-labeled sessions")
    print("    - Download calcium fluorescence traces")
    print("    - Run neuros-astro pipeline on real data")

    print("\n📊 Option 2: Use Your Existing Allen 2P Data")
    print("    You have preprocessed Allen sessions in:")
    print("    packages/neuros-mechint/examples/allen_data_demo/data/2p_sessions/")
    print("    These are trial-aligned. To use them:")
    print("    - Extract continuous fluorescence traces from Allen cache")
    print("    - Reformat as [n_cells, n_timepoints] array")
    print("    - Run event detection")

    print("\n📊 Option 3: DANDI Dataset Search")
    print("    Search DANDI for astrocyte calcium imaging datasets:")
    print("    neuros-astro scan-dandiset <DANDISET_ID>")

    print("\n🎨 Implement Visualization Module")
    print("    Create: packages/neuros-astro/neuros_astro/visualization/event_plots.py")
    print("    See: NEUROS_ASTRO_NEXT_STEPS.md for implementation details")

    print("\n🔬 neuroFMx Integration")
    print("    Next week: Connect astro tokens to neuroFMx")
    print("    See: NEUROS_ASTRO_PUBLICATION_ROADMAP.md Phase 3")

    print("\n📝 Documentation to Read:")
    print("    - NEUROS_ASTRO_PUBLICATION_ROADMAP.md  (full 3-week plan)")
    print("    - NEUROS_ASTRO_NEXT_STEPS.md          (today's tasks)")
    print("    - neuros_astro_whitepaper.md           (scientific background)")


def main():
    """Run starter demonstration."""
    print("=" * 70)
    print("  neuros-astro: Quick Start Demo")
    print("  Get started with publication-quality experiments TODAY!")
    print("=" * 70)

    # Run synthetic validation
    events, graphs, event_tokens, binned_tokens = run_synthetic_validation()

    # Create visualizations if possible
    visualize_results(events)

    # Show next steps
    show_next_steps()

    print("\n" + "=" * 70)
    print("  ✨ SUCCESS! You're ready to start experiments!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
