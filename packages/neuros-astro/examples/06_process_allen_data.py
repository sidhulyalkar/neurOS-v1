#!/usr/bin/env python3
"""
Quick script to process your existing Allen 2P data with neuros-astro.

This script:
1. Loads your preprocessed Allen sessions
2. Runs event detection
3. Builds networks
4. Generates visualizations
5. Exports tokens for neuroFMx

Usage:
    python examples/06_process_allen_data.py

    # Process specific session
    python examples/06_process_allen_data.py --session 2p_session_545446482.npz

    # Process all sessions
    python examples/06_process_allen_data.py --all

Compute: CPU only! Runs in seconds per session.
"""

import argparse
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from neuros_astro.io.allen_loader import (
    load_allen_session_from_npz,
    convert_trial_aligned_to_continuous,
)
from neuros_astro.events.event_detection import detect_events_from_traces
from neuros_astro.networks.functional_connectivity import build_event_coactivation_graph
from neuros_astro.tokenization.event_tokenizer import AstroEventTokenizer
from neuros_astro.visualization.event_plots import (
    plot_event_raster,
    plot_event_distributions,
    plot_event_statistics_summary,
    plot_network_graph,
)
from neuros_astro.export.to_neurofm import (
    save_tokenized_sequence_npz,
    build_neurofm_manifest,
    save_neurofm_manifest,
)
from neuros_astro.export.to_parquet import save_events_parquet


def process_session(session_path: Path, output_dir: Path, visualize: bool = True):
    """Process a single Allen session."""
    print(f"\n{'='*70}")
    print(f"Processing: {session_path.name}")
    print('='*70)

    session_id = session_path.stem
    session_output = output_dir / session_id
    session_output.mkdir(exist_ok=True, parents=True)

    # Load session
    print("\n[1/6] Loading session...")
    trial_responses, metadata = load_allen_session_from_npz(session_path)
    print(f"  ✓ {metadata['n_trials']} trials, {metadata['n_cells']} cells")

    # Convert to continuous
    print("\n[2/6] Converting to continuous traces...")
    traces, timestamps = convert_trial_aligned_to_continuous(
        trial_responses=trial_responses,
        trial_duration_s=0.5,
        frame_rate_hz=30.0,
    )
    print(f"  ✓ {traces.shape[0]} cells, {traces.shape[1]} timepoints")
    print(f"  ✓ Duration: {timestamps[-1] - timestamps[0]:.1f}s")

    # Detect events
    print("\n[3/6] Detecting calcium events...")
    events = detect_events_from_traces(
        traces=traces,
        frame_rate_hz=30.0,
        session_id=session_id,
        z_threshold=2.0,
        min_duration_s=1.0,  # Look for slow events
        merge_gap_s=0.5,
    )
    print(f"  ✓ Detected {len(events)} events")

    if len(events) == 0:
        print("  ⚠️  No events detected. Try adjusting detection parameters.")
        return None

    # Statistics
    plot_event_statistics_summary(events)

    # Save events
    events_path = session_output / "events.parquet"
    save_events_parquet(events, events_path)
    print(f"\n  ✓ Events saved: {events_path}")

    # Build networks
    print("\n[4/6] Building coactivation networks...")
    graphs = build_event_coactivation_graph(
        events=events,
        session_id=session_id,
        frame_rate_hz=30.0,
        window_size_s=5.0,
        stride_s=2.0,
        min_edge_weight=0.2,
    )
    print(f"  ✓ Generated {len(graphs)} network windows")

    # Tokenize
    print("\n[5/6] Tokenizing for foundation models...")
    tokenizer = AstroEventTokenizer(normalize=True)
    tokens = tokenizer.tokenize(events, session_id=session_id)
    print(f"  ✓ {len(tokens.tokens)} event tokens")

    # Save tokens
    tokens_path = session_output / "astro_tokens.npz"
    save_tokenized_sequence_npz(tokens, tokens_path)
    print(f"  ✓ Tokens saved: {tokens_path}")

    # Create manifest
    manifest = build_neurofm_manifest(
        session_id=session_id,
        modalities={
            "astro": {
                "type": "event_tokens",
                "path": str(tokens_path.name),
                "sampling": "irregular",
            }
        },
        metadata={
            "source": "Allen Brain Observatory 2P",
            "n_events": len(events),
            "n_cells": metadata['n_cells'],
            "n_graphs": len(graphs),
        }
    )

    manifest_path = session_output / "neurofm_manifest.json"
    save_neurofm_manifest(manifest, manifest_path)
    print(f"  ✓ Manifest saved: {manifest_path}")

    # Visualization
    if visualize:
        print("\n[6/6] Generating visualizations...")
        fig_dir = session_output / "figures"
        fig_dir.mkdir(exist_ok=True)

        # Import matplotlib
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        # Event raster
        try:
            fig, ax = plot_event_raster(
                events,
                frame_rate_hz=30.0,
                save_path=fig_dir / "event_raster.png"
            )
            if fig is not None:
                plt.close(fig)
        except Exception as e:
            print(f"  ⚠️  Raster plot failed: {e}")

        # Feature distributions
        try:
            fig = plot_event_distributions(
                events,
                save_path=fig_dir / "event_distributions.png"
            )
            if fig is not None:
                plt.close(fig)
        except Exception as e:
            print(f"  ⚠️  Distribution plot failed: {e}")

        # Network (if available)
        if len(graphs) > 0:
            try:
                fig, ax = plot_network_graph(
                    graphs[0],
                    save_path=fig_dir / "network.png"
                )
                if fig is not None:
                    plt.close(fig)
            except Exception as e:
                print(f"  ⚠️  Network plot failed: {e}")

        print(f"  ✓ Figures saved: {fig_dir}")

    print(f"\n{'='*70}")
    print(f"✅ Session {session_id} complete!")
    print(f"   Outputs: {session_output}/")
    print('='*70)

    return {
        'session_id': session_id,
        'n_cells': metadata['n_cells'],
        'n_events': len(events),
        'n_graphs': len(graphs),
        'output_dir': str(session_output),
    }


def main():
    parser = argparse.ArgumentParser(description='Process Allen 2P data with neuros-astro')
    parser.add_argument('--session', type=str, help='Specific session NPZ file to process')
    parser.add_argument('--all', action='store_true', help='Process all sessions')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization')
    parser.add_argument('--allen-dir', type=str,
                       default='../neuros-mechint/examples/allen_data_demo/data/2p_sessions',
                       help='Path to Allen 2P sessions directory')
    parser.add_argument('--output', type=str, default='./allen_processed',
                       help='Output directory')

    args = parser.parse_args()

    # Check matplotlib availability
    if not args.no_viz:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️  matplotlib not found. Disabling visualization.")
            print("   Install with: pip install matplotlib")
            args.no_viz = True

    # Find Allen data
    allen_dir = Path(args.allen_dir)
    if not allen_dir.exists():
        print(f"❌ Allen data directory not found: {allen_dir}")
        print("\nTrying alternative paths...")

        # Try alternative paths
        alternatives = [
            Path("../../neuros-mechint/examples/allen_data_demo/data/2p_sessions"),
            Path("../../../neuros-mechint/examples/allen_data_demo/data/2p_sessions"),
        ]

        for alt in alternatives:
            if alt.exists():
                allen_dir = alt
                print(f"✓ Found: {allen_dir.absolute()}")
                break
        else:
            print("❌ Could not find Allen data.")
            print("\nPlease specify path with --allen-dir")
            return

    # Get sessions to process
    if args.session:
        sessions = [allen_dir / args.session]
        if not sessions[0].exists():
            print(f"❌ Session not found: {sessions[0]}")
            return
    elif args.all:
        sessions = sorted(allen_dir.glob("*.npz"))
    else:
        # Process first session as demo
        sessions = sorted(allen_dir.glob("*.npz"))[:1]

    if len(sessions) == 0:
        print(f"❌ No NPZ files found in {allen_dir}")
        return

    print(f"\n🔬 Found {len(sessions)} session(s) to process")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Process sessions
    results = []
    for i, session_path in enumerate(sessions, 1):
        print(f"\n\n{'#'*70}")
        print(f"Session {i}/{len(sessions)}")
        print('#'*70)

        try:
            result = process_session(
                session_path,
                output_dir,
                visualize=not args.no_viz
            )

            if result:
                results.append(result)

        except Exception as e:
            print(f"\n❌ Error processing {session_path.name}:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n\n{'='*70}")
    print("PROCESSING COMPLETE")
    print('='*70)
    print(f"\nSuccessfully processed: {len(results)}/{len(sessions)} sessions")

    if len(results) > 0:
        print(f"\nOutputs saved to: {output_dir.absolute()}/")
        print("\nSummary:")
        total_events = sum(r['n_events'] for r in results)
        total_cells = sum(r['n_cells'] for r in results)
        print(f"  Total events detected: {total_events}")
        print(f"  Total cells analyzed: {total_cells}")
        print(f"  Average events/cell: {total_events/total_cells:.2f}")

        print("\n📁 Output structure:")
        print(f"  {output_dir}/")
        print(f"    └── <session_id>/")
        print(f"        ├── events.parquet")
        print(f"        ├── astro_tokens.npz")
        print(f"        ├── neurofm_manifest.json")
        print(f"        └── figures/")
        print(f"            ├── event_raster.png")
        print(f"            ├── event_distributions.png")
        print(f"            └── network.png")

        print("\n🎯 Next steps:")
        print("  1. Review figures in */figures/ directories")
        print("  2. Check event statistics")
        print("  3. Integrate tokens with neuroFMx")
        print("  4. Run ablation experiments!")

    print()


if __name__ == "__main__":
    main()
