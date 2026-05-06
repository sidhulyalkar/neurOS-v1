#!/usr/bin/env python
"""
Process Allen Visual Coding NWB files with neuros-astro pipeline.

This script processes continuous 2-photon calcium imaging data from NWB files,
detecting astrocyte-like events and building functional networks.

Usage:
    python process_allen_nwb.py
    python process_allen_nwb.py --nwb-dir /path/to/nwb/files
    python process_allen_nwb.py --session-id 545446482
    python process_allen_nwb.py --max-sessions 3 --z-threshold 2.0
"""

import argparse
import numpy as np
import json
import h5py
from pathlib import Path

from neuros_astro.events.event_detection import detect_events_from_traces
from neuros_astro.networks.functional_connectivity import build_event_coactivation_graph
from neuros_astro.analysis.statistics import compute_event_statistics
from neuros_astro.analysis.network_metrics import compute_temporal_network_metrics, compute_network_stability
from neuros_astro.visualization.event_plots import (
    plot_event_raster,
    plot_event_distributions,
    plot_network_graph,
)
from neuros_astro.export.to_parquet import save_events_parquet
from neuros_astro.export.to_neurofm import build_neurofm_manifest
from neuros_astro.tokenization.event_tokenizer import AstroEventTokenizer


def find_nwb_files(nwb_dir=None):
    """Find Allen NWB files in standard locations."""
    cwd = Path.cwd()

    search_paths = [
        Path(nwb_dir) if nwb_dir else None,
        cwd / "../neuros-mechint/examples/allen_data_demo/allen_validation_cache/ophys_experiment_data",
        cwd / "../../neuros-mechint/examples/allen_data_demo/allen_validation_cache/ophys_experiment_data",
        cwd / "../../../neuros-mechint/examples/allen_data_demo/allen_validation_cache/ophys_experiment_data",
        cwd / "allen_validation_cache/ophys_experiment_data",
        cwd / "ophys_experiment_data",
        Path("/mnt/c/Users/sidso/Documents/neurOS-v1/packages/neuros-mechint/examples/allen_data_demo/allen_validation_cache/ophys_experiment_data"),
    ]

    for path in search_paths:
        if path and path.exists():
            files = list(path.glob("*.nwb"))
            if files:
                return path.resolve(), files

    return None, []


def load_nwb_dff(nwb_path):
    """
    Load dF/F traces from Allen NWB file.

    Returns:
        traces: np.ndarray of shape (n_rois, n_timepoints)
        frame_rate_hz: float
        metadata: dict
    """
    def decode_value(val):
        """Helper to decode bytes and convert numpy types to native Python types."""
        if isinstance(val, bytes):
            return val.decode('utf-8')
        elif isinstance(val, np.generic):
            return val.item()
        return val

    with h5py.File(nwb_path, 'r') as f:
        # Extract dF/F data
        dff_data = f['processing/brain_observatory_pipeline/DfOverF/imaging_plane_1/data'][:]

        # Get timestamps to determine frame rate
        timestamps = f['processing/brain_observatory_pipeline/DfOverF/imaging_plane_1/timestamps'][:]
        frame_rate_hz = 1.0 / np.median(np.diff(timestamps))

        # Extract metadata
        session_id = decode_value(f['identifier'][()])
        session_desc = decode_value(f['session_description'][()]) if 'session_description' in f else 'N/A'

        # Get imaging plane metadata if available
        imaging_metadata = {}
        if 'general/optophysiology/imaging_plane_1' in f:
            imaging_plane = f['general/optophysiology/imaging_plane_1']
            if 'imaging_rate' in imaging_plane:
                # imaging_rate can be a string like "31Hz", just store as string
                imaging_metadata['imaging_rate_str'] = decode_value(imaging_plane['imaging_rate'][()])
            if 'indicator' in imaging_plane:
                imaging_metadata['indicator'] = decode_value(imaging_plane['indicator'][()])

        metadata = {
            'session_id': str(session_id),
            'session_description': str(session_desc),
            'n_rois': int(dff_data.shape[0]),
            'n_timepoints': int(dff_data.shape[1]),
            'frame_rate_hz': float(frame_rate_hz),
            **{k: decode_value(v) for k, v in imaging_metadata.items()},
        }

    return dff_data, frame_rate_hz, metadata


def process_nwb_session(nwb_path, output_base_dir, z_threshold=2.5, min_duration_s=1.0):
    """Process a single Allen NWB session."""

    session_id = nwb_path.stem
    output_dir = output_base_dir / f"session_{session_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print(f"Processing: {session_id}")
    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # 1. Load NWB Data
    # -------------------------------------------------------------------------
    print("[1/8] Loading NWB file...")

    try:
        traces, frame_rate_hz, metadata = load_nwb_dff(nwb_path)
        n_cells = traces.shape[0]
        n_timepoints = traces.shape[1]
        duration_s = n_timepoints / frame_rate_hz

        print(f"  ✓ Session: {metadata['session_id']}")
        print(f"  ✓ ROIs: {n_cells}")
        print(f"  ✓ Timepoints: {n_timepoints:,}")
        print(f"  ✓ Duration: {duration_s/60:.1f} min ({duration_s:.1f}s)")
        print(f"  ✓ Frame rate: {frame_rate_hz:.2f} Hz")
        if 'indicator' in metadata:
            print(f"  ✓ Indicator: {metadata['indicator']}")
    except Exception as e:
        print(f"  ✗ Error loading NWB: {e}")
        import traceback
        traceback.print_exc()
        return None

    # -------------------------------------------------------------------------
    # 2. Detect Events
    # -------------------------------------------------------------------------
    print(f"\n[2/8] Detecting calcium events (z_threshold={z_threshold})...")

    events = detect_events_from_traces(
        traces=traces,
        frame_rate_hz=frame_rate_hz,
        session_id=session_id,
        z_threshold=z_threshold,
        min_duration_s=min_duration_s,
    )

    print(f"  ✓ Detected {len(events)} events")

    if len(events) == 0:
        print("\n  ⚠️  No events detected!")
        print(f"  Try lowering z_threshold (current: {z_threshold})")
        print("  Example: --z-threshold 2.0 or --z-threshold 1.5")
        return None

    # -------------------------------------------------------------------------
    # 3. Compute Statistics
    # -------------------------------------------------------------------------
    print("\n[3/8] Computing event statistics...")

    stats = compute_event_statistics(events, recording_duration_s=duration_s)

    print(f"  ✓ Regions with events: {stats.n_regions}")
    print(f"  ✓ Event rate: {stats.event_rate_hz:.4f} Hz")
    print(f"  ✓ Events per ROI: {stats.n_events / n_cells:.2f}")
    print(f"  ✓ Duration: {stats.duration_mean:.2f} ± {stats.duration_std:.2f} s")
    print(f"  ✓ Amplitude: {stats.amplitude_mean:.3f} ± {stats.amplitude_std:.3f}")

    # -------------------------------------------------------------------------
    # 4. Build Networks
    # -------------------------------------------------------------------------
    print("\n[4/8] Building functional connectivity networks...")

    graphs = build_event_coactivation_graph(
        events=events,
        session_id=session_id,
        frame_rate_hz=frame_rate_hz,
        window_size_s=60.0,  # 1-minute windows
        bin_size_s=1.0,  # 1-second bins for coactivation
    )

    print(f"  ✓ Built {len(graphs)} temporal networks")

    if graphs:
        metrics = compute_temporal_network_metrics(graphs)
        stability = compute_network_stability(graphs, method='jaccard')

        mean_density = np.mean([m.density for m in metrics])
        mean_clustering = np.mean([m.global_clustering for m in metrics])
        mean_stability = np.mean(stability) if len(stability) > 0 else 0.0

        print(f"  ✓ Mean density: {mean_density:.3f}")
        print(f"  ✓ Mean clustering: {mean_clustering:.3f}")
        print(f"  ✓ Network stability: {mean_stability:.3f}")

    # -------------------------------------------------------------------------
    # 5. Generate Figures
    # -------------------------------------------------------------------------
    print("\n[5/8] Generating visualizations...")

    # Event raster
    plot_event_raster(
        events, frame_rate_hz,
        save_path=figures_dir / "event_raster.png",
    )
    print(f"  ✓ Event raster")

    # Event distributions
    plot_event_distributions(
        events,
        save_path=figures_dir / "event_distributions.png",
    )
    print(f"  ✓ Event distributions")

    # Network graph (first window)
    if graphs:
        plot_network_graph(
            graphs[0],
            save_path=figures_dir / "network_graph.png",
            layout='spring',
            show_weights=True,
        )
        print(f"  ✓ Network graph")

    print(f"  ✓ Figures saved to figures/")

    # -------------------------------------------------------------------------
    # 6. Tokenize and Export
    # -------------------------------------------------------------------------
    print("\n[6/8] Tokenizing and exporting results...")

    # Save events to Parquet
    events_path = output_dir / "events.parquet"
    save_events_parquet(events, events_path)
    print(f"  ✓ Events: events.parquet")

    # Tokenize events
    tokenizer = AstroEventTokenizer()
    token_seq = tokenizer.tokenize(events)

    # Save tokens
    tokens_path = output_dir / "astro_tokens.npz"
    np.savez(
        tokens_path,
        tokens=token_seq.tokens,
        timestamps=token_seq.timestamps_s,
        metadata=token_seq.metadata,
    )
    print(f"  ✓ Tokens: astro_tokens.npz ({len(token_seq.tokens)} tokens)")

    # Build neuroFMx manifest
    manifest = build_neurofm_manifest(
        session_id=session_id,
        modalities={
            'astro': {
                'token_path': str(tokens_path),
                'n_tokens': len(events),
                'modality_type': 'astrocyte_events',
            }
        },
        metadata={
            'n_rois': n_cells,
            'duration_s': duration_s,
            'frame_rate_hz': frame_rate_hz,
            'n_events': len(events),
            'n_networks': len(graphs),
            'event_rate_hz': stats.event_rate_hz,
            'mean_density': mean_density if graphs else 0.0,
            'mean_clustering': mean_clustering if graphs else 0.0,
            'network_stability': mean_stability if graphs else 0.0,
            **{k: v for k, v in metadata.items() if k not in ['n_rois', 'n_timepoints']},
        }
    )

    manifest_path = output_dir / "neurofm_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  ✓ Manifest: neurofm_manifest.json")

    # -------------------------------------------------------------------------
    # 7. Summary Report
    # -------------------------------------------------------------------------
    print("\n[7/8] Generating summary report...")

    summary = {
        'session_id': session_id,
        'nwb_path': str(nwb_path),
        'n_rois': n_cells,
        'n_timepoints': n_timepoints,
        'duration_s': duration_s,
        'duration_min': duration_s / 60,
        'frame_rate_hz': frame_rate_hz,
        'n_events': len(events),
        'n_networks': len(graphs),
        'statistics': {
            'event_rate_hz': stats.event_rate_hz,
            'events_per_roi': stats.n_events / n_cells,
            'duration_mean': stats.duration_mean,
            'duration_std': stats.duration_std,
            'amplitude_mean': stats.amplitude_mean,
            'amplitude_std': stats.amplitude_std,
        },
        'network_metrics': {
            'mean_density': mean_density if graphs else 0.0,
            'mean_clustering': mean_clustering if graphs else 0.0,
            'stability': mean_stability if graphs else 0.0,
        },
        'outputs': {
            'events': 'events.parquet',
            'tokens': 'astro_tokens.npz',
            'manifest': 'neurofm_manifest.json',
            'figures': 'figures/',
        }
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  ✓ Summary: summary.json")

    # Print text summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"\nSession: {session_id}")
    print(f"Recording: {duration_s/60:.1f} min, {n_cells} ROIs")
    print(f"Events: {len(events)} ({stats.event_rate_hz:.4f} Hz, {stats.n_events/n_cells:.2f} per ROI)")
    print(f"Networks: {len(graphs)} (stability: {mean_stability:.3f})")
    print(f"\nAll outputs saved to: {output_dir.name}/")
    print()

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Process Allen Visual Coding NWB files with neuros-astro"
    )
    parser.add_argument('--nwb-dir', type=str, help='Directory containing NWB files')
    parser.add_argument('--session-id', type=str, help='Process specific session ID')
    parser.add_argument('--output-dir', type=str, default='./allen_nwb_results',
                       help='Output directory')
    parser.add_argument('--z-threshold', type=float, default=2.5,
                       help='Z-score threshold for event detection')
    parser.add_argument('--min-duration', type=float, default=1.0,
                       help='Minimum event duration (seconds)')
    parser.add_argument('--max-sessions', type=int, default=None,
                       help='Maximum number of sessions to process')

    args = parser.parse_args()

    print("=" * 80)
    print("neuros-astro: Allen NWB Processing Pipeline")
    print("=" * 80)
    print()

    # Find NWB files
    print("[Setup] Locating NWB files...")
    nwb_dir, nwb_files = find_nwb_files(args.nwb_dir)

    if not nwb_files:
        print("\n✗ No NWB files found!")
        print(f"\nCurrent directory: {Path.cwd()}")
        print("\nSearched in:")
        cwd = Path.cwd()
        search_locs = [
            cwd / "../neuros-mechint/examples/allen_data_demo/allen_validation_cache/ophys_experiment_data",
            cwd / "../../neuros-mechint/examples/allen_data_demo/allen_validation_cache/ophys_experiment_data",
            cwd / "allen_validation_cache/ophys_experiment_data",
        ]
        for loc in search_locs:
            exists = "✓" if loc.exists() else "✗"
            has_files = ""
            if loc.exists():
                files = list(loc.glob("*.nwb"))
                if files:
                    has_files = f" ({len(files)} files found)"
            print(f"  {exists} {loc}{has_files}")
        print("\nPlease specify NWB directory:")
        print("  python process_allen_nwb.py --nwb-dir /path/to/nwb/files")
        return

    print(f"✓ Found {len(nwb_files)} NWB file(s) in {nwb_dir}")
    print()

    # Filter by session ID if specified
    if args.session_id:
        nwb_files = [f for f in nwb_files if args.session_id in f.stem]
        if not nwb_files:
            print(f"✗ No session matching '{args.session_id}' found")
            return
        print(f"Processing specific session: {args.session_id}")

    # Limit number of sessions
    if args.max_sessions:
        nwb_files = sorted(nwb_files)[:args.max_sessions]
        print(f"Processing first {len(nwb_files)} session(s)")

    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each session
    results = []

    for i, nwb_file in enumerate(nwb_files, 1):
        print(f"\n{'='*80}")
        print(f"Session {i}/{len(nwb_files)}")
        print(f"{'='*80}\n")

        try:
            summary = process_nwb_session(
                nwb_file,
                output_dir,
                z_threshold=args.z_threshold,
                min_duration_s=args.min_duration,
            )

            if summary:
                results.append(summary)

        except Exception as e:
            print(f"\n✗ Error processing {nwb_file.stem}: {e}")
            import traceback
            traceback.print_exc()

    # Overall summary
    print("\n" + "=" * 80)
    print("ALL SESSIONS PROCESSED")
    print("=" * 80)
    print(f"\nSuccessfully processed: {len(results)}/{len(nwb_files)} sessions")
    print(f"Output directory: {output_dir.absolute()}")

    if results:
        total_events = sum(r['n_events'] for r in results)
        total_networks = sum(r['n_networks'] for r in results)
        total_duration_min = sum(r['duration_min'] for r in results)

        print(f"\nTotal recording time: {total_duration_min:.1f} min")
        print(f"Total events detected: {total_events}")
        print(f"Total networks built: {total_networks}")
        print(f"\nMean event rate: {np.mean([r['statistics']['event_rate_hz'] for r in results]):.4f} Hz")
        print(f"Mean events per ROI: {np.mean([r['statistics']['events_per_roi'] for r in results]):.2f}")
        print(f"Mean network stability: {np.mean([r['network_metrics']['stability'] for r in results]):.3f}")

    # Save overall summary
    if results:
        overall_summary = {
            'n_sessions_processed': len(results),
            'total_recording_time_min': sum(r['duration_min'] for r in results),
            'total_events': sum(r['n_events'] for r in results),
            'total_networks': sum(r['n_networks'] for r in results),
            'sessions': results,
        }

        with open(output_dir / 'overall_summary.json', 'w') as f:
            json.dump(overall_summary, f, indent=2)

        print(f"\n✓ Overall summary: overall_summary.json")

    print("\n✅ Pipeline complete! Ready for neuroFMx integration.")
    print()


if __name__ == "__main__":
    main()
