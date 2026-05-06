#!/usr/bin/env python3
"""
End-to-end pipeline example for neuros-astro.

This script demonstrates the complete workflow:
1. Generate synthetic astrocyte data
2. Detect events from traces
3. Build functional networks
4. Tokenize for foundation models
5. Export to neuroFMx-compatible format
"""

from pathlib import Path
import numpy as np

from neuros_astro.io.synthetic import generate_synthetic_astro_traces
from neuros_astro.events.event_detection import detect_events_from_traces
from neuros_astro.networks.functional_connectivity import build_event_coactivation_graph
from neuros_astro.tokenization.event_tokenizer import AstroEventTokenizer
from neuros_astro.export.to_parquet import save_events_parquet
from neuros_astro.export.to_neurofm import (
    save_tokenized_sequence_npz,
    build_neurofm_manifest,
    save_neurofm_manifest,
)


def main():
    """Run end-to-end pipeline."""
    print("=" * 70)
    print("neuros-astro: End-to-End Pipeline Example")
    print("=" * 70)

    # Configuration
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    session_id = "demo_session"
    n_regions = 10
    duration_s = 60.0
    frame_rate_hz = 10.0

    # Step 1: Generate synthetic data
    print("\n[1/5] Generating synthetic astrocyte traces...")
    traces, gt_events = generate_synthetic_astro_traces(
        n_regions=n_regions,
        duration_s=duration_s,
        frame_rate_hz=frame_rate_hz,
        n_events_per_region=5,
        seed=42,
    )
    print(f"  ✓ Generated traces: shape {traces.shape}")
    print(f"  ✓ Ground truth events: {len(gt_events)}")

    # Save traces
    traces_path = output_dir / "synthetic_traces.npy"
    np.save(traces_path, traces)
    print(f"  ✓ Saved to: {traces_path}")

    # Step 2: Detect events
    print("\n[2/5] Detecting astrocyte calcium events...")
    events = detect_events_from_traces(
        traces=traces,
        frame_rate_hz=frame_rate_hz,
        session_id=session_id,
        z_threshold=2.0,
        min_duration_s=1.0,
        merge_gap_s=0.5,
    )
    print(f"  ✓ Detected {len(events)} events")

    # Show some event statistics
    if len(events) > 0:
        durations = [e.duration_s for e in events]
        amplitudes = [e.peak_dff for e in events]
        print(f"  ✓ Duration range: {min(durations):.2f}s - {max(durations):.2f}s")
        print(f"  ✓ Amplitude range: {min(amplitudes):.3f} - {max(amplitudes):.3f}")

    # Save events
    events_path = output_dir / "events.parquet"
    save_events_parquet(events, events_path)
    print(f"  ✓ Saved to: {events_path}")

    # Step 3: Build functional networks
    print("\n[3/5] Building astrocyte coactivation networks...")
    graphs = build_event_coactivation_graph(
        events=events,
        session_id=session_id,
        frame_rate_hz=frame_rate_hz,
        window_size_s=30.0,
        stride_s=5.0,
        min_edge_weight=0.1,
    )
    print(f"  ✓ Built {len(graphs)} network graphs")

    # Show network statistics
    if len(graphs) > 0:
        from neuros_astro.networks.graph_features import compute_graph_summary_features

        first_graph = graphs[0]
        features = compute_graph_summary_features(first_graph)
        print(f"  ✓ Example graph: {features['n_nodes']} nodes, {features['n_edges']} edges")
        print(f"  ✓ Network density: {features['density']:.3f}")

    # Step 4: Tokenize events
    print("\n[4/5] Tokenizing events for foundation models...")
    tokenizer = AstroEventTokenizer(normalize=True)
    token_sequence = tokenizer.tokenize(events, session_id=session_id)

    print(f"  ✓ Generated {len(token_sequence.tokens)} tokens")
    print(f"  ✓ Features: {', '.join(token_sequence.feature_names)}")

    # Save tokens
    tokens_path = output_dir / "astro_tokens.npz"
    save_tokenized_sequence_npz(token_sequence, tokens_path)
    print(f"  ✓ Saved to: {tokens_path}")

    # Step 5: Create neuroFMx manifest
    print("\n[5/5] Creating neuroFMx manifest...")
    manifest = build_neurofm_manifest(
        session_id=session_id,
        modalities={
            "astro": {
                "type": "event_tokens",
                "path": str(tokens_path.name),
                "sampling": "irregular",
                "timestamp_key": "timestamps_s",
                "feature_names": token_sequence.feature_names,
            }
        },
        metadata={
            "n_events": len(events),
            "n_graphs": len(graphs),
            "duration_s": duration_s,
            "frame_rate_hz": frame_rate_hz,
        },
    )

    manifest_path = output_dir / "neurofm_manifest.json"
    save_neurofm_manifest(manifest, manifest_path)
    print(f"  ✓ Saved to: {manifest_path}")

    print("\n" + "=" * 70)
    print("Pipeline complete! Outputs saved to:", output_dir)
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Load tokens in neuroFMx using the manifest")
    print("  2. Run ablation experiments (neural-only vs neural+astro)")
    print("  3. Analyze whether astro signals improve prediction/decoding")
    print()


if __name__ == "__main__":
    main()
