#!/usr/bin/env python3
"""
Example of using neuros-astro Python API.

This demonstrates how to use neuros-astro programmatically without the CLI.
"""

import numpy as np
from neuros_astro.io.synthetic import generate_synthetic_astro_traces
from neuros_astro.events.event_detection import detect_events_from_traces
from neuros_astro.networks.functional_connectivity import build_event_coactivation_graph
from neuros_astro.networks.graph_features import compute_graph_summary_features
from neuros_astro.tokenization.event_tokenizer import AstroEventTokenizer


def main():
    """Demonstrate Python API usage."""
    print("=" * 70)
    print("neuros-astro: Python API Example")
    print("=" * 70)

    # Generate synthetic data
    print("\nGenerating synthetic astrocyte traces...")
    traces, ground_truth = generate_synthetic_astro_traces(
        n_regions=5,
        duration_s=30.0,
        frame_rate_hz=10.0,
        n_events_per_region=3,
        event_rise_time_s=(1.0, 2.0),
        event_decay_time_s=(3.0, 8.0),
        seed=42,
    )

    print(f"  Traces shape: {traces.shape}")
    print(f"  Ground truth events: {len(ground_truth)}")

    # Detect events
    print("\nDetecting calcium events...")
    events = detect_events_from_traces(
        traces=traces,
        frame_rate_hz=10.0,
        session_id="api_example",
        z_threshold=2.0,
        min_duration_s=1.0,
    )

    print(f"  Detected {len(events)} events")

    # Analyze events
    if len(events) > 0:
        print("\nEvent analysis:")
        durations = [e.duration_s for e in events]
        amplitudes = [e.peak_dff for e in events]

        print(f"  Average duration: {np.mean(durations):.2f}s ± {np.std(durations):.2f}s")
        print(f"  Average amplitude: {np.mean(amplitudes):.3f} ± {np.std(amplitudes):.3f}")

        # Count events per region
        from collections import Counter

        region_counts = Counter(e.region_id for e in events if e.region_id)
        print(f"  Events per region: {dict(region_counts)}")

    # Build networks
    print("\nBuilding coactivation networks...")
    graphs = build_event_coactivation_graph(
        events=events,
        session_id="api_example",
        frame_rate_hz=10.0,
        window_size_s=15.0,
        stride_s=5.0,
        min_edge_weight=0.2,
    )

    print(f"  Generated {len(graphs)} time-windowed graphs")

    # Analyze network properties
    if len(graphs) > 0:
        print("\nNetwork analysis:")

        for i, graph in enumerate(graphs[:3]):  # Show first 3 graphs
            features = compute_graph_summary_features(graph)
            print(f"\n  Graph {i+1} [{graph.window_start_s:.1f}s - {graph.window_end_s:.1f}s]:")
            print(f"    Nodes: {features['n_nodes']}")
            print(f"    Edges: {features['n_edges']}")
            print(f"    Density: {features['density']:.3f}")
            if features['n_edges'] > 0:
                print(f"    Mean edge weight: {features['mean_edge_weight']:.3f}")
                print(f"    Mean degree: {features['degree_mean']:.2f}")

    # Tokenize for models
    print("\nTokenizing events...")
    tokenizer = AstroEventTokenizer(normalize=True, image_height=128, image_width=128)
    token_sequence = tokenizer.tokenize(events, session_id="api_example")

    print(f"  Token shape: ({len(token_sequence.tokens)}, {len(token_sequence.feature_names)})")
    print(f"  Features: {', '.join(token_sequence.feature_names)}")

    if len(tokenizer.norm_stats) > 0:
        print("\nNormalization statistics:")
        for feature, stats in list(tokenizer.norm_stats.items())[:3]:
            print(f"  {feature}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

    print("\n" + "=" * 70)
    print("Complete! Ready for integration with neuroFMx.")
    print("=" * 70)


if __name__ == "__main__":
    main()
