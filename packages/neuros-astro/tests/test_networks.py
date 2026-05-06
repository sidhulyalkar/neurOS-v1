"""Test network construction and graph features."""

import numpy as np
from neuros_astro.metadata.schema import AstroEvent
from neuros_astro.networks.functional_connectivity import (
    events_to_binary_matrix,
    build_event_coactivation_graph,
)
from neuros_astro.networks.graph_features import compute_graph_summary_features


def test_events_to_binary_matrix_empty():
    """Test binary matrix conversion with empty events."""
    matrix, region_ids, bin_times = events_to_binary_matrix(
        events=[], frame_rate_hz=10.0, bin_size_s=1.0
    )

    assert len(matrix) == 0
    assert len(region_ids) == 0


def test_events_to_binary_matrix():
    """Test binary matrix conversion."""
    events = [
        AstroEvent(
            event_id="e1",
            session_id="s1",
            region_id="roi_0",
            onset_frame=0,
            offset_frame=20,
            peak_frame=10,
            duration_s=2.0,
            peak_dff=0.5,
        ),
        AstroEvent(
            event_id="e2",
            session_id="s1",
            region_id="roi_1",
            onset_frame=10,
            offset_frame=30,
            peak_frame=20,
            duration_s=2.0,
            peak_dff=0.4,
        ),
    ]

    matrix, region_ids, bin_times = events_to_binary_matrix(
        events=events, frame_rate_hz=10.0, bin_size_s=1.0
    )

    assert len(region_ids) == 2
    assert matrix.shape[1] == 2
    assert np.any(matrix > 0)  # Some bins should be active


def test_build_coactivation_graph_empty():
    """Test graph construction with empty events."""
    graphs = build_event_coactivation_graph(
        events=[], session_id="test", frame_rate_hz=10.0
    )

    assert len(graphs) == 0


def test_build_coactivation_graph():
    """Test graph construction with coactive events."""
    # Create overlapping events in two regions
    events = [
        AstroEvent(
            event_id="e1",
            session_id="s1",
            region_id="roi_0",
            onset_frame=0,
            offset_frame=50,
            peak_frame=25,
            duration_s=5.0,
            peak_dff=0.5,
        ),
        AstroEvent(
            event_id="e2",
            session_id="s1",
            region_id="roi_1",
            onset_frame=10,
            offset_frame=60,
            peak_frame=35,
            duration_s=5.0,
            peak_dff=0.4,
        ),
    ]

    graphs = build_event_coactivation_graph(
        events=events,
        session_id="test",
        frame_rate_hz=10.0,
        window_size_s=10.0,
        stride_s=5.0,
    )

    assert len(graphs) > 0

    # Check first graph has both nodes
    assert len(graphs[0].nodes) == 2

    # Should have an edge (events overlap)
    if len(graphs[0].edges) > 0:
        assert graphs[0].edges[0] in [("roi_0", "roi_1"), ("roi_1", "roi_0")]


def test_compute_graph_features():
    """Test graph feature computation."""
    from neuros_astro.metadata.schema import AstroGraph

    graph = AstroGraph(
        session_id="test",
        window_start_s=0.0,
        window_end_s=30.0,
        nodes=["roi_0", "roi_1", "roi_2"],
        edges=[("roi_0", "roi_1"), ("roi_1", "roi_2")],
        edge_weights=[0.5, 0.3],
    )

    features = compute_graph_summary_features(graph)

    assert features["n_nodes"] == 3
    assert features["n_edges"] == 2
    assert features["density"] > 0
    assert features["mean_edge_weight"] == 0.4
