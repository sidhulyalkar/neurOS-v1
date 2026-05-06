"""Test core data schemas and validation."""

import pytest
from neuros_astro.metadata.schema import (
    AstroSession,
    AstroEvent,
    AstroGraph,
    TokenizedAstroSequence,
)


def test_astro_event_valid():
    """Test valid AstroEvent creation."""
    event = AstroEvent(
        event_id="evt_001",
        session_id="ses_001",
        region_id="roi_01",
        onset_frame=100,
        offset_frame=150,
        peak_frame=120,
        duration_s=5.0,
        peak_dff=0.25,
        confidence=0.9,
    )

    assert event.event_id == "evt_001"
    assert event.duration_s == 5.0


def test_astro_event_invalid_frame_ordering():
    """Test that invalid frame ordering raises error."""
    with pytest.raises(ValueError, match="offset_frame must be >= onset_frame"):
        AstroEvent(
            event_id="evt_001",
            session_id="ses_001",
            onset_frame=150,
            offset_frame=100,  # Invalid: before onset
            peak_frame=120,
            duration_s=5.0,
            peak_dff=0.25,
        )


def test_astro_event_invalid_peak_frame():
    """Test that peak_frame outside onset-offset raises error."""
    with pytest.raises(ValueError, match="peak_frame must be between"):
        AstroEvent(
            event_id="evt_001",
            session_id="ses_001",
            onset_frame=100,
            offset_frame=150,
            peak_frame=200,  # Invalid: after offset
            duration_s=5.0,
            peak_dff=0.25,
        )


def test_astro_event_invalid_confidence():
    """Test that confidence outside [0, 1] raises error."""
    with pytest.raises(ValueError):
        AstroEvent(
            event_id="evt_001",
            session_id="ses_001",
            onset_frame=100,
            offset_frame=150,
            peak_frame=120,
            duration_s=5.0,
            peak_dff=0.25,
            confidence=1.5,  # Invalid: > 1.0
        )


def test_astro_graph_valid():
    """Test valid AstroGraph creation."""
    graph = AstroGraph(
        session_id="ses_001",
        window_start_s=0.0,
        window_end_s=30.0,
        nodes=["roi_01", "roi_02", "roi_03"],
        edges=[("roi_01", "roi_02"), ("roi_02", "roi_03")],
        edge_weights=[0.5, 0.3],
    )

    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2
    assert len(graph.edge_weights) == 2


def test_astro_graph_invalid_window():
    """Test that invalid window raises error."""
    with pytest.raises(ValueError, match="window_end_s must be > window_start_s"):
        AstroGraph(
            session_id="ses_001",
            window_start_s=30.0,
            window_end_s=10.0,  # Invalid: before start
            nodes=["roi_01"],
            edges=[],
            edge_weights=[],
        )


def test_astro_graph_mismatched_edges_weights():
    """Test that mismatched edges and weights raises error."""
    with pytest.raises(ValueError, match="edges and edge_weights must have same length"):
        AstroGraph(
            session_id="ses_001",
            window_start_s=0.0,
            window_end_s=30.0,
            nodes=["roi_01", "roi_02"],
            edges=[("roi_01", "roi_02")],
            edge_weights=[0.5, 0.3],  # Invalid: too many weights
        )


def test_tokenized_sequence_valid():
    """Test valid TokenizedAstroSequence creation."""
    seq = TokenizedAstroSequence(
        session_id="ses_001",
        tokens=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        timestamps_s=[0.0, 1.0],
        region_ids=["roi_01", "roi_02"],
        feature_names=["feature_1", "feature_2", "feature_3"],
    )

    assert len(seq.tokens) == 2
    assert len(seq.timestamps_s) == 2
    assert len(seq.feature_names) == 3


def test_tokenized_sequence_dimension_mismatch():
    """Test that dimension mismatches raise errors."""
    with pytest.raises(ValueError, match="timestamps_s length must match"):
        TokenizedAstroSequence(
            session_id="ses_001",
            tokens=[[1.0, 2.0], [3.0, 4.0]],
            timestamps_s=[0.0],  # Too few timestamps
            region_ids=["roi_01", "roi_02"],
            feature_names=["f1", "f2"],
        )
