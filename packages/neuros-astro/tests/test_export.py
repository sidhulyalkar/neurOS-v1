"""Test export and import functionality."""

import tempfile
from pathlib import Path
import numpy as np

from neuros_astro.metadata.schema import AstroEvent, TokenizedAstroSequence
from neuros_astro.export.to_parquet import (
    events_to_dataframe,
    dataframe_to_events,
    save_events_parquet,
    load_events_parquet,
)
from neuros_astro.export.to_neurofm import (
    save_tokenized_sequence_npz,
    load_tokenized_sequence_npz,
    build_neurofm_manifest,
)


def test_events_to_dataframe_roundtrip():
    """Test events to DataFrame conversion preserves data."""
    events = [
        AstroEvent(
            event_id="e1",
            session_id="s1",
            region_id="roi_0",
            onset_frame=100,
            offset_frame=150,
            peak_frame=120,
            duration_s=5.0,
            peak_dff=0.3,
            area_px=50.0,
            centroid_yx=(32.0, 64.0),
            confidence=0.9,
        ),
    ]

    df = events_to_dataframe(events)
    recovered_events = dataframe_to_events(df)

    assert len(recovered_events) == 1
    assert recovered_events[0].event_id == "e1"
    assert recovered_events[0].peak_dff == 0.3
    assert recovered_events[0].centroid_yx == (32.0, 64.0)


def test_save_load_events_parquet():
    """Test saving and loading events to/from Parquet."""
    events = [
        AstroEvent(
            event_id=f"e{i}",
            session_id="s1",
            region_id=f"roi_{i}",
            onset_frame=i * 100,
            offset_frame=i * 100 + 50,
            peak_frame=i * 100 + 25,
            duration_s=5.0,
            peak_dff=0.3,
            confidence=0.9,
        )
        for i in range(3)
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "events.parquet"
        save_events_parquet(events, path)

        loaded_events = load_events_parquet(path)

        assert len(loaded_events) == 3
        assert loaded_events[0].event_id == "e0"
        assert loaded_events[2].event_id == "e2"


def test_save_load_tokenized_sequence():
    """Test saving and loading tokenized sequences."""
    sequence = TokenizedAstroSequence(
        session_id="test",
        tokens=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        timestamps_s=[0.0, 1.0],
        region_ids=["roi_0", "roi_1"],
        feature_names=["f1", "f2", "f3"],
        metadata={"test": "value"},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "tokens.npz"
        save_tokenized_sequence_npz(sequence, path)

        loaded_sequence = load_tokenized_sequence_npz(path)

        assert loaded_sequence.session_id == "test"
        assert len(loaded_sequence.tokens) == 2
        assert loaded_sequence.feature_names == ["f1", "f2", "f3"]
        assert loaded_sequence.metadata["test"] == "value"

        # Check arrays are equal
        np.testing.assert_array_almost_equal(sequence.tokens, loaded_sequence.tokens)


def test_build_neurofm_manifest():
    """Test manifest building."""
    manifest = build_neurofm_manifest(
        session_id="test_session",
        modalities={
            "astro": {
                "type": "event_tokens",
                "path": "astro_tokens.npz",
                "sampling": "irregular",
            }
        },
        metadata={"experiment": "test"},
    )

    assert manifest["session_id"] == "test_session"
    assert "astro" in manifest["modalities"]
    assert manifest["modalities"]["astro"]["type"] == "event_tokens"
    assert manifest["metadata"]["experiment"] == "test"
