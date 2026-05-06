"""Test tokenization of events and graphs."""

from neuros_astro.metadata.schema import AstroEvent
from neuros_astro.tokenization.event_tokenizer import AstroEventTokenizer
from neuros_astro.tokenization.astro_tokenizer import BinnedAstroTokenizer


def test_event_tokenizer_empty():
    """Test tokenizer with empty events."""
    tokenizer = AstroEventTokenizer()
    tokens = tokenizer.tokenize([], session_id="test")

    assert len(tokens.tokens) == 0
    assert len(tokens.timestamps_s) == 0
    assert len(tokens.feature_names) > 0


def test_event_tokenizer():
    """Test event tokenization."""
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
        AstroEvent(
            event_id="e2",
            session_id="s1",
            region_id="roi_1",
            onset_frame=200,
            offset_frame=250,
            peak_frame=220,
            duration_s=5.0,
            peak_dff=0.4,
            area_px=60.0,
            centroid_yx=(48.0, 80.0),
            confidence=0.8,
        ),
    ]

    tokenizer = AstroEventTokenizer(normalize=False)
    tokens = tokenizer.tokenize(events, session_id="test")

    assert len(tokens.tokens) == 2
    assert len(tokens.timestamps_s) == 2
    assert len(tokens.feature_names) == len(tokens.tokens[0])


def test_event_tokenizer_normalization():
    """Test that normalization works."""
    events = [
        AstroEvent(
            event_id=f"e{i}",
            session_id="s1",
            region_id=f"roi_{i}",
            onset_frame=i * 100,
            offset_frame=i * 100 + 50,
            peak_frame=i * 100 + 25,
            duration_s=5.0,
            peak_dff=0.3 + i * 0.1,
            confidence=0.9,
        )
        for i in range(5)
    ]

    tokenizer = AstroEventTokenizer(normalize=True)
    tokens = tokenizer.tokenize(events, session_id="test")

    # Check normalization stats were computed
    assert len(tokenizer.norm_stats) > 0


def test_binned_tokenizer():
    """Test binned tokenization."""
    events = [
        AstroEvent(
            event_id="e1",
            session_id="s1",
            region_id="roi_0",
            onset_frame=10,
            offset_frame=20,
            peak_frame=15,
            duration_s=1.0,
            peak_dff=0.3,
            area_px=50.0,
            confidence=0.9,
        ),
    ]

    tokenizer = BinnedAstroTokenizer(bin_size_s=1.0, normalize=False)
    tokens = tokenizer.tokenize(events, duration_s=10.0, session_id="test")

    assert len(tokens.tokens) == 10  # 10 seconds / 1 second bins
    assert len(tokens.timestamps_s) == 10
