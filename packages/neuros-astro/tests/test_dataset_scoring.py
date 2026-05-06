"""Test dataset triage and scoring."""

from neuros_astro.metadata.dataset_scoring import score_dataset_metadata


def test_high_score_dataset():
    """Test that high-value astro dataset scores well."""
    metadata = {
        "description": "Two-photon calcium imaging of GFAP-labeled astrocytes using GCaMP6f",
        "has_raw_movie": True,
        "has_masks": True,
        "has_behavior": True,
    }

    result = score_dataset_metadata(metadata, session_id="test_high")

    # Should score high
    assert result.astro_reanalysis_score > 0.7
    assert "GFAP" in result.matched_astro_terms
    assert "GCaMP" in result.matched_calcium_terms or "GCaMP6" in result.matched_calcium_terms
    assert "two-photon" in result.matched_modality_terms
    assert result.recommended_next_step == "run_event_detection"


def test_low_score_dataset():
    """Test that neuron-only dataset scores low."""
    metadata = {
        "description": "Neuropixels recording of visual cortex neurons",
    }

    result = score_dataset_metadata(metadata, session_id="test_low")

    # Should score low
    assert result.astro_reanalysis_score < 0.3
    assert len(result.matched_astro_terms) == 0
    assert result.recommended_next_step in ["reject_low_value", "inspect_metadata"]


def test_medium_score_dataset():
    """Test dataset with some but not all indicators."""
    metadata = {
        "description": "Calcium imaging with GCaMP in visual cortex",
        "has_behavior": True,
    }

    result = score_dataset_metadata(metadata, session_id="test_medium")

    # Should score medium
    assert 0.2 <= result.astro_reanalysis_score <= 0.6
    assert len(result.matched_calcium_terms) > 0


def test_plain_text_scoring():
    """Test scoring from plain text."""
    text = "This dataset contains two-photon imaging of astrocytes with GCaMP in mouse cortex"

    result = score_dataset_metadata(text, session_id="text_test")

    assert result.astro_reanalysis_score > 0.4
    assert len(result.matched_astro_terms) > 0
    assert len(result.matched_calcium_terms) > 0
