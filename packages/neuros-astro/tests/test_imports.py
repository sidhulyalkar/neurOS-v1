"""Test that all modules can be imported."""

def test_import_main_package():
    """Test main package import."""
    import neuros_astro
    assert neuros_astro.__version__ == "0.1.0"


def test_import_schemas():
    """Test schema imports."""
    from neuros_astro.metadata.schema import (
        AstroSession,
        AstroRegion,
        AstroEvent,
        AstroGraph,
        TokenizedAstroSequence,
        DatasetTriageResult,
    )
    assert AstroSession is not None
    assert AstroEvent is not None


def test_import_controlled_terms():
    """Test controlled terms import."""
    from neuros_astro.metadata.controlled_terms import (
        ASTRO_TERMS,
        CALCIUM_TERMS,
        MODALITY_TERMS,
    )
    assert len(ASTRO_TERMS) > 0
    assert len(CALCIUM_TERMS) > 0


def test_import_dataset_scoring():
    """Test dataset scoring import."""
    from neuros_astro.metadata.dataset_scoring import score_dataset_metadata
    assert callable(score_dataset_metadata)


def test_import_synthetic():
    """Test synthetic data generation import."""
    from neuros_astro.io.synthetic import (
        generate_synthetic_astro_traces,
        generate_synthetic_astro_movie,
    )
    assert callable(generate_synthetic_astro_traces)
    assert callable(generate_synthetic_astro_movie)


def test_import_event_detection():
    """Test event detection import."""
    from neuros_astro.events.event_detection import (
        detect_events_from_trace,
        detect_events_from_traces,
        detect_candidate_events_from_movie,
    )
    assert callable(detect_events_from_trace)


def test_import_networks():
    """Test network construction import."""
    from neuros_astro.networks.functional_connectivity import (
        build_event_coactivation_graph,
    )
    assert callable(build_event_coactivation_graph)


def test_import_tokenization():
    """Test tokenization import."""
    from neuros_astro.tokenization.event_tokenizer import AstroEventTokenizer
    from neuros_astro.tokenization.astro_tokenizer import BinnedAstroTokenizer

    assert AstroEventTokenizer is not None
    assert BinnedAstroTokenizer is not None


def test_import_export():
    """Test export utilities import."""
    from neuros_astro.export.to_parquet import save_events_parquet
    from neuros_astro.export.to_neurofm import save_tokenized_sequence_npz

    assert callable(save_events_parquet)
    assert callable(save_tokenized_sequence_npz)
