from __future__ import annotations

import numpy as np
import pytest

from neuros_mechint.representations import (
    PrecomputedTemporalSSLRepresentation,
    SequenceBatch,
    pairwise_distance_rank_preservation,
    temporal_continuity_ratio,
)


def test_sequence_batch_rejects_complex_values_before_method_dispatch() -> None:
    values = np.ones((5, 3), dtype=np.complex128) * (1 + 2j)
    with pytest.raises(TypeError, match="real numeric"):
        SequenceBatch(sequences=(values,), sequence_ids=("complex",))


def test_nested_metadata_keys_are_not_coerced_to_strings() -> None:
    with pytest.raises(ValueError, match="metadata keys"):
        SequenceBatch(
            sequences=(np.ones((5, 3)),),
            sequence_ids=("s1",),
            metadata={"nested": {7: "ambiguous"}},
        )


def test_external_ssl_rejects_complex_latent_coordinates() -> None:
    with pytest.raises(TypeError, match="real numeric"):
        PrecomputedTemporalSSLRepresentation(
            {"eval": np.ones((5, 2), dtype=np.complex128)},
            model_id="ssl",
            model_version="1",
        )


def test_public_geometry_metrics_reject_complex_inputs_without_projection() -> None:
    source = np.arange(15, dtype=float).reshape(5, 3)
    complex_embedding = np.ones((5, 2), dtype=np.complex128) * (1 + 1j)
    with pytest.raises(TypeError, match="real numeric"):
        pairwise_distance_rank_preservation(source, complex_embedding)
    with pytest.raises(TypeError, match="real numeric"):
        temporal_continuity_ratio(complex_embedding)
