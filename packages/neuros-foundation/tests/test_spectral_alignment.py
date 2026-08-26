from __future__ import annotations

import numpy as np
import pytest

from neuros.foundation_models.spectral_alignment import (
    SPECTRAL_METHOD_ID,
    spectral_alignment_evidence,
    verify_snap_invariant_reference,
)


REPRESENTATIONS = np.asarray(
    [
        [1.0, 0.2],
        [-0.4, 1.2],
        [0.7, -0.8],
        [-1.1, -0.3],
        [0.3, 0.9],
    ],
    dtype=np.float64,
)
TARGETS = np.asarray(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [-1.0, 0.5],
        [0.5, -0.5],
    ],
    dtype=np.float64,
)

# Invariant quantities from the SNAP reference construction for the matrices
# above. Null-space per-mode power is deliberately omitted because its basis is
# not unique. The first two modes and aggregate residual are basis invariant.
SNAP_INVARIANT_FIXTURE = {
    "positive_eigenvalues": [
        0.6371340246436072,
        0.48926597535641314,
    ],
    "target_power_by_mode": [
        0.30292378828622346,
        0.2799058693942279,
    ],
    "cumulative_captured_target_power": [
        0.30292378828622346,
        0.5828296576804515,
    ],
    "effective_dimension": 1.9661176974633945,
    "residual_target_power": 0.4171703423195485,
}


def test_spectral_alignment_matches_snap_invariant_reference_values():
    evidence = spectral_alignment_evidence(REPRESENTATIONS, TARGETS, centered=True)
    result = verify_snap_invariant_reference(evidence, SNAP_INVARIANT_FIXTURE)

    assert evidence.method_id == SPECTRAL_METHOD_ID
    assert evidence.feature_rank == 2
    assert evidence.n_samples == 5
    assert evidence.n_features == 2
    assert evidence.target_dim == 2
    assert evidence.residual_target_power == pytest.approx(0.4171703423195485)
    assert result["conformant"] is True
    assert result["null_space_basis_invariant"] is True


def test_spectral_evidence_is_deterministic_and_hashes_inputs():
    first = spectral_alignment_evidence(REPRESENTATIONS, TARGETS)
    second = spectral_alignment_evidence(REPRESENTATIONS.copy(), TARGETS.copy())

    assert first == second
    assert len(first.representation_sha256) == 64
    assert len(first.target_sha256) == 64
    assert len(first.evidence_sha256) == 64

    changed = REPRESENTATIONS.copy()
    changed[0, 0] += 1e-3
    changed_evidence = spectral_alignment_evidence(changed, TARGETS)
    assert changed_evidence.representation_sha256 != first.representation_sha256
    assert changed_evidence.evidence_sha256 != first.evidence_sha256


def test_spectral_evidence_aggregates_null_space_instead_of_exposing_arbitrary_basis():
    evidence = spectral_alignment_evidence(REPRESENTATIONS, TARGETS)

    assert len(evidence.positive_eigenvalues) == evidence.feature_rank
    assert len(evidence.target_power_by_mode) == evidence.feature_rank
    assert len(evidence.cumulative_captured_target_power) == evidence.feature_rank
    assert 0.0 <= evidence.residual_target_power <= 1.0
    assert evidence.to_dict()["claim_boundary"]["null_space_basis_invariant"] is True


def test_spectral_alignment_fails_closed_on_invalid_or_undefined_inputs():
    with pytest.raises(ValueError, match="same aligned samples"):
        spectral_alignment_evidence(REPRESENTATIONS, TARGETS[:-1])

    with pytest.raises(ValueError, match="NaN or infinite"):
        bad = REPRESENTATIONS.copy()
        bad[0, 0] = np.nan
        spectral_alignment_evidence(bad, TARGETS)

    with pytest.raises(ValueError, match="zero positive rank"):
        spectral_alignment_evidence(np.ones((5, 2)), TARGETS, centered=True)

    with pytest.raises(ValueError, match="zero-power dimension"):
        spectral_alignment_evidence(REPRESENTATIONS, np.ones((5, 1)), centered=True)

    with pytest.raises(ValueError, match="rank_tolerance"):
        spectral_alignment_evidence(REPRESENTATIONS, TARGETS, rank_tolerance=-1.0)


def test_reference_verifier_detects_a_changed_invariant_quantity():
    evidence = spectral_alignment_evidence(REPRESENTATIONS, TARGETS)
    fixture = dict(SNAP_INVARIANT_FIXTURE)
    fixture["residual_target_power"] = 0.0

    result = verify_snap_invariant_reference(evidence, fixture)
    assert result["conformant"] is False
    assert result["comparisons"]["residual_target_power"] is False
