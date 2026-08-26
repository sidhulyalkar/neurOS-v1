"""Backend-stable spectral representation evidence derived from SNAP.

SNAP (Spectral theory of Neural Alignment and Prediction) motivates a useful
vocabulary for relating representation eigenspectra to task/neural targets. The
upstream research implementation eigendecomposes the sample kernel and projects
targets onto its eigenvectors.

neurOS preserves the scientifically meaningful part of that construction while
hardening one important numerical ambiguity: eigenvectors inside an exact kernel
null space are not unique. Different valid LAPACK/Torch backends can therefore
redistribute target power among zero-eigenvalue modes. neurOS reports only
positive-rank modes individually and aggregates all out-of-span target power
into one residual term. This makes the evidence invariant to arbitrary null-space
basis rotations while retaining the SNAP interpretation.

The implementation is dependency-light NumPy and does not claim reproduced SNAP
paper results, biological alignment, or mechanistic equivalence. Those stronger
claims require separate evidence artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

SPECTRAL_METHOD_ID = "neuros-snap-invariant-spectral-alignment-v1"
SNAP_REFERENCE_REPOSITORY = "https://github.com/chung-neuroai-lab/SNAP"
SNAP_REFERENCE_PAPER = "Canatar et al., NeurIPS 2023, A Spectral Theory of Neural Prediction and Alignment"


def _matrix(values: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix; got shape {matrix.shape}")
    if matrix.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _participation_ratio(values: np.ndarray, *, eps: float = 1e-15) -> float:
    vector = np.asarray(values, dtype=np.float64)
    denominator = float(np.square(vector).sum())
    if denominator <= eps:
        return 0.0
    return float(vector.sum()) ** 2 / denominator


def _canonical_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class SpectralAlignmentEvidence:
    """Immutable spectral evidence for one aligned representation/target matrix."""

    method_id: str
    reference_method: str
    reference_repository: str
    centered: bool
    rank_tolerance: float
    n_samples: int
    n_features: int
    target_dim: int
    feature_rank: int
    positive_eigenvalues: tuple[float, ...]
    target_power_by_mode: tuple[float, ...]
    cumulative_captured_target_power: tuple[float, ...]
    effective_dimension: float
    task_tail_effective_dimension: float
    residual_target_power: float
    representation_sha256: str
    target_sha256: str
    evidence_sha256: str
    upstream_invariant_conformance_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "reference_method": self.reference_method,
            "reference_repository": self.reference_repository,
            "centered": self.centered,
            "rank_tolerance": self.rank_tolerance,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "target_dim": self.target_dim,
            "feature_rank": self.feature_rank,
            "positive_eigenvalues": list(self.positive_eigenvalues),
            "target_power_by_mode": list(self.target_power_by_mode),
            "cumulative_captured_target_power": list(self.cumulative_captured_target_power),
            "effective_dimension": self.effective_dimension,
            "task_tail_effective_dimension": self.task_tail_effective_dimension,
            "residual_target_power": self.residual_target_power,
            "representation_sha256": self.representation_sha256,
            "target_sha256": self.target_sha256,
            "evidence_sha256": self.evidence_sha256,
            "upstream_invariant_conformance_verified": self.upstream_invariant_conformance_verified,
            "claim_boundary": {
                "snap_invariant_definitions_implemented": True,
                "null_space_basis_invariant": True,
                "upstream_invariant_conformance_verified": self.upstream_invariant_conformance_verified,
                "snap_paper_results_reproduced": False,
                "biological_alignment_established": False,
                "mechanistic_equivalence_established": False,
            },
        }


def spectral_alignment_evidence(
    representations: Any,
    targets: Any,
    *,
    centered: bool = True,
    rank_tolerance: float | None = None,
) -> SpectralAlignmentEvidence:
    """Compute backend-stable SNAP-derived spectral alignment evidence.

    ``representations`` is sample x feature. ``targets`` is sample x target and
    may be one-dimensional. Positive representation modes are obtained through
    an SVD, which is equivalent to the positive eigenspectrum of ``X X^T / P``.
    Target power outside the positive-rank column space is aggregated as one
    residual quantity instead of being assigned to arbitrary null eigenvectors.

    No resampling, whitening, normalization, label encoding, filtering, or
    train/test reuse is performed by this function.
    """

    if isinstance(centered, np.bool_):
        centered = bool(centered)
    if not isinstance(centered, bool):
        raise ValueError("centered must be boolean")

    x_original = _matrix(representations, name="representations")
    y_original = _matrix(targets, name="targets")
    if x_original.shape[0] != y_original.shape[0]:
        raise ValueError("representations and targets must contain the same aligned samples")

    x = x_original.copy()
    y = y_original.copy()
    if centered:
        x -= x.mean(axis=0, keepdims=True)
        y -= y.mean(axis=0, keepdims=True)

    n_samples, n_features = x.shape
    target_dim = y.shape[1]
    left_vectors, singular_values, _ = np.linalg.svd(x, full_matrices=False)

    if rank_tolerance is None:
        largest = float(singular_values[0]) if singular_values.size else 0.0
        tolerance = max(x.shape) * np.finfo(np.float64).eps * largest
    else:
        if isinstance(rank_tolerance, bool) or not isinstance(rank_tolerance, (int, float)):
            raise ValueError("rank_tolerance must be numeric or None")
        tolerance = float(rank_tolerance)
        if tolerance < 0 or not math.isfinite(tolerance):
            raise ValueError("rank_tolerance must be finite and non-negative")

    feature_rank = int(np.count_nonzero(singular_values > tolerance))
    if feature_rank == 0:
        raise ValueError("representation matrix has zero positive rank after requested centering")

    u = left_vectors[:, :feature_rank]
    positive_singular = singular_values[:feature_rank]
    positive_eigenvalues = np.square(positive_singular) / float(n_samples)

    # SNAP's scaled sample-kernel eigenvectors are U * sqrt(P), so its target
    # weights are U.T @ Y / sqrt(P). We use exactly that positive-rank quantity.
    weights = (u.T @ y) / np.sqrt(float(n_samples))
    captured_power = np.square(weights)
    total_target_power = np.sum(np.square(y), axis=0) / float(n_samples)
    if np.any(total_target_power <= 1e-15):
        raise ValueError(
            "targets contain a zero-power dimension after requested centering; "
            "spectral task alignment is undefined"
        )

    normalized_captured = captured_power / total_target_power[None, :]
    captured_fraction = normalized_captured.sum(axis=0)
    residual_by_target = np.clip(1.0 - captured_fraction, 0.0, 1.0)
    residual_target_power = float(residual_by_target.mean())

    target_power_by_mode = normalized_captured.mean(axis=1)
    cumulative_captured = np.cumsum(normalized_captured, axis=0).mean(axis=1)

    # Preserve SNAP's "effective dimension of remaining task power" idea, but
    # make the residual one aggregate bin instead of an arbitrary null basis.
    task_tail_dimensions: list[float] = []
    for column in range(target_dim):
        stable_power = np.concatenate(
            [normalized_captured[:, column], residual_by_target[column : column + 1]]
        )
        cumulative = np.cumsum(stable_power)
        tail = np.maximum(1.0 - cumulative, 0.0)
        task_tail_dimensions.append(_participation_ratio(tail))

    payload: dict[str, Any] = {
        "method_id": SPECTRAL_METHOD_ID,
        "reference_method": SNAP_REFERENCE_PAPER,
        "reference_repository": SNAP_REFERENCE_REPOSITORY,
        "centered": centered,
        "rank_tolerance": tolerance,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "target_dim": int(target_dim),
        "feature_rank": feature_rank,
        "positive_eigenvalues": [float(value) for value in positive_eigenvalues],
        "target_power_by_mode": [float(value) for value in target_power_by_mode],
        "cumulative_captured_target_power": [float(value) for value in cumulative_captured],
        "effective_dimension": float(_participation_ratio(positive_eigenvalues)),
        "task_tail_effective_dimension": float(np.mean(task_tail_dimensions)),
        "residual_target_power": residual_target_power,
        "representation_sha256": _array_sha256(x_original),
        "target_sha256": _array_sha256(y_original),
        "upstream_invariant_conformance_verified": False,
    }
    payload["evidence_sha256"] = _canonical_sha256(payload)

    return SpectralAlignmentEvidence(
        method_id=str(payload["method_id"]),
        reference_method=str(payload["reference_method"]),
        reference_repository=str(payload["reference_repository"]),
        centered=bool(payload["centered"]),
        rank_tolerance=float(payload["rank_tolerance"]),
        n_samples=int(payload["n_samples"]),
        n_features=int(payload["n_features"]),
        target_dim=int(payload["target_dim"]),
        feature_rank=int(payload["feature_rank"]),
        positive_eigenvalues=tuple(payload["positive_eigenvalues"]),
        target_power_by_mode=tuple(payload["target_power_by_mode"]),
        cumulative_captured_target_power=tuple(payload["cumulative_captured_target_power"]),
        effective_dimension=float(payload["effective_dimension"]),
        task_tail_effective_dimension=float(payload["task_tail_effective_dimension"]),
        residual_target_power=float(payload["residual_target_power"]),
        representation_sha256=str(payload["representation_sha256"]),
        target_sha256=str(payload["target_sha256"]),
        evidence_sha256=str(payload["evidence_sha256"]),
        upstream_invariant_conformance_verified=False,
    )


def verify_snap_invariant_reference(
    evidence: SpectralAlignmentEvidence,
    expected: dict[str, Any],
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """Verify invariant quantities against a frozen SNAP-reference execution.

    A valid reference fixture should contain only quantities that are invariant
    to null-space basis rotation: positive eigenvalues, positive-rank target
    power, cumulative captured target power, representation effective dimension,
    and aggregate residual target power.
    """

    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    required = (
        "positive_eigenvalues",
        "target_power_by_mode",
        "cumulative_captured_target_power",
        "effective_dimension",
        "residual_target_power",
    )
    missing = [name for name in required if name not in expected]
    if missing:
        raise ValueError(f"SNAP invariant fixture missing fields: {', '.join(missing)}")

    vector_fields = required[:3]
    comparisons: dict[str, bool] = {}
    max_abs_error: dict[str, float] = {}
    for name in vector_fields:
        observed = np.asarray(getattr(evidence, name), dtype=np.float64)
        reference = np.asarray(expected[name], dtype=np.float64)
        if observed.shape != reference.shape:
            comparisons[name] = False
            max_abs_error[name] = float("inf")
            continue
        comparisons[name] = bool(np.allclose(observed, reference, rtol=rtol, atol=atol))
        max_abs_error[name] = float(np.max(np.abs(observed - reference))) if observed.size else 0.0

    for name in required[3:]:
        observed_scalar = float(getattr(evidence, name))
        reference_scalar = float(expected[name])
        comparisons[name] = bool(np.isclose(observed_scalar, reference_scalar, rtol=rtol, atol=atol))
        max_abs_error[name] = abs(observed_scalar - reference_scalar)

    return {
        "method_id": evidence.method_id,
        "conformant": all(comparisons.values()),
        "comparisons": comparisons,
        "max_abs_error": max_abs_error,
        "rtol": float(rtol),
        "atol": float(atol),
        "null_space_basis_invariant": True,
    }
