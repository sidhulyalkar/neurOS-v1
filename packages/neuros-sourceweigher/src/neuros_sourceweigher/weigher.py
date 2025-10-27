"""Source weight estimator for domain adaptation.

This module defines the :class:`SourceWeigher` class, which takes a
set of moment vectors from source domains and a target moment vector
and computes a non‑negative weight vector summing to one.  The
weights minimise the squared deviation between the weighted source
moments and the target moments.  The optimisation is solved by
computing the unconstrained least–squares solution and projecting it
onto the probability simplex to enforce the constraints.

The projection algorithm is based on the efficient method described by
Wang and Carreira‑Perpiñán (2013), ``Projection onto the probability
simplex: An efficient algorithm with a simple proof, and an
application``.

The implementation avoids external optimisation libraries, making it
suitable for lightweight deployments.
"""

from __future__ import annotations

from typing import Iterable
import numpy as np


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project a vector onto the probability simplex.

    Given a vector ``v`` in ℝ^n, this function returns ``w`` such
    that ``w >= 0`` and ``w.sum() == 1`` and ``w`` is the Euclidean
    projection of ``v`` onto the simplex.

    Parameters
    ----------
    v : np.ndarray
        The input vector to project.  Must be one‑dimensional.

    Returns
    -------
    np.ndarray
        The projected vector lying on the probability simplex.
    """
    if v.ndim != 1:
        raise ValueError("Input to simplex projection must be a 1‑D array")
    n = v.size
    # sort v in descending order
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # find rho where u_i + (1 - sum_{j<=i} u_j) / i > 0
    rho_candidates = (u + (1.0 - cssv) / np.arange(1, n + 1)) > 0
    if not np.any(rho_candidates):
        rho = n - 1
    else:
        rho = np.where(rho_candidates)[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = v - theta
    # clamp at zero to enforce non‑negativity
    w = np.maximum(w, 0.0)
    # normalise in case of numerical error
    sum_w = w.sum()
    if sum_w > 0:
        w /= sum_w
    else:
        # if the projected vector is all zeros (should rarely happen),
        # fall back to uniform weights
        w = np.ones_like(w) / w.size
    return w


class SourceWeigher:
    """Estimate mixture weights between source and target domains.

    The estimator solves a constrained least–squares problem of the
    form::

        min    ||Psi * pi - c||^2
        s.t.   pi >= 0,  sum(pi) = 1,

    where ``Psi`` stacks the moment vectors of each source domain
    column‑wise and ``c`` is the target moment vector.  The solution
    returns a weight vector ``pi`` whose entries sum to one and are
    non‑negative.
    """

    def __init__(self) -> None:
        pass

    def estimate_weights(self, source_moments: np.ndarray, target_moments: np.ndarray) -> np.ndarray:
        """Compute mixture weights for the given source and target moments.

        Parameters
        ----------
        source_moments : np.ndarray
            An array of shape ``(J, M)`` where ``J`` is the number of
            source domains and ``M`` is the number of moment features.
            Each row ``j`` contains the moment vector for source ``j``.

        target_moments : np.ndarray
            A one‑dimensional array of length ``M`` representing the
            moments of the target domain.

        Returns
        -------
        np.ndarray
            A one‑dimensional array of length ``J`` containing
            non‑negative weights that sum to one.

        Raises
        ------
        ValueError
            If the dimensions of ``source_moments`` and
            ``target_moments`` are incompatible.
        """
        if source_moments.ndim != 2:
            raise ValueError("source_moments must be a 2‑D array")
        if target_moments.ndim != 1:
            raise ValueError("target_moments must be a 1‑D array")
        J, M = source_moments.shape
        if target_moments.size != M:
            raise ValueError(
                f"Mismatch: target_moments has length {target_moments.size} but expected {M}"
            )
        if J == 0:
            raise ValueError("No source domains provided")

        # Transpose to shape (M, J) for computation: Psi @ pi approximates target_moments
        Psi = source_moments.T  # shape (M, J)
        c = target_moments      # shape (M,)
        # Solve unconstrained least squares: minimise ||Psi * pi - c||
        # Using numpy.linalg.lstsq to obtain the least squares solution
        try:
            pi_unconstrained, *_ = np.linalg.lstsq(Psi, c, rcond=None)
        except np.linalg.LinAlgError:
            # If the system is singular, fall back to uniform weights
            pi_unconstrained = np.ones(J) / J
        # Ensure 1‑D vector
        pi_unconstrained = np.asarray(pi_unconstrained).flatten()
        # Project onto probability simplex
        pi_projected = _project_to_simplex(pi_unconstrained)
        return pi_projected