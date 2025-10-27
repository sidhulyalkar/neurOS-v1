"""FastAPI microservice for computing source mixture weights.

This module defines a FastAPI application exposing an endpoint
``/weigh`` that accepts source and target moment vectors and returns
mixture weights.  It leverages the :class:`SourceWeigher` class
defined in :mod:`neuros_sourceweigher.weigher` to perform the
optimisation.

The service is designed to be stateless and lightweight.  It does not
persist any data and performs computation purely in memory.  The
request schema allows optional regularisation parameters to be passed
in, though the current implementation does not yet support
regularisation beyond the simplex constraints.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .weigher import SourceWeigher


app = FastAPI(title="SourceWeigher Service", version="1.0")


class WeightRequest(BaseModel):
    """Schema for weight estimation requests.

    The request contains a list of lists representing the moment
    vectors for each source domain and a list representing the target
    domain moments.  All inner lists must have the same length.

    Attributes
    ----------
    source_moments : List[List[float]]
        A 2‑D list where each sublist contains the moments for a
        single source domain.  The outer list length equals the number
        of sources (J) and the inner list length equals the number of
        moment features (M).

    target_moments : List[float]
        A 1‑D list of length M containing the target domain moments.

    """

    source_moments: List[List[float]] = Field(..., description="Moment vectors for each source domain")
    target_moments: List[float] = Field(..., description="Moment vector for the target domain")


class WeightResponse(BaseModel):
    """Response schema for weight estimation.

    Attributes
    ----------
    weights : List[float]
        The estimated mixture weights for each source domain.  The
        weights are guaranteed to be non‑negative and sum to one.

    ess : float
        The effective sample size (ESS) of the mixture.  ESS is
        computed as 1 / sum(weights^2) and reflects the number of
        equally weighted samples that would give the same variance as
        the weighted mixture.

    residual : float
        The ℓ2 norm of the difference between the weighted source
        moments and the target moments.  A smaller residual indicates
        a closer match.
    """

    weights: List[float]
    ess: float
    residual: float


@app.post("/weigh", response_model=WeightResponse)
def weigh(req: WeightRequest) -> WeightResponse:
    """Compute mixture weights for a given set of moments.

    This endpoint accepts source moment vectors and a target moment
    vector and returns the optimal mixture weights along with
    diagnostics.  It performs basic validation on the input shapes and
    delegates the optimisation to :class:`SourceWeigher`.

    Raises
    ------
    HTTPException
        If the input lists have inconsistent dimensions or if no
        sources are provided.
    """
    source_array = np.array(req.source_moments, dtype=float)
    target_array = np.array(req.target_moments, dtype=float)
    # Validate dimensions
    if source_array.ndim != 2:
        raise HTTPException(status_code=400, detail="source_moments must be a 2‑D list")
    if target_array.ndim != 1:
        raise HTTPException(status_code=400, detail="target_moments must be a 1‑D list")
    if source_array.shape[0] == 0 or source_array.shape[1] == 0:
        raise HTTPException(status_code=400, detail="source_moments cannot be empty")
    if source_array.shape[1] != target_array.size:
        raise HTTPException(status_code=400, detail="source and target moments dimensions mismatch")
    try:
        weigher = SourceWeigher()
        weights = weigher.estimate_weights(source_array, target_array)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # compute effective sample size
    ess = float(1.0 / np.sum(weights ** 2))
    # compute residual
    residual_vector = source_array.T @ weights - target_array
    residual = float(np.linalg.norm(residual_vector))
    return WeightResponse(weights=weights.tolist(), ess=ess, residual=residual)