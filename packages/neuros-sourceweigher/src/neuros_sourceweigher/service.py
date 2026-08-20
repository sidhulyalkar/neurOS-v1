"""Optional FastAPI boundary for remote source-weight estimation."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .weigher import SourceWeigher


class WeightRequest(BaseModel):
    source_moments: List[List[float]] = Field(
        ..., description="One moment/summary vector per source domain"
    )
    target_moments: List[float] = Field(..., description="Target-domain summary vector")
    prior: Optional[List[float]] = None
    quality_scores: Optional[List[float]] = None
    source_ids: Optional[List[str]] = None
    ridge: float = 1e-3
    quality_strength: float = 0.0
    standardize: bool = True
    min_weight: float = 0.0


class WeightResponse(BaseModel):
    weights: List[float]
    source_ids: List[str]
    ess: float
    residual: float
    diagnostics: dict


def create_app() -> FastAPI:
    app = FastAPI(
        title="neurOS SourceWeigher",
        version="0.2.0",
        description="Reliability-aware source selection and domain-mixture estimation.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/weigh", response_model=WeightResponse)
    def weigh(req: WeightRequest) -> WeightResponse:
        try:
            estimator = SourceWeigher(
                ridge=req.ridge,
                quality_strength=req.quality_strength,
                standardize=req.standardize,
                min_weight=req.min_weight,
            )
            result = estimator.estimate(
                np.asarray(req.source_moments, dtype=float),
                np.asarray(req.target_moments, dtype=float),
                prior=None if req.prior is None else np.asarray(req.prior, dtype=float),
                quality_scores=(
                    None
                    if req.quality_scores is None
                    else np.asarray(req.quality_scores, dtype=float)
                ),
                source_ids=req.source_ids,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return WeightResponse(
            weights=result.weights.tolist(),
            source_ids=list(result.source_ids),
            ess=result.ess,
            residual=result.residual,
            diagnostics=result.diagnostics.to_dict(),
        )

    return app


app = create_app()
