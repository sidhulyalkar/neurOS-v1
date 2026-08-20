"""Local-first client for SourceWeigher."""
from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from .weigher import SourceWeigher, WeightingDiagnostics, WeightingResult


class SourceWeightClient:
    """Use a local estimator or a remote ``/weigh`` endpoint."""

    def __init__(
        self,
        *,
        estimator: Any | None = None,
        url: str | None = None,
        timeout: float = 10.0,
        fallback: str = "raise",
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if fallback not in {"raise", "uniform"}:
            raise ValueError("fallback must be 'raise' or 'uniform'")
        self.estimator = estimator or SourceWeigher()
        self.url = url
        self.timeout = float(timeout)
        self.fallback = fallback

    def estimate(
        self,
        source_moments: np.ndarray,
        target_moments: np.ndarray,
        **kwargs: Any,
    ) -> WeightingResult:
        source = np.asarray(source_moments, dtype=float)
        target = np.asarray(target_moments, dtype=float)
        if self.url is None:
            return self.estimator.estimate(source, target, **kwargs)

        payload = {
            "source_moments": source.tolist(),
            "target_moments": target.tolist(),
        }
        for key in ("prior", "quality_scores", "source_ids"):
            if key in kwargs and kwargs[key] is not None:
                value = kwargs[key]
                payload[key] = value.tolist() if isinstance(value, np.ndarray) else list(value)
        request = Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            weights = np.asarray(data["weights"], dtype=float)
            diag_data = data.get("diagnostics", {})
            diagnostics = WeightingDiagnostics(
                method=str(diag_data.get("method", "remote")),
                residual=float(diag_data.get("residual", data.get("residual", np.nan))),
                scaled_residual=float(diag_data.get("scaled_residual", np.nan)),
                objective=float(diag_data.get("objective", np.nan)),
                effective_sample_size=float(
                    diag_data.get(
                        "effective_sample_size",
                        data.get("ess", 1.0 / np.sum(weights**2)),
                    )
                ),
                entropy=float(diag_data.get("entropy", 0.0)),
                max_weight=float(diag_data.get("max_weight", weights.max())),
                iterations=int(diag_data.get("iterations", 0)),
                converged=bool(diag_data.get("converged", True)),
                condition_number=float(diag_data.get("condition_number", np.nan)),
                source_distances=tuple(
                    float(x) for x in diag_data.get("source_distances", [])
                ),
                excluded_sources=tuple(
                    int(x) for x in diag_data.get("excluded_sources", [])
                ),
                metadata=diag_data.get("metadata", {}),
            )
            return WeightingResult(
                weights,
                diagnostics,
                tuple(data.get("source_ids", kwargs.get("source_ids", ()))),
            )
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
            if self.fallback == "raise":
                raise RuntimeError(f"SourceWeigher request failed: {exc}") from exc
            n = source.shape[0]
            weights = np.full(n, 1.0 / n)
            diagnostics = WeightingDiagnostics(
                method="uniform_fallback",
                residual=float(np.linalg.norm(source.T @ weights - target)),
                scaled_residual=float("nan"),
                objective=float("nan"),
                effective_sample_size=float(n),
                entropy=float(np.log(n)),
                max_weight=float(1.0 / n),
                iterations=0,
                converged=False,
                condition_number=float("nan"),
                source_distances=tuple(),
                metadata={"error": str(exc)},
            )
            return WeightingResult(weights, diagnostics)
