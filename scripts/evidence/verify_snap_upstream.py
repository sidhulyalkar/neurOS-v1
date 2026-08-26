#!/usr/bin/env python3
"""Verify neurOS invariant spectral evidence against pinned upstream SNAP code.

This script deliberately loads ``snap/metrics.py`` directly from an exact
checkout instead of importing the SNAP package. The upstream package __init__
imports a broad paper-reproduction environment that is irrelevant to the
spectral reference functions being checked here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neuros.foundation_models.spectral_alignment import spectral_alignment_evidence

SNAP_REFERENCE_REVISION = "76570574eab7b3115ed4503f8c4ecefaa2a7c5e6"

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


def _load_metrics(root: Path) -> Any:
    metrics_path = root / "snap" / "metrics.py"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"SNAP metrics.py not found at {metrics_path}")
    spec = importlib.util.spec_from_file_location("neuros_snap_reference_metrics", metrics_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not create import spec for {metrics_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _upstream_invariants(metrics: Any) -> dict[str, Any]:
    x = torch.as_tensor(REPRESENTATIONS, dtype=torch.float64)
    y = torch.as_tensor(TARGETS, dtype=torch.float64)
    eig, weights, residuals, _, p, n, _ = metrics.kernel_spectrum_from_feat(
        x,
        {"target": y},
        cent=True,
    )
    rank = min(p, n)
    weight = np.asarray(weights["target"], dtype=np.float64)
    power = np.square(weight)
    total = power.sum(axis=0)
    if np.any(total <= 0):
        raise RuntimeError("upstream SNAP fixture produced zero target power")
    normalized = power / total[None, :]
    return {
        "positive_eigenvalues": np.asarray(eig[:rank], dtype=np.float64),
        "target_power_by_mode": normalized[:rank].mean(axis=1),
        "cumulative_captured_target_power": np.cumsum(normalized[:rank], axis=0).mean(axis=1),
        "effective_dimension": float(metrics.eff_dimension(np.asarray(eig[:rank], dtype=np.float64))),
        "residual_target_power": float(np.mean(np.asarray(residuals["target"], dtype=np.float64))),
    }


def verify(root: Path, *, rtol: float = 1e-7, atol: float = 1e-9) -> dict[str, Any]:
    metrics = _load_metrics(root)
    upstream = _upstream_invariants(metrics)
    ours = spectral_alignment_evidence(REPRESENTATIONS, TARGETS, centered=True)

    comparisons = {
        "positive_eigenvalues": bool(
            np.allclose(ours.positive_eigenvalues, upstream["positive_eigenvalues"], rtol=rtol, atol=atol)
        ),
        "target_power_by_mode": bool(
            np.allclose(ours.target_power_by_mode, upstream["target_power_by_mode"], rtol=rtol, atol=atol)
        ),
        "cumulative_captured_target_power": bool(
            np.allclose(
                ours.cumulative_captured_target_power,
                upstream["cumulative_captured_target_power"],
                rtol=rtol,
                atol=atol,
            )
        ),
        "effective_dimension": bool(
            np.isclose(ours.effective_dimension, upstream["effective_dimension"], rtol=rtol, atol=atol)
        ),
        "residual_target_power": bool(
            np.isclose(ours.residual_target_power, upstream["residual_target_power"], rtol=rtol, atol=atol)
        ),
    }
    payload = {
        "reference_repository": "https://github.com/chung-neuroai-lab/SNAP",
        "reference_revision": SNAP_REFERENCE_REVISION,
        "reference_file": "snap/metrics.py",
        "neurOS_method_id": ours.method_id,
        "null_space_basis_invariant": True,
        "comparisons": comparisons,
        "conformant": all(comparisons.values()),
        "rtol": rtol,
        "atol": atol,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snap-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = verify(args.snap_root)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if result["conformant"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
