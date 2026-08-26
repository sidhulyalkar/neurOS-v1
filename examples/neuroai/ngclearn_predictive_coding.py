"""Minimal evidence-producing ngc-learn predictive-coding example.

Run after installing the optional integration:

    pip install "neuros-foundation[ngclearn]"
    python examples/neuroai/ngclearn_predictive_coding.py

The fixture is intentionally tiny and deterministic. It demonstrates the
qualified software/integration surface; it is not a real-neural-data benchmark.
"""

from __future__ import annotations

import json

import numpy as np

from neuros.foundation_models import NgcLearnPredictiveCodingTransform


def main() -> None:
    samples = np.asarray(
        [
            [0.50, -0.25],
            [1.00, 0.50],
            [-0.75, 0.25],
            [0.20, 0.80],
        ],
        dtype=np.float64,
    )

    # Identity weights make the expected reconstruction behavior transparent:
    # the latent has enough capacity to represent both observed channels, while
    # the circuit still has to reach that representation through iterative
    # ngc-learn residual-feedback dynamics.
    transform = NgcLearnPredictiveCodingTransform(
        latent_dim=2,
        settling_steps=80,
        settling_dt_ms=1.0,
        tau_m_ms=20.0,
        prior_gamma=0.0,
        activation="identity",
        integration_type="euler",
        output="linear",
        weights=np.eye(2, dtype=np.float64),
        seed=11,
    )
    result = transform.transform(samples, sample_rate_hz=250.0)

    payload = {
        "latent_shape": list(result.values.shape),
        "reconstruction_shape": list(result.reconstruction.shape),
        "mean_squared_error_by_step": result.mean_squared_error_by_step.tolist(),
        "evidence": result.evidence.to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
