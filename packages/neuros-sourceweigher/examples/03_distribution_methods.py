"""Compare distribution-level source weighting strategies."""
from __future__ import annotations

import numpy as np

from neuros_sourceweigher import MMDSourceWeigher, RiemannianCovarianceWeigher

rng = np.random.default_rng(11)
target = rng.normal(size=(300, 4))

sources = {
    "near": rng.normal(loc=0.05, size=(300, 4)),
    "mean-shifted": rng.normal(loc=2.0, size=(300, 4)),
    "cov-shifted": rng.normal(size=(300, 4)) @ np.diag([3.0, 0.4, 1.0, 1.0]),
}

mmd = MMDSourceWeigher(temperature=0.03, max_samples=256).estimate(sources, target)
riemann = RiemannianCovarianceWeigher(temperature=0.6).estimate(sources, target)

print("MMD weights:", mmd.by_source())
print("MMD discrepancies:", mmd.diagnostics.source_distances)
print("Riemannian weights:", riemann.by_source())
print("Riemannian distances:", riemann.diagnostics.source_distances)
