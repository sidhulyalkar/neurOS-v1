"""Weight source subjects from frozen neural representations."""
from __future__ import annotations

import numpy as np

from neuros_sourceweigher import RepresentationSourceWeigher

rng = np.random.default_rng(7)

target = rng.normal(loc=0.0, scale=1.0, size=(250, 16))
sources = {
    "subject-near": rng.normal(loc=0.08, scale=1.05, size=(300, 16)),
    "subject-shifted": rng.normal(loc=1.2, scale=1.0, size=(300, 16)),
    "subject-noisy": rng.normal(loc=0.2, scale=2.0, size=(300, 16)),
}

weigher = RepresentationSourceWeigher()
result = weigher.estimate(sources, target)

print("weights:", result.by_source())
print("ESS:", round(result.ess, 3))
print("residual:", round(result.residual, 5))
print("diagnostics:", result.diagnostics.to_dict())
