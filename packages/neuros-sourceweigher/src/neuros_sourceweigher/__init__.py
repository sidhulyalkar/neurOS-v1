"""Top‑level package for the neurOS SourceWeigher.

This package contains a lightweight microservice and helper class for
estimating mixture weights in domain adaptation scenarios.  It
provides a `SourceWeigher` class that solves a constrained least
squares problem to determine how to combine source domains when
adapting to a new target domain, and a FastAPI app for deploying the
estimator as a microservice.

You can import the estimator or run the service directly:

```
from neuros_sourceweigher import SourceWeigher, app

# compute weights offline
weigher = SourceWeigher()
weights = weigher.estimate_weights(source_moments, target_moments)

# or mount the FastAPI app under your own server
from fastapi import FastAPI
from neuros_sourceweigher.service import app as weigher_app
my_app = FastAPI()
my_app.mount("/weigher", weigher_app)
```
"""

from .weigher import SourceWeigher  # noqa: F401
from .service import app  # noqa: F401

__all__ = ["SourceWeigher", "app"]