# Neuros SourceWeigher Microservice

This package provides a simple microservice for estimating **mixture weights**
for domain adaptation in neurOS.  The goal of the service is to choose
continuous weights over a set of source subjects so that the weighted
combination of their *moment vectors* best matches the target subject's
moment vector.  These weights can then be used to bias training towards
source datasets that are most similar to the target, reducing negative
transfer when fine‑tuning models on new subjects.

The service exposes a FastAPI application with a single endpoint:

* `/weigh` — accepts a JSON payload containing a matrix of source moments
  (one row per source, one column per moment) and a target moment vector.
  It returns a vector of non‑negative weights summing to one, the
  resulting effective sample size and the residual between the weighted
  source moments and the target moments.

Internally, weights are computed by solving a constrained least squares
problem:

$$
\min_{\pi}\; \lVert \Psi \pi - c \rVert_2^2 \quad\text{such that}\quad
\pi \ge 0,\;\sum_j \pi_j = 1,
$$

where \(\Psi \in \mathbb{R}^{M\times J}\) stacks the moment vectors of all
sources column‑wise and \(c \in \mathbb{R}^M\) is the target moment vector.
The implementation first computes the unconstrained least–squares
solution and then projects it onto the probability simplex to satisfy
the non‑negativity and sum–to–one constraints.  This closed‑form
approach avoids heavy optimisation dependencies such as CVXPy.

The microservice is designed to be lightweight and stateless.  It does
not persist any information or depend on a database.  If persistent
storage of moment vectors or weight estimates is required, it is
recommended to implement that logic in the calling code (e.g. in your
training loop or orchestration layer).

## Usage

To run the service locally for development:

```bash
pip install fastapi uvicorn numpy
uvicorn neuros_sourceweigher.service:app --reload
```

Send a POST request to `/weigh` with JSON data of the form:

```json
{
  "source_moments": [[0.9, 0.1], [0.2, 0.8], [0.4, 0.6]],
  "target_moments": [0.5, 0.5]
}
```

The response will include a `weights` field containing the mixture
weights, an `ess` field (the effective sample size of the resulting
weighted distribution) and a `residual` field describing how closely
the weighted combination matches the target moments.

## Why not DataJoint?

The mixture weight computation performed by this service is a purely
numerical optimisation that does not benefit from a relational data
pipeline.  While neurOS supports DataJoint for experiments that
produce large volumes of trial data and metrics, this service only
needs access to small in‑memory arrays.  Using a full‑featured ETL
framework such as DataJoint would add unnecessary complexity and
dependencies.  Instead, the calling code (e.g. `neurofmxx_trainer.py`)
is responsible for extracting moments from its own data loaders and
passing them to the microservice.  Should you decide to persist
weights or moments for analysis, a simple file‑based or in‑memory
solution is sufficient and demonstrated in `packages/neuros-neurofm/src/neuros_neurofm/etl.py`.