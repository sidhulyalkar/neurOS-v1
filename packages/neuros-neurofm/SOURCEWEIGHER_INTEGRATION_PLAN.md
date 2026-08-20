# SourceWeigher integration status

> **Superseded by `neuros-sourceweigher` v0.2.**

The original integration plan described SourceWeigher v0.1 as production-ready and recommended accepting the package largely as-is. That assessment is no longer accurate.

## What changed

The v0.1 implementation solved unconstrained least squares and projected that solution onto the probability simplex once. In general, that is not equivalent to solving the stated simplex-constrained least-squares problem. It also made FastAPI a mandatory dependency, treated a network service as the primary training interface, silently fell back to uniform weights on service failures in NeuroFM trainers, and had no package-level regression suite or CI gate.

SourceWeigher v0.2 therefore reframes the package as the neurOS **source reliability and transfer-risk layer**. The numerical core is local-first and NumPy-only; the default estimator uses iterative simplex-constrained optimization; the package includes distribution-level and online strategies, diagnostics, a foundation-representation bridge, and a runtime fusion operator. FastAPI remains an optional deployment boundary.

## NeuroFM integration direction

Do not make `NeuroFMXXTrainer` or `NeuroFMXXXTrainer` depend on an HTTP URL by default. A follow-up trainer refactor should inject a `SourceWeightClient` or estimator object and record the resulting `WeightingResult` with the experiment manifest.

For foundation-model transfer, prefer weighting source subjects/sessions from held-out target calibration representations through `RepresentationSourceWeigher`, MMD, or a predeclared task-risk estimator. The current `[loss, pseudo-accuracy]` examples are pedagogical and should not be treated as a scientifically validated universal transfer metric.

For class-conditional weighting, avoid materializing entire datasets into Python lists. Compute class/domain summaries incrementally or through existing data-loader abstractions and keep class-specific calibration separate from the final test set.

## Package boundary

`neuros-sourceweigher` owns source-selection policies, transfer-risk routing, reliability-weighted fusion operators, and source-mixture diagnostics. It does not own model architectures, preprocessing, hardware acquisition, mechanistic interpretation, or generic runtime scheduling.

See `packages/neuros-sourceweigher/README.md` and its examples for the canonical API and roadmap.
