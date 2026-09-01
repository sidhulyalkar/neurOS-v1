# Neural Representation Benchmark Authority

This document defines the neurOS research contract for comparing temporal neural representations across PCA, a train-only autoencoder, upstream T-PHATE, and fixed external temporal self-supervised representations.

The benchmark is intentionally not a leaderboard. These methods operate under different information regimes, and collapsing them into one scalar winner would hide the most important scientific distinction: **what data each representation was allowed to observe while it was constructed**.

## Why T-PHATE is interesting

Temporal PHATE (T-PHATE) was introduced by Busch et al. in *Nature Computational Science* (2023), “Multi-view manifold learning of human brain-state trajectories” (DOI: 10.1038/s43588-023-00419-0).

The upstream implementation is maintained at:

- https://github.com/KrishnaswamyLab/TPHATE

At a high level, T-PHATE constructs two views of an ordered trajectory:

1. a PHATE-style geometry diffusion operator derived from similarity in observed feature space;
2. a temporal transition operator derived from the mean autocorrelation function across features.

The method combines those views, diffuses the resulting transition operator, converts the diffusion probabilities to an information-potential representation, and embeds that potential using MDS.

The temporal view is the key distinction from ordinary PHATE. Nearby timepoints can reinforce one another even when observation noise makes them less similar in the raw feature space.

## neurOS integration boundary

neurOS does **not** vendor, copy, or reimplement T-PHATE.

At integration time, the upstream repository's `LICENSE.md` contains a Yale Non-Commercial License, while upstream package metadata still advertises a GPL classifier. neurOS itself is MIT licensed. Because those surfaces do not present a clean permissive dependency boundary, neurOS treats T-PHATE as an optional external package:

- `tphate` is not a default neurOS dependency;
- normal neurOS CI does not install it;
- the adapter lazily imports a separately installed upstream package;
- missing T-PHATE is represented as an explicit `unavailable` method outcome;
- users are responsible for reviewing the current upstream terms for their intended use.

This is an engineering boundary, not legal advice or a license grant.

## Sequence authority

T-PHATE assumes rows form one ordered temporal sequence. Its temporal kernel uses sample-index lag as temporal distance.

Therefore neurOS never creates a T-PHATE input by concatenating independent subjects, sessions, runs, or trials into one artificial timeline.

`SequenceBatch` preserves trajectories as separate arrays with explicit unique sequence IDs. The T-PHATE adapter creates a **fresh upstream estimator for each evaluation trajectory**.

This is stricter than the older exploratory `ManifoldAnalyzer`, whose generic 3-D path can flatten multiple trajectories for order-independent geometry analyses. That flattening must not be used as T-PHATE temporal authority.

## Fit regimes

Every method exposes a `FitRegime`.

| Method | neurOS fit regime | What observes evaluation structure? | Coordinate frame |
| --- | --- | --- | --- |
| PCA | `train_only_inductive` | only the fixed train-fitted transform is applied | shared train-fitted axes |
| Autoencoder | `train_only_inductive` | only the fixed train-fitted encoder is applied | shared train-fitted encoder |
| T-PHATE | `transductive_target_observed` | the unlabeled evaluation trajectory is itself fit | independently fit MDS frame per trajectory |
| Temporal SSL | `external_pretrained` | depends on the external model and its pretraining/adaptation history | shared external encoder when the provider is fixed |

A T-PHATE result must not be described as train-only zero-shot transfer. Its target trajectory participates in representation construction even when no target labels are used.

Likewise, an external temporal SSL embedding is not automatically clean zero-shot evidence. neurOS records a model ID, model version, known pretraining datasets, and a pretraining-lineage audit status. Dataset overlap remains a scientific-authority question outside the geometry metric itself.

## T-PHATE parameter evidence

The adapter forwards the maintained upstream estimator parameters and disables landmarking so temporal continuity is not routed through a landmark approximation.

`n_pca` receives extra evidence because upstream defaults can exceed a short trajectory's dimensions. neurOS records:

- `requested_n_pca`;
- `effective_n_pca_by_sequence`;
- `n_pca_policy`.

When requested `n_pca` is not strictly smaller than both dimensions of a trajectory, the adapter disables upstream PCA for that trajectory and records the effective value as `None`. This prevents a necessary compatibility adjustment from becoming invisible preprocessing drift.

## Upstream failure boundary

The current upstream temporal-kernel path finds the first negative crossing of a smoothed mean autocorrelation function. A trajectory whose smoothed ACF never crosses zero can therefore fail during the upstream dropoff calculation.

neurOS does not invent a replacement dropoff or silently switch to a different temporal kernel. The adapter translates that failure into a method-scoped `TPHATEEmbeddingError` explaining that no negative crossing was available.

More generally, a single unavailable or failed representation method does not erase the other benchmark results. Every requested method receives one explicit outcome:

- `ok`;
- `failed`;
- `unavailable`.

## Real-valued representation contract

The common comparison surface accepts finite **real-valued** arrays only.

Complex inputs are rejected before method dispatch. This matters because otherwise NumPy linear algebra could preserve complex values while a torch autoencoder or real-valued geometry metric silently projected them to a different space.

Caller-owned arrays are copied and made read-only. Nested metadata is detached recursively, and metadata keys must already be explicit nonblank strings rather than being coerced.

## Geometry metrics

The initial common metrics are deliberately coordinate-frame agnostic.

### Local neighborhood preservation

For each trajectory, compare the source-space `k` nearest neighbors of each timepoint with its latent-space neighbors. The metric is the mean retained-neighbor fraction.

### Pairwise-distance rank preservation

Compute Spearman correlation between pairwise distances in source and latent space. The metric uses a deterministic temporal subsample for very long trajectories.

### Temporal continuity ratio

Compare median adjacent-step distance in latent space with median distance between non-adjacent timepoints.

Lower values mean adjacent states are relatively close, but this is **not** a universal optimization objective. A representation can score extremely smoothly by over-smoothing away meaningful state transitions.

### Known-reference geometry

Controlled simulations may supply a separate `reference=SequenceBatch` containing the known clean latent trajectory. Reference IDs and timepoint counts must exactly match the evaluation batch.

The benchmark then adds:

- `reference_local_knn_preservation`;
- `reference_pairwise_distance_rank`.

These metrics are inspired by the same scientific question as controlled-manifold fidelity analyses such as DeMAP: does an embedding recover known latent geometry despite noisy observations? neurOS does not claim these two metrics are a reimplementation of DeMAP.

## Why metrics are trajectory-local

An independently fitted MDS embedding can rotate, reflect, or translate without changing its geometry. T-PHATE embeddings fit separately to different trajectories therefore do not share meaningful raw axes by default.

The initial benchmark computes distance/neighborhood metrics inside each trajectory and aggregates the scalar metrics afterward. It does **not** concatenate latent coordinates across independently fitted trajectories.

Cross-subject coordinate comparison requires a separate alignment protocol with explicit anchors or correspondence assumptions.

## PCA baseline

`PCARepresentation` is a dependency-light NumPy/SVD baseline.

It concatenates training trajectories only for the order-independent PCA fit, records the train mean/components, and applies the resulting fixed transform separately to each evaluation trajectory.

Changing evaluation observations cannot change the fitted PCA state.

## Autoencoder baseline

`AutoencoderRepresentation` is a small deterministic torch MLP autoencoder.

It is intentionally a baseline rather than a claim that one MLP architecture represents the entire autoencoder literature. The model:

- standardizes using training observations only;
- uses a seeded CPU training path;
- trains only on the declared training sequences;
- applies the fixed encoder to evaluation trajectories;
- records optimizer/training configuration and final training loss.

Reconstruction loss is not used as a universal cross-method score because T-PHATE and arbitrary SSL embeddings do not necessarily define inverse decoders.

## Temporal SSL / foundation representations

`PrecomputedTemporalSSLRepresentation` binds fixed external embeddings to exact evaluation sequence IDs and timepoint counts.

This is intentionally a consumption adapter, not a training framework. A representation generated by EEGPT, LaBraM, REVE, CSBrain, LUNA, NeuroFM-X, or another temporal model should be extracted under that model's own qualified pipeline, then supplied with explicit provenance.

Required identity includes:

- `model_id`;
- `model_version`;
- exact per-sequence embedding arrays.

Pretraining metadata includes known pretraining dataset IDs and one explicit lineage status:

- `disjoint_verified`;
- `overlap_detected`;
- `possible_overlap`;
- `unknown_lineage`;
- `not_audited`.

The benchmark must not reinterpret `unknown_lineage` as clean transfer evidence.

## Controlled example

A deterministic synthetic benchmark is available at:

```bash
python packages/neuros-mechint/examples/11_representation_benchmark.py
```

It creates a clean temporal latent trajectory, maps it nonlinearly into a higher-dimensional observation space, adds Gaussian noise, and compares the observed embeddings with both observed-space and known-reference geometry.

Without a separately installed T-PHATE package, the output will contain an explicit `tphate: unavailable` record while PCA and the autoencoder continue to run.

An external temporal SSL embedding can be supplied as an `.npz` file containing an `eval` array with one row per evaluation timepoint. The model/version and pretraining-lineage fields should identify the actual external representation rather than a synthetic stand-in.

## Noise-sweep experiment design

The highest-value first scientific experiment is a controlled noise sweep rather than a single visualization.

For several fixed noise levels and random seeds:

1. generate train and evaluation observations from a known smooth latent trajectory;
2. fit PCA and the autoencoder on train observations only;
3. fit T-PHATE transductively on each evaluation trajectory;
4. consume a fixed temporal SSL representation only when a genuine model embedding is available;
5. report both observed-space and known-reference geometry metrics;
6. retain fit-regime and lineage fields next to every metric;
7. plot metric distributions/curves with uncertainty across seeds, not one cherry-picked embedding.

This directly tests the hypothesis that T-PHATE's temporal prior is especially useful when observation noise damages raw geometric neighborhood structure.

## No scalar winner

`RepresentationBenchmarkResult` intentionally has no `winner` field.

Different metrics answer different questions, and the methods use different information regimes. A useful result might show, for example:

- T-PHATE recovers reference geometry best under high noise but is transductive;
- PCA transfers most cleanly across held-out trajectories but misses nonlinear geometry;
- the autoencoder improves nonlinear compression after sufficient train data;
- a pretrained temporal SSL model transfers well, but its pretraining lineage is only partially known.

Those statements should remain visible rather than being collapsed into a single number.

## Qualification boundary

Normal neurOS qualification validates:

- sequence identity and immutability;
- PCA and autoencoder train-only fitting;
- failure-preserving orchestration;
- rigid-transform-invariant metrics;
- known-reference identity;
- T-PHATE adapter parameter forwarding through a fake upstream seam;
- missing-upstream behavior;
- upstream-error translation;
- temporal SSL identity and lineage metadata;
- import/build behavior without T-PHATE installed.

Normal CI does **not** install the upstream non-commercial T-PHATE package. Therefore a green neurOS workflow proves adapter contracts, not compatibility with every current/future upstream T-PHATE release.

A separately authorized upstream-compatibility study should name the exact upstream T-PHATE version/commit and environment it exercised.

## Claim boundaries

This benchmark establishes representation-analysis plumbing and evidence discipline. It does **not** establish:

- that one representation method is universally superior;
- a decoder-performance advantage;
- causal neural mechanisms;
- biological interpretation of embedding axes;
- clinical validity or safety;
- cross-subject alignment from unaligned T-PHATE coordinates;
- zero-shot status for transductive T-PHATE;
- clean transfer for a pretrained model with unknown/overlapping pretraining lineage;
- commercial permission to use upstream T-PHATE.

## Related neurOS surfaces

- `docs/SCIENTIFIC_AUTHORITY_V2.md` defines dataset/model lineage and evidence-domain rules.
- `neuros_mechint.dynamics.ManifoldAnalyzer` remains a generic exploratory manifold tool; it is not the T-PHATE sequence-authority boundary.
- `neuros.foundation_models` catalogs temporal/foundation representation methods and their integration/access status.
- issue #141 owns this representation benchmark tranche.
