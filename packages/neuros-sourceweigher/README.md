# neurOS SourceWeigher

**Reliability-aware source selection, transfer-risk estimation, and adaptive fusion for neural systems.**

`neuros-sourceweigher` answers a question that appears everywhere in practical neuroscience ML:

> Given several subjects, sessions, sites, devices, models, or sensor streams, which sources should the current target trust, and by how much?

The original package solved a small moment-matching problem behind a FastAPI endpoint. v0.2 turns that idea into a local-first reliability engine for neurOS. The package now supports source-domain weighting for transfer learning, foundation-model embedding transfer, distribution-level similarity, online drift adaptation, and runtime sensor fusion while keeping the numerical core dependency-light.

## Why keep this as a package?

Source weighting is a distinct concern from both model architecture and data acquisition:

```text
neuros-drivers            acquire streams
      |
      v
neuros-core               runtime / synchronization / graph execution
      |
      +--------------------------+
      |                          |
      v                          v
neuros-foundation        neuros-sourceweigher
representation space     source trust / transfer risk
      |                          |
      +-------------+------------+
                    v
              neuros-models
              task decoder
                    |
                    v
                  ORION
```

`neuros-foundation` asks what a representation contains. SourceWeigher asks which source representation, cohort, session, or stream is most relevant and reliable for the target. That distinction is useful enough to preserve as an independent package.

## What changed in v0.2

The package is no longer HTTP-first and no longer treats a one-shot projection of an unconstrained least-squares solution as the constrained optimum.

The default `SourceWeigher` now solves the actual convex simplex-constrained objective with projected gradient descent:

```text
min_w  1/2 ||A w - b||^2
     + ridge/2 ||w - prior||^2
     - quality_strength * quality^T w

subject to
     w_j >= min_weight
     sum_j w_j = 1
```

The result includes diagnostics, not just a vector:

- residual and standardized residual;
- objective value;
- effective source count (ESS);
- entropy and maximum source concentration;
- condition number;
- source-to-target distances;
- excluded/non-finite sources;
- convergence metadata.

The base install is NumPy-only. FastAPI is now an optional deployment boundary.

## Installation

```bash
pip install neuros-sourceweigher
```

For the optional service:

```bash
pip install "neuros-sourceweigher[service]"
uvicorn neuros_sourceweigher.service:app --host 0.0.0.0 --port 8000
```

## 1. Moment or summary matching

```python
import numpy as np
from neuros_sourceweigher import SourceWeigher

sources = np.array([
    [0.10, 0.20, 0.95],
    [0.80, 0.70, 0.40],
    [0.25, 0.30, 0.85],
])
target = np.array([0.18, 0.27, 0.88])

result = SourceWeigher(ridge=1e-2).estimate(
    sources,
    target,
    source_ids=["subject-A", "subject-B", "subject-C"],
)

print(result.by_source())
print(result.diagnostics.to_dict())
```

Use this when each source can be represented by the same scientifically meaningful summary vector. Examples include representation means/spreads, calibration metrics, spectral summaries, task-conditioned losses, or other predeclared transfer features.

### Reliability priors and quality evidence

Source similarity and source quality are not the same quantity. v0.2 keeps them separate:

```python
result = SourceWeigher(
    ridge=0.05,
    quality_strength=0.15,
).estimate(
    sources,
    target,
    prior=np.array([0.5, 0.25, 0.25]),
    quality_scores=np.array([0.95, 0.30, 0.80]),
)
```

A prior can encode historical confidence or cohort knowledge. `quality_scores` can encode independently measured stream/device quality. Do not use downstream test performance as a quality score.

## 2. Foundation-model representation transfer

The most useful neurOS integration is to weight source subjects or sessions directly in a foundation-model embedding space.

```python
from neuros_sourceweigher import RepresentationSourceWeigher

weigher = RepresentationSourceWeigher()
result = weigher.estimate(
    source_embeddings={
        "subject-01": z_subject_01,
        "subject-02": z_subject_02,
        "subject-03": z_subject_03,
    },
    target_embeddings=z_target_calibration,
)

print(result.by_source())
```

By default each domain is summarized by feature means and log standard deviations. You can also use robust summaries such as median and IQR.

This is a natural companion to `neuros-foundation`: model-specific preprocessing and representation extraction stay upstream, while SourceWeigher operates on comparable embedding matrices.

## 3. Distribution-level weighting with MMD

First/second-moment summaries can miss multimodal or nonlinear shifts. `MMDSourceWeigher` compares entire source and target feature distributions with an RBF-kernel maximum mean discrepancy.

```python
from neuros_sourceweigher import MMDSourceWeigher

result = MMDSourceWeigher(temperature=0.05).estimate(
    {
        "site-a": embeddings_a,
        "site-b": embeddings_b,
        "site-c": embeddings_c,
    },
    target_embeddings,
)
```

This is useful when the representation dimension is moderate and source/target distributions have similar semantics but different shapes.

## 4. Riemannian covariance weighting

EEG and MEG pipelines often carry useful domain information in covariance structure. `RiemannianCovarianceWeigher` compares regularized covariance matrices with the affine-invariant SPD distance:

```python
from neuros_sourceweigher import RiemannianCovarianceWeigher

result = RiemannianCovarianceWeigher().estimate(
    {
        "session-1": eeg_features_1,
        "session-2": eeg_features_2,
        "session-3": eeg_features_3,
    },
    target_eeg_features,
)
```

Covariance geometry is complementary to mean/distribution matching. It is not a universal similarity metric, especially when mean shift or task-label shift dominates.

## 5. Weight from measured transfer risk

When you have a clean calibration split and a comparable per-source risk estimate, `GibbsRiskWeigher` provides a simple entropy-regularized selector:

```python
from neuros_sourceweigher import GibbsRiskWeigher

result = GibbsRiskWeigher(temperature=0.1).estimate(
    risks=np.array([0.18, 0.51, 0.23]),
    source_ids=["decoder-a", "decoder-b", "decoder-c"],
)
```

Lower-risk sources receive more mass. This is useful for decoder ensembles, checkpoint selection, or source-specific transfer heads. The calibration data used to estimate risk must remain separate from the final evaluation set.

## 6. Online adaptation under drift

BCI source reliability changes over time. Electrodes move, impedance changes, attention changes, and devices drop channels. `OnlineSourceWeigher` smooths any estimator and bounds abrupt mixture changes:

```python
from neuros_sourceweigher import DistanceWeigher, OnlineSourceWeigher

online = OnlineSourceWeigher(
    DistanceWeigher(temperature=0.2),
    adaptation_rate=0.2,
    max_l1_step=0.25,
)

for target_summary in streaming_target_summaries:
    result = online.update(source_summaries, target_summary)
    deploy_weights(result.weights)
```

The instantaneous mixture is preserved in diagnostics so adaptation can be audited rather than becoming a hidden feedback loop.

## 7. Streaming feature summaries

For large or continuous recordings, source moments do not need to be accumulated in memory:

```python
from neuros_sourceweigher import RunningFeatureSummary

summary = RunningFeatureSummary(n_features=64)
for batch in embedding_batches:
    summary.update(batch)

target_vector = summary.vector(log_std=True)
```

The implementation uses pooled Welford updates for numerically stable online mean and variance estimation.

## 8. Reliability-aware fusion inside the neurOS runtime

`neuros-core` fusion nodes call an operator's `fuse(latest)` method when one is supplied. SourceWeigher provides a compatible operator without making `neuros-core` depend on this package:

```python
from neuros.runtime import NodeKind, RuntimeNode
from neuros_sourceweigher import ReliabilityWeightedFusion

fusion = ReliabilityWeightedFusion(
    {
        "transform:eeg": 0.65,
        "transform:emg": 0.25,
        "transform:imu": 0.10,
    },
    mode="scale_concat",
)

node = RuntimeNode("fusion:reliability", NodeKind.FUSION, fusion)
```

`scale_concat` is the safe default for heterogeneous modalities because it preserves each modality's coordinates while scaling its contribution. `weighted_mean` is only valid when all incoming arrays represent the same coordinate space and have identical shape.

This operator can be updated online with `fusion.set_weights(...)` when an external quality/reliability monitor produces new evidence.

## 9. Stability and influence diagnostics

A source mixture should be interrogated, not blindly trusted.

```python
from neuros_sourceweigher import (
    leave_one_source_out_stability,
    target_perturbation_sensitivity,
)

loo = leave_one_source_out_stability(weigher, source_summaries, target_summary)
noise = target_perturbation_sensitivity(
    weigher,
    source_summaries,
    target_summary,
    noise_scale=0.02,
)
```

These reports identify mixtures that depend precariously on one source or change dramatically under small target-summary perturbations.

## Local first, service second

For normal neurOS training, prefer an in-process estimator:

```python
from neuros_sourceweigher import SourceWeightClient

client = SourceWeightClient()  # local, no network
result = client.estimate(source_summaries, target_summary)
```

Use the service only when process/language isolation, remote orchestration, or centralized policy genuinely requires it:

```python
client = SourceWeightClient(
    url="http://sourceweigher:8000/weigh",
    fallback="raise",
)
```

The historical behavior of silently returning uniform weights after any network failure is intentionally not the default. Uniform fallback is still available, but it must be requested explicitly with `fallback="uniform"` and its failure reason is written into diagnostics.

## Which algorithm should I use?

| Situation | Recommended starting point | Main assumption |
|---|---|---|
| Comparable compact summaries | `SourceWeigher` | target lies near a convex source mixture |
| Need transparent nearest-domain baseline | `DistanceWeigher` | Euclidean summary distance is meaningful |
| Reliable calibration risk exists | `GibbsRiskWeigher` | risk estimates transfer to deployment |
| Nonlinear distribution shift | `MMDSourceWeigher` | kernel discrepancy captures relevant shift |
| EEG/MEG covariance structure matters | `RiemannianCovarianceWeigher` | covariance geometry reflects transferability |
| Reliability changes over time | `OnlineSourceWeigher` | gradual updates are safer than abrupt routing |

No weighting method should be assumed to improve transfer. Compare against uniform pooling and target-only baselines under the exact deployment split.

## Best neurOS use cases

### Multi-subject / multi-session foundation-model adaptation

Rank and sample source subjects using frozen embedding distributions before fine-tuning a target decoder. This directly complements the model-agnostic probes in `neuros-foundation`.

### Site and device transfer

For EEG data pooled across hospitals, headsets, montages, or acquisition protocols, use source weighting to avoid giving every domain equal influence when the target resembles only a subset.

### Multi-modal runtime degradation

Use independently measured signal quality and drift summaries to update a runtime fusion operator when one modality becomes unreliable. SourceWeigher should route trust, not manufacture confidence.

### Federated or privacy-preserving cohorts

Sites can exchange compact distribution summaries or approved calibration metrics rather than raw neural recordings. Weight estimation can remain central while data remains local.

### Ensemble and checkpoint routing

Treat independently trained models as sources and use held-out calibration risk, shift scores, or representation agreement to weight their outputs.

### Active calibration

Use mixture uncertainty and instability to decide which source/target conditions need additional calibration data rather than treating calibration duration as fixed.

## Scientific guardrails

1. **Split by the deployment unit.** Cross-subject claims require subject-disjoint evaluation; cross-device claims require device-disjoint evaluation.
2. **Separate similarity from task utility.** A source can look distributionally close yet transfer poorly because label mechanisms differ.
3. **Never use final test labels to choose weights.** Weighting is model selection and must happen inside training/calibration data.
4. **Always report uniform and target-only baselines.** Weighting should earn its complexity.
5. **Audit concentration.** ESS close to 1 means the system is effectively betting on one source.
6. **Audit stability.** Leave-one-source-out and perturbation analyses should accompany important deployment decisions.
7. **Keep provenance.** Source IDs, method, configuration, split, and diagnostics should be stored with trained checkpoints and experiment manifests.
8. **Do not turn quality heuristics into fake probabilities.** Stream quality, domain similarity, task risk, and predictive uncertainty are different signals.

## Package boundaries

`neuros-sourceweigher` intentionally does **not** own:

- raw device acquisition (`neuros-drivers`);
- generic runtime scheduling (`neuros-core`);
- foundation-model implementations (`neuros-foundation` / `neuros-neurofm`);
- mechanistic interpretability (`neuros-mechint`);
- task decoder architecture (`neuros-models`).

It owns source/domain reliability policies and the diagnostics necessary to make those policies inspectable.

## Research roadmap

The next high-value algorithms are not more variants of Euclidean moment matching. Strong candidates include:

- entropic Wasserstein / optimal-transport source matching;
- distributionally robust source mixtures that optimize against uncertainty sets;
- conformal risk gating with abstention when no source is sufficiently trustworthy;
- Bayesian source reliability with posterior uncertainty over weights;
- hierarchical priors over subject -> site -> device -> session;
- class/task-conditional mixture routing without materializing datasets in memory;
- change-point detection that triggers weight recalibration only when drift is statistically meaningful;
- differentiable routing for end-to-end mixture-of-experts training, kept separate from the transparent default estimators;
- causal/interventional source diagnostics to distinguish covariate shift from changes in the neural-to-behavior mapping;
- federated summary protocols with privacy/accounting guarantees.

The design principle is to keep transparent estimators as the baseline and add sophisticated algorithms only when they improve transfer under controlled domain-shift benchmarks.

## References and methodological roots

The package draws on several established ideas rather than claiming one universal source-selection theory: simplex-constrained mixture estimation, multi-source domain adaptation and moment matching, maximum mean discrepancy, Riemannian covariance geometry, entropy/Gibbs weighting, and online smoothing under drift.

For research comparisons, treat each strategy as a hypothesis about what makes a source transferable and validate that hypothesis on subject/site/device-disjoint experiments.

## Development

```bash
pip install -e packages/neuros-sourceweigher
pytest -q packages/neuros-sourceweigher/tests

python packages/neuros-sourceweigher/examples/01_subject_transfer.py
python packages/neuros-sourceweigher/examples/02_online_drift.py
python packages/neuros-sourceweigher/examples/03_distribution_methods.py
```

The core test suite is designed to run without FastAPI, PyTorch, or a neural dataset.
