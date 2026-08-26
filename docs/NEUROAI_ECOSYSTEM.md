# NeuroAI Ecosystem Evidence

neurOS integrates external neuroscience and NeuroAI projects according to one rule: **an integration claim must be no stronger than the executable evidence behind it**.

The goal is not to collect logos. The goal is to make neurOS a dependable seam between acquisition, neural computation, learned representations, deployment, and scientific evidence while keeping `neuros-core` small and stable.

## Integration classes

External projects fall into four classes.

1. **Runtime/interoperability integration**: a real upstream package or object crosses a neurOS boundary under executable tests.
2. **Scientific-method integration**: neurOS implements or delegates a published analysis under frozen numerical definitions and explicit provenance.
3. **Benchmark/reference target**: a project is scientifically relevant but remains external until its artifacts, licensing, environment, and protocol are qualified.
4. **Deliberate non-integration**: a useful project may inspire product or community design without belonging in the neural runtime or scientific dependency graph.

This distinction prevents research-code environments from becoming hidden kernel dependencies.

## SNAP-derived invariant spectral evidence

The Chung NeuroAI Lab's SNAP work introduces a spectral vocabulary for relating representation geometry to neural or task targets. neurOS exposes a dependency-light NumPy implementation through:

```python
from neuros.foundation_models import spectral_alignment_evidence

evidence = spectral_alignment_evidence(representations, targets)
print(evidence.effective_dimension)
print(evidence.residual_target_power)
```

The evidence object records positive representation eigenvalues, target power captured by positive-rank modes, cumulative captured target power, effective dimensions, aggregate residual target power, input/target SHA-256 identities, and a deterministic evidence digest.

### Why neurOS does not expose every SNAP null-space mode

For a rank-deficient sample kernel, the eigenvectors associated with zero eigenvalues are not unique. Two correct linear-algebra backends may rotate that null space and assign different target power to individual zero-eigenvalue vectors even though the represented subspace is identical.

neurOS therefore leaves positive-rank modes explicit and aggregates all target power outside their span into one invariant residual quantity. The dedicated CI lane also checks out a pinned SNAP revision and executes the authors' real `snap/metrics.py` against the neurOS implementation for invariant quantities.

The current public claim is a **numerical method / upstream conformance claim**. It does not imply that SNAP's published experiments were reproduced, that a model is biologically aligned, or that a representation is mechanistically equivalent to a brain circuit.

## ngc-learn interoperability

`ngc-learn` is a JAX toolkit for computational neuroscience and biologically plausible NeuroAI, including graded neural dynamics, spiking cells, predictive-coding building blocks, and Hebbian/STDP synapses.

neurOS keeps it optional:

```bash
pip install "neuros-foundation[ngclearn]"
```

The qualified upstream surface is deliberately narrower than the upstream library.

### RateCell execution

```python
from neuros.foundation_models import NgcLearnRateCellTransform

transform = NgcLearnRateCellTransform(
    tau_m_ms=10.0,
    activation="identity",
    integration_type="euler",
)
result = transform.transform(samples, sample_rate_hz=250.0)
```

This surface records exact ngc-learn/JAX identity, explicit time-by-channel geometry, integration step, parameters, and deterministic input/output/evidence hashes. It performs no hidden resampling, filtering, normalization, padding, fitting, or channel reordering.

### Predictive reconstruction

neurOS now also qualifies an **inference-only fixed-weight predictive-coding circuit** using real ngc-learn 3.2 components:

```text
input target x -> GaussianErrorCell e0 -> transpose feedback E -> latent RateCell z
                    ^                                         |
                    |                                         v
                    +----------- fixed generative W <---------+
```

```python
from neuros.foundation_models import NgcLearnPredictiveCodingTransform

pc = NgcLearnPredictiveCodingTransform(
    latent_dim=4,
    settling_steps=30,
    seed=7,
)
result = pc.transform(samples, sample_rate_hz=250.0)

latents = result.values
reconstruction = result.reconstruction
print(result.evidence.error_reduction_fraction)
```

Each observation begins from a reset circuit. The observation is clamped as the `GaussianErrorCell` target; the latent `RateCell` iteratively changes under residual feedback; a fixed generative `StaticSynapse` reconstructs the input; and the feedback matrix is explicitly tied to the transpose of that fixed generative matrix.

The result records:

- exact ngc-learn and JAX runtime identity;
- actual `RateCell`, `GaussianErrorCell`, and `StaticSynapse` class identities;
- latent and reconstruction geometry;
- settling count, settling timestep, membrane constant, prior, activation, and integration method;
- reset-per-observation and tied-transpose-feedback semantics;
- input, weight, latent, reconstruction, and error-trajectory SHA-256 identities;
- initial/final reconstruction MSE and reduction fraction;
- fraction of observations that improve during settling.

The real-upstream CI contract goes beyond object construction. With an identity generative dictionary, the installed ngc-learn 3.2 circuit must reduce known reconstruction error by more than 90%, and repeated executions must be bit-identical after reset. That behavior passed on the qualified Python 3.10 and 3.11 lanes.

The circuit does **not** learn its weights. That is intentional. Fixed weights make inference dynamics, state reset, geometry, and provenance independently testable before introducing another source of state mutation.

### Current ngc-learn evidence ladder

1. **Qualified:** RateCell upstream execution and geometry.
2. **Qualified:** fixed-weight predictive reconstruction with iterative Gaussian residual feedback.
3. **Not yet qualified:** Hebbian/STDP synaptic adaptation and explicit adaptation/evaluation authority.
4. **Not yet qualified:** spiking-network representation contracts.
5. **Not yet qualified:** real neural-data utility under deployment-unit-disjoint evaluation.
6. **Not yet qualified:** comparison against ORION under matched downstream capacity and calibration budgets.
7. **Not yet qualified:** hardware or closed-loop behavior.

This ordering matters. A working predictive-coding circuit is not evidence that biologically local learning improves a BCI, and neither is evidence of a biological mechanism.

## Why predictive coding matters to ORION

The value of the ngc-learn bridge is not merely adding another model family. It gives ORION a scientifically different representation baseline.

A conventional frozen encoder, an ORION tokenizer/encoder, and a predictive-coding latent can eventually be compared under the same evaluation authority for:

- held-out task utility;
- user-specific calibration examples/minutes;
- cross-session and cross-subject transfer;
- representation effective dimension and task-aligned spectral power;
- residual target power;
- artifact, channel, montage, and jitter sensitivity;
- uncertainty calibration;
- latency and memory;
- intervention/adaptation stability;
- immutable data/model/evidence identity.

That allows a useful question: **does iterative error-correcting inference provide transferable neural structure, and if so, does it reduce calibration cost compared with simpler or learned alternatives?**

## Other evaluated NeuroAI projects

### IBM NeuroAIKit

IBM NeuroAIKit remains a **planned isolated reference worker**, not a neurOS dependency. Its SNU work is scientifically useful as a historical biologically inspired baseline, while its TensorFlow-era environment should not constrain the current neurOS Python/runtime matrix.

### NeuroAI Lab projects

The `neuroailab` organization is not represented as one integration because its repositories cover very different scientific problems and environments.

- **mouse-vision** is a planned neural-predictivity benchmark, with future neurOS evidence preferably bound to authoritative Allen/public-data identities.
- **TDANN** is a planned topographic-representation benchmark. Licensing and reproducible artifact identity must be resolved before implementation code is reused.
- older paper-specific environments can remain isolated reference workers rather than forcing legacy TensorFlow/Python dependencies into neurOS.

### Chung NeuroAI Lab

SNAP is the first method integrated because its representation geometry maps directly onto the neurOS evidence plane. Other work on manifold geometry, feature learning, and spontaneous retinal representations is valuable input to future ORION evaluation but is not automatically a package dependency.

### Neuro SAN Studio

Cognizant's Neuro SAN Studio is an LLM multi-agent orchestration product, not a neuroscience runtime. neurOS therefore does **not** list it as scientific compatibility. Its declarative workflows, traceability, project scaffolding, security/support documentation, and visual tooling are useful product-design references for a future neurOS Studio.

## Toward evidence conformance

The long-term pattern is:

```text
reference method / upstream package
             |
             v
      frozen input identity
             |
             v
   neurOS adapter/operator
             |
             v
 numerical/runtime comparison
             |
             v
 evidence conformance artifact
```

A future `EvidenceConformanceManifest` should bind method/citation, upstream repository and revision, license, upstream environment, neurOS operator/revision, fixture/data hashes, numerical semantics/tolerances, comparison result, and explicit claim boundary.

This is the difference between saying **"neurOS has an implementation"** and saying **"this implementation has executable evidence connecting it to the referenced method."**

## ORION connection

ORION should consume this ecosystem through stable neurOS contracts rather than bespoke research scripts. A future ORION representation card should pair task performance and calibration cost with spectral geometry, cross-session/subject invariance, domain leakage, montage/device robustness, causal/adaptation stability, uncertainty, runtime, and immutable identities.

The research objective remains practical: achieve the same or better held-out neural performance with materially less user-specific calibration, while showing *why* the representation transfers and where the claim stops.

## Design invariant

External ecosystems may evolve quickly. neurOS runtime contracts should evolve slowly.

```text
external research package
        |
        v
optional adapter / evidence worker
        |
        v
stable neurOS neural contracts
        |
        +----> runtime / replay
        +----> ORION
        +----> evidence / qualification
```

No integration should reverse that dependency direction merely because importing it is convenient.
