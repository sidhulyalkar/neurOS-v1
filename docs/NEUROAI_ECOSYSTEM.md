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

The Chung NeuroAI Lab's SNAP work introduces a spectral vocabulary for relating representation geometry to neural or task targets. neurOS now exposes a dependency-light NumPy implementation through:

```python
from neuros.foundation_models import spectral_alignment_evidence

evidence = spectral_alignment_evidence(representations, targets)
print(evidence.effective_dimension)
print(evidence.residual_target_power)
```

The v1 evidence object records:

- positive representation eigenvalues;
- target power captured by each positive-rank mode;
- cumulative captured target power;
- participation-ratio effective dimension;
- an effective dimension of remaining task power;
- aggregate target power outside the representation span;
- input and target SHA-256 identities;
- a deterministic evidence digest.

### Why neurOS does not expose every SNAP null-space mode

For a rank-deficient sample kernel, the eigenvectors associated with zero eigenvalues are not unique. Two correct linear-algebra backends may rotate that null space and therefore assign different target power to individual zero-eigenvalue vectors even though the represented subspace is identical.

neurOS treats that as an evidence-design problem rather than hiding it behind a tolerance. Positive-rank modes remain explicit and all target power outside their span is aggregated into one invariant residual quantity.

That gives the evidence a stable interpretation across valid eigensolver implementations.

The current public claim is a **software-contract numerical-method claim**. It does not imply that SNAP's paper results were reproduced, that a model is biologically aligned, or that a representation is mechanistically equivalent to a brain circuit.

## ngc-learn interoperability

`ngc-learn` is an actively maintained JAX toolkit for computational neuroscience and biologically plausible NeuroAI, including graded neural dynamics, spiking cells, predictive-coding building blocks, Hebbian/STDP synapses, and neural input encoders.

neurOS keeps it optional:

```bash
pip install "neuros-foundation[ngclearn]"
```

The first qualified upstream surface is intentionally narrow: the **ngc-learn 3.2.x `RateCell`**.

```python
from neuros.foundation_models import NgcLearnRateCellTransform

transform = NgcLearnRateCellTransform(
    tau_m_ms=10.0,
    activation="identity",
    integration_type="euler",
)
result = transform.transform(samples, sample_rate_hz=250.0)
```

The bridge records:

- exact ngc-learn version;
- exact JAX version and backend;
- JAX x64 state when observable;
- upstream component identity;
- time-by-channel input and output geometry;
- sample rate and derived integration step;
- RateCell parameters and seed;
- input/output hashes and an evidence digest.

It performs **no hidden resampling, filtering, normalization, padding, fitting, or channel reordering**.

The integration CI exercises the real upstream package. neurOS does not infer support for all of ngc-learn from one successful component test.

### Planned ngc-learn evidence ladder

1. RateCell upstream execution and geometry;
2. an explicit predictive-coding circuit with frozen dynamics/evidence semantics;
3. spiking-cell and STDP/Hebbian learning contracts;
4. real neural-data evaluation under deployment-unit-disjoint protocols;
5. comparison against ORION representations and adaptation strategies;
6. closed-loop qualification only if a complete sensing-to-action system is actually measured.

Each step must land with its own evidence before the compatibility claim is promoted.

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

A future `EvidenceConformanceManifest` should bind:

- method name and citation;
- upstream repository and revision;
- license;
- upstream environment identity;
- neurOS operator/revision;
- fixture/data hashes;
- numerical tolerances or exact semantic rules;
- comparison result;
- explicit claim boundary.

This is the difference between saying **"neurOS has an implementation"** and saying **"this implementation has executable evidence connecting it to the referenced method."**

## ORION connection

ORION should consume this ecosystem through stable neurOS contracts rather than bespoke research scripts. A future ORION representation card should pair task performance and calibration cost with evidence such as:

- spectral effective dimension;
- task-aligned mode power;
- residual target power;
- cross-session and cross-subject geometry;
- domain leakage;
- montage/device robustness;
- causal intervention stability;
- uncertainty calibration;
- runtime latency;
- immutable dataset/model/evidence identities.

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