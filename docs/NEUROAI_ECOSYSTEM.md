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

The qualified upstream surface remains deliberately narrower than the upstream library.

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

### Fixed-weight predictive reconstruction

neurOS qualifies an inference-only predictive-coding circuit using real ngc-learn 3.2 components:

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
```

Each observation begins from a reset circuit. The observation is clamped as the `GaussianErrorCell` target; the latent `RateCell` iteratively changes under residual feedback; a fixed generative `StaticSynapse` reconstructs the input; and the feedback matrix is explicitly tied to the transpose of the fixed generative matrix.

The result records exact runtime/component identity, latent/reconstruction geometry, settling semantics, input/weight/output hashes, and reconstruction-error trajectories. The real-upstream identity-dictionary test requires the installed ngc-learn 3.2 circuit to reduce reconstruction error by more than 90%. That behavior is qualified on Python 3.10 and 3.11.

The fixed circuit intentionally does not learn. It establishes inference and reset semantics independently before mutable state is introduced.

### Governed Hebbian predictive adaptation

The next evidence rung adds a real ngc-learn `HebbianSynapse` M-step while preserving the same predictive-inference structure:

```text
observation
    |
    v
predictive E-step / settling
    |
    +--> latent zF -------------------+
    |                                 |
    +--> Gaussian residual dmu        |
                                      v
                         upstream HebbianSynapse.evolve()
                                      |
                                      v
                              generative weights
```

The public learner is available from the dependency-light foundation namespace:

```python
from neuros.foundation_models import NgcLearnHebbianPredictiveCoding

learner = NgcLearnHebbianPredictiveCoding(
    latent_dim=8,
    settling_steps=20,
    learning_rate=1e-3,
    optimizer="adam",
    seed=7,
)

before = learner.infer(calibration_samples, sample_rate_hz=250.0)
adaptation = learner.adapt(calibration_samples, sample_rate_hz=250.0, epochs=2)
after = learner.infer(qualification_samples, sample_rate_hz=250.0)
```

The integration executes the real upstream two-factor learning rule after each predictive E-step:

- `latent.zF` is the Hebbian **pre** statistic;
- `GaussianErrorCell.dmu` is the Hebbian **post** residual statistic;
- `sign_value=-1` gives the reconstruction-minimization direction used by the qualified fixture;
- feedback is retied to the current generative transpose before each inference;
- no hidden post-update row normalization is performed;
- inference is required to preserve the complete learning-state identity.

#### Complete mutable-state identity

A learned model is not just its visible weight matrix. Adam, for example, contains first/second moments and an update counter that change future learning even when weights are restored separately.

neurOS therefore defines the adaptive state as:

```text
Hebbian learning state
    |
    +--> exact upstream weight array identity
    |
    +--> exact optimizer pytree identity
    |
    +--> combined state SHA-256
```

Snapshots preserve the upstream weight dtype and include the optimizer state. Rollback validation checks checkpoint content before mutating the learner, restores both weight and optimizer state, reties feedback, resets transient neural activity, and then verifies the exact combined state again.

The real-upstream tests exercise both SGD and Adam. They require deterministic independent learners, read-only inference after learning, exact Adam rollback, replay of the same future learning trajectory after rollback, and failure of corrupt checkpoints before the live learner is changed.

#### ORION adaptation authority binding

The learning mechanism stays in `neuros-foundation`. ORION stays independent. Their composition occurs in the evidence layer:

```text
AdaptationAuthority
  | exact ordered calibration rows
  | exact ordered qualification rows
  | processed-data identity
  v
scripts/evidence/run_ngclearn_hebbian_authority.py
  |
  +--> calibration-only proposal evidence
  +--> governed approval
  +--> ngc-learn adapt(exact calibration rows)
  +--> learner input SHA == authority-selected input SHA
  +--> read-only qualification before/after
  +--> retain OR exact full-state rollback
  v
deterministic evidence JSON
```

The worker verifies that the canonical time-by-channel matrix selected by the authority is exactly the matrix whose SHA-256 is recorded by the learner. The public worker is run twice in CI and the complete JSON output must be byte-identical.

This is an **integration/process-integrity qualification**, not an efficacy result. The current deterministic fixture proves that real upstream learning obeys the authority and rollback contract. It does not prove the learned representation helps real neural decoding.

#### Qualification data is not final-assessment data

The authority used by the worker has adaptation rows and held-out qualification rows. The qualification rows are read-only with respect to learning, but their metric may decide whether the update is retained or rolled back. Therefore they are part of **model-state selection**.

They must not later be described as an untouched final test set.

A real efficacy experiment should use three authorities:

```text
historical/source data
        |
target calibration partition
        |
        v
state-changing adaptation
        |
retention / rollback qualification partition
        |
        v
selected frozen state
        |
independent final-assessment partition
        |
        v
scientific efficacy claim
```

This distinction is required before neurOS can make claims about calibration reduction, transfer, or superiority over ORION/frozen baselines.

### Current ngc-learn evidence ladder

1. **Qualified:** RateCell upstream execution and geometry.
2. **Qualified:** fixed-weight predictive reconstruction with iterative Gaussian residual feedback.
3. **Qualified at integration tier:** real `HebbianSynapse` predictive adaptation with exact weight + optimizer-state identity, deterministic replay, ORION adaptation-authority binding, and exact retain-or-rollback semantics.
4. **Not yet qualified:** STDP or spiking-network adaptation contracts.
5. **Not yet qualified:** real neural-data adaptation utility under a three-way calibration/qualification/final-assessment protocol.
6. **Not yet qualified:** calibration-reduction or transfer superiority versus ORION under matched downstream capacity and budgets.
7. **Not yet qualified:** hardware or closed-loop adaptive behavior.

This ordering matters. A working local-learning circuit is not evidence that local learning improves a BCI, and neither is evidence that the circuit is a biological mechanism.

## Why predictive/local learning matters to ORION

The value of the ngc-learn bridge is not merely adding another model family. It gives ORION scientifically different representation and adaptation baselines.

A conventional frozen encoder, fixed predictive-coding latent, locally adapted predictive-coding latent, and ORION representation can eventually be compared under the same independent evidence authority for:

- held-out task utility;
- user-specific calibration examples/minutes;
- cross-session and cross-subject transfer;
- representation effective dimension and task-aligned spectral power;
- residual target power;
- domain leakage;
- artifact, channel, montage, and jitter sensitivity;
- uncertainty calibration;
- latency and memory;
- intervention/adaptation stability;
- immutable data/model/evidence identity.

That makes the commercially meaningful ORION question falsifiable: **can ORION preserve or improve neural utility with materially less user-specific calibration than frozen or locally adapted alternatives, and can neurOS show exactly where that advantage survives?**

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
