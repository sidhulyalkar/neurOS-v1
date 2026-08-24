# Research extensions

neurOS deliberately keeps its stable runtime independent from fast-moving research packages. External
projects can depend on neurOS contracts and register sources, transforms, tokenizers, encoders,
decoders, sinks, and monitors through Python entry points without becoming dependencies of
`neuros-core`.

This page tracks research extensions that use that boundary for a concrete scientific workflow.
Inclusion here documents interoperability, not empirical validation or endorsement of a scientific
claim.

## QuantumBCI

[QuantumBCI](https://github.com/sidhulyalkar/QuantumBCI) is a falsifiable workbench for classical,
quantum-inspired, quantum-algorithm, and separately gated physical-quantum hypotheses about neural
signals.

The intended division of responsibility is:

```text
neurOS
  neural-data ABI / sources / replay / provenance
  grouped and longitudinal evidence boundaries
  foundation-model interoperability
  mechanistic-evidence tooling
        |
        v
QuantumBCI
  density geometry / open-system dynamics / contextual models
  matched classical falsification controls
  quantum-resource accounting
        |
        v
neuros-mechint
  interventions / held-out evidence / replication / evidence packs
```

### External transform plugin

QuantumBCI v0.3 registers a `quantumbci-density` transform through neurOS's standard
`neuros.transforms` entry-point group. Once the QuantumBCI integration release is installed, a neurOS
runtime config can use it without any QuantumBCI dependency in neurOS itself:

```yaml
streams:
  - id: eeg
    source:
      plugin: mock
      options:
        sampling_rate: 250.0
        channels: 8
    transforms:
      - plugin: smoothing
        options:
          window_size: 3
      - plugin: quantumbci-density
        options:
          sample_axis: -1
          output: observables
```

The transform preserves a `SignalFrame` and records an explicit
`quantumbci_claim_class=quantum_inspired` metadata field. It does not reinterpret the frame as evidence
for a microscopic physical quantum state.

### Shared evidence contracts

QuantumBCI's E001 density-geometry experiments are designed to reuse neurOS
`GroupedEvaluationData`, `EvaluationPartition`, `chronological_partition`, and
`NestedCalibrationSplit` instead of defining another split implementation. This provides a useful
cross-project invariant:

- neurOS freezes the deployment-realistic evidence boundary;
- QuantumBCI changes only the representation/mechanism under test;
- raw-data checksum, neurOS partition fingerprint, calibration-split fingerprint, QuantumBCI source
  revision, and neurOS source revision all enter the final scientific run identity.

This is especially valuable for longitudinal EEG because it allows QuantumBCI methods to be compared
against neurOS model-ladder evidence on the same sample authority rather than on a look-alike split.

### Foundation-model bridge

QuantumBCI can wrap a runnable `neuros-foundation` registry adapter behind its small encoder protocol.
The neurOS registry remains authoritative for availability and fails closed when an upstream model is
cataloged but not runnable. QuantumBCI then receives the exact embedding tensor and tests density or
other mechanism layers on top of it.

This means a foundation model receives credit for representation quality while QuantumBCI is evaluated
only for incremental mechanism value.

### Mechanistic evidence bridge

The next cross-project layer should translate QuantumBCI interventions into `neuros-mechint` evidence
contracts instead of reproducing the mechanistic-interpretability stack. Examples include:

| QuantumBCI intervention | Shared evidence interpretation |
| --- | --- |
| zero density off-diagonals | explicit representation ablation |
| scramble density eigenbasis | counterfactual representation intervention |
| remove one Hamiltonian coupling | component/parameter ablation |
| sweep a dephasing rate | dose-response intervention |
| force contextual operators to commute | mechanism substitution |

A successful intervention can support the stated quantum-inspired mechanism class. It does not, by
itself, promote the claim to a physical quantum neural mechanism.

## Stability status

The neurOS side of this integration uses stable public contracts already present in `neuros-core`,
`neuros-foundation`, and `neuros-mechint`. The QuantumBCI plugin/bridge is being qualified in its v0.3
experiment-orchestration work. Keep this page descriptive until that release is merged; do not make
QuantumBCI an installation requirement of the neurOS kernel.

## Extension design rule

A good neurOS research extension should satisfy all four rules:

1. **one-way dependency:** the extension may depend on neurOS; stable neurOS packages do not depend on it;
2. **explicit plugin boundary:** runtime participation uses standard neurOS entry points or public APIs;
3. **shared evidence authority:** the extension reuses neurOS provenance/split contracts rather than
   redefining them invisibly;
4. **claim isolation:** the extension's scientific claims remain owned and falsified by the extension,
   not implied by successful neurOS execution.

This pattern lets the ecosystem grow without turning the runtime kernel into a research dependency knot.
