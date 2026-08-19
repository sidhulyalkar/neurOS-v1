# neuros-mechint v1 release contract

v1 freezes the **software/evidence contract** while keeping empirical claims conditional on actual real-study artifacts.

## Stable v1 software contract

A mergeable v1 release requires:

- Python 3.10-3.12 maintained tests;
- optional TransformerLens, NNsight, and SAELens import checks;
- ORION/NeuroFM integration tests;
- nine synthetic scientific/falsification gates;
- manifest schema v3;
- frozen artifact catalog for evidence packs, factorial studies, feature correspondence, hierarchical replication, and dose response;
- backwards migration for manifest v2 and pre-v1 artifact envelopes;
- self-checking full-content hashes;
- deterministic scientific fingerprints separate from execution-specific run hashes;
- tolerance-based independent reproduction contracts;
- CPU tutorial execution in release/evidence CI;
- a machine-readable release status that cannot silently mark missing real evidence as complete.

## The identity split

```text
scientific fingerprint
  = frozen design/model/data/method identity

run hash
  = scientific identity + timestamped/environment execution provenance

filename/model alias
  = mutable convenience label, never scientific identity
```

Two independent executions should usually share the first and differ in the second.

## Empirical closure is separate

`neuros-mechint release-status --json` reports both:

- `software_contract_ready`;
- `empirical_evidence_complete`.

The first can be certified by repository code and CI. The second requires real neural/model evidence artifacts and therefore remains pending until those experiments are actually executed.

This distinction is a feature, not an embarrassment. It prevents synthetic benchmark success from being narrated as evidence that Relative-ISI, Event, Transformer, SSM, ORION, or any discovered circuit is superior on real neuroscience data.
