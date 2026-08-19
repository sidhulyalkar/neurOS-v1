# v1 empirical evidence status

The v1 software contract is implemented in PR #14. The real empirical evidence program is intentionally **not marked complete** merely because the framework can express it.

Current machine-readable status:

```bash
neuros-mechint release-status --json
```

Pending empirical requirements are:

1. a held-out real-model circuit-faithfulness evidence pack;
2. a matched Transformer × SSM × Event × Relative-ISI study on real neural data;
3. at least one held-out causal feature correspondence;
4. replication of a correspondence across independent model-training seeds;
5. subject/session-level uncertainty where supported by the dataset;
6. at least one real cross-session or cross-dataset causal transfer study;
7. a real intervention dose-response study;
8. a stronger manifold-aware substitution control;
9. independent reproduction through a distinct execution path;
10. publication of supported negative results.

## Why these remain pending

The repository currently contains the model/data infrastructure and the causal evidence machinery, but a completed real 2 × 2 × seed evidence family is not present in PR #14. Marking these boxes complete without those artifacts would convert implementation capability into a scientific claim.

## Promotion rule

A requirement should move from `pending` only when the corresponding self-checking artifact fingerprint can be attached to the evidence status. For higher-level claims, the artifact must also contain the independent-unit hierarchy that the claim names.
