# Strict execution foundation for Circuit Auditor

The base `MechanisticExperiment` API remains backward-compatible. For stronger
evidence requirements, opt in to strict execution on a plain PyTorch-backed
model. This is **software execution qualification**, not proof of a mechanism.

```python
from neuros_mechint import (
    ComponentRef, CounterfactualPair, MechanisticExperiment,
    OutputMetric, PatchIntervention, PyTorchAdapter,
)

model.eval()
experiment = MechanisticExperiment(
    adapter=PyTorchAdapter(model),
    pair=CounterfactualPair(clean, corrupted),
    metric=OutputMetric(lambda output: output.mean(), name="mean_output"),
    experiment_name="candidate-route",
    model_id="my-model",
    dataset_id="my-frozen-dataset",
    strict_execution=True,
)
result = experiment.run([PatchIntervention(ComponentRef("layer_name"))])
print(result.manifest.scientific_fingerprint)
```

## What strict mode enforces

- The model must be in evaluation mode, with a PyTorch-backed adapter.
- Caller PyTorch CPU/CUDA RNG state is restored on success or failure.
- Baseline and each intervention are replayed to detect non-deterministic outputs.
- Model state is content-hashed before and after execution; state mutation fails.
- Input and model content hashes are mandatory and frozen before any forward pass;
  silent input mutation is checked after the full audit.
- Every intervention/control must be a dataclass, and its full dataclass
  configuration plus implementation identity participates in the scientific
  fingerprint, not just its human-readable name.
- Non-finite scalar metrics are rejected.
- Strict runs use method version 2; legacy runs retain method version 1.

This is intentionally opt-in: a stricter default would break prior research
scripts that rely on stochastic training-mode execution. A future major release
can revisit the default only after migration coverage is complete.

## Limitations and scientific boundary

The replay checks are equality checks over repeated evaluations, not statistical
guarantees about all stochastic model behavior. Strict mode does not validate
the biological meaning of a hook point, interventions on shared-module call
occurrences, off-manifold replacements, hidden dataset leakage, or deployment
robustness. It also does not replace v0.6 evidence packs, random controls,
held-out validation, or independent replication.

For evidence beyond synthetic demonstrations, feed candidates into the existing
`run_adapter_evidence_pack` pipeline with frozen model/data revisions and
independent validation examples. Do not promote a scientific claim based
on this execution policy alone.
