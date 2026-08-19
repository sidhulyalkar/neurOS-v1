# Migration from the pre-v0.2 research package

This document records the major compatibility boundary introduced by the v0.2 refactor. It remains relevant historical guidance in v0.7 because the package still retains exploratory pre-v0.2 modules for provenance.

## Stable package identity after v0.2

`neuros-mechint` is no longer treated as a second runtime or a grab bag for every computational-neuroscience analysis.

The maintained package identity is:

> a causal experiment and learning framework for testing how computation is implemented, learned, and shared across artificial and neural-data models.

The Stable/maintained path now progresses through:

```text
v0.2 causal experiment kernel
v0.3 comparative causal maps
v0.4 neural tokenizer / checkpoint laboratories
v0.5 quantitative circuit faithfulness
v0.6 held-out evidence packs
v0.7 matched architecture × tokenizer factorial evidence
```

## Top-level imports

The stable top-level namespace intentionally stays small. Historical names resolve lazily where compatibility is practical so importing `neuros_mechint` does not eagerly load every optional research dependency.

Prefer importing maintained scientific objects from their owning namespaces, for example:

```python
from neuros_mechint.core import MechanisticExperiment
from neuros_mechint.benchmarks import FactorialMechanismSpec
```

## Historical Phase-2 modules

Broad historical analyses remain available as research artifacts, including older dynamics, thermodynamics, topology, biophysical, counterfactual, cross-species, and pipeline modules.

Their presence does not imply current Stable method maturity or maintained test coverage.

Before promoting a historical method into new work:

1. define the exact scientific object it estimates;
2. state the claim boundary;
3. add matched controls;
4. add known-ground-truth positive and negative fixtures;
5. declare optional dependencies cleanly;
6. add maintained `test_mechint_*` coverage;
7. assign a method card/maturity level.

## Circuit naming

The historical ACDC implementation is now explicitly an ACDC-inspired module-pruning baseline rather than canonical edge-level ACDC.

Activation patching and module-level path patching are separate methods with separate claim boundaries.

Do not use historical names to imply a more faithful algorithm than the implementation supports.

## Provenance

New maintained experiments should use repository-aligned manifests and content hashing rather than building an independent provenance stack.

Real mechanism claims should progress through v0.6 evidence packs. Comparative architecture/tokenizer claims should additionally pass the v0.7 estimability layer.

## Tutorials and experiments

Maintained educational notebooks live under:

```text
tutorials/mechint/
```

Exploratory and real-study artifacts belong under root `experiments/`, including:

```text
experiments/mechint/evidence_packs/
experiments/mechint/factorial_studies/
```

The historical package example/notebook collection remains provenance rather than the supported teaching contract.
