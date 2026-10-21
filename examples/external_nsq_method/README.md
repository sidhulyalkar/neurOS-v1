# External NSQ method example

This is a deliberately small, non-promoted example of an external method
participating in the neurOS Neural System Qualification (NSQ) referee.

It is ordinary scikit-learn code:

```text
StandardScaler -> LogisticRegression
```

The fixture is deterministic and synthetic. Its scores are software-contract
evidence only; this example makes no claim about neural decoding efficacy,
biological meaning, competitiveness, or the frozen Kumar2024 comparison.

## Run it

From the repository root, after installing the repository's research/test
dependencies:

```bash
python -m examples.external_nsq_method.demo
pytest -q examples/external_nsq_method/test_example.py
```

To use this as a starting point in another repository, copy `method.py` and
`demo.py`, keep the public `neuros.foundation_models` imports, and replace
`ExternalDecoder` with your own estimator implementation.

## Contract boundary

The external implementation owns:

- feature transforms and estimator settings;
- fitting and prediction;
- probability semantics and class-column order;
- deterministic learned-state serialization and hashing.

neurOS owns:

- the frozen protocol and dataset lineage;
- source, calibration, and untouched final-assessment observation roles;
- the calibration budget frontier;
- score semantics, run identity, provenance, and failure preservation.

The factory returns a fresh estimator for every qualification budget. The
content-addressable learned state is an explicit SHA-256 over the fitted
scikit-learn arrays and class labels; no pickle serialization is used.

The method specification records the installed scikit-learn version. This is
intentional provenance: changing that dependency changes the method-spec SHA,
even when the source code and estimator settings are unchanged.
