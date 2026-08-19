# neuros-mechint test policy

Files named `test_mechint_*.py` are the maintained v0.9 test surface. Run:

```bash
cd packages/neuros-mechint
pytest
```

The maintained suite covers the causal experiment kernel, evidence tiers/method cards, localization, path patching, causal-map stability, tokenizer/shared-computation studies, checkpoint emergence, circuit faithfulness, ecosystem adapter protocols, ORION/NeuroFM integration, v0.6 held-out evidence packs, v0.7 matched factorial comparison, v0.8 causal feature correspondence, and v0.9 hierarchical replication/dose-response analysis.

v0.8 correspondence tests retain coverage for:

- true held-out causal correspondence recovery;
- rejection of a nearly perfectly predictive, semantically matched but causally unused decoy;
- discovery-only access for correspondence fitting;
- semantic-trial discovery/validation leakage rejection;
- undeclared source/target context-difference rejection;
- one-to-one, one-to-many, and subspace correspondence;
- source and target causal relevance requirements;
- shuffled semantic-pair controls;
- same-cardinality random-source controls;
- real `PyTorchAdapter` feature capture, ablation, and target substitution;
- model-state mutation guards;
- artifact round-trip/tamper detection.

v0.9 specifically tests:

- model-seed claims count independent seeds rather than raw trials;
- heavily unbalanced trial counts do not reweight the seed-level estimate;
- hundreds of trials from one model seed are rejected as model-seed replication;
- an estimable multi-seed result with 50/50 sign disagreement is rejected;
- missing hierarchy coordinates make the declared claim non-estimable;
- replication-family mismatches are rejected;
- hierarchical bootstrap output is deterministic for a fixed seed;
- v0.7 factorial bridges preserve non-estimable status and rejection reasons;
- v0.8 correspondence bridges preserve held-out causal metrics;
- replication artifact round-trip/tamper detection;
- monotonic planted dose-response recovery;
- non-monotonic dose-response rejection;
- generative/conditional manifold donors require fit-partition provenance;
- maintained notebook JSON validity including `11_hierarchical_replication.ipynb`;
- eight CLI scientific gates.

Optional ORION tests skip cleanly when absent and are exercised explicitly in the ORION + NeuroFM CI job. TransformerLens, NNsight, and SAELens additionally receive real-package import/solver jobs.

CI does not download pretrained model weights. Real evidence belongs under:

```text
experiments/mechint/evidence_packs/
experiments/mechint/factorial_studies/
experiments/mechint/correspondence_studies/
experiments/mechint/replication_studies/
```

The scientific gates are deliberately two-sided. The framework must recover planted mechanisms and also reject known invalid, confounded, overfit, similarity-only, or pseudoreplicated cases. A framework that always returns a positive mechanism, correspondence, or replication result is scientifically incorrect.

## Historical Phase-2 tests

Older files such as `test_integration_phase2.py`, `test_advanced_integration.py`, and `test_fractals.py` are retained for provenance while their broad research APIs are audited and migrated. They target pre-v0.2 contracts or exploratory methods and are not evidence that those methods are Stable.

They remain outside the default `python_files` pattern. Promote one into the maintained prefix only when it executes against the current API, has an explicit scientific claim boundary, includes required controls, declares optional dependencies correctly, and fits the current maturity/evidence policy.
