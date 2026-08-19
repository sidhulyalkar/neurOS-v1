# neuros-mechint 1.0.0 release notes

v1.0 closes the software/reproducibility contract built across v0.3-v0.9.

## New in v1

- manifest schema v3 with deterministic `scientific_fingerprint` and execution-specific `run_hash`;
- backwards migration from manifest schema v2;
- frozen artifact schema catalog for evidence packs, factorial studies, correspondence, hierarchical replication, and dose response;
- backwards migration of pre-v1 artifact envelopes without rewriting scientific results;
- self-checking dose-response artifacts;
- `ReproductionSpec` and tolerance-aware independent execution comparison;
- duplicate-run and qualitative-decision-flip falsification controls;
- `neuros-mechint schemas --json`;
- `neuros-mechint release-status --json`;
- `neuros-mechint verify-dose-response-artifact ...`;
- `neuros-mechint v1-ground-truth --json`, the ninth maintained synthetic scientific gate;
- executable CPU tutorial evidence CI;
- tutorial 12 on independent reproduction and evidence closure;
- explicit empirical evidence status and a concrete real-study landing zone.

## Scientific boundary

Version 1.0.0 means the **framework contract is stable enough to run and reproduce the intended experiments**. It does not mean the repository has already demonstrated a universal neural mechanism or a superior tokenizer. The real Transformer × SSM × Event × Relative-ISI evidence program remains explicitly pending until its artifacts are produced.
