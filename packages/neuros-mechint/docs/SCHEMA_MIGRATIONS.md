# v1 schema freeze and migrations

## Experiment manifests

v1 uses manifest schema `3`.

Schema `2` remains readable through `migrate_manifest_payload(...)`. The migration adds:

- `scientific_identity`;
- `scientific_fingerprint`;
- schema version `3`.

Runtime benchmark metadata is deliberately excluded from the scientific fingerprint.

## Artifact envelopes

The v0.6-v0.9 result schemas remain frozen and valid:

- `neuros-mechint.evidence-pack.v1`;
- `neuros-mechint.factorial-mechanism-study.v1`;
- `neuros-mechint.feature-correspondence-study.v1`;
- `neuros-mechint.hierarchical-replication-study.v1`.

v1 adds:

- `neuros-mechint.dose-response-study.v1`;
- canonical envelope metadata `neuros-mechint.artifact-envelope.v1`.

`migrate_artifact_envelope(...)` validates the historical family schema and full-content hash, then attaches the canonical v1 contract block. It does **not** rewrite the scientific result and does not change its `artifact_hash`.

Unknown future schema versions fail loudly rather than being guessed.
