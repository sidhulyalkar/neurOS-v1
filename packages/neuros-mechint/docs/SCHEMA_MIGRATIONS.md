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

## Nonestimable evidence-pack results (v2)

The historical `neuros-mechint.evidence-pack.v1` result schema is **frozen**.
Existing all-valid and partially-valid v1 artifacts remain byte-for-byte
scientifically compatible. A pack containing at least one entire unestimable
aggregate (discovery, validation, or magnitude comparator) is instead
serialized as `neuros-mechint.evidence-pack.v2`.

In v2, aggregates with zero valid cases carry `estimable: false` and
`nonestimable_reason`; unmeasured statistics, including percentiles,
intervals and pass rates, are JSON `null`, **never** zero or NaN.
Case-level `invalid_reason` records are preserved. Promotion is always
rejected if discovery or validation is not estimable, or when the required
magnitude comparison is unestimable.

The envelope's historical v1 content-hash wrapper remains valid. The
`read_evidence_pack_artifact` reader accepts **both** v1 and v2 result
schemas, verifies their content hash, and rejects unknown schema families.
The v2 outcome remains unsuitable for numeric factorial-cell conversion
when required contrasts are missing. Consumers of v2 must check
`aggregate["estimable"]` or non-null estimates before numeric operations.
