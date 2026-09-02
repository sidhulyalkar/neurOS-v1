# Release Policy

This policy governs public neurOS/ORION software releases and the evidence claims attached to them.

## Release unit

The repository is a multi-distribution workspace. Individual Python distributions own their package versions under `packages/*/pyproject.toml`. A repository release is a **platform release manifest**, not an assertion that every internal distribution shares one version number.

A platform release must record the exact versions and wheel hashes of every included distribution.

## Workspace membership is not release eligibility

`tool.uv.workspace.members` answers only whether a distribution is developed and validated in this monorepo. It does **not** authorize that distribution for publication.

`release/package-policy.json` is the machine-readable release authority. It classifies every workspace member exactly once across two independent dimensions:

- `release_tier`: whether the package is part of the default public runtime, a separately qualified integration, a research extension, or an internal preview;
- `scientific_maturity`: what kind of evidence the package has actually earned.

Only entries with `publish_candidate=true` enter the default release-candidate wheel set. The policy validator fails closed if a workspace package is unclassified, a distribution name drifts from package metadata, a non-runtime package enters the default release set, or the SDK dependency closure is missing.

A package may remain fully maintained in the monorepo while intentionally not being a default publication candidate. Version number, workspace membership, test coverage, and scientific maturity are not interchangeable promotion signals.

The release manifest checksum-binds the exact package-policy file and the selected package inventory so the artifact set can be reconstructed from evidence rather than inferred from repository layout.

## Distribution and namespace ownership

Multiple distributions may contribute subpackages to the shared `neuros` Python namespace, but **two distributions must never own the same installed file path**.

Component distributions use PEP 420 implicit namespace portions. The user-facing `neuros` SDK is the sole owner of `neuros/__init__.py` because that initializer defines the public top-level SDK API.

Release qualification scans the payload of every built release wheel, normalizes wheel `.data/purelib` and `.data/platlib` entries to their real install destinations, and fails if any destination has multiple owners. The resulting `neuros.wheel_ownership.v1` manifest is checksum-bound into the release-candidate bundle.

This is an install-integrity requirement, not a style preference. Package managers track files per distribution; if two wheels record the same path, uninstalling either distribution can delete a file the other still requires.

## Versioning

Promoted public Python APIs follow semantic-versioning intent:

- **patch**: backward-compatible fixes, evidence/qualification hardening that does not remove a public contract, documentation corrections;
- **minor**: backward-compatible capabilities, adapters, evidence operators, optional integrations, or new public contracts;
- **major**: intentional incompatible changes to promoted public APIs or persisted canonical formats that cannot be handled through a compatible migration path.

Research packages and explicitly experimental APIs may evolve faster, but must remain labeled experimental and must not silently masquerade as stable kernel contracts.

## Deprecation

For promoted public APIs:

1. introduce the replacement before removing the old path when feasible;
2. emit a clear deprecation warning/documented migration;
3. keep the deprecated path for at least one subsequent minor release unless security, data-integrity, or safety concerns require faster removal;
4. describe removals in `CHANGELOG.md` and migration documentation.

Persisted replay/archive/qualification formats require stronger discipline. A format change must either remain readable or provide an explicit migration/compatibility boundary.

## Release candidate qualification

Before a public release is published, CI must:

- validate that every workspace member is explicitly classified by the release package policy;
- build every distribution explicitly authorized for the default release set from the release source;
- run package metadata validation (`twine check` or equivalent);
- prove that built wheels have no overlapping installed-file ownership;
- prove core/drivers/models work as implicit namespace portions without the SDK initializer;
- prove the installed SDK owns and exposes the documented top-level `neuros` API;
- checksum-bind the release package policy and selected inventory into the release manifest;
- generate SHA-256 checksums for release artifacts and the wheel-ownership manifest;
- smoke-install the user-facing `neuros` distribution from the built wheel set;
- execute a minimal `neuros doctor` / compatibility smoke path;
- build documentation in strict mode;
- pass the applicable runtime, scientific/evidence, interoperability, and qualification workflows on the exact release commit.

A release is not considered qualified because a prior commit was green.

## Artifact identity

Release notes should include:

- Git tag and commit SHA;
- component package/version manifest;
- release package-policy SHA-256 and selected inventory;
- SHA-256 manifest for built artifacts;
- wheel-ownership manifest proving unique installed-file ownership;
- strongest supported evidence tiers and important claim boundaries;
- breaking/deprecated behavior;
- known limitations.

Qualification/evidence bundles retain their own roots and should not be replaced by a release checksum.

## Publishing authority

Pull-request CI must not possess package-publishing credentials.

PyPI publication should use trusted publishing / short-lived OIDC identity when configured. Until the repository and PyPI project are explicitly configured for trusted publishing, release workflows stop after building and validating artifacts. Adding a long-lived PyPI API token merely to automate releases is not an acceptable substitute.

## Release provenance

When publishing is enabled, prefer GitHub artifact attestations or an equivalent provenance mechanism that binds the published artifact to the source workflow and commit. Provenance augments, but does not replace, package checksums and scientific qualification artifacts.

## Support window

The current `main` branch and most recent tagged public release receive priority for bug/security fixes. Additional backports are best effort unless a future long-term-support release is explicitly designated.

## Scientific evidence and releases

A software release never automatically promotes scientific evidence. A new version may contain a method capable of producing real-dataset or hardware evidence while the method itself remains only software-contract/integration qualified until those stronger artifacts exist.

`docs/SCIENTIFIC_CLAIMS.md` is authoritative for claim language.

## Reproducible citation

Research publications should record the exact release tag or commit and relevant evidence artifact roots. `CITATION.cff` intentionally does not invent a DOI before an archival release deposit exists.
