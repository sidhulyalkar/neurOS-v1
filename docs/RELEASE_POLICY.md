# Release Policy

This policy governs public neurOS/ORION software releases and the evidence claims attached to them.

## Release unit

The repository is a multi-distribution workspace. Individual Python distributions own their package versions under `packages/*/pyproject.toml`. A repository release is a **platform release manifest**, not an assertion that every internal distribution shares one version number.

A platform release must record the exact versions and wheel hashes of every included distribution.

## Distribution and namespace ownership

Multiple distributions may contribute subpackages to the shared `neuros` Python namespace, but **two distributions must never own the same installed file path**.

Component distributions use PEP 420 implicit namespace portions. The user-facing `neuros` SDK is the sole owner of `neuros/__init__.py` because that initializer defines the public top-level SDK API.

Release qualification scans the payload of every built workspace wheel, normalizes wheel `.data/purelib` and `.data/platlib` entries to their real install destinations, and fails if any destination has multiple owners. The resulting `neuros.wheel_ownership.v1` manifest is checksum-bound into the release-candidate bundle.

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

- build every intended workspace wheel from the release source;
- run package metadata validation (`twine check` or equivalent);
- prove that built wheels have no overlapping installed-file ownership;
- prove core/drivers/models work as implicit namespace portions without the SDK initializer;
- prove the installed SDK owns and exposes the documented top-level `neuros` API;
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
