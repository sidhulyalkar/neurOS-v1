# Release authority

This directory contains repository-level release policy that is intentionally independent from workspace membership and package version numbers.

- `package-policy.json` classifies every maintained workspace distribution.
- `scripts/list_release_packages.py` validates the policy fail-closed and emits the default release-candidate set.
- `.github/workflows/release-candidate.yml` builds only that explicit set and checksum-binds the policy snapshot into the release manifest.

A package can remain maintained, tested, and available to research workflows without being authorized for default publication. Scientific maturity is a separate axis from packaging eligibility.
