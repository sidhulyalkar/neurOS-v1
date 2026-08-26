# Governance

neurOS and ORION are currently maintainer-led open-source projects with evidence-first technical governance. The goal is to move quickly without allowing scientific meaning, runtime guarantees, or compatibility claims to become implicit.

## Decision principles

Changes are evaluated in this order:

1. **Scientific and operational correctness.** Does the change preserve the actual meaning of neural data, timing, models, and evidence?
2. **Explicit contracts.** Are inputs, outputs, lifecycle, provenance, and failure behavior inspectable?
3. **Minimal stable coupling.** Can the capability live behind an existing plugin/adapter/evidence boundary instead of expanding the kernel?
4. **Executable evidence.** Is the strongest public claim backed by a test, conformance fixture, real dataset, hardware manifest, or stronger evidence tier?
5. **User utility.** Does the change solve a repeatable research/deployment problem rather than merely increase feature count?
6. **Long-term maintainability.** Can the boundary survive upstream package and hardware evolution without forcing unrelated users to install it?

## Maintainer authority

The repository maintainer is responsible for merge decisions, release cuts, evidence-tier promotion, security response, and resolving architectural deadlocks. Maintainer authority does not override the evidence rules: a claim should still be reduced when the repository cannot support it.

As additional sustained contributors emerge, ownership should move toward documented package/domain maintainers rather than informal permission.

## Change classes

### Routine changes

Bug fixes, docs corrections, tests, narrowly scoped adapters, and backward-compatible implementation work can proceed through ordinary pull-request review.

### Contract changes

Changes to `SignalFrame`, runtime lifecycle, clocks, replay/archive semantics, plugin interfaces, ORION public contracts, qualification schemas, or public evidence semantics require:

- explicit migration/backward-compatibility analysis;
- relevant cross-package regression tests;
- documentation in the same pull request;
- a clear strongest evidence tier;
- exact-head CI qualification before merge.

### RFC-level changes

Open a design issue/RFC before implementation when a proposal would:

- reverse a documented package dependency direction;
- add a required heavyweight ecosystem dependency to a stable package;
- change canonical timestamp, replay, qualification, or artifact identity semantics;
- introduce a new public evidence tier or redefine an existing one;
- create a new top-level product surface;
- remove a public API without the deprecation path in `docs/RELEASE_POLICY.md`.

An RFC should state the problem, non-goals, alternatives, proposed contracts, failure modes, evidence plan, and migration strategy.

## Scientific claim governance

`docs/SCIENTIFIC_CLAIMS.md` governs public language. The weakest accurate evidence tier wins. A result can be scientifically interesting while remaining experimental.

Promoting an integration or claim requires executable evidence at the promoted tier. Organization names, package installation, citations, or passing unit tests are not themselves qualification.

## Merge discipline

Foundational changes should remain draft until the exact head is green. Consequential merges should use an expected-head guard when tooling permits. Tests should be strengthened or semantics fixed when failures expose ambiguity; tests should not be weakened merely to obtain green CI.

## Releases

Releases follow `docs/RELEASE_POLICY.md`. Release artifacts must be buildable from the tagged source, pass metadata checks, and carry cryptographic checksums. Package publication credentials are intentionally separate from ordinary pull-request CI.

## Conflicts of interest and upstream work

Contributors should disclose material affiliations when advocating an integration, benchmark, hardware platform, or scientific conclusion where that affiliation could reasonably affect evaluation. Upstream licenses and citations must be respected. neurOS should prefer conformance against upstream methods over copying code when a clean-room/numerical implementation is sufficient.
