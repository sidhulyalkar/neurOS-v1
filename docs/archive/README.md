# Historical Archive

This directory preserves development history that is useful for provenance but is **not current product documentation**.

Contents may include:

- migration and modularization plans,
- cleanup/completion reports,
- dated session summaries,
- superseded architecture/API/performance documents,
- research evaluation references that no longer belong at repository root.

## Source-of-truth rule

When archived material conflicts with current code or documentation, use the current source of truth in this order:

1. tested code and versioned schemas/contracts,
2. `README.md`, `docs/ARCHITECTURE.md`, and `docs/API_REFERENCE.md`,
3. current configuration and quality profiles,
4. current `ROADMAP.md`,
5. archived material only for historical context.

Archived documents should retain their original wording where practical so development decisions remain traceable. Do not edit an archived report to make it appear current.

New session notes, migration reports, or one-time completion summaries should be stored here or represented as GitHub issues/PRs rather than placed at repository root.
