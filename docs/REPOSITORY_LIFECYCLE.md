# Repository lifecycle policy

neurOS uses exact-source qualification and evidence-bearing pull requests. That makes history valuable, but it also makes stale stacked branches unusually easy to mistake for current authority. This policy keeps provenance without letting historical work become the active project map.

## 1. `main` is the integration authority

`main` is the only long-lived integration line. A production PR should normally target current `main` directly. Short stacks are acceptable while one coherent tranche is being developed, but a deep stack is not a substitute for integration.

If `main` has materially advanced or a stack has grown beyond roughly two dependent production PRs, prefer a fresh current-main consolidation PR that reconstructs the intended net change and requalifies it there.

Historical qualification proves the historical source. It does not automatically transfer to a reconstructed or rebased candidate.

## 2. Classify every PR by lifecycle

Every PR should be one of three things:

### Production candidate

A coherent change intended for `main`. It owns its exact source identity, tests, claim boundary, and merge gate.

### Experiment or staging

A disposable branch used to answer a bounded question, benchmark an implementation, reproduce a failure, or harden a candidate. Its evidence may inform production work, but the staging branch is not promoted directly unless it is explicitly reconstructed as a production candidate and freshly qualified.

Use clear branch/PR naming such as `experiment/` or `[STAGING]` when possible. Close the PR after the result has been recorded.

### Historical or superseded provenance

A PR whose useful semantics have already been consolidated onto `main`, replaced by a stronger architecture, or made obsolete by a newer authority boundary. Keep the PR and its exact-head evidence as provenance, but close it rather than continually rebasing it.

## 3. Exact-head qualification is immutable evidence

Qualification belongs to the exact commit that was tested.

Any authority-bearing source change after qualification resets the qualification state. Documentation-only changes may be treated separately only when the PR explicitly demonstrates that no production/test semantics changed and the repository's governing workflows support that distinction.

Construction evidence, staging evidence, historical green runs, and current promotion evidence must be labeled separately.

## 4. Scientific authority stays narrower than software success

Green CI can establish software, contract, reproducibility, transport, or execution semantics. It does not by itself establish neural efficacy, biological mechanism, hardware validity, clinical validity, or ORION superiority.

A PR that changes any of the following must name the changed authority explicitly:

- dataset or participant identity;
- preprocessing or materialization;
- train/evaluation split or reveal order;
- model or representation identity;
- metric semantics or comparison authority;
- retry, settlement, or provider execution authority;
- external-floor interpretation;
- ORION comparison or mechanism claims.

## 5. Merge authority is source-bound

Authority-bearing merges should use an expected-head SHA guard. If the resulting `main` commit itself becomes part of an execution or evidence identity, fresh post-merge qualification is required before that identity is promoted.

Do not merge an old stack one PR at a time merely because each historical head was once green. Qualify the intended current composition.

## 6. Keep the active queue small

Prefer fewer than five active production PRs. Experiments should have bounded questions and short lifetimes. Once a consolidation PR lands, close the superseded stack promptly.

A useful open PR should answer at least one of these questions:

1. What current product/scientific capability will this add?
2. What current defect will this remove?
3. What bounded experiment is still awaiting an answer?

If none applies, the PR is probably provenance rather than active work.

## 7. Repository settings should match the source policy

The repository should use a `main` ruleset or branch protection with pull-request review flow and required current CI checks. Automatic deletion of merged branches is preferred once branch provenance is safely retained through merged PRs and commits.

`CODEOWNERS` expresses ownership, not independence. A solo maintainer reviewing their own code is still one reviewer. As neurOS gains maintainers, scientific authority, runtime/data-plane authority, and release/governance ownership should be split across people where practical.

## 8. Cleanup is preservation, not erasure

Closing an obsolete PR does not erase its discussion, commits, workflow records, or scientific reasoning. The goal is to make GitHub's *open* state accurately represent what can still change the future of neurOS.
