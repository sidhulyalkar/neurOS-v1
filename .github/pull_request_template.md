## Purpose

<!-- What user, scientific, runtime, or architectural problem does this change solve? Keep one primary responsibility. -->

## Exact candidate

- Base branch / SHA:
- Head branch / SHA:
- Supersedes or depends on:
- Production candidate, experiment/staging, or historical provenance:

## Architectural boundary

- [ ] This change has one primary responsibility.
- [ ] Dependency direction remains consistent with `docs/ARCHITECTURE.md`.
- [ ] New hardware/model/research functionality enters through an existing contract/plugin/adapter boundary, or this PR explicitly justifies a contract change.
- [ ] Stable kernel code does not gain an implicit dependency on research implementations.

## Compatibility and migration

<!-- Describe API/config/artifact compatibility, deprecations, migration behavior, and rollback. -->

- [ ] Existing stable behavior is preserved or an explicit migration path is documented.
- [ ] Configuration/schema changes are versioned rather than silently reinterpreted.
- [ ] Model or representation changes preserve honest algorithm/capability identity.

## Evidence tier

Select the strongest evidence actually produced by this PR. Do not imply a stronger tier.

- [ ] Unit
- [ ] Contract
- [ ] Integration
- [ ] Replay
- [ ] Scientific synthetic
- [ ] Real dataset
- [ ] Hardware qualification
- [ ] Closed-loop qualification
- [ ] Clinical evidence (normally outside this software repository)

### Validation performed

<!-- Exact commands, tests, datasets/fixtures, hardware if applicable, and important results. -->

## Scientific and BCI claim boundary

- [ ] Train/fit/adaptation and held-out evaluation data are separated where relevant.
- [ ] Subject/session/site/device/montage split semantics are explicit where relevant.
- [ ] Missing uncertainty/confidence is not fabricated.
- [ ] Attribution/attention/sparse features are not described as causal mechanism without intervention evidence.
- [ ] Synthetic/software evidence is not described as hardware, biological, clinical, or safety validation.
- [ ] Any change to data identity, preprocessing, model, metric, reveal order, retry policy, provider authority, or ORION comparison authority is called out explicitly below.

### Authority changed, if any

<!-- State the exact authority changed and the strongest claim the resulting evidence can support. -->

## Reliability, replay, and provenance

- [ ] Runtime changes expose failures, queue loss, timing, and overload behavior rather than hiding them.
- [ ] Recording changes preserve canonical sequence/timing/quality/provenance semantics.
- [ ] Promoted model/data/benchmark artifacts have immutable or reproducible identity where applicable.
- [ ] A replay/regression path exists for consequential runtime/model changes where practical.

## Qualification integrity

- Exact head being qualified:
- Required workflows:
- Focused/adversarial tests:
- Known skipped or unavailable surfaces:

- [ ] Construction/staging evidence is not being reused as promotion evidence.
- [ ] Qualification evidence belongs to the exact head above.
- [ ] Any authority-bearing code change after qualification will require fresh exact-head qualification.

## Merge gate

<!-- State what must be true before merge. Authority-bearing merges should use an expected-head SHA guard. -->

- [ ] PR is current with the intended `main` authority or explicitly documents why it is a historical stack.
- [ ] No unresolved review thread changes the claimed authority boundary.
- [ ] Post-merge qualification requirements are stated when the resulting `main` SHA itself becomes an authority identity.

## Documentation and developer experience

- [ ] Current docs are updated in the same PR when public behavior changes.
- [ ] Supported examples use current APIs and are executable.
- [ ] Optional dependencies fail with actionable errors and do not silently alter algorithm identity.
- [ ] `python scripts/check_repo_hygiene.py` passes for structural changes.

## Lifecycle

- [ ] Production candidate intended for `main`.
- [ ] Experiment/staging only; close after evidence is recorded and do not merge directly.
- [ ] Historical/superseded; retain for provenance and close rather than rebasing indefinitely.

## Known limitations / next layer

<!-- What remains deliberately out of scope? What evidence or qualification is still missing? -->
