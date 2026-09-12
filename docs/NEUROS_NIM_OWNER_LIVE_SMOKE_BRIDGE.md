# neurOS NVIDIA NIM owner live-smoke bridge

This control-plane bridge exists only to dispatch the already-frozen bounded NIM hosted smoke without exposing provider credentials or scientific execution inputs to an issue comment.

## Exact command

The bridge listens for a newly created comment on merged control PR `#183`. The complete comment body must be exactly:

```text
/nim-swarm-live-smoke
```

No suffix, arguments, JSON, model names, case IDs, budgets, seeds, prompts, or other execution parameters are accepted.

## Redundant owner authorization

The command is eligible only when all of the following are true:

- the event is a new `issue_comment` on PR `#183`;
- PR `#183` is closed and independently re-read as merged;
- its frozen merge identity remains `3fb2ab351cd26ee462495afaf2baa94fb6d0c7e8`;
- `github.actor` equals the repository owner;
- the comment author's login equals the repository owner;
- GitHub reports the comment author's association as `OWNER`;
- the complete comment text equals `/nim-swarm-live-smoke`.

The bridge receives no NVIDIA credential. Its token permissions are limited to `actions: write` and `contents: read`.

## Frozen hosted-workflow authority

Before dispatch, the issue-comment event SHA must still equal the current `main` SHA. The bridge then reads the two hosted-smoke control-plane files from that exact SHA and requires their Git blob identities to remain:

- `.github/workflows/nim-swarm-live-eval.yml`: `71297f625419a38a85c0157c85ad86173290c180`;
- `.github/workflows/nim-live-artifact-policy.yml`: `28e61cc9e450eb99d24c2c4b6b018a98ebbeabe8`.

If either file changes in the future, this command fails closed until the bridge is deliberately updated and independently requalified. An old owner comment therefore cannot authorize an evolved hosted workflow.

## Fresh-main qualification gate

Owner identity and frozen workflow bytes are not enough. Before any hosted dispatch, the bridge independently queries every push-triggered workflow observed for its captured `main` SHA.

It requires the following critical workflows to exist:

- `NIM Owner Live-Smoke Bridge Policy`;
- `Public Trust Contracts`;
- `neurOS CI`.

Every observed push workflow for the captured SHA must be completed with `success`. A missing critical workflow, a queued/running workflow, or any non-success conclusion rejects the command. This removes the human assumption that the owner remembered to wait for fresh-main qualification before commenting.

Both the fresh-main history query and the prior hosted-smoke history query request at most 100 runs and compare GitHub's `total_count` with the returned list length. If the result is paginated or otherwise incomplete, authorization fails closed rather than deciding from a truncated run set.

## Dispatch semantics

After fresh-main qualification, the bridge rejects any active hosted smoke and any already-successful hosted smoke for the same exact source SHA. It then dispatches only `.github/workflows/nim-swarm-live-eval.yml` with the exact fixed control-plane payload:

```json
{"ref":"main","return_run_details":true}
```

`return_run_details` is not a hosted-review input. It explicitly asks GitHub's workflow-dispatch API to return the created run ID and URLs so dispatch correlation does not depend on a changing API default. The target workflow itself exposes no dispatch inputs. All case selection, model qualification, reviewer roles, request parameters, and call limits remain owned by the promoted target workflow.

The dispatch response must contain a positive `workflow_run_id` plus non-empty API and HTML run URLs. The bridge then independently re-reads that run and requires:

- event `workflow_dispatch`;
- branch `main`;
- `head_sha` equal to the owner-comment event's captured SHA;
- workflow ID `355385233`;
- path `.github/workflows/nim-swarm-live-eval.yml`.

A mismatch is treated as a dispatch/main race or target substitution. The bridge attempts to cancel the target run and fails closed.

## Independent policy

`.github/workflows/nim-owner-live-smoke-bridge-policy.yml` statically checks the exact owner command, authorization gates, fresh-main qualification gate, complete run-history checks, fixed dispatch payload, explicit run-detail receipt, race cancellation, absence of NVIDIA secrets/scientific controls, and the frozen target workflow identities. It also independently computes the checked-out Git blob SHA for the target live workflow and artifact-policy workflow and requires them to match the bridge constants.

## Interpretation boundary

Dispatch authorization is not scientific authority. A successful command means only that the frozen bounded hosted smoke may execute on one exact, fresh-main-qualified SHA. Its three public cases remain operational transport/schema/telemetry qualification only. The bridge grants no merge authority, provider ranking, future provider-execution authority, Kumar2024 authorization, scientific promotion, clinical claim, or ORION comparison authority.

Tracks #178 and #183.
