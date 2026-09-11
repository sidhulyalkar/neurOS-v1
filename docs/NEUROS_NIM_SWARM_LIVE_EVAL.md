# neurOS NVIDIA swarm live evaluation

This tranche measures whether the advisory neurOS reviewer swarm can produce useful,
verifiable regression-review evidence at bounded operational cost. It does **not** create a
new scientific or execution authority.

## Promotion dependency

The harness is built on promoted benchmark main:

- source base: `271b0df596f2ce4464e46c7a708aebde027c6c35`;
- public regression corpus commitment:
  `122d862b2d43078723b858d5d5520a4ffb24e9e497d0090a52865c20f3aa5e0a`.

The benchmark cases and taxonomy are public regression material. They are not a hidden
holdout and cannot establish general model capability, neuroscience validity, clinical
utility, or provider superiority.

## Additive observation contract

`nim_observed.py` leaves the existing `NimCallRecord` serialization unchanged. An observed
call returns that same record plus a separate `NimTokenUsage` object.

Provider token fields are retained only when the hosted response actually reports them.
Missing or malformed counts remain unavailable and are never estimated.

The observed machine-output path is intentionally stricter than the older proposal parser.
After trimming whitespace, the complete response must be exactly one JSON object. Free-form
prefix/suffix prose, non-object JSON roots, and duplicate object keys are rejected. This keeps
machine-validated evidence from depending on permissive extraction behavior and prevents the
evaluator from reproducing benchmark defect D06.

## Truthful failure accounting

Each `EvaluationCallReceipt` classifies one reviewer attempt as exactly one of:

- `success`: provider transport and strict review validation succeeded;
- `provider_failure`: no usable hosted response reached review validation;
- `review_validation_failure`: a hosted response arrived but failed exact JSON, schema, or
  public-taxonomy validation.

A validation-failed response may still have consumed latency and tokens. Its exact transport
identities and provider-reported usage are preserved rather than making malformed output look
artificially cheap.

## Frozen reviewer configurations

Live-qualified model ordering is frozen before benchmark outcomes are observed. The first
qualified route becomes the preregistered reference route. This is an outcome-blind reference,
not a claim that it is the strongest model.

The v1 smoke reconstructs:

1. `single-reference-v1`: one generalist reviewer on the reference route;
2. `homogeneous-five-role-v1`: five specialist roles on that same route;
3. `heterogeneous-five-role-v1`: the same five roles distributed across qualified routes,
   only when at least two routes qualify.

The third versus second configuration is the cleaner operational contrast for route diversity
because reviewer count and role prompts remain fixed.

The second versus first configuration does **not** isolate the causal value of role
specialization. It simultaneously changes reviewer count, independent sampling budget, and
role-specific prompts. If the five-role council improves, v1 can say that the complete council
configuration improved on these regression cases, not that specialization alone caused the
improvement. A future ablation should add five same-model generalist reviewers to separate
ensemble-size effects from specialist-role effects.

## Exact request reconstruction

The credential-free scorer independently reconstructs the reviewer configuration from the
live-qualified `selected_models`. For every successful or validation-failed attempt it then
rebuilds the frozen task prompt and canonical NIM request:

- exact member system prompt;
- exact benchmark user prompt;
- exact model route;
- `temperature=0.1`;
- `max_tokens=1200`;
- `stream=false`;
- `enable_thinking=false`.

The receipt's prompt and request fingerprints must match those reconstructed values. A
self-rehashed artifact therefore cannot silently substitute a different prompt or request.

## Provider qualification reconstruction

The scorer does not trust `qualified_models` merely because the qualification object hashes.
It independently requires:

- the pinned endpoint `https://integrate.api.nvidia.com/v1`;
- the exact documented Nemotron candidate roster;
- one ordered probe per documented route;
- valid probe status/failure semantics;
- `qualified_models` derived exactly from successful probes;
- a valid complete qualification fingerprint.

This prevents cryptographically self-consistent but semantically invented provider rosters from
becoming benchmark inputs.

## Counterbalanced hosted schedule

Running every baseline call first and every council call later would confound configuration with
provider load or route drift. The smoke therefore uses the frozen
`deterministic_rotating_configuration_order_v1` policy.

For each successive case, configuration order rotates by one position. The exact schedule is
serialized into the raw artifact and independently reconstructed by the scorer. It is fixed
before any hosted outcome exists.

With two configurations and three cases this is not perfectly balanced, and with three cases no
small schedule can remove all time/provider effects. The rotation simply removes the most obvious
block-order confound. Results remain operational smoke evidence, not causal model comparisons.

## Wall latency versus model-call work

Five council reviewers execute concurrently. Summing their individual call latencies would
therefore exaggerate user-visible wait time.

Each configuration manifest binds both:

- per-reviewer call latency;
- per-case end-to-end wall latency.

The scorer reports both, plus outcome counts and provider-reported token totals/coverage. Dollar
cost is not estimated in v1. A future monetary layer must bind an independently frozen price
schedule rather than retroactively applying current prices.

## Model-facing / scorer separation

The hosted-review process imports only `swarm_benchmark_cases.py`. It never imports scorer-side
`swarm_benchmark.py` or `_GROUND_TRUTH`. It emits unscored provider-neutral `CouncilRun` payloads,
receipts, and case-wall timings.

A separate credential-free scorer verifies, before importing ground truth:

- the outer raw artifact fingerprint;
- provider qualification semantics and fingerprint;
- selected-model ordering;
- the exact counterbalanced schedule and reviewer-call budget;
- reconstructed reviewer IDs, roles, models, and prompt hashes;
- exact prompt/request identities;
- manifest hashes and false authority flags;
- exact case × reviewer receipt coverage;
- receipt schema, outcome semantics, token provenance, endpoint, and task binding;
- council-run reconstruction and run fingerprints;
- successful receipt-to-review parsed-response identity;
- failed reviewer-to-error-class consistency;
- exact case-wall timing coverage.

The offline Python 3.10/3.11/3.12 matrix also fabricates complete content-addressed synthetic
raw evidence and executes the real scorer CLI. Adversarial unit tests rehash forged inner
artifacts after semantic substitutions and still require rejection.

## Bounded v1 smoke

The hosted job is manual `workflow_dispatch` only. Pull requests and pushes are credential-free
and never call NVIDIA.

The first hosted smoke is frozen to:

- `case-001`;
- `case-015`;
- `case-029`.

With one qualified route, the maximum is 18 reviewer attempts. With at least two qualified routes,
the maximum is 33 attempts. A full 29-case A/B/C run would require up to 319 reviewer attempts and
is intentionally disabled in v1.

Three public cases are sufficient to qualify transport, structured-output behavior, telemetry,
scorer reconstruction, and obvious operational failure modes. They are not sufficient to estimate
stable precision/recall differences between reviewer configurations.

## Credential boundary

The workflow reuses the existing repository secret aliases and prefers `NVIDIA_API_KEY`:

- `NVIDIA_API_KEY`;
- `NVIDIA_NIM_API_KEY`;
- `NVAPI_KEY`.

Never place a key in source, issue text, artifacts, or chat. The key is scoped only to the hosted
review step and scoring executes without the credential.

Raw hosted evidence is fail-closed for preservation. After `raw.json` is written, the hosted step
searches it for the exact active credential. Only a clean artifact receives `RAW_SAFE.sha256`.
Later evidence preservation re-verifies that seal, copies only sealed files into a separate
`nim-swarm-live-eval-upload/` staging directory, and creates `UPLOAD_READY` there only after the
copy succeeds. The upload action can see only that staged directory. If credential leakage is
detected, the safety seal is never created and **no raw evidence is uploaded**, even though the
job fails. If a later scoring or verification step fails after the raw artifact was safely sealed,
the sealed raw evidence may still be preserved for diagnosis without exposing the provider key.

A separate CI policy workflow checks these upload invariants whenever the hosted workflow changes.

## Authority boundary

This tranche grants no merge authority, no future provider-execution authority, no Kumar2024
execution authorization, no scientific-promotion authority, and no ORION comparison authority.
Reviewer agreement is measurement data, not a vote that creates truth.

Tracks #178.
