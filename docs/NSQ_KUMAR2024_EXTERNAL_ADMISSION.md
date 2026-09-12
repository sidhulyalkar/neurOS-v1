# NSQ Kumar2024 external systems admission

Status: **score-blind admission contract. No efficacy, fleet, or ORION claim.**

This layer sits after the independent verifier from PR #163. It exists so a successful real external classical-worker qualification can be accepted mechanically without a human or controller opening score-bearing worker files.

## Authority flow

```text
frozen binding + archived source + one authorized CSP shard
                     |
                     v
         provider-neutral external runner
                     |
                     v
          sealed qualification bundle
                     |
                     v
       independent stdlib verifier
                     |
        narrow score-free return value
                     |
                     v
             admission adapter
                     |
                     v
      immutable systems-admission receipt
```

The adapter never reads `case_result.json`, never reads a model metric, and never imports neurOS scientific code. It calls the already-independent verifier and consumes only its narrow return mapping.

## Exact verifier output contract

The admission adapter requires the verifier to return exactly:

- `verified`
- `transport_provider`
- `binding_input_mode`
- `transport_source_revision`
- `transport_script_sha256`
- `external_bundle_sha256`
- `qualification_sha256`
- `worker_bundle_sha256`
- `shard_result_sha256`
- `numerical_result_interpretable`
- `global_analysis_performed`
- `external_floor_claim_generated`
- `orion_comparison_permitted`

Unexpected keys fail closed. This prevents a future verifier from silently widening the admission surface to include scientific values.

The four claim flags must remain `false`.

## Admission receipt

A successful admission binds:

- frozen scientific source revision;
- binding run/artifact/ZIP/bundle identity;
- exact EnvironmentAuthority;
- raw and study materialization identities;
- execution-plan and authorized shard identities;
- transport provider and binding-input mode;
- transport source revision;
- transport-script SHA-256;
- external bundle SHA-256;
- qualification SHA-256;
- worker bundle SHA-256;
- promoted shard-result SHA-256;
- independent verifier-script SHA-256.

It positively states only:

```text
transport_structurally_qualified = true
```

and preserves:

```text
scientific_outcomes_inspected = false
numerical_result_interpretable = false
global_analysis_performed = false
external_floor_claim_generated = false
production_fleet_authorized = false
orion_comparison_permitted = false
```

The receipt is domain-separated and content-addressed by `admission_sha256`.

## Verifier version binding

The admission records the exact SHA-256 of `verify_kumar2024_external_qualification.py`.

By default `verify-admission` requires that identity to equal the verifier bytes in the current checkout. Historical admission receipts may be inspected with an explicit `--allow-historical-verifier` flag, but that is an audit convenience, not permission to reinterpret or upgrade the historical claim.

## Write-once admission

The `admit` command creates the requested admission path with create-if-absent semantics. Existing admission files are never overwritten.

Example after a real external qualification:

```bash
python3 scripts/evidence/admit_kumar2024_external_qualification.py admit \
  /path/to/qualification \
  --transport-script scripts/evidence/run_kumar2024_external_qualification.sh \
  --expected-transport-revision <EXACT_PR163_TRANSPORT_SHA> \
  --output /new/write-once/path/external_systems_admission.json
```

Then independently check the receipt itself:

```bash
python3 scripts/evidence/admit_kumar2024_external_qualification.py verify-admission \
  /path/to/external_systems_admission.json
```

Neither command should be used to inspect the numerical contents of the worker artifact.

## Acceptance boundary for issue #166

A future issue #166 success can be recorded from the immutable admission receipt plus the underlying independently verified sealed bundle. No human score review is required for the transport go/no-go.

A structurally admitted one-shard CSP execution proves only that the frozen worker can execute outside GitHub Actions while preserving the authority graph.

It does not authorize the 1,350-shard comparison, does not create an external scientific floor, and does not permit ORION comparison.
