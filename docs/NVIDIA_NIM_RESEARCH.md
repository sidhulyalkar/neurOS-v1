# NVIDIA NIM research adapter

`neuros-research` can use NVIDIA's hosted NIM API as an optional research-proposal
layer. NIM is deliberately **outside** neurOS evidence authority.

## Trust boundary

The adapter may receive only explicitly approved public or aggregate material:

- source code;
- schemas;
- aggregate metrics;
- de-identified plots;
- public metadata.

It must not receive raw participant data, participant identifiers, hidden targets,
private-leaderboard feedback, credentials, or any other payload prohibited by the
experiment's `ExternalDispatchPolicy`.

The API credential is carried only in the HTTPS `Authorization` header. The client
fails closed unless the endpoint host is exactly `integrate.api.nvidia.com`; a caller
cannot redirect the secret to another host.

NIM output is proposal material. It is not `ExperimentEvidence`, cannot attach a
promotion decision, and cannot rewrite dataset, split, temporal-alignment, metric, or
promotion authority.

## Live tournament

`.github/workflows/nim-research-tournament.yml` runs on the research-authority branch,
the isolated `feat/nim-live-calibration` branch, or by explicit workflow dispatch. The
calibration branch exists only to qualify the external-provider path without spawning
the research PR's full monorepo workflow fan-out.

The workflow:

1. checks out and verifies the exact clean source revision;
2. qualifies the local `neuros-research` contracts before any network request;
3. requires one of `NVIDIA_API_KEY`, `NVIDIA_NIM_API_KEY`, or `NVAPI_KEY`;
4. queries NVIDIA's `/v1/models` endpoint;
5. admits only the explicitly qualified hosted `nvidia/nemotron-3-super-120b-a12b`
   model for the first calibration;
6. runs generator, adversarial-critic, and program-synthesis roles as separate calls;
7. structurally validates every proposed experiment against the frozen dispatch and
   development-metric menus;
8. fingerprints prompts, requests, responses, public context, candidates, and the
   complete tournament artifact;
9. verifies the credential is absent from the artifact;
10. uploads an immutable artifact named by the exact GitHub source SHA.

The workflow intentionally has `contents: read` permission. A model cannot push code,
open hidden data, alter a PR, or promote itself.

## First Algonauts objective

The initial prompt asks the NIM tournament to generate development-only experiments
around one prospective scientific question:

> Can neural-geometry evidence measured before OOD truth is opened predict which
> frozen representation families later generalize best?

The proposal menu includes representation, temporal alignment, neural geometry,
readout, fusion, and generalization experiments. Every candidate must include a
falsification test, changed variables, approved development metrics, payload classes,
compute tier, and expected failure modes.

The first live tournament is a **proposal experiment**, not an efficacy result.
Candidate execution becomes scientific only after a real Algonauts adapter freezes an
`ExperimentPacket`, executes it against the bound evaluator, attaches
`ExperimentEvidence`, and lets `ResearchRegistry` adjudicate the predeclared gates.

## Model identity limitation

The artifact binds the exact API model ID, model-catalog fingerprint, request, prompt,
and response. NVIDIA's hosted API does not give this adapter an independently verified
weight hash for the backend deployment. Therefore model identity is strong enough for
proposal provenance but is not represented as cryptographic proof of immutable model
weights.
