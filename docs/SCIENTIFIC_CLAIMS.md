# Scientific Claims Policy

neurOS treats scientific and deployment claims as evidence-bearing contracts. The purpose of this policy is not to make language timid. It is to make strong claims expensive in exactly the right way: stronger claims require stronger evidence.

## Evidence ladder

Use the weakest tier that accurately describes the evidence.

| Tier | Meaning | Minimum evidence | Claims explicitly not granted |
| --- | --- | --- | --- |
| **unit** | local implementation behavior | deterministic unit test | upstream compatibility, scientific validity |
| **software contract** | a neurOS operator/adapter obeys defined local numerical or structural semantics | frozen fixtures, typed failures, deterministic identity | real upstream execution, real-data utility |
| **integration** | a real upstream package/object/API crosses the neurOS boundary correctly | pinned/identified upstream execution in CI | task superiority, hardware behavior |
| **replay** | recorded neurOS input reproduces declared execution semantics | verified archive + replay artifact | live hardware performance |
| **scientific synthetic** | a hypothesis survives controlled synthetic falsification tests | frozen generator/protocol + controls | real neural utility |
| **real dataset** | a named public/authorized neural dataset supports a result under frozen evaluation authority | dataset/protocol/model identities + leakage-controlled split | arbitrary population/device generality |
| **hardware** | a named physical device/firmware/transport/host configuration passes measured qualification | physical measurement origin + verified qualification bundle + thresholds | closed-loop or clinical safety |
| **closed loop** | the sensing-to-decision/action loop passes defined end-to-end constraints | measured complete-loop timing/failure/constraint evidence | clinical efficacy/safety |
| **clinical** | a claim is supported by an appropriate human clinical/regulatory evidence process | study/protocol/statistical/regulatory evidence appropriate to the claim | broader claims outside that evidence |

Repository code may expose additional subtiers, but public language should map back to this ladder.

## Claim dimensions are separate

A single scalar score must not silently stand in for all of these:

- task utility;
- calibration cost;
- subject/session/site/device transfer;
- representation geometry;
- robustness to montage/channel/artifact/jitter perturbations;
- uncertainty calibration;
- causal/mechanistic faithfulness;
- latency and resource use;
- data integrity/provenance;
- safety constraints.

For example, high decoding accuracy does not establish subject invariance. Similar representations do not establish predictive utility. Attribution does not establish mechanism. A causal model intervention does not automatically establish a biological mechanism.

## Required identities

A result intended for comparison or promotion should identify, as applicable:

- neurOS/ORION revision or package versions;
- model architecture/checkpoint/training identity;
- dataset and preprocessing identity;
- subject/session/run/site/device split authority;
- calibration/adaptation data authority;
- protocol/configuration identity;
- random seeds or deterministic policy;
- environment/upstream package identity;
- generated evidence/qualification artifact hashes.

Missing identities should reduce the strength of the claim rather than being inferred after the fact.

## Leakage and deployment units

Evaluation splits must match the unit across which the proposed system is expected to generalize. If the real deployment question is a new person, session, site, montage, or device, random trial-level splitting is not sufficient evidence for that claim.

Fit/training, representation learning, source selection, hyperparameter tuning, adaptation/calibration, mechanism discovery, and final evaluation authority should be separated when the method can learn from those partitions.

## Benchmark fairness

Comparisons should preserve the evaluation target while controlling major confounders where practical:

- same train/calibration/evaluation partitions;
- equivalent preprocessing authority;
- matched downstream decoder capacity when representation quality is the target;
- declared compute/parameter/latency differences;
- simple baselines and ablations;
- uncertainty across subjects/sessions/seeds where appropriate.

A more complicated method should earn its complexity through measurable utility.

## Mechanistic language

Use terms carefully:

- **association / correlation** for observational relationships;
- **attribution** for methods assigning importance without an intervention claim;
- **causal effect in the model** when controlled intervention changes model behavior under a defined experiment;
- **mechanistic evidence** when the proposed internal mechanism survives intervention, controls, replication, and held-out tests;
- **biological mechanism** only when empirical neuroscience evidence supports translation from model mechanism to biology.

## Hardware and clinical language

Software CI, simulators, prerecorded datasets, and synthetic devices must never self-promote a hardware claim. Hardware evidence must be physical and bound to the named configuration.

Hardware qualification does not imply closed-loop qualification. Closed-loop qualification does not imply medical-device certification, clinical efficacy, or clinical safety.

## Upstream methods

When neurOS implements a published numerical method, distinguish:

1. method-inspired/local contract tests;
2. numerical conformance against the authors' implementation;
3. reproduction of published experiments;
4. extension to new neural data/tasks.

These are different claims. The repository should say which one is supported.

## Promotion rule

A compatibility matrix, README, API docstring, paper, demo, or product surface may only advertise the strongest tier supported by executable evidence in the repository or an explicitly referenced immutable external artifact.

If evidence and wording disagree, the evidence tier wins.
