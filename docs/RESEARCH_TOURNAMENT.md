# neurOS Research Tournament Authority

## Purpose

neurOS should be able to support aggressive autonomous research without turning
scientific model selection into an opaque agent conversation.

The research-tournament layer separates two kinds of power:

1. **proposal power** — humans or models can generate hypotheses, implementations,
   ablations, critiques, and candidate representations;
2. **evidence authority** — only a deterministic evaluator operating on a frozen
   experiment packet can decide whether a candidate satisfies predeclared gates.

This architecture is deliberately provider-agnostic. GPT, Fable, Nemotron, DeepSeek,
Kimi, local models, and deterministic search programs can all participate without
becoming part of the scientific trust boundary.

## Why this belongs in neurOS

The immediate proving ground is game-fMRI modelling for the Algonauts program, but
the contracts are modality-agnostic. The same pattern is applicable to EEG decoder
benchmarks, MRI modelling, neural foundation-model comparisons, mechanistic
interpretability studies, and closed-loop BCI development.

The reusable unit is not an agent. It is the **experiment packet**.

```text
DatasetAuthority + EvaluationAuthority
                  |
                  v
             Hypothesis
                  |
                  v
          ExperimentPacket
                  |
        +---------+---------+
        |         |         |
      Agent A   Agent B   Agent C
        |         |         |
        +---------+---------+
                  |
                  v
          ExperimentEvidence
                  |
                  v
           EvidenceArbiter
                  |
          +-------+-------+
          |               |
       reject          promote
                          |
                          v
                     InsightCard
                          |
                          v
                  next isolated round
```

## Non-negotiable scientific rules

### The proposer cannot rewrite the referee

An `ExperimentPacket` binds:

- exact dataset/source fingerprint;
- exact split/evaluation fingerprint;
- declared metric names and evaluation domains;
- optimization boundary;
- forbidden feedback channels;
- agent/model/prompt identity;
- code revision;
- deterministic seeds;
- information regime;
- representation identity when applicable;
- compute budget;
- external-dispatch policy;
- explicit claim ceiling.

Evidence that does not cryptographically bind to that packet is non-evaluable.

### Failures remain evidence

Failed or unavailable executions are explicit `ExperimentEvidence` states. They cannot
carry promoted metric values. A high-performing sibling run does not erase a failed
case.

### Promotion is vector-valued

There is no `winner_score` in the research authority.

A promotion policy can independently require, for example:

- unseen-level Pearson improvement;
- validation stability;
- neural-geometry alignment;
- source revalidation;
- temporal-shift-null survival;
- runtime/storage constraints.

A candidate either satisfies the frozen evidence vector or it does not.

### Cross-pollination is lineage-bound

Agents should first explore independently. Only promoted evidence may be distilled into
an `InsightCard`. The card binds its originating packet, evidence, and promotion
decision. A child experiment that uses an insight declares the parent experiment ID.

This preserves epistemic diversity while making knowledge transfer inspectable.

### External LLMs are outside the data trust boundary

`ExternalDispatchPolicy` records what classes of material may leave the trusted
execution environment. The defaults permit source code, schemas, aggregate metrics,
de-identified plots, and public metadata, while prohibiting raw participant data,
participant identifiers, hidden targets, and credentials.

Dataset terms always remain authoritative. The contract is a fail-safe software
boundary, not permission to transmit restricted data.

## Algonauts proving program

The recommended first adopter is `algonaut-a-mario`.

### A0 — public Yale reproduction

Reproduce the public manifold/BCI methodology before using it as inspiration for
competition modelling. Track T-PHATE as transductive target-observed analysis, not as a
zero-shot predictive representation.

### A1 — public naturalistic fMRI

Use open naturalistic-video fMRI to qualify representation-to-neural-geometry
comparisons under temporal nulls.

### A2 — pseudo-hidden historical Algonauts

Create frozen video-level pseudo-hidden folds from a prior public challenge and test
whether representation, temporal, geometry, and specialist-selection ideas predict
held-out brain responses without leaderboard feedback.

### A3 — CNeuroMod Mario G2

For one authorized subject compare low-cost actions/state, frozen V-JEPA controls,
V-JEPA 2.1 candidates, and the Yale-inspired segment-aware neural-geometry track under
the same unseen-level referee.

### A4 — G3/G4

Only promoted candidates advance to Mario→Shinobi and held-out-subject tests. Cross-game
or held-out-subject truth never participates in representation selection.

### A5 — prospective geometry proxy

Preregister a rule that uses only train/validation neural-geometry alignment to predict
which representation family will transfer best. Then test that rule prospectively on
OOD evidence. If successful, this becomes a scientifically meaningful screening tool,
not merely another diagnostic plot.

## Phased implementation

### v0.1 — authority substrate

Implemented in `neuros-research`:

- immutable research contracts;
- deterministic full SHA-256 identities;
- failure-preserving evidence;
- vector promotion policies;
- pure deterministic arbiter;
- promoted-only insight cards;
- append-only hash-chained evidence ledger;
- external-dispatch policy.

### v0.2 — executor adapters

Next:

- subprocess/local Python executor adapter;
- generic artifact-directory verifier;
- `algonaut-a-mario` evidence adapter;
- deterministic registry CLI;
- experiment-packet JSON schema export.

### v0.3 — agent adapters

Only after v0.2:

- provider-neutral proposal interface;
- OpenAI/Fable/NVIDIA/local adapters as optional integrations;
- token/compute budgets;
- task packets that contain only explicitly approved material;
- independent-round orchestration;
- selective insight-card sharing.

No provider is allowed to become required for the core package.

### v0.4 — research memory

- searchable experiment DAG;
- repeated-failure and duplicated-hypothesis detection;
- complementarity/error-correlation evidence;
- experiment-value estimation based on uncertainty reduction and compute cost;
- prospective stopping rules.

## Adoption criterion

The system is successful if a researcher can remove every LLM integration and still
retain a complete, deterministic, inspectable record of:

- what was proposed;
- what data and feedback it was allowed to observe;
- exactly how it was evaluated;
- what failed;
- what survived;
- why a method was promoted;
- which later experiments inherited which discoveries.

That is the distinction between autonomous experimentation and autonomous science.
