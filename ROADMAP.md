# neurOS + ORION Roadmap

The broad platform refactor is complete. The next phase is not package expansion. It is **external scientific proof, product convergence, and progressively stronger qualification**.

See [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) for the current maturity map and [`docs/SCIENTIFIC_CLAIMS.md`](docs/SCIENTIFIC_CLAIMS.md) for evidence-language rules.

## North star

neurOS should become the open qualification and execution layer for neural AI systems:

```text
hardware / public data / replay
          |
          v
       neurOS
runtime + timing + replay + provenance
          |
          +--------------------+
          |                    |
          v                    v
        ORION              external methods
representation +           MNE / Braindecode /
adaptation                 future model plugins
          \                    /
           +--------+---------+
                    v
                    NSQ
 protocol + observation authority + scoring + artifact identity
                    |
                    v
          reproducible evidence bundle
```

The strategic ORION hypothesis is deliberately falsifiable:

> A learned neural representation/adaptation system can preserve or improve held-out neural utility while materially reducing per-user calibration cost, without trading away robustness, latency, uncertainty calibration, provenance, or representation stability.

Until real longitudinal evidence supports that statement, it remains a research hypothesis rather than a product claim.

---

# Phase 0: repository convergence

## Goal

Make the repository communicate one active program rather than every useful idea at once.

## Actions

- keep the four public concepts legible: neurOS, ORION, Evidence/NSQ, Studio;
- keep Arena explicitly inside Evidence;
- close completed bookkeeping issues rather than leaving historical milestones active;
- park historical stacked research PRs behind explicit re-entry issues;
- reduce CI workflow duplication and replace brittle hand-curated test lists with reusable qualification lanes and explicit coverage ownership;
- align package metadata with actual maturity, especially `neuros-ui` and `neuros-cloud`;
- migrate deprecated packaging license metadata before the setuptools deadline tracked in #59;
- protect `main` with repository rules that match exact-head qualification discipline;
- delete obsolete branches only after unique research has been rescued or explicitly rejected.

## Exit gate

A newcomer can understand the active scientific program from the README, project status, open PR list, and open issues without reconstructing repository history.

---

# Phase 1: NSQ Kumar2024 v1

Tracking issue: **#82**.

## Goal

Produce the first externally meaningful result using the production Neural System Qualification runner on a real longitudinal EEG dataset.

## Frozen first comparison

- MOABB Kumar2024 bar-feedback subset;
- participant as the independent inferential unit;
- prospective prior-session history;
- labeled calibration budgets `0, 1, 2, 5, 10` examples/class where the frozen authority permits them;
- untouched final assessment;
- exact preprocessing and dataset lineage;
- MNE/scikit-learn CSP + LDA;
- upstream Braindecode EEGNet;
- upstream EEGConformer where the pinned upstream API/input geometry supports it;
- balanced accuracy primary, with accuracy/AUC/Brier/ECE as semantically available;
- all failures/unavailable/OOM/nonconvergent cases retained.

## Product result

Ship a small immutable study bundle that another researcher can verify without trusting manually curated tables.

The important output is the **calibration frontier**, not a single leaderboard number.

## Exit gate

One command from clean released/built artifacts can reproduce a frozen public slice and independently verify the result identities.

---

# Phase 2: ORION earns its first real-data claim

## Goal

Evaluate ORION only after the external baseline floor is frozen.

## Comparison ladder

1. frozen external classical/deep baselines from Phase 1;
2. matched-capacity task decoder baselines;
3. frozen upstream foundation representations when lineage/licensing permit honest evaluation;
4. ORION tokenizer/representation variants;
5. SourceWeigher only as an explicitly tested transfer strategy;
6. governed adaptation only after target observation roles are frozen independently.

## Required axes

- performance vs labeled calibration cost;
- separately declared unlabeled target cost;
- cross-session and cross-subject transfer;
- montage/channel-drop robustness;
- artifact and preprocessing sensitivity;
- representation geometry and deployment-unit leakage;
- uncertainty calibration;
- model/artifact identity and pretraining-overlap verdict;
- latency/resource cost;
- failure rate.

## Scientific rule

Matched downstream capacity and identical observation authority are mandatory. A representation may not choose a friendlier split or adaptation budget than a competing method.

## Exit gate

Either:

- ORION demonstrates a reproducible advantage on a predeclared deployment-relevant metric, or
- the experiment clearly falsifies the current ORION hypothesis and tells us what to change.

Both outcomes are valuable.

---

# Phase 3: adaptive NSQ authority

Tracking issue: **#81**.

## Goal

Freeze target observations into independent scientific roles before enabling unlabeled adaptation claims.

Required pairwise-disjoint roles:

```text
source history
labeled target calibration
unlabeled target adaptation
qualification / state selection
untouched final assessment
```

Every role must have exact indices and a content-bound SHA. Labeled and unlabeled budgets must be independent. “Whatever target rows are left” is not an unlabeled adaptation protocol.

## Exit gate

The NSQ runner can execute an external adaptive method while proving that no target observation silently changes scientific role across the calibration frontier.

---

# Phase 4: real hardware qualification

## Goal

Qualify one named EEG hardware/firmware/transport/host combination end to end instead of expanding a generic driver catalog.

## Measure

- channel/montage identity;
- sample-rate behavior;
- device and host clocks;
- packet/sequence loss;
- reconnect/restart behavior;
- synchronization uncertainty and drift;
- queue pressure/drop behavior;
- sustained recording integrity;
- source-to-host and end-to-end latency;
- exact configuration/software identities.

Runtime artifact binding / descriptor propagation in **#74** should be completed before artifact-backed execution is treated as fully compositional through transforms.

## Exit gate

A public machine-readable qualification bundle supports the exact named hardware configuration. No blanket vendor or device-family claim is inferred.

---

# Phase 5: external adoption and methods-paper readiness

Tracking issue: **#76**.

## Goal

Turn internal rigor into external trust.

## Required proof

- at least three independent users reproduce the small NSQ task from public instructions and built artifacts;
- at least one external model participates without neurOS training code;
- at least one genuinely external plugin/repository passes the conformance path;
- at least one substantive external contribution lands;
- reproducibility failures are published rather than hidden;
- a concise methods paper explains the qualification problem and state of the field instead of presenting package count as novelty.

## Exit gate

An external researcher can reasonably say:

> I do not need neurOS to train my model. I use it because it makes my neural-system claim harder to accidentally overstate and easier to reproduce.

---

# Phase 6: resume Arena only against concrete falsification targets

Tracking issue: **#83**.

The parked Arena v2 stack contains valuable causal work around sample-indexed artifacts, participant response, presentation epochs, and measured display evidence. Preserve it, but do not let synthetic sophistication outrun the real-data NSQ program.

Resume when either:

- the first NSQ real-data baseline is frozen and Arena perturbations can test a concrete robustness hypothesis, or
- a physical display/device experiment needs the measured-evidence boundary.

When resumed, reconstruct the semantics cleanly on current `main` rather than carrying stale stacked ancestry forward.

---

# Phase 7: Studio and closed-loop safety

These are downstream product surfaces, not current proof priorities.

## Studio

Studio should visualize runtime state, recordings/replay, NSQ results, ORION representations/adaptation, artifact lineage, latency/quality, and claim/evidence status through existing APIs. It must not become a second executor.

## Closed-loop safety

Before consequential actuation, add first-class policy for:

- stale-data rejection;
- signal-quality/confidence gates;
- action bounds/rate limits;
- deadman/fallback states;
- emergency stop;
- explicit advisory-vs-actuation semantics;
- hardware-in-the-loop fault qualification.

Software tests do not constitute medical-device validation.

---

# What we should not build now

Do not spend the next cycle on:

- another generic EEG preprocessing stack;
- another broad decoder/model zoo;
- a competing BIDS/NWB/MOABB catalog;
- dozens of device drivers already covered upstream;
- another foundation-model architecture without a frozen benchmark and trained artifact;
- UI/cloud polish that outruns evidence;
- synthetic physiological complexity with no real-data falsification target;
- opaque AutoML or adaptation that weakens information-role authority.

Every new subsystem should satisfy the scope-firewall logic in #77: identify the established ecosystem owner, explain why integration is insufficient, state neurOS's unique authority, define an external falsification target, quantify maintenance cost, and name the removal condition.

---

# Company/product direction

The open-source wedge should remain the trusted local execution + qualification standard. Commercial value should accumulate above stable open contracts rather than by hiding scientific semantics:

- team/organization artifact registry;
- qualification history and comparison service;
- fleet/device compatibility evidence;
- reproducibility and audit workflows;
- collaborative Studio;
- managed benchmark execution;
- private deployment governance;
- regulated/enterprise support when justified.

That creates a credible business path without making the scientific core less inspectable.

---

# Next three moves

1. **Finish #82 and publish the first frozen NSQ Kumar2024 baseline artifact.**
2. **Run ORION against that exact authority and find out whether it actually reduces calibration burden.**
3. **Qualify one real EEG system and recruit external users to reproduce the same evidence workflow.**

Everything else should be judged by whether it accelerates one of those three moves.
