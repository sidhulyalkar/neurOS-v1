# Support Policy

neurOS is an open research and engineering platform. Support is organized around reproducible technical surfaces rather than bespoke environment debugging.

## Before asking for help

Please capture:

- the exact neurOS/ORION package version or Git commit;
- Python and operating-system versions;
- relevant optional package versions;
- the smallest configuration or code path that reproduces the problem;
- whether the problem occurs on synthetic/replay data, a public dataset, or physical hardware;
- the relevant `neuros doctor --json`, compatibility, qualification, or evidence output when applicable.

Never attach credentials or identifiable participant data to a public issue.

## Supported question types

Good public issues include:

- installation/build failures on documented Python versions;
- runtime, replay, configuration, plugin, interoperability, or evidence-contract regressions;
- scientifically incorrect implementation behavior with a reproducible counterexample;
- documentation that no longer matches an executable public surface;
- requests for a new integration accompanied by the concrete scientific or operational problem it would solve.

Exploratory research questions are welcome when they can become a benchmark, falsifiable hypothesis, protocol, or clearly scoped design discussion.

## What support does not imply

Maintainer help does not imply:

- clinical advice or medical-device validation;
- guaranteed compatibility with an unqualified device, dataset, model, or vendor SDK;
- a hardware/closed-loop claim based only on software reproduction;
- private consulting or guaranteed response times;
- maintenance of arbitrary historical notebook environments.

## Triage labels

Issues should be triaged by responsibility where possible: runtime, driver, interoperability, ORION, model, evidence, documentation, security, or research. Scientific claims should also identify their strongest accurate evidence tier.

## Integration requests

A proposed integration should explain why neurOS needs the boundary and why an external package cannot remain a user-owned call outside neurOS. Strong candidates provide one or more of:

- a canonical data/runtime conversion;
- a reproducibility or conformance problem;
- an evidence method used across multiple models/datasets;
- a hardware or synchronization boundary;
- a meaningful ORION transfer/adaptation comparison;
- a concrete reduction in duplicated glue code or hidden scientific assumptions.

The project prefers narrow, executable adapters over broad dependency coupling.
