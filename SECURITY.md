# Security Policy

neurOS processes neural data and may sit on paths that eventually control devices. Security reports therefore need the same claim discipline as scientific evidence: exact affected versions, reproducible conditions, and no accidental escalation from a software defect to an unsupported safety claim.

## Supported security surface

Security fixes prioritize the current `main` branch and the most recent tagged public release. Older research snapshots may receive fixes when the affected code is still shared with a supported surface, but are not guaranteed independent backports.

A package version being installable does not mean a historical release receives ongoing security maintenance. Release support is governed by `docs/RELEASE_POLICY.md`.

## Reporting a vulnerability

Do **not** open a public issue containing credentials, private neural/health data, a working exploit against a deployed system, or details that would materially increase risk before a fix is available.

Preferred reporting path:

1. use GitHub private vulnerability reporting / Security Advisories for this repository when available;
2. otherwise contact the repository maintainer privately through GitHub and provide a minimal description sufficient to establish a private channel;
3. include the affected neurOS/ORION package versions or Git commit, operating system, Python version, relevant optional dependencies, configuration, and a minimal reproducer when safe to share.

Do not include real participant data when synthetic or redacted data can reproduce the issue.

## What counts as security-sensitive

Examples include:

- arbitrary code execution or unsafe deserialization;
- path traversal or unintended file disclosure;
- credential/token leakage;
- privilege or authorization bypass in optional service/cloud surfaces;
- integrity failures that allow replay/evidence artifacts to be silently substituted;
- malicious plugin behavior crossing an advertised isolation boundary;
- denial-of-service behavior that defeats documented resource bounds;
- vulnerabilities that could permit unauthorized control of a connected actuator or closed-loop application.

A numerical bug, model-performance regression, or unqualified scientific claim is usually not a security vulnerability unless it creates one of the security consequences above.

## Response principles

Maintainers will attempt to reproduce the issue, establish the affected boundary, develop regression tests, and release the narrowest accurate advisory/fix. We will not promise a disclosure timeline before the impact and remediation path are understood.

Security fixes must not weaken provenance, qualification, replay integrity, or safety-related fail-closed behavior merely to preserve backward compatibility.

## Deployment responsibility

neurOS is an active research and engineering platform, not a medical device or safety-certified control system. Deployers remain responsible for authentication, network boundaries, operating-system hardening, secrets management, participant privacy, independent safety controls, and regulatory obligations appropriate to their use case.
