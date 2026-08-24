"""Evidence-aware ecosystem compatibility inventory for neurOS.

The registry is intentionally conservative. A supported integration names the
strongest evidence tier currently present in the repository. Planned and
indirect integrations are visible without being presented as qualified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

IntegrationStatus = Literal["supported", "experimental", "indirect", "planned"]
EvidenceTier = Literal[
    "software-contract",
    "integration",
    "real-dataset",
    "hardware",
    "closed-loop",
    "clinical",
]


@dataclass(frozen=True)
class IntegrationRecord:
    """One public neurOS ecosystem compatibility statement."""

    integration_id: str
    name: str
    category: str
    status: IntegrationStatus
    capabilities: tuple[str, ...]
    evidence_tier: EvidenceTier | None
    evidence_paths: tuple[str, ...]
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_REGISTRY: tuple[IntegrationRecord, ...] = (
    IntegrationRecord(
        integration_id="brainflow",
        name="BrainFlow",
        category="live acquisition",
        status="supported",
        capabilities=("source", "continuous-stream", "device-metadata"),
        evidence_tier="software-contract",
        evidence_paths=(
            "tests/test_brainflow_driver.py",
            ".github/workflows/drivers-ci.yml",
        ),
        notes=(
            "Board-aware acquisition is fail-closed and contract-tested. No physical "
            "board/firmware/transport combination is yet claimed as hardware-qualified."
        ),
    ),
    IntegrationRecord(
        integration_id="lsl",
        name="Lab Streaming Layer",
        category="live acquisition + synchronization",
        status="supported",
        capabilities=("source", "continuous-stream", "clock-correction"),
        evidence_tier="software-contract",
        evidence_paths=(
            "tests/test_lsl_driver.py",
            ".github/workflows/drivers-ci.yml",
        ),
        notes=(
            "Continuous regular-rate streams use deterministic discovery and explicit "
            "raw-timestamp + time-correction semantics. Network/hardware timing remains "
            "deployment-specific evidence."
        ),
    ),
    IntegrationRecord(
        integration_id="nwb",
        name="NWB / PyNWB",
        category="recording interoperability",
        status="supported",
        capabilities=("export", "round-trip-evidence-boundary"),
        evidence_tier="integration",
        evidence_paths=(
            "tests/test_recording_exports.py",
            "tests/test_session_archive.py",
            ".github/workflows/ci.yml",
        ),
        notes="NWB export is exercised in the recording interoperability CI lane.",
    ),
    IntegrationRecord(
        integration_id="zarr",
        name="Zarr",
        category="recording interoperability",
        status="supported",
        capabilities=("export", "round-trip-evidence-boundary"),
        evidence_tier="integration",
        evidence_paths=(
            "tests/test_recording_exports.py",
            "tests/test_session_archive.py",
            ".github/workflows/ci.yml",
        ),
        notes="Zarr export is exercised in the recording interoperability CI lane.",
    ),
    IntegrationRecord(
        integration_id="moabb",
        name="MOABB",
        category="offline benchmark",
        status="experimental",
        capabilities=("dataset-adapter", "longitudinal-authority", "model-ladder"),
        evidence_tier="software-contract",
        evidence_paths=(
            "scripts/evidence/run_moabb_model_ladder.py",
            "packages/neuros-foundation/tests/test_longitudinal_model_ladder.py",
            "docs/LONGITUDINAL_MODEL_LADDER.md",
        ),
        notes=(
            "The benchmark authority and model ladder are contract-tested. Public dataset "
            "runs are intentionally manual and must preserve upstream dataset/split identity."
        ),
    ),
    IntegrationRecord(
        integration_id="mne",
        name="MNE-Python",
        category="offline signal interoperability",
        status="planned",
        capabilities=("signal-object-adapter", "preprocessing-bridge"),
        evidence_tier=None,
        evidence_paths=(),
        notes="A first-class SignalFrame <-> MNE object adapter is not yet qualified.",
    ),
    IntegrationRecord(
        integration_id="braindecode",
        name="Braindecode",
        category="decoder ecosystem",
        status="planned",
        capabilities=("model-adapter", "training-bridge", "decoder-bridge"),
        evidence_tier=None,
        evidence_paths=(),
        notes="Use faithful adapters rather than copying the Braindecode model zoo.",
    ),
    IntegrationRecord(
        integration_id="neuralbench",
        name="Meta NeuralBench",
        category="external benchmark",
        status="planned",
        capabilities=("benchmark-worker", "evidence-extension"),
        evidence_tier=None,
        evidence_paths=(),
        notes=(
            "Planned as an optional isolated benchmark worker so its Python/runtime "
            "requirements do not become kernel dependencies."
        ),
    ),
    IntegrationRecord(
        integration_id="dandi",
        name="DANDI",
        category="public neural data",
        status="planned",
        capabilities=("dataset-discovery", "artifact-provenance"),
        evidence_tier=None,
        evidence_paths=(),
        notes="Dataset identity/provenance integration is planned; no public support claim yet.",
    ),
    IntegrationRecord(
        integration_id="spikeinterface",
        name="SpikeInterface",
        category="invasive electrophysiology",
        status="planned",
        capabilities=("recording-adapter", "analyzer-bridge"),
        evidence_tier=None,
        evidence_paths=(),
        notes="Planned invasive/offline interoperability lane.",
    ),
    IntegrationRecord(
        integration_id="py-neuromodulation",
        name="py_neuromodulation",
        category="closed-loop transforms",
        status="planned",
        capabilities=("feature-transform-adapter",),
        evidence_tier=None,
        evidence_paths=(),
        notes="Planned transform adapter; no closed-loop qualification is claimed.",
    ),
    IntegrationRecord(
        integration_id="openbci",
        name="OpenBCI",
        category="reference hardware",
        status="indirect",
        capabilities=("brainflow-device-family",),
        evidence_tier=None,
        evidence_paths=("packages/neuros-drivers/src/neuros/drivers/brainflow_driver.py",),
        notes=(
            "Reachable through BrainFlow, but no named OpenBCI board/firmware/transport "
            "configuration has completed neurOS hardware qualification."
        ),
    ),
    IntegrationRecord(
        integration_id="open-ephys",
        name="Open Ephys",
        category="invasive acquisition",
        status="planned",
        capabilities=("source-adapter", "plugin-bridge"),
        evidence_tier=None,
        evidence_paths=(),
        notes="A first-class qualified source/plugin bridge is not yet implemented.",
    ),
)


def compatibility_inventory() -> tuple[IntegrationRecord, ...]:
    """Return the immutable, deterministically ordered compatibility registry."""

    return _REGISTRY


def get_integration(integration_id: str) -> IntegrationRecord:
    """Return one integration or raise ``KeyError`` for an unknown identifier."""

    normalized = integration_id.strip().lower()
    for record in _REGISTRY:
        if record.integration_id == normalized:
            return record
    known = ", ".join(item.integration_id for item in _REGISTRY)
    raise KeyError(f"Unknown integration {integration_id!r}. Known integrations: {known}")


def compatibility_payload(integration_id: str | None = None) -> list[dict[str, object]]:
    """Return JSON-friendly compatibility records for the CLI and external tooling."""

    records = _REGISTRY if integration_id is None else (get_integration(integration_id),)
    return [record.to_dict() for record in records]
