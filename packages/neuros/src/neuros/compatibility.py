"""Evidence-aware ecosystem compatibility inventory for neurOS.

The registry is deliberately conservative: an integration advertises only the
strongest evidence tier currently exercised by this repository. Planned and
indirect integrations remain visible without being presented as qualified.

This module lives in the user-facing ``neuros`` distribution rather than the
kernel. The kernel must never need to know which third-party ecosystems happen
to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class IntegrationStatus(str, Enum):
    """Public support state for an external ecosystem integration."""

    SUPPORTED = "supported"
    EXPERIMENTAL = "experimental"
    INDIRECT = "indirect"
    PLANNED = "planned"


class EvidenceTier(str, Enum):
    """Strongest evidence tier attached to an integration claim."""

    SOFTWARE_CONTRACT = "software-contract"
    INTEGRATION = "integration"
    REAL_DATASET = "real-dataset"
    HARDWARE = "hardware"
    CLOSED_LOOP = "closed-loop"
    CLINICAL = "clinical"


@dataclass(frozen=True, slots=True)
class IntegrationRecord:
    """One machine-readable neurOS ecosystem compatibility statement."""

    integration_id: str
    name: str
    category: str
    status: IntegrationStatus
    capabilities: tuple[str, ...]
    evidence_tier: EvidenceTier | None
    evidence_paths: tuple[str, ...]
    notes: str
    install_hint: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-friendly representation."""

        return {
            "integration_id": self.integration_id,
            "name": self.name,
            "category": self.category,
            "status": self.status.value,
            "capabilities": list(self.capabilities),
            "evidence_tier": self.evidence_tier.value if self.evidence_tier else None,
            "evidence_paths": list(self.evidence_paths),
            "notes": self.notes,
            "install_hint": self.install_hint,
        }


_REGISTRY: tuple[IntegrationRecord, ...] = (
    IntegrationRecord(
        integration_id="brainflow",
        name="BrainFlow",
        category="live acquisition",
        status=IntegrationStatus.SUPPORTED,
        capabilities=("source", "continuous-stream", "device-metadata"),
        evidence_tier=EvidenceTier.SOFTWARE_CONTRACT,
        evidence_paths=("tests/test_brainflow_driver.py", ".github/workflows/drivers-ci.yml"),
        notes=(
            "Board-aware acquisition is fail-closed and contract-tested. No physical "
            "board/firmware/transport combination is yet claimed as hardware-qualified."
        ),
        install_hint='pip install "neuros-drivers[eeg]"',
    ),
    IntegrationRecord(
        integration_id="lsl",
        name="Lab Streaming Layer",
        category="live acquisition + synchronization",
        status=IntegrationStatus.SUPPORTED,
        capabilities=("source", "continuous-stream", "clock-correction"),
        evidence_tier=EvidenceTier.SOFTWARE_CONTRACT,
        evidence_paths=("tests/test_lsl_driver.py", ".github/workflows/drivers-ci.yml"),
        notes=(
            "Continuous regular-rate streams use deterministic discovery and explicit "
            "raw-timestamp plus time-correction semantics. Network timing remains a "
            "deployment-specific qualification concern."
        ),
        install_hint='pip install "neuros-drivers[lsl]"',
    ),
    IntegrationRecord(
        integration_id="mne",
        name="MNE-Python",
        category="offline signal interoperability",
        status=IntegrationStatus.SUPPORTED,
        capabilities=("raw-adapter", "signalframe-bridge", "stream-descriptor"),
        evidence_tier=EvidenceTier.INTEGRATION,
        evidence_paths=("tests/test_mne_interop.py", ".github/workflows/compatibility-ci.yml"),
        notes=(
            "MNE Raw objects can be converted to provenance-rich SignalFrame chunks and "
            "reconstructed from unambiguous sample-by-channel frames. This is an object "
            "interoperability claim, not validation of arbitrary preprocessing pipelines."
        ),
        install_hint='pip install "neuros[interop-mne]"',
    ),
    IntegrationRecord(
        integration_id="nwb",
        name="NWB / PyNWB",
        category="recording interoperability",
        status=IntegrationStatus.SUPPORTED,
        capabilities=("export", "session-provenance"),
        evidence_tier=EvidenceTier.INTEGRATION,
        evidence_paths=(
            "tests/test_recording_exports.py",
            "tests/test_session_archive.py",
            ".github/workflows/ci.yml",
        ),
        notes=(
            "NWB export is exercised in CI. The canonical neurOS archive remains the "
            "lossless replay authority for runtime semantics."
        ),
        install_hint='pip install "neuros[recording]"',
    ),
    IntegrationRecord(
        integration_id="zarr",
        name="Zarr",
        category="recording interoperability",
        status=IntegrationStatus.SUPPORTED,
        capabilities=("export", "session-provenance"),
        evidence_tier=EvidenceTier.INTEGRATION,
        evidence_paths=(
            "tests/test_recording_exports.py",
            "tests/test_session_archive.py",
            ".github/workflows/ci.yml",
        ),
        notes="Zarr export is exercised in the recording interoperability CI lane.",
        install_hint='pip install "neuros[recording]"',
    ),
    IntegrationRecord(
        integration_id="moabb",
        name="MOABB",
        category="offline benchmark",
        status=IntegrationStatus.EXPERIMENTAL,
        capabilities=("dataset-adapter", "longitudinal-authority", "model-ladder"),
        evidence_tier=EvidenceTier.REAL_DATASET,
        evidence_paths=(
            "scripts/evidence/run_moabb_model_ladder.py",
            "docs/LONGITUDINAL_MODEL_LADDER.md",
            ".github/workflows/longitudinal-evidence-study.yml",
        ),
        notes=(
            "The longitudinal evidence program preserves subject/session/run identity and "
            "runs real MOABB datasets under frozen evaluation authority. Experimental "
            "status reflects the still-evolving public benchmark surface, not synthetic data."
        ),
        install_hint='pip install "neuros-foundation[moabb]"',
    ),
    IntegrationRecord(
        integration_id="braindecode",
        name="Braindecode",
        category="decoder ecosystem",
        status=IntegrationStatus.EXPERIMENTAL,
        capabilities=("neural-window", "model-adapter", "training-bridge", "decoder-bridge"),
        evidence_tier=EvidenceTier.INTEGRATION,
        evidence_paths=("tests/test_braindecode_adapter.py", ".github/workflows/braindecode-ci.yml"),
        notes=(
            "Braindecode 1.7 is integrated by delegation for a qualified raw-window "
            "whitelist (EEGNet, EEGConformer, ShallowFBCSPNet, Deep4Net). The adapter "
            "preserves exact window geometry and upstream training identity without hidden "
            "resampling or preprocessing. Real-dataset utility, stable mechanistic hook "
            "paths, and hardware/closed-loop behavior remain separate qualification steps."
        ),
        install_hint='pip install "neuros[braindecode]"',
    ),
    IntegrationRecord(
        integration_id="snap",
        name="SNAP spectral alignment",
        category="representation evidence",
        status=IntegrationStatus.EXPERIMENTAL,
        capabilities=(
            "positive-rank-spectrum",
            "task-power",
            "residual-target-power",
            "null-space-invariant-evidence",
        ),
        evidence_tier=EvidenceTier.SOFTWARE_CONTRACT,
        evidence_paths=(
            "packages/neuros-foundation/src/neuros/foundation_models/spectral_alignment.py",
            "packages/neuros-foundation/tests/test_spectral_alignment.py",
            ".github/workflows/neuroai-ecosystem-ci.yml",
        ),
        notes=(
            "neurOS implements dependency-light SNAP-derived invariant spectral quantities. "
            "Positive-rank modes are explicit while null-space target power is aggregated "
            "because the null eigenbasis is non-unique across valid linear-algebra backends. "
            "This is a numerical-method contract, not a reproduced-paper or biological claim."
        ),
    ),
    IntegrationRecord(
        integration_id="ngclearn",
        name="ngc-learn",
        category="computational neuroscience / NeuroAI",
        status=IntegrationStatus.EXPERIMENTAL,
        capabilities=(
            "rate-cell-transform",
            "predictive-reconstruction",
            "iterative-error-feedback",
            "jax-identity",
            "biological-dynamics-bridge",
        ),
        evidence_tier=EvidenceTier.INTEGRATION,
        evidence_paths=(
            "packages/neuros-foundation/src/neuros/foundation_models/ngclearn_bridge.py",
            "packages/neuros-foundation/src/neuros/foundation_models/ngclearn_predictive_coding.py",
            "packages/neuros-foundation/tests/test_ngclearn_bridge.py",
            "packages/neuros-foundation/tests/test_ngclearn_predictive_coding.py",
            ".github/workflows/neuroai-ecosystem-ci.yml",
        ),
        notes=(
            "Qualified upstream ngc-learn 3.2.x surfaces include RateCell execution and a "
            "fixed-weight predictive reconstruction circuit using real RateCell, "
            "GaussianErrorCell, and StaticSynapse residual-feedback dynamics. The circuit "
            "resets per observation, ties feedback to the generative-weight transpose, and "
            "records reconstruction-error trajectories and artifact identities. Hebbian/STDP "
            "learning, online adaptation, spiking networks, real-data utility, hardware, and "
            "closed-loop behavior remain unqualified until separate evidence lands."
        ),
        install_hint='pip install "neuros-foundation[ngclearn]"',
    ),
    IntegrationRecord(
        integration_id="neuralbench",
        name="Meta NeuralBench",
        category="external benchmark",
        status=IntegrationStatus.PLANNED,
        capabilities=("benchmark-worker", "evidence-extension"),
        evidence_tier=None,
        evidence_paths=(),
        notes=(
            "Planned as an isolated optional benchmark worker so upstream Python/runtime "
            "requirements never become kernel dependencies."
        ),
    ),
    IntegrationRecord(
        integration_id="neuroaikit",
        name="IBM NeuroAIKit",
        category="historical NeuroAI reference",
        status=IntegrationStatus.PLANNED,
        capabilities=("isolated-snu-reference-worker",),
        evidence_tier=None,
        evidence_paths=(),
        notes=(
            "The TensorFlow-era SNU toolkit is scientifically useful as a historical/reference "
            "baseline, but its legacy dependency surface will remain isolated from neurOS core."
        ),
    ),
    IntegrationRecord(
        integration_id="mouse-vision",
        name="NeuroAI Lab mouse-vision",
        category="neural predictivity benchmark",
        status=IntegrationStatus.PLANNED,
        capabilities=("external-model-benchmark", "allen-neural-response-evidence"),
        evidence_tier=None,
        evidence_paths=(),
        notes=(
            "Planned as a cross-species representation/predictivity benchmark. neurOS will "
            "prefer authoritative Allen/public-data identities over adopting research pickles "
            "as a canonical runtime format."
        ),
    ),
    IntegrationRecord(
        integration_id="tdann",
        name="NeuroAI Lab TDANN",
        category="topographic representation benchmark",
        status=IntegrationStatus.PLANNED,
        capabilities=("topographic-representation-evidence",),
        evidence_tier=None,
        evidence_paths=(),
        notes=(
            "Planned as an external representation/topography benchmark, not a runtime "
            "dependency. Licensing and reproducible artifact identity must be resolved before "
            "any implementation code is reused."
        ),
    ),
    IntegrationRecord(
        integration_id="dandi",
        name="DANDI",
        category="public neural data",
        status=IntegrationStatus.PLANNED,
        capabilities=("dataset-discovery", "artifact-provenance"),
        evidence_tier=None,
        evidence_paths=(),
        notes="Dataset identity/provenance integration is planned; no support claim yet.",
    ),
    IntegrationRecord(
        integration_id="spikeinterface",
        name="SpikeInterface",
        category="invasive electrophysiology",
        status=IntegrationStatus.PLANNED,
        capabilities=("recording-adapter", "analyzer-bridge"),
        evidence_tier=None,
        evidence_paths=(),
        notes="Planned invasive/offline interoperability lane.",
    ),
    IntegrationRecord(
        integration_id="py-neuromodulation",
        name="py_neuromodulation",
        category="closed-loop transforms",
        status=IntegrationStatus.PLANNED,
        capabilities=("feature-transform-adapter",),
        evidence_tier=None,
        evidence_paths=(),
        notes="Planned transform adapter; no closed-loop qualification is claimed.",
    ),
    IntegrationRecord(
        integration_id="openbci",
        name="OpenBCI",
        category="reference hardware",
        status=IntegrationStatus.INDIRECT,
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
        status=IntegrationStatus.PLANNED,
        capabilities=("source-adapter", "plugin-bridge"),
        evidence_tier=None,
        evidence_paths=(),
        notes="A first-class qualified source/plugin bridge is not yet implemented.",
    ),
)


def _validate_registry(records: Iterable[IntegrationRecord]) -> tuple[IntegrationRecord, ...]:
    materialized = tuple(records)
    seen: set[str] = set()
    for record in materialized:
        if not record.integration_id or record.integration_id != record.integration_id.lower():
            raise ValueError(f"Invalid integration id: {record.integration_id!r}")
        if record.integration_id in seen:
            raise ValueError(f"Duplicate integration id: {record.integration_id}")
        seen.add(record.integration_id)
        if record.status is IntegrationStatus.PLANNED and record.evidence_tier is not None:
            raise ValueError(f"Planned integration {record.integration_id} cannot claim evidence")
        if record.status is IntegrationStatus.PLANNED and record.evidence_paths:
            raise ValueError(f"Planned integration {record.integration_id} cannot publish evidence paths")
        if record.status is IntegrationStatus.SUPPORTED:
            if record.evidence_tier is None or not record.evidence_paths:
                raise ValueError(
                    f"Supported integration {record.integration_id} requires evidence paths and tier"
                )
        if record.evidence_tier is not None and not record.evidence_paths:
            raise ValueError(f"Evidence-bearing integration {record.integration_id} requires evidence paths")
    return materialized


_REGISTRY = _validate_registry(_REGISTRY)


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


def compatibility_payload(
    integration_id: str | None = None,
    *,
    status: IntegrationStatus | str | None = None,
) -> list[dict[str, object]]:
    """Return filtered JSON-friendly compatibility records for CLI/tooling use."""

    records = _REGISTRY if integration_id is None else (get_integration(integration_id),)
    if status is not None:
        normalized_status = status if isinstance(status, IntegrationStatus) else IntegrationStatus(status)
        records = tuple(record for record in records if record.status is normalized_status)
    return [record.to_dict() for record in records]
