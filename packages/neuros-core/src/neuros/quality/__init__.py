"""Scientific validity, fault injection, provenance, and runtime quality gates."""

from .faults import FaultProfile, PerturbedSource, perturb_frame
from .manifest import BenchmarkManifest, content_hash
from .metrics import QualityGateResult, QualityThresholds, evaluate_runtime_snapshot
from .scientific import (
    FrequencyProbeResult,
    expected_eeg_band,
    frequency_selectivity_probe,
    synthetic_tone,
)

__all__ = [
    "BenchmarkManifest",
    "FaultProfile",
    "FrequencyProbeResult",
    "PerturbedSource",
    "QualityGateResult",
    "QualityThresholds",
    "content_hash",
    "evaluate_runtime_snapshot",
    "expected_eeg_band",
    "frequency_selectivity_probe",
    "perturb_frame",
    "synthetic_tone",
]
