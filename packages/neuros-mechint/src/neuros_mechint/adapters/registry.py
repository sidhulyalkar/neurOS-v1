"""Lazy ecosystem-adapter discovery without hard optional dependencies."""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IntegrationStatus:
    name: str
    import_name: str
    available: bool
    role: str
    status: str


_OPTIONAL = {
    "orion": (
        "orion",
        "neural token and representation causal audits",
        "integrated",
    ),
    "transformer_lens": (
        "transformer_lens",
        "TransformerLens/TransformerBridge hook-point tracing and interventions",
        "integrated",
    ),
    "nnsight": (
        "nnsight",
        "general model tracing and activation interventions",
        "integrated",
    ),
    "sae_lens": (
        "sae_lens",
        "sparse-autoencoder feature encoding, reconstruction, and interventions",
        "integrated",
    ),
    "circuit_tracer": (
        "circuit_tracer",
        "feature-level attribution graph normalization and candidate extraction",
        "integrated",
    ),
}


def integration_status() -> Iterable[IntegrationStatus]:
    results = []
    for name, (import_name, role, status) in _OPTIONAL.items():
        results.append(
            IntegrationStatus(
                name=name,
                import_name=import_name,
                available=importlib.util.find_spec(import_name) is not None,
                role=role,
                status=status,
            )
        )
    return tuple(results)


def integration_status_dict() -> dict[str, bool]:
    return {item.name: item.available for item in integration_status()}
