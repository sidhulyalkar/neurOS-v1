"""Fail-closed orchestration authority for promoted Kumar2024 execution.

Public fleet construction derives only from a verified binding, and public worker
ingestion accepts only fully verified persisted worker bundles. Infrastructure
retries remain code-owned and cannot be selected from scientific outcomes.
"""
from __future__ import annotations

from ._kumar2024_promoted_fleet_common import (
    PromotedFleetAuthority,
    PromotedShardLease,
    build_promoted_fleet_authority_from_binding,
)
from ._kumar2024_promoted_fleet_records import (
    PromotedFleetAcceptedResult,
    PromotedFleetAttemptRecord,
)
from ._kumar2024_promoted_fleet_ledger import (
    PromotedFleetLedger,
    record_infrastructure_failure,
    record_verified_worker_bundle,
)
from ._kumar2024_promoted_fleet_assembly import (
    PromotedFleetAssemblyManifest,
    assemble_promoted_fleet,
)

__all__ = [
    "PromotedFleetAcceptedResult",
    "PromotedFleetAssemblyManifest",
    "PromotedFleetAttemptRecord",
    "PromotedFleetAuthority",
    "PromotedFleetLedger",
    "PromotedShardLease",
    "assemble_promoted_fleet",
    "build_promoted_fleet_authority_from_binding",
    "record_infrastructure_failure",
    "record_verified_worker_bundle",
]
