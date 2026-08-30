"""Complete-fleet assembly for Kumar2024 FleetAuthority."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .kumar2024_comparison import Kumar2024ComparisonPlan
from .kumar2024_promoted_execution import (
    PromotedExecutionPlan,
    PromotedShardResult,
    _exact_nonnegative_int,
    _identity_sha256,
    _sha256,
    assemble_promoted_execution,
    validate_promoted_shard_result,
)
from ._kumar2024_promoted_fleet_ledger import PromotedFleetLedger
from ._kumar2024_promoted_fleet_records import PromotedFleetAcceptedResult

@dataclass(frozen=True, slots=True)
class PromotedFleetAssemblyManifest:
    """Content-addressed proof that a complete fleet was assembled exactly once."""

    fleet_authority_sha256: str
    fleet_ledger_sha256: str
    execution_plan_sha256: str
    comparison_plan_sha256: str
    accepted_results: tuple[PromotedFleetAcceptedResult, ...]
    execution_assembly_sha256: str
    attempted_records: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("PromotedFleetAssemblyManifest schema_version must be 1")
        for name in (
            "fleet_authority_sha256",
            "fleet_ledger_sha256",
            "execution_plan_sha256",
            "comparison_plan_sha256",
            "execution_assembly_sha256",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        object.__setattr__(
            self,
            "attempted_records",
            _exact_nonnegative_int("attempted_records", self.attempted_records),
        )
        accepted = tuple(sorted(self.accepted_results, key=lambda item: item.shard_id))
        if not accepted:
            raise ValueError("fleet assembly manifest requires accepted worker artifacts")
        if any(not isinstance(item, PromotedFleetAcceptedResult) for item in accepted):
            raise TypeError(
                "accepted_results must contain PromotedFleetAcceptedResult objects"
            )
        ids = [item.shard_id for item in accepted]
        shas = [item.shard_spec_sha256 for item in accepted]
        if len(set(ids)) != len(ids) or len(set(shas)) != len(shas):
            raise ValueError("fleet assembly accepted results must be unique")
        object.__setattr__(self, "accepted_results", accepted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fleet_authority_sha256": self.fleet_authority_sha256,
            "fleet_ledger_sha256": self.fleet_ledger_sha256,
            "execution_plan_sha256": self.execution_plan_sha256,
            "comparison_plan_sha256": self.comparison_plan_sha256,
            "accepted_results": [
                {**item.to_dict(), "accepted_result_sha256": item.sha256}
                for item in self.accepted_results
            ],
            "expected_shards": len(self.accepted_results),
            "attempted_records": self.attempted_records,
            "execution_assembly_sha256": self.execution_assembly_sha256,
            "global_analysis_performed": True,
            "numerical_result_interpretable": False,
            "external_floor_claim_generated": False,
            "orion_comparison_permitted": False,
            "interpretation_boundary": (
                "assembly proves orchestration completeness and delegates statistics "
                "to the preregistered comparison authority; numerical interpretation, "
                "external-floor claims, and ORION comparison require separate audited "
                "claim-authority steps"
            ),
        }

    @property
    def sha256(self) -> str:
        return _identity_sha256(
            "neuros.kumar2024_promoted_fleet_assembly_manifest.v1",
            self.to_dict(),
        )


def assemble_promoted_fleet(
    ledger: PromotedFleetLedger,
    shard_results: Sequence[PromotedShardResult],
    *,
    execution_plan: PromotedExecutionPlan,
    comparison_plan: Kumar2024ComparisonPlan,
) -> dict[str, Any]:
    """Assemble only a complete fleet whose accepted artifact identities match."""

    if not isinstance(ledger, PromotedFleetLedger):
        raise TypeError("ledger must be PromotedFleetLedger")
    if ledger.authority.execution_plan_sha256 != execution_plan.sha256:
        raise ValueError("fleet authority and execution plan differ")
    if execution_plan.template.comparison_plan_sha256 != comparison_plan.sha256:
        raise ValueError("execution plan and comparison plan differ")
    if not ledger.complete:
        raise ValueError(
            "cannot assemble incomplete promoted fleet: "
            f"accepted={len(ledger.accepted_results)}, "
            f"expected={ledger.authority.expected_shards}, "
            f"exhausted={len(ledger.exhausted_without_artifact)}"
        )

    accepted = ledger.accepted_result_map
    observed: dict[str, PromotedShardResult] = {}
    for result in shard_results:
        if not isinstance(result, PromotedShardResult):
            raise TypeError("shard_results must contain PromotedShardResult objects")
        shard_sha = result.shard_spec_sha256
        if shard_sha in observed:
            raise ValueError(f"duplicate fleet shard result for {shard_sha}")
        if shard_sha not in accepted:
            raise ValueError(f"foreign fleet shard result for {shard_sha}")
        validate_promoted_shard_result(
            result,
            execution_plan=execution_plan,
            comparison_plan=comparison_plan,
        )
        if result.sha256 != accepted[shard_sha].shard_result_sha256:
            raise ValueError("fleet shard result content differs from accepted ledger identity")
        observed[shard_sha] = result

    missing = sorted(set(accepted) - set(observed))
    if missing:
        raise ValueError(
            f"fleet assembly missing {len(missing)} accepted shard result(s): {missing[:5]}"
        )

    execution_assembly = assemble_promoted_execution(
        tuple(observed[sha] for sha in sorted(observed)),
        execution_plan=execution_plan,
        comparison_plan=comparison_plan,
    )
    execution_assembly_sha = _identity_sha256(
        "neuros.kumar2024_promoted_execution_assembly.v1",
        execution_assembly,
    )
    manifest = PromotedFleetAssemblyManifest(
        fleet_authority_sha256=ledger.authority.sha256,
        fleet_ledger_sha256=ledger.sha256,
        execution_plan_sha256=execution_plan.sha256,
        comparison_plan_sha256=comparison_plan.sha256,
        accepted_results=ledger.accepted_results,
        execution_assembly_sha256=execution_assembly_sha,
        attempted_records=len(ledger.attempts),
    )
    return {
        "schema_version": 1,
        "fleet_assembly_manifest": {
            **manifest.to_dict(),
            "fleet_assembly_manifest_sha256": manifest.sha256,
        },
        "execution_assembly": execution_assembly,
    }

__all__ = ["PromotedFleetAssemblyManifest", "assemble_promoted_fleet"]
