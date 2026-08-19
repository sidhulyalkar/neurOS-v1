"""Bridge v0.4 causal maps and v0.6 evidence packs into v0.7 factorial studies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from neuros_mechint.benchmarks.evidence_pack import EvidencePackResult
from neuros_mechint.benchmarks.factorial import (
    FactorialCellOutcome,
    FactorialMechanismReport,
    FactorialMechanismSpec,
    analyze_factorial_mechanisms,
    outcome_from_evidence_pack,
)


@dataclass(frozen=True, slots=True)
class FactorialEvidenceCellInput:
    """One factorial cell backed by a completed held-out evidence pack.

    ``effect_record`` may be a v0.3/v0.4 causal record such as
    ``CausalEffectRecord`` or ``TokenizerEffectRecord``. The bridge only requires
    an ``effect_map`` attribute and optionally validates common context fields.
    """

    cell_id: str
    evidence_pack: EvidencePackResult
    effect_record: Any | None = None

    def __post_init__(self) -> None:
        if not self.cell_id:
            raise ValueError("cell_id must be non-empty")


def _validate_pack_against_cell(
    spec: FactorialMechanismSpec,
    item: FactorialEvidenceCellInput,
) -> None:
    declared = {cell.cell_id: cell for cell in spec.cells}
    if item.cell_id not in declared:
        raise ValueError(f"factorial input references undeclared cell {item.cell_id!r}")
    cell = declared[item.cell_id]
    if not cell.available:
        raise ValueError(f"factorial input supplied for missing cell {item.cell_id!r}")

    pack = item.evidence_pack.spec
    checks = {
        "model_id": (pack.model_id, cell.model_id),
        "model_revision": (pack.model_revision, cell.model_revision),
        "tokenizer_id": (pack.tokenizer_id, cell.tokenizer_id),
        "tokenizer_revision": (pack.tokenizer_revision, cell.tokenizer_revision),
        "dataset_id": (pack.dataset_id, cell.dataset_id),
        "dataset_revision": (pack.dataset_revision, cell.dataset_revision),
        "metric_name": (pack.metric_name, cell.metric_name),
        "discovery_method": (pack.discovery_method, cell.discovery_method),
    }
    mismatches = [
        name for name, (observed, expected) in checks.items() if observed != expected
    ]
    if mismatches:
        raise ValueError(
            f"evidence pack for cell {item.cell_id!r} disagrees on: {mismatches}"
        )
    if set(pack.target_universe) != set(cell.target_universe):
        raise ValueError(
            f"evidence pack target universe does not match cell {item.cell_id!r}"
        )

    metadata_checks = {
        "architecture": cell.architecture,
        "checkpoint": cell.checkpoint,
        "checkpoint_maturity": cell.checkpoint_maturity,
        "discovery_partition_id": cell.discovery_partition_id,
        "session_id": cell.session_id,
        "subject_id": cell.subject_id,
        "training_seed": cell.training_seed,
        "validation_partition_id": cell.validation_partition_id,
    }
    metadata_checks.update(dict(cell.covariates))
    missing_metadata = sorted(name for name in metadata_checks if name not in pack.metadata)
    if missing_metadata:
        raise ValueError(
            f"evidence pack for cell {item.cell_id!r} is missing factorial metadata: "
            f"{missing_metadata}"
        )
    metadata_mismatches = [
        name for name, expected in metadata_checks.items() if pack.metadata[name] != expected
    ]
    if metadata_mismatches:
        raise ValueError(
            f"evidence pack for cell {item.cell_id!r} has mismatched factorial metadata: "
            f"{metadata_mismatches}"
        )


def _effect_map(item: FactorialEvidenceCellInput, *, cell: Any) -> Mapping[str, float]:
    record = item.effect_record
    if record is None:
        return {}
    effects = getattr(record, "effect_map", None)
    if not isinstance(effects, Mapping):
        raise TypeError("effect_record must expose a mapping-valued effect_map")

    context = getattr(record, "context", None)
    if context is not None:
        validations = {
            "architecture": getattr(context, "architecture", None),
            "dataset_id": getattr(context, "dataset_id", None),
            "session_id": getattr(context, "session_id", None),
            "subject_id": getattr(context, "subject_id", None),
            "checkpoint": getattr(context, "checkpoint", None),
        }
        expected = {
            "architecture": cell.architecture,
            "dataset_id": cell.dataset_id,
            "session_id": cell.session_id,
            "subject_id": cell.subject_id,
            "checkpoint": cell.checkpoint,
        }
        mismatches = [
            name
            for name, value in validations.items()
            if value is not None and value != expected[name]
        ]
        if mismatches:
            raise ValueError(
                f"effect record for cell {item.cell_id!r} disagrees on: {mismatches}"
            )
    return {str(key): float(value) for key, value in effects.items()}


def run_factorial_evidence_study(
    spec: FactorialMechanismSpec,
    cells: Sequence[FactorialEvidenceCellInput],
) -> FactorialMechanismReport:
    """Analyze a preregistered factorial study from completed evidence packs."""

    inputs = tuple(cells)
    ids = [item.cell_id for item in inputs]
    if len(ids) != len(set(ids)):
        raise ValueError("factorial evidence cell IDs must be unique")

    declared = {cell.cell_id: cell for cell in spec.cells}
    outcomes: dict[str, FactorialCellOutcome] = {}
    for item in inputs:
        _validate_pack_against_cell(spec, item)
        cell = declared[item.cell_id]
        outcomes[item.cell_id] = outcome_from_evidence_pack(
            item.evidence_pack,
            effect_map=_effect_map(item, cell=cell),
        )
    return analyze_factorial_mechanisms(spec, outcomes)
