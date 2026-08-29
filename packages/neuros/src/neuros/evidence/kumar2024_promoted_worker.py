"""Artifact-consuming worker authority for the promoted Kumar2024 study.

A promoted worker accepts only a sealed binding bundle and one content-addressed
``shard_spec_sha256``. Participant/session/split/model-seed/budget choices are
restored from the binding rather than accepted as scheduler-owned flags.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import kumar2024 as base
from .kumar2024_comparison import Kumar2024ComparisonPlan, promoted_external_floor_plan
from .kumar2024_materialization import (
    build_case_result_observation_roles,
    build_processed_subject_shard,
)
from .kumar2024_materialized_study import _runtime_authority
from .kumar2024_promoted_binding import (
    _binding_from_payload,
    _method_spec_from_payload,
    _template_from_payload,
    promoted_materialization_config,
    verify_promoted_binding_bundle,
)
from .kumar2024_promoted_execution import (
    PromotedExecutionPlan,
    PromotedExecutionShardSpec,
    PromotedShardResult,
    validate_promoted_shard_result,
)

PROMOTED_WORKER_BUNDLE_FILES = (
    "worker_manifest.json",
    "case_result.json",
    "observation_roles.json",
    "shard_result.json",
)


@dataclass(frozen=True, slots=True)
class PromotedWorkerAssignment:
    binding_root: Path
    binding_bundle_sha256: str
    comparison_plan: Kumar2024ComparisonPlan
    execution_plan: PromotedExecutionPlan
    shard: PromotedExecutionShardSpec
    case_authority: Any
    archived_method_spec: Any
    manifest: Mapping[str, Any]
    materialization: Mapping[str, Any]
    archived_processed_shard: Mapping[str, Any]
    archived_raw_selections: tuple[Mapping[str, Any], ...]
    archived_raw_files: tuple[Mapping[str, Any], ...]

    @property
    def expected_method_spec_sha256(self) -> str:
        return self.execution_plan.expected_method_spec_sha256(self.shard)

    @property
    def expected_case_authority_sha256(self) -> str:
        return self.execution_plan.expected_case_authority_sha256(self.shard)


def _load_json(root: Path, name: str) -> dict[str, Any]:
    payload = json.loads((root / name).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _execution_plan(payload: Mapping[str, Any]) -> PromotedExecutionPlan:
    plan = PromotedExecutionPlan(
        template=_template_from_payload(payload["template"]),
        binding=_binding_from_payload(payload["binding"]),
    )
    if payload.get("execution_plan_sha256") != plan.sha256:
        raise ValueError("serialized promoted execution-plan SHA mismatch")
    return plan


def _protocol_from_manifest(manifest: Mapping[str, Any]):
    from neuros.foundation_models.qualification import QualificationProtocolSpec

    raw = dict(manifest["protocol"])
    declared = raw.pop("protocol_sha256", None)
    protocol = QualificationProtocolSpec(
        protocol_id=raw["protocol_id"],
        dataset_id=raw["dataset_id"],
        dataset_lineage_sha256=raw["dataset_lineage_sha256"],
        task_id=raw["task_id"],
        independent_unit=raw["independent_unit"],
        grouping_hierarchy=tuple(raw["grouping_hierarchy"]),
        calibration_budgets_per_class=tuple(raw["calibration_budgets_per_class"]),
        primary_metric=raw.get("primary_metric", "balanced_accuracy"),
        secondary_metrics=tuple(raw.get("secondary_metrics", ())),
        metric_scorecard_sha256=raw.get("metric_scorecard_sha256"),
        robustness_axes=tuple(raw.get("robustness_axes", ())),
        final_assessment_role=raw.get("final_assessment_role", "untouched_final_assessment"),
        protocol_status=raw.get("protocol_status", "draft"),
        metadata=raw.get("metadata", {}),
        schema_version=raw.get("schema_version", 1),
    )
    if declared is not None and declared != protocol.sha256:
        raise ValueError("embedded protocol SHA is inconsistent")
    if protocol.sha256 != manifest["protocol_sha256"]:
        raise ValueError("binding protocol payload differs from protocol SHA")
    return protocol


def _preprocessing_from_manifest(manifest: Mapping[str, Any]) -> base.Kumar2024PreprocessingSpec:
    fixed = manifest["preprocessing_authority"]["fixed_preprocessing"]
    return base.Kumar2024PreprocessingSpec(
        fmin_hz=fixed["fmin_hz"],
        fmax_hz=fixed["fmax_hz"],
        resample_hz=fixed.get("resample_hz"),
        additional_normalization=fixed.get("additional_normalization", "none"),
        return_epochs=fixed.get("return_epochs", True),
    )


def _subject_raw_archive(materialization: Mapping[str, Any], subject: int):
    raw_selection = materialization["raw_selection"]
    selections = tuple(
        sorted(
            (
                item
                for item in raw_selection["selections"]
                if int(item["subject"]) == int(subject)
            ),
            key=lambda item: str(item["logical_path"]),
        )
    )
    if not selections:
        raise ValueError(f"binding contains no raw selections for subject {subject}")
    logical_paths = {str(item["logical_path"]) for item in selections}
    files = tuple(
        sorted(
            (
                item
                for item in materialization["authority"]["raw_materialization"]["files"]
                if str(item["logical_path"]) in logical_paths
            ),
            key=lambda item: str(item["logical_path"]),
        )
    )
    if {str(item["logical_path"]) for item in files} != logical_paths:
        raise ValueError("binding raw selections and raw file authority disagree")
    return selections, files


def load_promoted_worker_assignment(
    binding_root: str | Path,
    shard_spec_sha256: str,
) -> PromotedWorkerAssignment:
    """Select archived authorities for one worker without loading data or a model."""

    root = Path(binding_root).resolve()
    verified = verify_promoted_binding_bundle(root)
    manifest = _load_json(root, "binding_manifest.json")
    materialization = _load_json(root, "materialization.json")
    execution = _execution_plan(_load_json(root, "execution_plan.json"))
    if execution.sha256 != verified["execution_plan_sha256"]:
        raise ValueError("verified binding and execution plan differ")
    shard = execution.shard_by_sha256.get(str(shard_spec_sha256))
    if shard is None:
        raise ValueError("requested shard_spec_sha256 is not present in the promoted binding")

    cases = _load_json(root, "case_authorities.json")["authorities"]
    matched = [
        raw
        for raw in cases
        if int(raw["case_metadata"]["subject"]) == shard.subject
        and str(raw["case_metadata"]["held_out_session"]) == shard.target_session
        and int(raw["case_metadata"]["split_seed"]) == shard.split_seed
    ]
    if len(matched) != 1:
        raise ValueError("binding must contain exactly one case authority for worker shard")
    from neuros.foundation_models.longitudinal_authority import LongitudinalCaseAuthority

    authority = LongitudinalCaseAuthority.from_dict(matched[0])
    if authority.authority_sha256 != execution.expected_case_authority_sha256(shard):
        raise ValueError("selected case authority differs from execution binding")

    records = _load_json(root, "method_specs.json")["method_specs"]
    methods = [
        item for item in records if str(item["realization_key"]) == shard.method_realization_key
    ]
    if len(methods) != 1:
        raise ValueError("binding must contain exactly one method spec for worker realization")
    method_payload = dict(methods[0]["method_spec"])
    declared_method_sha = method_payload.pop("method_spec_sha256")
    method_spec = _method_spec_from_payload(method_payload)
    if method_spec.sha256 != declared_method_sha:
        raise ValueError("archived worker method-spec payload does not match its SHA")
    if method_spec.sha256 != execution.expected_method_spec_sha256(shard):
        raise ValueError("selected method spec differs from execution binding")

    processed = [
        item
        for item in materialization["authority"]["processed_shards"]
        if str(item["shard_id"]) == f"subject={shard.subject}"
    ]
    if len(processed) != 1:
        raise ValueError("binding must contain exactly one processed shard for worker subject")
    raw_selections, raw_files = _subject_raw_archive(materialization, shard.subject)
    return PromotedWorkerAssignment(
        binding_root=root,
        binding_bundle_sha256=str(verified["bundle_sha256"]),
        comparison_plan=promoted_external_floor_plan(),
        execution_plan=execution,
        shard=shard,
        case_authority=authority,
        archived_method_spec=method_spec,
        manifest=manifest,
        materialization=materialization,
        archived_processed_shard=processed[0],
        archived_raw_selections=raw_selections,
        archived_raw_files=raw_files,
    )


def _factory_for_assignment(assignment: PromotedWorkerAssignment, sample_rate_hz: float):
    from neuros.foundation_models.qualification_baselines import (
        MNECSPLDAFactory,
        RiemannianTangentLogRegFactory,
        UpstreamBraindecodeFactory,
    )

    config = promoted_materialization_config(assignment.comparison_plan)
    shard = assignment.shard
    if shard.method_id == "mne-csp-lda":
        factory = MNECSPLDAFactory(n_components=config.csp_components)
    elif shard.method_id == "pyriemann-rg-lr":
        factory = RiemannianTangentLogRegFactory()
    elif shard.method_id == "braindecode-eegnet":
        if shard.model_seed is None:
            raise ValueError("promoted EEGNet worker requires an archived model seed")
        factory = UpstreamBraindecodeFactory(
            model_name="EEGNet",
            sample_rate_hz=float(sample_rate_hz),
            optimizer_name=config.braindecode_optimizer,
            learning_rate=config.braindecode_learning_rate,
            weight_decay=config.braindecode_weight_decay,
            n_epochs=config.braindecode_epochs,
            batch_size=config.braindecode_batch_size,
            device=config.device,
            random_state=int(shard.model_seed),
            validation_fraction=config.braindecode_validation_fraction,
            validation_seed=config.braindecode_validation_seed,
            early_stopping_patience=config.braindecode_early_stopping_patience,
            early_stopping_threshold=0.0,
            restore_best=True,
            source_reference=(
                "Braindecode maintained MOABB cross-session EEGNet training family; "
                "Kumar2024 preprocessing remains the shared NSQ authority"
            ),
        )
    else:
        raise ValueError(f"unsupported promoted worker method {shard.method_id!r}")
    if factory.method_spec.sha256 != assignment.expected_method_spec_sha256:
        raise RuntimeError("current decoder does not reproduce the archived method-spec authority")
    if factory.method_spec.to_dict() != assignment.archived_method_spec.to_dict():
        raise RuntimeError("current decoder method-spec payload differs from archived binding")
    return factory


def _assert_runtime_authority(assignment: PromotedWorkerAssignment):
    revision = base._git_revision()
    expected_revision = assignment.execution_plan.binding.source_revision
    if revision is None:
        raise RuntimeError("promoted worker requires an exact Git checkout")
    if revision != expected_revision:
        raise RuntimeError(
            f"promoted worker source revision differs: expected={expected_revision}, observed={revision}"
        )
    environment = _runtime_authority(promoted_materialization_config(assignment.comparison_plan))
    expected_environment = assignment.execution_plan.binding.environment_authority_sha256
    if environment.sha256 != expected_environment:
        raise RuntimeError(
            "promoted worker environment authority differs from archived execution binding: "
            f"expected={expected_environment}, observed={environment.sha256}"
        )
    return environment


def _verify_subject_raw_materialization(assignment: PromotedWorkerAssignment, raw_evidence: Any):
    if raw_evidence.loader_contract != assignment.materialization["raw_selection"]["loader_contract"]:
        raise RuntimeError("worker raw loader contract differs from archived binding")
    current_selections = tuple(
        sorted((item.to_dict() for item in raw_evidence.selections), key=lambda x: x["logical_path"])
    )
    current_files = tuple(
        sorted((item.to_dict() for item in raw_evidence.authority.files), key=lambda x: x["logical_path"])
    )
    if current_selections != assignment.archived_raw_selections:
        raise RuntimeError("worker raw subject selection differs from archived binding")
    if current_files != assignment.archived_raw_files:
        raise RuntimeError("worker consumed raw subject bytes differ from archived binding")


def _prepare_output(output: str | Path, overwrite: bool) -> Path:
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    managed = [root / name for name in (*PROMOTED_WORKER_BUNDLE_FILES, "artifact_hashes.json")]
    existing = [path for path in managed if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite promoted worker evidence: "
            + ", ".join(path.name for path in existing)
        )
    return root


def _seal_worker_bundle(output: Path) -> dict[str, Any]:
    files = {name: base._file_sha256(output / name) for name in PROMOTED_WORKER_BUNDLE_FILES}
    root = base._identity_sha256(
        "neuros.nsq_kumar2024_promoted_worker_bundle.v1", {"files": files}
    )
    payload = {"schema_version": 1, "files": files, "bundle_sha256": root}
    base._json_dump(output / "artifact_hashes.json", payload)
    return payload


def _shard_result_from_payload(payload: Mapping[str, Any]) -> PromotedShardResult:
    raw = dict(payload)
    declared = raw.pop("shard_result_sha256", None)
    result = PromotedShardResult(
        execution_plan_sha256=raw["execution_plan_sha256"],
        shard_spec_sha256=raw["shard_spec_sha256"],
        comparison_plan_sha256=raw["comparison_plan_sha256"],
        study_materialization_sha256=raw["study_materialization_sha256"],
        environment_authority_sha256=raw["environment_authority_sha256"],
        raw_materialization_sha256=raw["raw_materialization_sha256"],
        dataset_lineage_sha256=raw["dataset_lineage_sha256"],
        protocol_sha256=raw["protocol_sha256"],
        preprocessing_authority_sha256=raw["preprocessing_authority_sha256"],
        case_authority_sha256=raw["case_authority_sha256"],
        method_spec_sha256=raw["method_spec_sha256"],
        rows=tuple(raw["rows"]),
        schema_version=raw.get("schema_version", 1),
    )
    if declared is not None and declared != result.sha256:
        raise ValueError("serialized promoted shard-result SHA mismatch")
    return result


def _verify_serialized_qualification_result(payload: Mapping[str, Any]) -> str:
    """Recompute every nested NSQ v3 identity from serialized rich evidence."""

    result_payload = dict(payload)
    declared_result_sha = result_payload.pop("result_sha256", None)
    if not isinstance(declared_result_sha, str):
        raise ValueError("serialized NSQ case result is missing result_sha256")
    rows = result_payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("serialized NSQ case result must contain budget rows")
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError("serialized NSQ budget result must be a mapping")
        row_payload = dict(raw_row)
        declared_row_sha = row_payload.pop("result_sha256", None)
        if not isinstance(declared_row_sha, str):
            raise ValueError("serialized NSQ budget result is missing result_sha256")
        expected_row_sha = base._identity_sha256(
            "neuros.qualification_budget_result.v3", row_payload
        )
        if declared_row_sha != expected_row_sha:
            raise ValueError("serialized NSQ budget-result SHA mismatch")
    expected_result_sha = base._identity_sha256(
        "neuros.qualification_case_result.v3", result_payload
    )
    if declared_result_sha != expected_result_sha:
        raise ValueError("serialized NSQ case-result SHA mismatch")
    return declared_result_sha


def verify_promoted_worker_bundle(output: str | Path, *, binding_root: str | Path):
    root = Path(output).resolve()
    hashes = _load_json(root, "artifact_hashes.json")
    if hashes.get("schema_version") != 1 or set(hashes.get("files", {})) != set(
        PROMOTED_WORKER_BUNDLE_FILES
    ):
        raise ValueError("invalid promoted worker hash manifest")
    actual = {name: base._file_sha256(root / name) for name in PROMOTED_WORKER_BUNDLE_FILES}
    if actual != hashes["files"]:
        raise ValueError("promoted worker bundle file hashes differ from sealed manifest")
    bundle_sha = base._identity_sha256(
        "neuros.nsq_kumar2024_promoted_worker_bundle.v1", {"files": actual}
    )
    if hashes.get("bundle_sha256") != bundle_sha:
        raise ValueError("promoted worker bundle root SHA mismatch")

    manifest = _load_json(root, "worker_manifest.json")
    shard_result = _shard_result_from_payload(_load_json(root, "shard_result.json"))
    assignment = load_promoted_worker_assignment(binding_root, shard_result.shard_spec_sha256)
    if manifest.get("binding_bundle_sha256") != assignment.binding_bundle_sha256:
        raise ValueError("worker manifest names a different promoted binding bundle")
    if manifest.get("source_revision") != assignment.execution_plan.binding.source_revision:
        raise ValueError("worker manifest source revision differs from execution binding")
    if manifest.get("execution_plan_sha256") != assignment.execution_plan.sha256:
        raise ValueError("worker manifest execution plan differs from assignment")
    if manifest.get("shard_spec_sha256") != assignment.shard.sha256:
        raise ValueError("worker manifest shard identity differs from assignment")
    rows = validate_promoted_shard_result(
        shard_result,
        execution_plan=assignment.execution_plan,
        comparison_plan=assignment.comparison_plan,
    )

    case_payload = _load_json(root, "case_result.json")
    expected_case_metadata = {
        "subject": assignment.shard.subject,
        "held_out_session": assignment.shard.target_session,
        "split_seed": assignment.shard.split_seed,
        "model_seed": assignment.shard.model_seed,
        "method_realization_key": assignment.shard.method_realization_key,
    }
    for key, expected in expected_case_metadata.items():
        if case_payload.get(key) != expected:
            raise ValueError(f"worker case-result {key} differs from shard assignment")
    result_payload = case_payload.get("result")
    if not isinstance(result_payload, Mapping):
        raise ValueError("worker case-result is missing the NSQ result object")
    verified_result_sha = _verify_serialized_qualification_result(result_payload)
    if manifest.get("result_sha256") != verified_result_sha:
        raise ValueError("worker manifest and case-result SHA differ")
    for name, expected in (
        ("protocol_sha256", shard_result.protocol_sha256),
        ("case_authority_sha256", shard_result.case_authority_sha256),
        ("method_spec_sha256", shard_result.method_spec_sha256),
    ):
        if result_payload.get(name) != expected:
            raise ValueError(f"worker case-result {name} differs from shard envelope")
    nsq_rows = result_payload.get("rows")
    if not isinstance(nsq_rows, list) or len(nsq_rows) != len(rows):
        raise ValueError("worker case-result rows differ from promoted frontier cardinality")
    nsq_by_budget = {int(item["calibration_per_class"]): item for item in nsq_rows}
    flat_by_budget = {int(item["calibration_per_class"]): item for item in rows}
    if set(nsq_by_budget) != set(assignment.shard.budgets_per_class):
        raise ValueError("worker case-result does not preserve every calibration budget")
    for budget, raw_row in nsq_by_budget.items():
        flat = flat_by_budget[budget]
        if raw_row.get("result_sha256") != flat.get("qualification_result_row_sha256"):
            raise ValueError("worker flattened row does not bind the NSQ budget-result SHA")
        if raw_row.get("status") != flat.get("status"):
            raise ValueError("worker flattened row status differs from NSQ budget result")
        if raw_row.get("case_authority_sha256") != flat.get("case_authority_sha256"):
            raise ValueError("worker flattened row case authority differs from NSQ budget result")
        if raw_row.get("method_spec_sha256") != shard_result.method_spec_sha256:
            raise ValueError("worker NSQ budget result method identity differs from shard envelope")
        score = raw_row.get("score")
        expected_balanced = None if score is None else score["metrics"]["balanced_accuracy"]
        if flat.get("balanced_accuracy") != expected_balanced:
            raise ValueError("worker flattened balanced accuracy differs from NSQ budget result")

    role_rows = _load_json(root, "observation_roles.json").get("rows")
    if not isinstance(role_rows, list) or len(role_rows) != len(rows):
        raise ValueError("worker observation-role rows differ from promoted frontier cardinality")
    role_by_budget = {int(item["calibration_per_class"]): item for item in role_rows}
    if set(role_by_budget) != set(assignment.shard.budgets_per_class):
        raise ValueError("worker observation roles do not cover every calibration budget")
    for row in rows:
        budget = int(row["calibration_per_class"])
        if role_by_budget[budget]["qualification_result_row_sha256"] != row.get(
            "qualification_result_row_sha256"
        ):
            raise ValueError("worker observation-role row SHA differs from flattened NSQ row")
    return {
        "verified": True,
        "schema_version": 1,
        "worker_bundle_sha256": bundle_sha,
        "binding_bundle_sha256": assignment.binding_bundle_sha256,
        "execution_plan_sha256": assignment.execution_plan.sha256,
        "shard_spec_sha256": assignment.shard.sha256,
        "shard_result_sha256": shard_result.sha256,
        "attempted_budgets": list(assignment.shard.budgets_per_class),
        "statuses": [str(row["status"]) for row in rows],
        "files": actual,
    }


def run_promoted_worker(
    binding_root: str | Path,
    shard_spec_sha256: str,
    output: str | Path,
    *,
    overwrite: bool = False,
):
    """Execute exactly one archived promoted shard and seal its evidence bundle."""

    from neuros.foundation_models.longitudinal import ordered_group_values
    from neuros.foundation_models.moabb_epochs import collect_moabb_epochs
    from neuros.foundation_models.moabb_longitudinal import (
        build_moabb_longitudinal_dataset,
        validate_observed_sessions,
    )
    from neuros.foundation_models.moabb_materialization import resolve_kumar2024_raw_materialization
    from neuros.foundation_models.qualification_runner import (
        QualificationExecutionContext,
        run_external_qualification_case,
    )

    assignment = load_promoted_worker_assignment(binding_root, shard_spec_sha256)
    environment = _assert_runtime_authority(assignment)
    root = _prepare_output(output, overwrite)
    preprocessing = _preprocessing_from_manifest(assignment.manifest)
    dataset_spec, dataset, paradigm = build_moabb_longitudinal_dataset(
        base.KUMAR2024_DATASET_KEY,
        fmin=preprocessing.fmin_hz,
        fmax=preprocessing.fmax_hz,
        resample=preprocessing.resample_hz,
    )
    raw_evidence = resolve_kumar2024_raw_materialization(
        dataset, subjects=[assignment.shard.subject]
    )
    _verify_subject_raw_materialization(assignment, raw_evidence)
    data, descriptor = collect_moabb_epochs(
        dataset,
        paradigm,
        subjects=[assignment.shard.subject],
        dataset_id=base.KUMAR2024_DATASET_ID,
    )
    archived_descriptor = assignment.manifest["subject_epoch_descriptors"][
        str(assignment.shard.subject)
    ]
    if descriptor.sha256 != archived_descriptor["descriptor_sha256"]:
        raise RuntimeError("worker MOABB epoch descriptor differs from archived binding")
    observed = validate_observed_sessions(
        dataset_spec, ordered_group_values(data, split_unit="session")
    )
    if observed != base.KUMAR2024_EXPECTED_SESSIONS:
        raise RuntimeError("worker Kumar2024 chronology differs from binding authority")

    preprocessing_sha = assignment.execution_plan.binding.preprocessing_authority_sha256
    current_processed = build_processed_subject_shard(
        data,
        subject=assignment.shard.subject,
        preprocessing_authority_sha256=preprocessing_sha,
    )
    if current_processed.to_dict() != assignment.archived_processed_shard:
        raise RuntimeError("worker processed participant shard differs from archived binding")
    assignment.case_authority.restore(data)
    protocol = _protocol_from_manifest(assignment.manifest)
    if protocol.sha256 != assignment.execution_plan.binding.protocol_sha256:
        raise RuntimeError("worker protocol differs from execution binding")
    factory = _factory_for_assignment(assignment, descriptor.sampling_rate_hz)
    context = QualificationExecutionContext(
        observed_dataset_lineage_sha256=assignment.execution_plan.binding.dataset_lineage_sha256,
        preprocessing_authority_sha256s=(preprocessing_sha,),
        metadata={
            "binding_bundle_sha256": assignment.binding_bundle_sha256,
            "execution_plan_sha256": assignment.execution_plan.sha256,
            "shard_spec_sha256": assignment.shard.sha256,
            "source_revision": assignment.execution_plan.binding.source_revision,
            "study_materialization_sha256": assignment.execution_plan.binding.study_materialization_sha256,
            "raw_materialization_sha256": assignment.execution_plan.binding.raw_materialization_sha256,
            "processed_shard_sha256": current_processed.sha256,
            "worker_scope": "one_atomic_promoted_frontier",
        },
    )
    result = run_external_qualification_case(
        data,
        assignment.case_authority,
        protocol,
        factory,
        execution_context=context,
    )
    flat_rows = []
    for row in result.rows:
        flat = base._flatten_result_row(row, assignment.case_authority)
        flat.update(
            {
                "split_seed": assignment.shard.split_seed,
                "model_seed": assignment.shard.model_seed,
                "method_realization_key": assignment.shard.method_realization_key,
                "qualification_result_row_sha256": row.sha256,
                "shard_spec_sha256": assignment.shard.sha256,
                "execution_plan_sha256": assignment.execution_plan.sha256,
                "binding_bundle_sha256": assignment.binding_bundle_sha256,
            }
        )
        flat_rows.append(flat)

    binding = assignment.execution_plan.binding
    shard_result = PromotedShardResult(
        execution_plan_sha256=assignment.execution_plan.sha256,
        shard_spec_sha256=assignment.shard.sha256,
        comparison_plan_sha256=assignment.comparison_plan.sha256,
        study_materialization_sha256=binding.study_materialization_sha256,
        environment_authority_sha256=binding.environment_authority_sha256,
        raw_materialization_sha256=binding.raw_materialization_sha256,
        dataset_lineage_sha256=binding.dataset_lineage_sha256,
        protocol_sha256=binding.protocol_sha256,
        preprocessing_authority_sha256=binding.preprocessing_authority_sha256,
        case_authority_sha256=assignment.case_authority.authority_sha256,
        method_spec_sha256=factory.method_spec.sha256,
        rows=tuple(flat_rows),
    )
    validate_promoted_shard_result(
        shard_result,
        execution_plan=assignment.execution_plan,
        comparison_plan=assignment.comparison_plan,
    )
    roles = build_case_result_observation_roles(
        authority=assignment.case_authority,
        shard=current_processed,
        result=result,
    )
    manifest = {
        "schema_version": 1,
        "artifact_kind": "promoted_atomic_worker_result",
        "binding_bundle_sha256": assignment.binding_bundle_sha256,
        "execution_plan_sha256": assignment.execution_plan.sha256,
        "shard_spec_sha256": assignment.shard.sha256,
        "shard_id": assignment.shard.shard_id,
        "source_revision": binding.source_revision,
        "environment_authority_sha256": environment.sha256,
        "study_materialization_sha256": binding.study_materialization_sha256,
        "raw_materialization_sha256": binding.raw_materialization_sha256,
        "processed_shard_sha256": current_processed.sha256,
        "case_authority_sha256": assignment.case_authority.authority_sha256,
        "method_spec_sha256": factory.method_spec.sha256,
        "method_realization_key": assignment.shard.method_realization_key,
        "budgets_per_class": list(assignment.shard.budgets_per_class),
        "result_sha256": result.sha256,
        "shard_result_sha256": shard_result.sha256,
        "global_analysis_performed": False,
        "claim_boundary": (
            "one atomic preregistered promoted worker frontier; global comparative claims "
            "require complete-shard assembly under the promoted execution authority"
        ),
    }
    base._json_dump(root / "worker_manifest.json", manifest)
    base._json_dump(
        root / "case_result.json",
        {
            "schema_version": 1,
            "subject": assignment.shard.subject,
            "held_out_session": assignment.shard.target_session,
            "split_seed": assignment.shard.split_seed,
            "model_seed": assignment.shard.model_seed,
            "method_realization_key": assignment.shard.method_realization_key,
            "result": result.to_dict(),
        },
    )
    base._json_dump(
        root / "observation_roles.json",
        {"schema_version": 1, "shard_spec_sha256": assignment.shard.sha256, "rows": roles},
    )
    base._json_dump(
        root / "shard_result.json",
        {**shard_result.to_dict(), "shard_result_sha256": shard_result.sha256},
    )
    sealed = _seal_worker_bundle(root)
    verified = verify_promoted_worker_bundle(root, binding_root=binding_root)
    return {
        "verified": verified["verified"],
        "worker_bundle_sha256": sealed["bundle_sha256"],
        "binding_bundle_sha256": assignment.binding_bundle_sha256,
        "execution_plan_sha256": assignment.execution_plan.sha256,
        "shard_spec_sha256": assignment.shard.sha256,
        "shard_result_sha256": shard_result.sha256,
        "attempted_budgets": list(assignment.shard.budgets_per_class),
        "statuses": [row.status for row in result.rows],
        "global_analysis_performed": False,
        "output": str(root),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute exactly one content-addressed Kumar2024 promoted worker shard."
    )
    parser.add_argument("--binding", required=True)
    parser.add_argument("--shard-spec-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        verify_promoted_worker_bundle(args.output, binding_root=args.binding)
        if args.verify_only
        else run_promoted_worker(
            args.binding,
            args.shard_spec_sha256,
            args.output,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PROMOTED_WORKER_BUNDLE_FILES",
    "PromotedWorkerAssignment",
    "load_promoted_worker_assignment",
    "main",
    "run_promoted_worker",
    "verify_promoted_worker_bundle",
]
