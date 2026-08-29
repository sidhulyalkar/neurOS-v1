"""No-model materialization and binding authority for the promoted Kumar2024 study.

This module freezes everything a distributed promoted execution is allowed to
consume before any worker fits a decoder or generates final-assessment
predictions. It deliberately performs two-pass data materialization and case
construction only.

The output bundle binds:

- the preregistered promoted comparison plan;
- the exact realized Python environment and a reviewer-facing environment lock;
- the exact consumed raw Kumar2024 GDF bytes;
- all 18 participant-native processed shards;
- all 270 participant/session/split LongitudinalCaseAuthority objects;
- the five external method-realization specifications;
- the exact 1,350-shard / 6,750-fit promoted execution plan.

No decoder ``create()``, ``fit()``, ``predict()``, ``predict_proba()`` or scoring
path is invoked by this module.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import kumar2024 as base
from .kumar2024_comparison import (
    Kumar2024ComparisonPlan,
    promoted_external_floor_plan,
)
from .kumar2024_materialization import (
    build_processed_subject_shard,
    materialization_manifest,
    verify_processed_subject_shard,
)
from .kumar2024_materialized_study import _runtime_authority
from .kumar2024_promoted_execution import (
    PromotedExecutionBinding,
    PromotedExecutionPlan,
    PromotedExecutionShardSpec,
    PromotedExecutionTemplate,
    bind_promoted_execution_template,
    build_promoted_execution_template,
)

PROMOTED_BINDING_BUNDLE_FILES = (
    "binding_manifest.json",
    "materialization.json",
    "environment_lock.json",
    "case_authorities.json",
    "method_specs.json",
    "execution_plan.json",
)

_LOCAL_SOURCE_DISTRIBUTIONS = frozenset(
    {
        "neuros",
        "neuros-core",
        "neuros-drivers",
        "neuros-foundation",
        "neuros-models",
        "neuros-orion",
    }
)


def promoted_materialization_config(
    plan: Kumar2024ComparisonPlan | None = None,
) -> base.Kumar2024StudyConfig:
    """Return the full-cohort config used only for materialization/protocol identity.

    The first preregistered split seed is an explicit anchor because
    ``Kumar2024StudyConfig`` describes one realization. The binding runner later
    constructs all case authorities for every split seed in ``plan``.
    """

    plan = plan or promoted_external_floor_plan()
    return base.Kumar2024StudyConfig(
        subjects=plan.subjects,
        target_sessions=plan.target_sessions,
        budgets_per_class=plan.budgets_per_class,
        methods=plan.methods,
        split_seed=plan.split_seeds[0],
        analysis_bootstrap_replicates=plan.bootstrap_replicates,
        analysis_seed=plan.analysis_seed,
        profile="full",
    )


def _method_spec_record(
    realization_key: str,
    spec: Any,
) -> dict[str, Any]:
    return {
        "realization_key": str(realization_key),
        "method_spec": {
            **spec.to_dict(),
            "method_spec_sha256": spec.sha256,
        },
    }


def promoted_method_spec_records(
    *,
    plan: Kumar2024ComparisonPlan,
    config: base.Kumar2024StudyConfig,
    sample_rate_hz: float,
) -> tuple[dict[str, Any], ...]:
    """Build the five frozen method identities without creating a decoder."""

    from neuros.foundation_models.qualification_baselines import (
        MNECSPLDAFactory,
        RiemannianTangentLogRegFactory,
        UpstreamBraindecodeFactory,
    )

    records: list[dict[str, Any]] = []
    for policy in plan.method_seed_policies:
        if policy.method_id == "mne-csp-lda":
            factory = MNECSPLDAFactory(n_components=config.csp_components)
            records.append(
                _method_spec_record("mne-csp-lda/deterministic", factory.method_spec)
            )
            continue
        if policy.method_id == "pyriemann-rg-lr":
            factory = RiemannianTangentLogRegFactory()
            records.append(
                _method_spec_record("pyriemann-rg-lr/deterministic", factory.method_spec)
            )
            continue
        if policy.method_id != "braindecode-eegnet":
            raise ValueError(
                f"promoted binding has no frozen method factory for {policy.method_id!r}"
            )
        for model_seed in policy.model_seeds:
            factory = UpstreamBraindecodeFactory(
                model_name="EEGNet",
                sample_rate_hz=float(sample_rate_hz),
                optimizer_name=config.braindecode_optimizer,
                learning_rate=config.braindecode_learning_rate,
                weight_decay=config.braindecode_weight_decay,
                n_epochs=config.braindecode_epochs,
                batch_size=config.braindecode_batch_size,
                device=config.device,
                random_state=int(model_seed),
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
            records.append(
                _method_spec_record(
                    f"braindecode-eegnet/model-seed-{int(model_seed)}",
                    factory.method_spec,
                )
            )

    records.sort(key=lambda item: str(item["realization_key"]))
    expected = set(build_promoted_execution_template(plan).method_realization_keys)
    observed = {str(item["realization_key"]) for item in records}
    if observed != expected:
        raise RuntimeError(
            "promoted method-spec records do not match execution template: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return tuple(records)


def _environment_lock_payload(environment: Any) -> dict[str, Any]:
    distributions = [item.to_dict() for item in environment.distributions]
    local = [
        item for item in distributions if str(item["name"]) in _LOCAL_SOURCE_DISTRIBUTIONS
    ]
    external = [
        item for item in distributions if str(item["name"]) not in _LOCAL_SOURCE_DISTRIBUTIONS
    ]
    return {
        "schema_version": 1,
        "environment_authority_sha256": environment.sha256,
        "python": {
            "implementation": environment.python_implementation,
            "version": environment.python_version,
        },
        "platform": {
            "system": environment.platform_system,
            "machine": environment.platform_machine,
        },
        "source_revision": environment.source_revision,
        "local_source_distributions": local,
        "external_distributions": external,
        "accelerator_runtime": dict(environment.accelerator_runtime),
        "deterministic_flags": dict(environment.deterministic_flags),
        "reproduction_policy": (
            "checkout source_revision; install local_source_distributions from that source; "
            "install every external distribution at the exact recorded version; then "
            "recompute EnvironmentAuthority and require the bound SHA before execution"
        ),
    }


def _method_spec_from_payload(payload: Mapping[str, Any]):
    from neuros.foundation_models.qualification import ExternalDecoderMethodSpec

    return ExternalDecoderMethodSpec(
        method_id=payload["method_id"],
        implementation=payload["implementation"],
        implementation_version=payload["implementation_version"],
        input_axes=tuple(payload["input_axes"]),
        probability_semantics=payload["probability_semantics"],
        target_adaptation_mode=payload.get("target_adaptation_mode", "none"),
        uncertainty_semantics=payload.get("uncertainty_semantics", "none"),
        model_lineage_sha256=payload.get("model_lineage_sha256"),
        source_reference=payload.get("source_reference"),
        metadata=payload.get("metadata", {}),
        schema_version=payload.get("schema_version", 1),
    )


def _template_from_payload(payload: Mapping[str, Any]) -> PromotedExecutionTemplate:
    shards: list[PromotedExecutionShardSpec] = []
    for item in payload["shards"]:
        shard = PromotedExecutionShardSpec(
            comparison_plan_sha256=item["comparison_plan_sha256"],
            subject=item["subject"],
            target_session=item["target_session"],
            split_seed=item["split_seed"],
            method_id=item["method_id"],
            model_seed=item.get("model_seed"),
            budgets_per_class=tuple(item["budgets_per_class"]),
            schema_version=item.get("schema_version", 1),
        )
        if item.get("shard_spec_sha256") != shard.sha256:
            raise ValueError(
                f"serialized promoted shard SHA mismatch for {shard.shard_id}"
            )
        shards.append(shard)
    template = PromotedExecutionTemplate(
        comparison_plan_sha256=payload["comparison_plan_sha256"],
        shards=tuple(shards),
        schema_version=payload.get("schema_version", 1),
    )
    if payload.get("template_sha256") != template.sha256:
        raise ValueError("serialized promoted execution template SHA mismatch")
    return template


def _binding_from_payload(payload: Mapping[str, Any]) -> PromotedExecutionBinding:
    cases = tuple(
        (
            item["subject"],
            item["target_session"],
            item["split_seed"],
            item["case_authority_sha256"],
        )
        for item in payload["case_authority_sha256_by_case"]
    )
    methods = tuple(
        (key, value)
        for key, value in payload["method_spec_sha256_by_realization"].items()
    )
    binding = PromotedExecutionBinding(
        comparison_plan_sha256=payload["comparison_plan_sha256"],
        template_sha256=payload["template_sha256"],
        study_materialization_sha256=payload["study_materialization_sha256"],
        environment_authority_sha256=payload["environment_authority_sha256"],
        raw_materialization_sha256=payload["raw_materialization_sha256"],
        dataset_lineage_sha256=payload["dataset_lineage_sha256"],
        protocol_sha256=payload["protocol_sha256"],
        preprocessing_authority_sha256=payload["preprocessing_authority_sha256"],
        source_revision=payload["source_revision"],
        case_authority_sha256_by_case=cases,
        method_spec_sha256_by_realization=methods,
        schema_version=payload.get("schema_version", 1),
    )
    if payload.get("binding_sha256") != binding.sha256:
        raise ValueError("serialized promoted execution binding SHA mismatch")
    return binding


def _environment_sha(payload: Mapping[str, Any]) -> str:
    return base._identity_sha256("neuros.environment_authority.v1", payload)


def _raw_materialization_sha(payload: Mapping[str, Any]) -> str:
    return base._identity_sha256("neuros.raw_materialization_authority.v1", payload)


def _processed_shard_sha(payload: Mapping[str, Any]) -> str:
    return base._identity_sha256("neuros.processed_materialization_shard.v1", payload)


def _study_materialization_sha(payload: Mapping[str, Any]) -> str:
    return base._identity_sha256("neuros.study_materialization_authority.v1", payload)


def _seal_binding_bundle(output: Path) -> dict[str, Any]:
    files = {
        name: base._file_sha256(output / name)
        for name in PROMOTED_BINDING_BUNDLE_FILES
    }
    root = base._identity_sha256(
        "neuros.nsq_kumar2024_promoted_binding_bundle.v1",
        {"files": files},
    )
    payload = {
        "schema_version": 1,
        "files": files,
        "bundle_sha256": root,
    }
    base._json_dump(output / "artifact_hashes.json", payload)
    return payload


def _verify_materialization_payload(
    materialization: Mapping[str, Any],
) -> dict[str, str]:
    authority = materialization["authority"]
    environment = authority["environment"]
    raw = authority["raw_materialization"]
    processed = authority["processed_shards"]

    environment_sha = _environment_sha(environment)
    if environment_sha != authority["environment_sha256"]:
        raise ValueError("environment authority SHA does not match serialized payload")
    raw_sha = _raw_materialization_sha(raw)
    if raw_sha != authority["raw_materialization_sha256"]:
        raise ValueError("raw materialization SHA does not match serialized payload")
    if len(processed) != 18:
        raise ValueError("promoted binding requires exactly 18 processed participant shards")
    processed_shas = [_processed_shard_sha(item) for item in processed]
    if processed_shas != list(authority["processed_shard_sha256s"]):
        raise ValueError("processed shard SHA list does not match serialized payloads")
    study_sha = _study_materialization_sha(authority)
    if study_sha != materialization["study_materialization_sha256"]:
        raise ValueError("study materialization SHA does not match serialized authority")
    return {
        "study_materialization_sha256": study_sha,
        "environment_authority_sha256": environment_sha,
        "raw_materialization_sha256": raw_sha,
    }


def verify_promoted_binding_bundle(
    output: str | Path,
    *,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify a no-model promoted binding artifact without loading neural data."""

    from neuros.foundation_models.longitudinal_authority import LongitudinalCaseAuthority

    root = Path(output).resolve()
    if payload is None:
        payload = json.loads(
            (root / "artifact_hashes.json").read_text(encoding="utf-8")
        )
    if payload.get("schema_version") != 1 or not isinstance(payload.get("files"), Mapping):
        raise ValueError("invalid promoted binding hash manifest")
    declared = dict(payload["files"])
    if set(declared) != set(PROMOTED_BINDING_BUNDLE_FILES):
        raise ValueError("promoted binding bundle file set differs from frozen contract")
    actual: dict[str, str] = {}
    for name in PROMOTED_BINDING_BUNDLE_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing promoted binding bundle file: {name}")
        digest = base._file_sha256(path)
        if digest != declared[name]:
            raise ValueError(f"promoted binding bundle hash mismatch for {name}")
        actual[name] = digest
    expected_root = base._identity_sha256(
        "neuros.nsq_kumar2024_promoted_binding_bundle.v1",
        {"files": actual},
    )
    if payload.get("bundle_sha256") != expected_root:
        raise ValueError("promoted binding bundle root SHA does not match file manifest")

    manifest = json.loads((root / "binding_manifest.json").read_text(encoding="utf-8"))
    materialization = json.loads((root / "materialization.json").read_text(encoding="utf-8"))
    environment_lock = json.loads((root / "environment_lock.json").read_text(encoding="utf-8"))
    authority_payload = json.loads((root / "case_authorities.json").read_text(encoding="utf-8"))
    method_payload = json.loads((root / "method_specs.json").read_text(encoding="utf-8"))
    execution_payload = json.loads((root / "execution_plan.json").read_text(encoding="utf-8"))

    plan = promoted_external_floor_plan()
    if manifest.get("comparison_plan_sha256") != plan.sha256:
        raise ValueError("binding manifest comparison-plan SHA is not the promoted plan")
    if manifest.get("comparison_plan") != plan.to_dict():
        raise ValueError("binding manifest comparison-plan payload drifted")
    if manifest.get("model_execution_performed") is not False:
        raise ValueError("promoted binding artifact must explicitly claim zero model execution")
    if manifest.get("final_assessment_predictions_generated") is not False:
        raise ValueError("promoted binding artifact must explicitly claim zero predictions")

    materialization_roots = _verify_materialization_payload(materialization)
    if manifest.get("study_materialization_sha256") != materialization_roots["study_materialization_sha256"]:
        raise ValueError("binding manifest and materialization root differ")
    environment_payload = materialization["authority"]["environment"]
    if environment_lock.get("environment_authority_sha256") != materialization_roots["environment_authority_sha256"]:
        raise ValueError("environment lock does not name the materialized environment")
    if environment_lock.get("source_revision") != manifest.get("source_revision"):
        raise ValueError("environment lock source revision differs from binding manifest")
    if environment_payload.get("source_revision") != manifest.get("source_revision"):
        raise ValueError("materialized environment source revision differs from binding manifest")
    if environment_lock.get("python") != environment_payload.get("python"):
        raise ValueError("environment lock Python identity differs from environment authority")
    if environment_lock.get("platform") != environment_payload.get("platform"):
        raise ValueError("environment lock platform identity differs from environment authority")
    if environment_lock.get("accelerator_runtime") != environment_payload.get("accelerator_runtime"):
        raise ValueError("environment lock accelerator runtime differs from environment authority")
    if environment_lock.get("deterministic_flags") != environment_payload.get("deterministic_flags"):
        raise ValueError("environment lock deterministic flags differ from environment authority")
    env_distributions = environment_payload["distributions"]
    lock_distributions = sorted(
        [
            *environment_lock["local_source_distributions"],
            *environment_lock["external_distributions"],
        ],
        key=lambda item: (item["name"], item["version"]),
    )
    if lock_distributions != sorted(
        env_distributions,
        key=lambda item: (item["name"], item["version"]),
    ):
        raise ValueError("environment lock distribution set differs from environment authority")

    preprocessing_sha = str(manifest["preprocessing_authority"]["sha256"])
    processed_by_subject: dict[int, Mapping[str, Any]] = {}
    for shard in materialization["authority"]["processed_shards"]:
        shard_id = str(shard["shard_id"])
        if not shard_id.startswith("subject="):
            raise ValueError(f"non-canonical promoted processed shard id {shard_id!r}")
        subject = int(shard_id.split("=", 1)[1])
        if subject in processed_by_subject:
            raise ValueError(f"duplicate promoted processed shard for subject {subject}")
        if list(shard["preprocessing_authority_sha256s"]) != [preprocessing_sha]:
            raise ValueError("processed shard preprocessing authority differs from binding manifest")
        observations = shard["observation_identity"]["observations"]
        if not observations:
            raise ValueError("promoted processed shard cannot have zero observations")
        if any(str(item["participant"]) != str(subject) for item in observations):
            raise ValueError("processed shard observation participant differs from shard id")
        processed_by_subject[subject] = shard
    if set(processed_by_subject) != set(plan.subjects):
        raise ValueError("processed participant shard set differs from promoted comparison cohort")

    raw_authorities = authority_payload.get("authorities")
    if not isinstance(raw_authorities, list) or len(raw_authorities) != 270:
        raise ValueError("promoted binding requires exactly 270 case authorities")
    case_map: dict[tuple[int, str, int], str] = {}
    for raw in raw_authorities:
        authority = LongitudinalCaseAuthority.from_dict(raw)
        metadata = authority.case_metadata
        key = (
            int(metadata["subject"]),
            str(metadata["held_out_session"]),
            int(metadata["split_seed"]),
        )
        if key in case_map:
            raise ValueError(f"duplicate promoted binding case authority for {key!r}")
        subject, session, split_seed = key
        shard = processed_by_subject.get(subject)
        if shard is None:
            raise ValueError(f"case authority names unknown promoted subject {subject}")
        expected_case_id = (
            f"{base.KUMAR2024_DATASET_ID}/subject-{subject}/"
            f"session-{session}/split-{split_seed}"
        )
        if authority.case_id != expected_case_id:
            raise ValueError("case authority id differs from canonical promoted case identity")
        if authority.seed != split_seed:
            raise ValueError("case authority split seed differs from case metadata")
        if authority.held_out_values != (session,):
            raise ValueError("case authority held-out session differs from case metadata")
        if authority.history_policy != "prior":
            raise ValueError("promoted case authority must use prior-session history policy")
        if authority.observed_group_order != base.KUMAR2024_EXPECTED_SESSIONS:
            raise ValueError("promoted case authority observed chronology differs from Kumar2024")
        expected_sources = tuple(
            value
            for value in base.KUMAR2024_EXPECTED_SESSIONS
            if int(value) < int(session)
        )
        if authority.source_group_values != expected_sources:
            raise ValueError("promoted case authority source history differs from prior sessions")
        if authority.processed_data_sha256 != shard["processed_data_sha256"]:
            raise ValueError("case authority processed-data SHA differs from participant shard")
        observation_count = len(shard["observation_identity"]["observations"])
        if authority.n_samples != observation_count or authority.input_shape[0] != observation_count:
            raise ValueError("case authority sample geometry differs from participant shard")
        case_map[key] = authority.authority_sha256
    expected_case_keys = {
        (subject, session, split_seed)
        for subject in plan.subjects
        for session in plan.target_sessions
        for split_seed in plan.split_seeds
    }
    if set(case_map) != expected_case_keys:
        raise ValueError("case authority keys do not cover the full promoted comparison plan")

    records = method_payload.get("method_specs")
    if not isinstance(records, list) or len(records) != 5:
        raise ValueError("promoted binding requires exactly five method-realization specs")
    method_map: dict[str, str] = {}
    for record in records:
        key = str(record["realization_key"])
        spec_payload = dict(record["method_spec"])
        declared_sha = spec_payload.pop("method_spec_sha256")
        spec = _method_spec_from_payload(spec_payload)
        if spec.sha256 != declared_sha:
            raise ValueError(f"method-spec SHA mismatch for realization {key!r}")
        if key in method_map:
            raise ValueError(f"duplicate method realization key {key!r}")
        if key.endswith("/deterministic"):
            expected_method_id = key[: -len("/deterministic")]
            if spec.method_id != expected_method_id:
                raise ValueError("deterministic realization key and method spec id differ")
        elif "/model-seed-" in key:
            expected_method_id, seed_text = key.rsplit("/model-seed-", 1)
            if spec.method_id != expected_method_id:
                raise ValueError("stochastic realization key and method spec id differ")
            if spec.metadata.get("model_seed") != int(seed_text):
                raise ValueError("stochastic realization key and method-spec model seed differ")
            if spec.metadata.get("final_assessment_used_for_state_selection") is not False:
                raise ValueError("promoted stochastic method may not select state on final assessment")
        else:
            raise ValueError(f"unsupported promoted method realization key {key!r}")
        method_map[key] = spec.sha256

    template_payload = execution_payload["template"]
    binding_payload = execution_payload["binding"]
    template = _template_from_payload(template_payload)
    binding = _binding_from_payload(binding_payload)
    execution_plan = PromotedExecutionPlan(template=template, binding=binding)
    if execution_payload.get("execution_plan_sha256") != execution_plan.sha256:
        raise ValueError("serialized promoted execution plan SHA mismatch")
    if template.comparison_plan_sha256 != plan.sha256:
        raise ValueError("execution template does not bind promoted comparison plan")
    if len(template.shards) != 1350 or template.expected_fit_attempts != 6750:
        raise ValueError("promoted execution cardinality differs from preregistered graph")
    if binding.case_authority_map != case_map:
        raise ValueError("execution binding case-authority map differs from case artifact")
    if binding.method_spec_map != method_map:
        raise ValueError("execution binding method-spec map differs from method artifact")
    if binding.study_materialization_sha256 != materialization_roots["study_materialization_sha256"]:
        raise ValueError("execution binding study materialization differs from artifact")
    if binding.environment_authority_sha256 != materialization_roots["environment_authority_sha256"]:
        raise ValueError("execution binding environment differs from artifact")
    if binding.raw_materialization_sha256 != materialization_roots["raw_materialization_sha256"]:
        raise ValueError("execution binding raw materialization differs from artifact")
    if manifest.get("environment_authority_sha256") != materialization_roots["environment_authority_sha256"]:
        raise ValueError("binding manifest environment authority differs from materialization")
    if manifest.get("raw_materialization_sha256") != materialization_roots["raw_materialization_sha256"]:
        raise ValueError("binding manifest raw materialization differs from materialization")
    if binding.dataset_lineage_sha256 != manifest.get("dataset_lineage_sha256"):
        raise ValueError("execution binding dataset lineage differs from binding manifest")
    if binding.protocol_sha256 != manifest.get("protocol_sha256"):
        raise ValueError("execution binding protocol differs from binding manifest")
    if binding.preprocessing_authority_sha256 != preprocessing_sha:
        raise ValueError("execution binding preprocessing authority differs from manifest")
    if binding.source_revision != manifest.get("source_revision"):
        raise ValueError("execution binding source revision differs from manifest")
    if manifest.get("execution_plan_sha256") != execution_plan.sha256:
        raise ValueError("binding manifest execution-plan SHA differs from execution artifact")
    counts = manifest.get("counts", {})
    if counts != {
        "participants": 18,
        "case_authorities": 270,
        "method_realizations": 5,
        "execution_shards": 1350,
        "planned_fit_attempts": 6750,
    }:
        raise ValueError("binding manifest promoted execution counts differ from frozen plan")
    if manifest.get("final_assessment_metrics_generated") is not False:
        raise ValueError("promoted binding artifact must explicitly claim zero metrics")

    return {
        "verified": True,
        "schema_version": 1,
        "bundle_sha256": expected_root,
        "comparison_plan_sha256": plan.sha256,
        "study_materialization_sha256": materialization_roots["study_materialization_sha256"],
        "environment_authority_sha256": materialization_roots["environment_authority_sha256"],
        "raw_materialization_sha256": materialization_roots["raw_materialization_sha256"],
        "execution_plan_sha256": execution_plan.sha256,
        "case_authorities": len(case_map),
        "method_realizations": len(method_map),
        "expected_shards": len(template.shards),
        "expected_fit_attempts": template.expected_fit_attempts,
        "files": actual,
    }


def run_promoted_binding(
    output: str | Path,
    *,
    overwrite: bool = False,
    plan: Kumar2024ComparisonPlan | None = None,
    preprocessing: base.Kumar2024PreprocessingSpec | None = None,
) -> dict[str, Any]:
    """Materialize and bind the full promoted study without model execution."""

    from neuros.foundation_models.longitudinal import ordered_group_values
    from neuros.foundation_models.materialization_authority import StudyMaterializationAuthority
    from neuros.foundation_models.moabb_epochs import collect_moabb_epochs
    from neuros.foundation_models.moabb_longitudinal import (
        build_moabb_longitudinal_dataset,
        validate_observed_sessions,
    )
    from neuros.foundation_models.moabb_materialization import (
        resolve_kumar2024_raw_materialization,
    )

    plan = plan or promoted_external_floor_plan()
    config = promoted_materialization_config(plan)
    preprocessing = preprocessing or base.Kumar2024PreprocessingSpec()
    output_path = base._prepare_output(Path(output), overwrite=overwrite)
    versions = base._runtime_versions()
    dataset_spec, dataset, paradigm = build_moabb_longitudinal_dataset(
        base.KUMAR2024_DATASET_KEY,
        fmin=preprocessing.fmin_hz,
        fmax=preprocessing.fmax_hz,
        resample=preprocessing.resample_hz,
    )
    raw_evidence = resolve_kumar2024_raw_materialization(
        dataset,
        subjects=config.subjects,
    )
    environment = _runtime_authority(config)

    first_descriptor = None
    preprocessing_authority = None
    frozen_shards = []
    shards_by_subject: dict[int, Any] = {}
    subject_descriptors: dict[str, Any] = {}

    # Pass 1 freezes exact processed participant shards and releases arrays.
    for subject in config.subjects:
        data, descriptor = collect_moabb_epochs(
            dataset,
            paradigm,
            subjects=[subject],
            dataset_id=base.KUMAR2024_DATASET_ID,
        )
        observed = validate_observed_sessions(
            dataset_spec,
            ordered_group_values(data, split_unit="session"),
        )
        if observed != base.KUMAR2024_EXPECTED_SESSIONS:
            raise RuntimeError(
                f"Kumar2024 chronology changed for subject {subject}: {observed}"
            )
        if first_descriptor is None:
            first_descriptor = descriptor
            preprocessing_authority = base._preprocessing_authority(
                preprocessing,
                descriptor,
                versions,
            )
        elif descriptor.signal_contract_sha256 != first_descriptor.signal_contract_sha256:
            raise RuntimeError(
                "processed MOABB signal contract changed across promoted participants"
            )
        assert preprocessing_authority is not None
        shard = build_processed_subject_shard(
            data,
            subject=subject,
            preprocessing_authority_sha256=preprocessing_authority["sha256"],
        )
        frozen_shards.append(shard)
        shards_by_subject[int(subject)] = shard
        subject_descriptors[str(subject)] = {
            **descriptor.to_dict(),
            "descriptor_sha256": descriptor.sha256,
        }
        del data
        gc.collect()

    if first_descriptor is None or preprocessing_authority is None:
        raise RuntimeError("promoted Kumar2024 binding produced no processed shards")
    if len(frozen_shards) != len(plan.subjects):
        raise RuntimeError("promoted materialization did not freeze every participant")

    materialization = StudyMaterializationAuthority(
        environment=environment,
        raw_materialization=raw_evidence.authority,
        processed_shards=tuple(frozen_shards),
    )
    lineage = base.build_dataset_lineage(
        config=config,
        preprocessing_authority=preprocessing_authority,
        versions=versions,
        raw_materialization_sha256=raw_evidence.authority.sha256,
    )
    protocol = base.build_protocol(
        config=config,
        dataset_lineage=lineage,
        preprocessing_authority_sha256=preprocessing_authority["sha256"],
    )

    method_records = promoted_method_spec_records(
        plan=plan,
        config=config,
        sample_rate_hz=first_descriptor.sampling_rate_hz,
    )
    method_map = {
        str(record["realization_key"]): str(record["method_spec"]["method_spec_sha256"])
        for record in method_records
    }

    # Pass 2 reproduces each processed shard, then freezes all split authorities.
    authorities: list[Any] = []
    case_map: dict[tuple[int, str, int], str] = {}
    for subject in config.subjects:
        data, descriptor = collect_moabb_epochs(
            dataset,
            paradigm,
            subjects=[subject],
            dataset_id=base.KUMAR2024_DATASET_ID,
        )
        if descriptor.sha256 != subject_descriptors[str(subject)]["descriptor_sha256"]:
            raise RuntimeError(
                f"second-pass MOABB epoch descriptor changed for subject {subject}"
            )
        verify_processed_subject_shard(
            data,
            shards_by_subject[int(subject)],
            subject=subject,
        )
        observed = validate_observed_sessions(
            dataset_spec,
            ordered_group_values(data, split_unit="session"),
        )
        if observed != base.KUMAR2024_EXPECTED_SESSIONS:
            raise RuntimeError(
                f"Kumar2024 chronology changed on binding pass for subject {subject}"
            )
        for split_seed in plan.split_seeds:
            split_config = replace(config, split_seed=int(split_seed))
            for target_session in plan.target_sessions:
                authority = base._make_case_authority(
                    data=data,
                    dataset_spec=dataset_spec,
                    subject=int(subject),
                    target_session=str(target_session),
                    config=split_config,
                )
                key = (int(subject), str(target_session), int(split_seed))
                if key in case_map:
                    raise RuntimeError(f"duplicate promoted case authority for {key!r}")
                case_map[key] = authority.authority_sha256
                authorities.append(authority)
        del data
        gc.collect()

    expected_cases = len(plan.subjects) * len(plan.target_sessions) * len(plan.split_seeds)
    if len(authorities) != expected_cases or expected_cases != 270:
        raise RuntimeError(
            f"promoted binding expected 270 case authorities, observed {len(authorities)}"
        )

    template = build_promoted_execution_template(plan)
    source_revision = base._git_revision()
    if source_revision is None:
        raise RuntimeError("promoted binding requires an exact Git source revision")
    execution_plan = bind_promoted_execution_template(
        template,
        study_materialization_sha256=materialization.sha256,
        environment_authority_sha256=environment.sha256,
        raw_materialization_sha256=raw_evidence.authority.sha256,
        dataset_lineage_sha256=lineage.lineage_sha256,
        protocol_sha256=protocol.sha256,
        preprocessing_authority_sha256=preprocessing_authority["sha256"],
        source_revision=source_revision,
        case_authority_sha256_by_case=case_map,
        method_spec_sha256_by_realization=method_map,
    )

    raw_selection = {
        "schema_version": raw_evidence.schema_version,
        "loader_contract": raw_evidence.loader_contract,
        "selections": [item.to_dict() for item in raw_evidence.selections],
    }
    manifest = {
        "schema_version": 1,
        "study": "nsq-kumar2024-promoted-external-floor-v1",
        "artifact_kind": "no_model_promoted_binding_authority",
        "comparison_plan": plan.to_dict(),
        "comparison_plan_sha256": plan.sha256,
        "materialization_config": config.to_dict(),
        "materialization_config_sha256": config.sha256,
        "materialization_split_seed_role": (
            "first preregistered split seed is a config anchor only; all split seeds are "
            "materialized independently in case_authorities.json"
        ),
        "preprocessing_authority": preprocessing_authority,
        "dataset_lineage": lineage.to_dict(),
        "dataset_lineage_sha256": lineage.lineage_sha256,
        "protocol": {**protocol.to_dict(), "protocol_sha256": protocol.sha256},
        "protocol_sha256": protocol.sha256,
        "study_materialization_sha256": materialization.sha256,
        "environment_authority_sha256": environment.sha256,
        "raw_materialization_sha256": raw_evidence.authority.sha256,
        "source_revision": source_revision,
        "execution_plan_sha256": execution_plan.sha256,
        "subject_epoch_descriptors": subject_descriptors,
        "counts": {
            "participants": len(plan.subjects),
            "case_authorities": len(authorities),
            "method_realizations": len(method_records),
            "execution_shards": len(template.shards),
            "planned_fit_attempts": template.expected_fit_attempts,
        },
        "model_execution_performed": False,
        "final_assessment_predictions_generated": False,
        "final_assessment_metrics_generated": False,
        "claim_boundary": (
            "authority-only artifact that freezes materialization, cases, method specs, and "
            "execution graph before promoted model execution; contains no efficacy result"
        ),
    }

    base._json_dump(output_path / "binding_manifest.json", manifest)
    base._json_dump(
        output_path / "materialization.json",
        materialization_manifest(materialization, raw_selection=raw_selection),
    )
    base._json_dump(
        output_path / "environment_lock.json",
        _environment_lock_payload(environment),
    )
    base._json_dump(
        output_path / "case_authorities.json",
        {
            "schema_version": 1,
            "comparison_plan_sha256": plan.sha256,
            "authorities": [item.to_dict() for item in authorities],
        },
    )
    base._json_dump(
        output_path / "method_specs.json",
        {
            "schema_version": 1,
            "comparison_plan_sha256": plan.sha256,
            "method_specs": list(method_records),
        },
    )
    base._json_dump(
        output_path / "execution_plan.json",
        {
            **execution_plan.to_dict(),
            "execution_plan_sha256": execution_plan.sha256,
        },
    )
    sealed = _seal_binding_bundle(output_path)
    verified = verify_promoted_binding_bundle(output_path)
    return {
        "bundle_schema_version": 1,
        "bundle_sha256": sealed["bundle_sha256"],
        "comparison_plan_sha256": plan.sha256,
        "study_materialization_sha256": materialization.sha256,
        "environment_authority_sha256": environment.sha256,
        "raw_materialization_sha256": raw_evidence.authority.sha256,
        "execution_plan_sha256": execution_plan.sha256,
        "case_authorities": len(authorities),
        "method_realizations": len(method_records),
        "expected_shards": len(template.shards),
        "expected_fit_attempts": template.expected_fit_attempts,
        "model_execution_performed": False,
        "verified": verified["verified"],
        "output": str(output_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the full Kumar2024 promoted execution binding without fitting models."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verify_only:
        result = verify_promoted_binding_bundle(args.output)
    else:
        result = run_promoted_binding(args.output, overwrite=args.overwrite)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "PROMOTED_BINDING_BUNDLE_FILES",
    "main",
    "promoted_materialization_config",
    "promoted_method_spec_records",
    "run_promoted_binding",
    "verify_promoted_binding_bundle",
]
