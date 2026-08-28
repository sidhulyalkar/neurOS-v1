"""Materialization-aware execution for the canonical Kumar2024 NSQ study.

Bundle generation v2 preserves the frozen scientific protocol while binding
the exact realized Python environment, consumed raw GDF bytes, participant-
native processed shards, and reviewer-facing observation-role manifests.

Execution is deliberately two-pass. Pass one freezes every processed shard
and the global study materialization root. Pass two reloads each participant
and must reproduce the frozen processed SHA and human observation identities
before any model may execute.
"""

from __future__ import annotations

import gc
import json
import platform
from pathlib import Path
from typing import Any, Mapping

from . import kumar2024 as base
from .kumar2024_materialization import (
    build_case_result_observation_roles,
    build_processed_subject_shard,
    materialization_manifest,
    verify_processed_subject_shard,
)
from neuros.foundation_models.materialization_authority import (
    StudyMaterializationAuthority,
    capture_environment_authority,
)
from neuros.foundation_models.moabb_materialization import (
    resolve_kumar2024_raw_materialization,
)

BUNDLE_FILES_V2 = (
    *base._BUNDLE_FILES,
    "materialization.json",
    "observation_roles.json",
)


def _runtime_authority(config: base.Kumar2024StudyConfig):
    accelerator_runtime: dict[str, str] = {"requested_device": str(config.device)}
    deterministic_flags: dict[str, str] = {}
    try:
        import torch
    except ImportError:
        accelerator_runtime["torch"] = "unavailable"
    else:
        accelerator_runtime.update(
            {
                "torch": str(torch.__version__),
                "cuda_runtime": str(torch.version.cuda or "none"),
                "cudnn_runtime": str(torch.backends.cudnn.version() or "none"),
            }
        )
        deterministic_flags.update(
            {
                "torch_deterministic_algorithms": str(
                    torch.are_deterministic_algorithms_enabled()
                ).lower(),
                "cudnn_deterministic": str(
                    bool(torch.backends.cudnn.deterministic)
                ).lower(),
                "cudnn_benchmark": str(
                    bool(torch.backends.cudnn.benchmark)
                ).lower(),
            }
        )
    return capture_environment_authority(
        source_revision=base._git_revision(),
        accelerator_runtime=accelerator_runtime,
        deterministic_flags=deterministic_flags,
    )


def _seal_bundle_v2(output: Path) -> dict[str, Any]:
    files = {name: base._file_sha256(output / name) for name in BUNDLE_FILES_V2}
    root = base._identity_sha256(
        "neuros.nsq_kumar2024_bundle.v2", {"files": files}
    )
    payload = {
        "schema_version": 2,
        "files": files,
        "bundle_sha256": root,
    }
    base._json_dump(output / "artifact_hashes.json", payload)
    return payload


def verify_bundle_v2(
    output: str | Path,
    *,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify a generation-v2 bundle without rerunning the study."""

    root = Path(output).resolve()
    if payload is None:
        payload = json.loads(
            (root / "artifact_hashes.json").read_text(encoding="utf-8")
        )
    if payload.get("schema_version") != 2 or not isinstance(
        payload.get("files"), Mapping
    ):
        raise ValueError("invalid Kumar2024 bundle-v2 hash manifest")
    declared = dict(payload["files"])
    if set(declared) != set(BUNDLE_FILES_V2):
        raise ValueError(
            "Kumar2024 bundle-v2 file set differs from the frozen evidence contract"
        )
    actual: dict[str, str] = {}
    for name in BUNDLE_FILES_V2:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing Kumar2024 bundle-v2 file: {name}")
        digest = base._file_sha256(path)
        if digest != declared[name]:
            raise ValueError(f"Kumar2024 bundle hash mismatch for {name}")
        actual[name] = digest
    expected_root = base._identity_sha256(
        "neuros.nsq_kumar2024_bundle.v2", {"files": actual}
    )
    if payload.get("bundle_sha256") != expected_root:
        raise ValueError("Kumar2024 bundle-v2 root SHA-256 does not match file manifest")
    return {
        "verified": True,
        "schema_version": 2,
        "bundle_sha256": expected_root,
        "files": actual,
    }


def _render_report_v2(
    *,
    config: base.Kumar2024StudyConfig,
    lineage: Any,
    protocol: Any,
    preprocessing_authority: Mapping[str, Any],
    rows: list[dict[str, Any]],
    analysis: Mapping[str, Any],
    materialization: StudyMaterializationAuthority,
) -> str:
    report = base._render_report(
        config=config,
        lineage=lineage,
        protocol=protocol,
        preprocessing_authority=preprocessing_authority,
        rows=rows,
        analysis=analysis,
    )
    return report + (
        "\n## Materialization authority\n\n"
        f"- Bundle generation: `2`\n"
        f"- Study materialization SHA-256: `{materialization.sha256}`\n"
        f"- Environment authority SHA-256: `{materialization.environment.sha256}`\n"
        f"- Raw materialization SHA-256: `{materialization.raw_materialization.sha256}`\n"
        f"- Processed participant shards: `{len(materialization.processed_shards)}`\n\n"
        "Every participant was processed once to freeze this authority, then loaded "
        "again before model execution. The second pass was required to reproduce the "
        "same processed-data and label-free observation-identity hashes exactly.\n"
    )


def run_materialized_study(
    output: str | Path,
    *,
    config: base.Kumar2024StudyConfig | None = None,
    preprocessing: base.Kumar2024PreprocessingSpec | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    from neuros.foundation_models.longitudinal import ordered_group_values
    from neuros.foundation_models.moabb_epochs import collect_moabb_epochs
    from neuros.foundation_models.moabb_longitudinal import (
        build_moabb_longitudinal_dataset,
        validate_observed_sessions,
    )
    from neuros.foundation_models.qualification_runner import (
        QualificationExecutionContext,
        run_external_qualification_case,
    )

    config = config or base.pilot_config()
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
    shards_by_subject = {}
    subject_descriptors: dict[str, Any] = {}

    # Pass 1: freeze processed participant shards without retaining the arrays.
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
                "processed MOABB signal contract changed across participants: "
                f"subject={subject}, reference={first_descriptor.signal_contract_sha256}, "
                f"observed={descriptor.signal_contract_sha256}"
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
        raise RuntimeError("Kumar2024 study produced no processed participant shards")

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
    context = QualificationExecutionContext(
        observed_dataset_lineage_sha256=lineage.lineage_sha256,
        preprocessing_authority_sha256s=(preprocessing_authority["sha256"],),
        metadata={
            "study": "nsq-kumar2024-v1",
            "bundle_generation": 2,
            "moabb_version": versions.get("moabb"),
            "mne_version": versions.get("mne"),
            "processed_signal_contract_sha256": first_descriptor.signal_contract_sha256,
            "study_materialization_sha256": materialization.sha256,
            "environment_authority_sha256": environment.sha256,
            "raw_materialization_sha256": raw_evidence.authority.sha256,
        },
    )
    factories = base._method_factories(
        config=config,
        sample_rate_hz=first_descriptor.sampling_rate_hz,
    )

    authorities: list[Any] = []
    case_results: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    observation_roles: list[dict[str, Any]] = []

    # Pass 2: reload, verify exact materialization, then execute models.
    for subject in config.subjects:
        data, descriptor = collect_moabb_epochs(
            dataset,
            paradigm,
            subjects=[subject],
            dataset_id=base.KUMAR2024_DATASET_ID,
        )
        frozen_descriptor = subject_descriptors[str(subject)]
        if descriptor.sha256 != frozen_descriptor["descriptor_sha256"]:
            raise RuntimeError(
                f"second-pass MOABB epoch descriptor changed for subject {subject}"
            )
        shard = shards_by_subject[int(subject)]
        verify_processed_subject_shard(data, shard, subject=subject)
        observed = validate_observed_sessions(
            dataset_spec,
            ordered_group_values(data, split_unit="session"),
        )
        if observed != base.KUMAR2024_EXPECTED_SESSIONS:
            raise RuntimeError(
                f"Kumar2024 chronology changed on execution pass for subject {subject}: {observed}"
            )

        for target_session in config.target_sessions:
            authority = base._make_case_authority(
                data=data,
                dataset_spec=dataset_spec,
                subject=subject,
                target_session=target_session,
                config=config,
            )
            authorities.append(authority)
            for factory in factories:
                result = run_external_qualification_case(
                    data,
                    authority,
                    protocol,
                    factory,
                    execution_context=context,
                )
                case_results.append(
                    {
                        "subject": subject,
                        "original_protocol": authority.case_metadata["original_protocol"],
                        "held_out_session": target_session,
                        "method_spec": {
                            **factory.method_spec.to_dict(),
                            "method_spec_sha256": factory.method_spec.sha256,
                        },
                        "result": result.to_dict(),
                    }
                )
                flat_rows.extend(
                    base._flatten_result_row(row, authority) for row in result.rows
                )
                observation_roles.extend(
                    build_case_result_observation_roles(
                        authority=authority,
                        shard=shard,
                        result=result,
                    )
                )
        del data
        gc.collect()

    analysis = base.summarize_rows(flat_rows, config=config)
    method_specs = []
    for factory in factories:
        spec = factory.method_spec
        method_specs.append(
            {**spec.to_dict(), "method_spec_sha256": spec.sha256}
        )
    study_identity_payload = {
        "config_sha256": config.sha256,
        "dataset_lineage_sha256": lineage.lineage_sha256,
        "protocol_sha256": protocol.sha256,
        "preprocessing_authority_sha256": preprocessing_authority["sha256"],
        "study_materialization_sha256": materialization.sha256,
        "method_spec_sha256s": [
            item["method_spec_sha256"] for item in method_specs
        ],
        "case_authority_sha256s": [
            item.authority_sha256 for item in authorities
        ],
    }
    manifest = {
        "schema_version": 2,
        "study": "nsq-kumar2024-v1",
        "evidence_tier": "real_dataset",
        "bundle_generation": 2,
        "study_sha256": base._identity_sha256(
            "neuros.nsq_kumar2024_study.v2",
            study_identity_payload,
        ),
        "study_materialization_sha256": materialization.sha256,
        "environment_authority_sha256": environment.sha256,
        "raw_materialization_sha256": raw_evidence.authority.sha256,
        "config": config.to_dict(),
        "config_sha256": config.sha256,
        "preprocessing_authority": preprocessing_authority,
        "dataset_lineage": lineage.to_dict(),
        "protocol": {
            **protocol.to_dict(),
            "protocol_sha256": protocol.sha256,
        },
        "execution_context": {
            **context.to_dict(),
            "execution_context_sha256": context.sha256,
        },
        "method_specs": method_specs,
        "case_authority_sha256s": [
            item.authority_sha256 for item in authorities
        ],
        "case_result_sha256s": [
            item["result"]["result_sha256"] for item in case_results
        ],
        "subject_epoch_descriptors": subject_descriptors,
        "package_versions": versions,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_revision": base._git_revision(),
        "claim_boundary": (
            "offline comparative evidence for this exact materialized MOABB "
            "bar-feedback subset and prospective longitudinal protocol only"
        ),
        "not_claimed": [
            "reproduction of the original online GR/PAR intervention",
            "physiological mechanism",
            "hardware qualification",
            "online BCI efficacy",
            "clinical benefit",
            "ORION superiority",
        ],
    }

    raw_selection = {
        "schema_version": raw_evidence.schema_version,
        "loader_contract": raw_evidence.loader_contract,
        "selections": [item.to_dict() for item in raw_evidence.selections],
    }
    base._json_dump(output_path / "study_manifest.json", manifest)
    base._json_dump(
        output_path / "case_authorities.json",
        {
            "schema_version": 1,
            "authorities": [item.to_dict() for item in authorities],
        },
    )
    base._json_dump(
        output_path / "case_results.json",
        {"schema_version": 1, "case_results": case_results},
    )
    base._write_csv(output_path / "results.csv", flat_rows)
    base._json_dump(output_path / "analysis.json", analysis)
    base._json_dump(
        output_path / "materialization.json",
        materialization_manifest(
            materialization,
            raw_selection=raw_selection,
        ),
    )
    base._json_dump(
        output_path / "observation_roles.json",
        {
            "schema_version": 1,
            "study_materialization_sha256": materialization.sha256,
            "entries": observation_roles,
        },
    )
    (output_path / "report.md").write_text(
        _render_report_v2(
            config=config,
            lineage=lineage,
            protocol=protocol,
            preprocessing_authority=preprocessing_authority,
            rows=flat_rows,
            analysis=analysis,
            materialization=materialization,
        ),
        encoding="utf-8",
    )
    sealed = _seal_bundle_v2(output_path)
    verified = verify_bundle_v2(output_path)
    return {
        "study_sha256": manifest["study_sha256"],
        "bundle_sha256": sealed["bundle_sha256"],
        "bundle_schema_version": 2,
        "study_materialization_sha256": materialization.sha256,
        "environment_authority_sha256": environment.sha256,
        "raw_materialization_sha256": raw_evidence.authority.sha256,
        "protocol_sha256": protocol.sha256,
        "dataset_lineage_sha256": lineage.lineage_sha256,
        "cases": len(authorities),
        "result_rows": len(flat_rows),
        "verified": verified["verified"],
        "output": str(output_path),
    }


__all__ = [
    "BUNDLE_FILES_V2",
    "run_materialized_study",
    "verify_bundle_v2",
]
