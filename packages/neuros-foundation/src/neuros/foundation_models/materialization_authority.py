"""Content-addressed study materialization authority for promoted NSQ evidence.

This module answers a narrower question than model qualification itself:

    *what exact study materialization did this result come from?*

A promoted study should be able to bind the realized software environment, the
raw files actually consumed by the data loader, the ordered human-readable
identity of every processed observation, and the processed/preprocessing
contract without depending on machine-local cache roots or volatile host state.

The authority objects here deliberately contain no target labels. Observation
identity is evidence metadata, not a side channel through which a model may learn
anything about untouched final-assessment outcomes.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .real_world import GroupedEvaluationData

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DIST_NORMALIZE_RE = re.compile(r"[-_.]+")


def _canonical_sha256(schema: str, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        {"schema": schema, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _nonempty(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _sha256(name: str, value: Any) -> str:
    text = _nonempty(name, value).lower()
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _exact_nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _processed_observation_sha256(value: Any) -> str:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("processed observation must be numeric")
    if not np.isfinite(array).all():
        raise ValueError("processed observation must contain only finite values")
    contiguous = np.ascontiguousarray(array)
    payload = {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "bytes_sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }
    return _canonical_sha256("neuros.processed_observation.v1", payload)


def _normalized_distribution_name(value: Any) -> str:
    return _DIST_NORMALIZE_RE.sub("-", _nonempty("distribution name", value)).lower()


def _string_pairs(name: str, values: Mapping[str, Any] | Iterable[tuple[Any, Any]]) -> tuple[tuple[str, str], ...]:
    items = values.items() if isinstance(values, Mapping) else values
    normalized: dict[str, str] = {}
    for raw_key, raw_value in items:
        key = _nonempty(f"{name} key", raw_key)
        value = _nonempty(f"{name}[{key}]", raw_value)
        if key in normalized and normalized[key] != value:
            raise ValueError(f"{name} contains conflicting values for {key!r}")
        normalized[key] = value
    return tuple(sorted(normalized.items()))


@dataclass(frozen=True, slots=True)
class EnvironmentDistribution:
    """One exact installed Python distribution identity."""

    name: str
    version: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("EnvironmentDistribution schema_version must be 1")
        object.__setattr__(self, "name", _normalized_distribution_name(self.name))
        object.__setattr__(self, "version", _nonempty("distribution version", self.version))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentAuthority:
    """Canonical realized execution environment without volatile host noise.

    Hostname, timestamps, process IDs, temporary paths, runner IDs, free memory,
    and similar telemetry are intentionally absent. They may be logged elsewhere,
    but they are not reproducibility authority.
    """

    python_implementation: str
    python_version: str
    platform_system: str
    platform_machine: str
    distributions: tuple[EnvironmentDistribution, ...]
    source_revision: str | None = None
    accelerator_runtime: tuple[tuple[str, str], ...] = ()
    deterministic_flags: tuple[tuple[str, str], ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("EnvironmentAuthority schema_version must be 1")
        implementation = _nonempty("python_implementation", self.python_implementation)
        python_version = _nonempty("python_version", self.python_version)
        system = _nonempty("platform_system", self.platform_system)
        machine = _nonempty("platform_machine", self.platform_machine)
        distributions = tuple(
            sorted(self.distributions, key=lambda item: (item.name, item.version))
        )
        if not distributions:
            raise ValueError("environment authority requires at least one distribution")
        if any(not isinstance(item, EnvironmentDistribution) for item in distributions):
            raise TypeError("distributions must contain EnvironmentDistribution objects")
        names = [item.name for item in distributions]
        if len(set(names)) != len(names):
            raise ValueError("environment authority cannot contain duplicate distribution names")
        revision = None
        if self.source_revision is not None:
            revision = _nonempty("source_revision", self.source_revision)
        object.__setattr__(self, "python_implementation", implementation)
        object.__setattr__(self, "python_version", python_version)
        object.__setattr__(self, "platform_system", system)
        object.__setattr__(self, "platform_machine", machine)
        object.__setattr__(self, "distributions", distributions)
        object.__setattr__(self, "source_revision", revision)
        object.__setattr__(
            self,
            "accelerator_runtime",
            _string_pairs("accelerator_runtime", self.accelerator_runtime),
        )
        object.__setattr__(
            self,
            "deterministic_flags",
            _string_pairs("deterministic_flags", self.deterministic_flags),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "python": {
                "implementation": self.python_implementation,
                "version": self.python_version,
            },
            "platform": {
                "system": self.platform_system,
                "machine": self.platform_machine,
            },
            "distributions": [item.to_dict() for item in self.distributions],
            "source_revision": self.source_revision,
            "accelerator_runtime": dict(self.accelerator_runtime),
            "deterministic_flags": dict(self.deterministic_flags),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256("neuros.environment_authority.v1", self.to_dict())


def capture_environment_authority(
    *,
    distribution_names: Sequence[str] | None = None,
    source_revision: str | None = None,
    accelerator_runtime: Mapping[str, Any] | Iterable[tuple[Any, Any]] = (),
    deterministic_flags: Mapping[str, Any] | Iterable[tuple[Any, Any]] = (),
) -> EnvironmentAuthority:
    """Capture a canonical Python environment authority.

    When ``distribution_names`` is ``None`` every installed Python distribution
    is captured. Promoted studies should execute in a purpose-built environment
    so this complete realized set is scientifically meaningful rather than a
    random workstation's package collection. A caller may provide an explicit
    distribution set for narrower integration evidence.
    """

    distributions: list[EnvironmentDistribution] = []
    if distribution_names is None:
        seen: dict[str, str] = {}
        for distribution in importlib.metadata.distributions():
            raw_name = distribution.metadata.get("Name")
            if raw_name is None:
                continue
            name = _normalized_distribution_name(raw_name)
            version = _nonempty(f"distribution version for {name}", distribution.version)
            previous = seen.get(name)
            if previous is not None and previous != version:
                raise RuntimeError(
                    f"environment exposes conflicting installed versions for {name!r}: "
                    f"{previous!r} versus {version!r}"
                )
            seen[name] = version
        distributions = [
            EnvironmentDistribution(name=name, version=version)
            for name, version in sorted(seen.items())
        ]
    else:
        requested = sorted({_normalized_distribution_name(name) for name in distribution_names})
        for name in requested:
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError as exc:
                raise ValueError(
                    f"requested environment distribution {name!r} is not installed"
                ) from exc
            distributions.append(EnvironmentDistribution(name=name, version=version))

    return EnvironmentAuthority(
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        distributions=tuple(distributions),
        source_revision=source_revision,
        accelerator_runtime=_string_pairs("accelerator_runtime", accelerator_runtime),
        deterministic_flags=_string_pairs("deterministic_flags", deterministic_flags),
    )


@dataclass(frozen=True, slots=True)
class RawMaterializationFile:
    """Path-independent identity of one raw file consumed by a study loader."""

    logical_path: str
    size_bytes: int
    sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("RawMaterializationFile schema_version must be 1")
        path = PurePosixPath(_nonempty("logical_path", self.logical_path))
        if path.is_absolute() or ".." in path.parts or "." in path.parts:
            raise ValueError("logical_path must be a canonical dataset-relative path")
        normalized = path.as_posix()
        object.__setattr__(self, "logical_path", normalized)
        object.__setattr__(self, "size_bytes", _exact_nonnegative_int("size_bytes", self.size_bytes))
        object.__setattr__(self, "sha256", _sha256("raw file sha256", self.sha256))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "logical_path": self.logical_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class RawMaterializationAuthority:
    """Exact byte identity of the raw files materialized for one logical dataset."""

    dataset_id: str
    files: tuple[RawMaterializationFile, ...]
    upstream_identity: tuple[tuple[str, str], ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("RawMaterializationAuthority schema_version must be 1")
        dataset_id = _nonempty("dataset_id", self.dataset_id)
        files = tuple(sorted(self.files, key=lambda item: item.logical_path))
        if not files:
            raise ValueError("raw materialization authority requires at least one file")
        if any(not isinstance(item, RawMaterializationFile) for item in files):
            raise TypeError("files must contain RawMaterializationFile objects")
        logical_paths = [item.logical_path for item in files]
        if len(set(logical_paths)) != len(logical_paths):
            raise ValueError("raw materialization cannot contain duplicate logical paths")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "files", files)
        object.__setattr__(
            self,
            "upstream_identity",
            _string_pairs("upstream_identity", self.upstream_identity),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "upstream_identity": dict(self.upstream_identity),
            "files": [item.to_dict() for item in self.files],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256("neuros.raw_materialization_authority.v1", self.to_dict())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_raw_materialization(
    *,
    dataset_id: str,
    root: str | Path,
    relative_paths: Sequence[str | Path],
    upstream_identity: Mapping[str, Any] | Iterable[tuple[Any, Any]] = (),
) -> RawMaterializationAuthority:
    """Hash consumed raw files without binding a machine-local cache root."""

    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise ValueError("raw materialization root must be an existing directory")
    if not relative_paths:
        raise ValueError("relative_paths must name at least one consumed raw file")

    files: list[RawMaterializationFile] = []
    seen: set[str] = set()
    for raw_relative in relative_paths:
        logical = PurePosixPath(str(raw_relative).replace("\\", "/"))
        if logical.is_absolute() or ".." in logical.parts or "." in logical.parts:
            raise ValueError("raw paths must be canonical paths relative to root")
        logical_text = logical.as_posix()
        if logical_text in seen:
            raise ValueError(f"duplicate raw logical path {logical_text!r}")
        seen.add(logical_text)
        resolved = (base / Path(*logical.parts)).resolve()
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise ValueError(
                f"raw materialization path escapes declared root: {logical_text!r}"
            ) from exc
        if not resolved.is_file():
            raise ValueError(f"raw materialization file does not exist: {logical_text!r}")
        files.append(
            RawMaterializationFile(
                logical_path=logical_text,
                size_bytes=resolved.stat().st_size,
                sha256=_file_sha256(resolved),
            )
        )

    return RawMaterializationAuthority(
        dataset_id=dataset_id,
        files=tuple(files),
        upstream_identity=_string_pairs("upstream_identity", upstream_identity),
    )


@dataclass(frozen=True, slots=True)
class ObservationIdentity:
    """Human-auditable identity for one processed observation, excluding labels."""

    row_index: int
    participant: str
    session: str
    run: str | None
    local_epoch: int
    processed_observation_sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("ObservationIdentity schema_version must be 1")
        object.__setattr__(self, "row_index", _exact_nonnegative_int("row_index", self.row_index))
        object.__setattr__(self, "participant", _nonempty("participant", self.participant))
        object.__setattr__(self, "session", _nonempty("session", self.session))
        run = None if self.run is None else _nonempty("run", self.run)
        object.__setattr__(self, "run", run)
        object.__setattr__(self, "local_epoch", _exact_nonnegative_int("local_epoch", self.local_epoch))
        object.__setattr__(
            self,
            "processed_observation_sha256",
            _sha256("processed_observation_sha256", self.processed_observation_sha256),
        )

    @property
    def display_id(self) -> str:
        run = "<none>" if self.run is None else self.run
        return (
            f"participant={self.participant}/session={self.session}/"
            f"run={run}/epoch={self.local_epoch}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "row_index": self.row_index,
            "participant": self.participant,
            "session": self.session,
            "run": self.run,
            "local_epoch": self.local_epoch,
            "processed_observation_sha256": self.processed_observation_sha256,
            "display_id": self.display_id,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256("neuros.observation_identity.v1", self.to_dict())


@dataclass(frozen=True, slots=True)
class ObservationIdentityAuthority:
    """Ordered identity manifest for every processed observation in a study array."""

    dataset_id: str
    observations: tuple[ObservationIdentity, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("ObservationIdentityAuthority schema_version must be 1")
        dataset_id = _nonempty("dataset_id", self.dataset_id)
        observations = tuple(self.observations)
        if not observations:
            raise ValueError("observation identity authority cannot be empty")
        if any(not isinstance(item, ObservationIdentity) for item in observations):
            raise TypeError("observations must contain ObservationIdentity objects")
        for expected, observation in enumerate(observations):
            if observation.row_index != expected:
                raise ValueError(
                    "observation row_index must exactly match processed array order"
                )
        semantic = [
            (item.participant, item.session, item.run, item.local_epoch)
            for item in observations
        ]
        if len(set(semantic)) != len(semantic):
            raise ValueError("observation identities are ambiguous or duplicated")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "observations", observations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "observations": [item.to_dict() for item in self.observations],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256("neuros.observation_identity_authority.v1", self.to_dict())

    def role(self, role: str, row_indices: Sequence[int]) -> "ObservationRoleAuthority":
        indices = tuple(_exact_nonnegative_int("row index", value) for value in row_indices)
        if len(set(indices)) != len(indices):
            raise ValueError("observation role cannot contain duplicate row indices")
        if any(index >= len(self.observations) for index in indices):
            raise ValueError("observation role row index exceeds identity authority")
        selected = tuple(self.observations[index] for index in indices)
        return ObservationRoleAuthority(
            dataset_id=self.dataset_id,
            observation_identity_authority_sha256=self.sha256,
            role=role,
            row_indices=indices,
            observation_sha256s=tuple(item.sha256 for item in selected),
            display_ids=tuple(item.display_id for item in selected),
        )


def observation_identities_from_grouped_data(
    data: GroupedEvaluationData,
) -> ObservationIdentityAuthority:
    """Derive deterministic MOABB-style identities from grouped processed rows.

    Local epoch ordinals are counted within participant/session/run in processed
    row order. Labels are deliberately not read or serialized.
    """

    if not isinstance(data, GroupedEvaluationData):
        raise TypeError("data must be GroupedEvaluationData")
    groups = data.groups
    if "subject" not in groups or "session" not in groups:
        raise ValueError(
            "human observation identity requires subject and session groups"
        )
    participant = groups["subject"]
    session = groups["session"]
    run_values = groups.get("run")
    counters: dict[tuple[str, str, str | None], int] = {}
    observations: list[ObservationIdentity] = []
    for row_index in range(len(data.X)):
        participant_id = str(participant[row_index])
        session_id = str(session[row_index])
        run_id = None if run_values is None else str(run_values[row_index])
        key = (participant_id, session_id, run_id)
        local_epoch = counters.get(key, 0)
        counters[key] = local_epoch + 1
        observations.append(
            ObservationIdentity(
                row_index=row_index,
                participant=participant_id,
                session=session_id,
                run=run_id,
                local_epoch=local_epoch,
                processed_observation_sha256=_processed_observation_sha256(
                    data.X[row_index]
                ),
            )
        )
    return ObservationIdentityAuthority(
        dataset_id=data.dataset_id,
        observations=tuple(observations),
    )


@dataclass(frozen=True, slots=True)
class ObservationRoleAuthority:
    """Inspectable role membership for one exact ordered subset of observations."""

    dataset_id: str
    observation_identity_authority_sha256: str
    role: str
    row_indices: tuple[int, ...]
    observation_sha256s: tuple[str, ...]
    display_ids: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("ObservationRoleAuthority schema_version must be 1")
        object.__setattr__(self, "dataset_id", _nonempty("dataset_id", self.dataset_id))
        object.__setattr__(
            self,
            "observation_identity_authority_sha256",
            _sha256(
                "observation_identity_authority_sha256",
                self.observation_identity_authority_sha256,
            ),
        )
        object.__setattr__(self, "role", _nonempty("role", self.role))
        indices = tuple(_exact_nonnegative_int("row index", value) for value in self.row_indices)
        hashes = tuple(_sha256("observation sha256", value) for value in self.observation_sha256s)
        display = tuple(_nonempty("display_id", value) for value in self.display_ids)
        if len(set(indices)) != len(indices):
            raise ValueError("observation role authority cannot duplicate rows")
        if not (len(indices) == len(hashes) == len(display)):
            raise ValueError("observation role fields must have identical lengths")
        object.__setattr__(self, "row_indices", indices)
        object.__setattr__(self, "observation_sha256s", hashes)
        object.__setattr__(self, "display_ids", display)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "observation_identity_authority_sha256": self.observation_identity_authority_sha256,
            "role": self.role,
            "row_indices": list(self.row_indices),
            "observation_sha256s": list(self.observation_sha256s),
            "display_ids": list(self.display_ids),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256("neuros.observation_role_authority.v1", self.to_dict())


@dataclass(frozen=True, slots=True)
class ProcessedMaterializationShard:
    """One independently loaded processed shard within a materialized study.

    Longitudinal EEG studies are commonly loaded participant-by-participant.
    Preserving that native boundary avoids manufacturing global row offsets and
    keeps case-local NSQ indices directly compatible with the evidence manifest.
    """

    shard_id: str
    processed_data_sha256: str
    observation_identity: ObservationIdentityAuthority
    preprocessing_authority_sha256s: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("ProcessedMaterializationShard schema_version must be 1")
        object.__setattr__(self, "shard_id", _nonempty("shard_id", self.shard_id))
        object.__setattr__(
            self,
            "processed_data_sha256",
            _sha256("processed_data_sha256", self.processed_data_sha256),
        )
        if not isinstance(self.observation_identity, ObservationIdentityAuthority):
            raise TypeError("observation_identity must be ObservationIdentityAuthority")
        preprocessing = tuple(
            _sha256("preprocessing authority sha256", value)
            for value in self.preprocessing_authority_sha256s
        )
        if not preprocessing:
            raise ValueError("processed shard requires preprocessing authority")
        if len(set(preprocessing)) != len(preprocessing):
            raise ValueError("preprocessing authorities cannot contain duplicates")
        object.__setattr__(self, "preprocessing_authority_sha256s", preprocessing)

    @property
    def dataset_id(self) -> str:
        return self.observation_identity.dataset_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "shard_id": self.shard_id,
            "processed_data_sha256": self.processed_data_sha256,
            "preprocessing_authority_sha256s": list(
                self.preprocessing_authority_sha256s
            ),
            "observation_identity": self.observation_identity.to_dict(),
            "observation_identity_sha256": self.observation_identity.sha256,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256("neuros.processed_materialization_shard.v1", self.to_dict())


@dataclass(frozen=True, slots=True)
class StudyMaterializationAuthority:
    """Strict composed identity of one realized, potentially sharded study."""

    environment: EnvironmentAuthority
    raw_materialization: RawMaterializationAuthority
    processed_shards: tuple[ProcessedMaterializationShard, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("StudyMaterializationAuthority schema_version must be 1")
        if not isinstance(self.environment, EnvironmentAuthority):
            raise TypeError("environment must be EnvironmentAuthority")
        if not isinstance(self.raw_materialization, RawMaterializationAuthority):
            raise TypeError("raw_materialization must be RawMaterializationAuthority")
        shards = tuple(sorted(self.processed_shards, key=lambda item: item.shard_id))
        if not shards:
            raise ValueError("study materialization requires at least one processed shard")
        if any(not isinstance(item, ProcessedMaterializationShard) for item in shards):
            raise TypeError("processed_shards must contain ProcessedMaterializationShard objects")
        shard_ids = [item.shard_id for item in shards]
        if len(set(shard_ids)) != len(shard_ids):
            raise ValueError("processed study shards must have unique shard_id values")
        if any(item.dataset_id != self.raw_materialization.dataset_id for item in shards):
            raise ValueError("raw and processed shard authorities must describe one dataset_id")
        object.__setattr__(self, "processed_shards", shards)

    @property
    def dataset_id(self) -> str:
        return self.raw_materialization.dataset_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "environment": self.environment.to_dict(),
            "environment_sha256": self.environment.sha256,
            "raw_materialization": self.raw_materialization.to_dict(),
            "raw_materialization_sha256": self.raw_materialization.sha256,
            "processed_shards": [item.to_dict() for item in self.processed_shards],
            "processed_shard_sha256s": [item.sha256 for item in self.processed_shards],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256("neuros.study_materialization_authority.v1", self.to_dict())


__all__ = [
    "EnvironmentAuthority",
    "EnvironmentDistribution",
    "ObservationIdentity",
    "ObservationIdentityAuthority",
    "ObservationRoleAuthority",
    "ProcessedMaterializationShard",
    "RawMaterializationAuthority",
    "RawMaterializationFile",
    "StudyMaterializationAuthority",
    "capture_environment_authority",
    "hash_raw_materialization",
    "observation_identities_from_grouped_data",
]
