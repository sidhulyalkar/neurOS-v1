"""Raw-file selection authority for MOABB evidence datasets.

MOABB datasets frequently download archives containing more material than a
particular dataset adapter actually exposes. Qualification must hash the files
that the installed loader consumes, not every byte that happens to share its
cache directory.

The Kumar2024 resolver below intentionally follows the installed MOABB
``Kumar2024`` adapter's own directory-resolution helpers and its documented
``*.gdf``/``*.GDF`` run selection. If that upstream contract changes, promoted
evidence fails closed until this adapter is reviewed against the new loader.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .materialization_authority import (
    RawMaterializationAuthority,
    hash_raw_materialization,
)

KUMAR2024_EXPECTED_RUNS = {
    "0": 4,
    "1": 4,
    "2": 3,
    "3": 3,
    "4": 3,
    "5": 3,
}
KUMAR2024_PAPER_DOI = "10.1093/pnasnexus/pgae076"
KUMAR2024_DATA_DOI = "10.5281/zenodo.10694880"
KUMAR2024_ZENODO_RECORD = "https://zenodo.org/records/10694880"


def _nonempty(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _gdf_files(session_dir: Path) -> tuple[Path, ...]:
    files = tuple(sorted(session_dir.glob("*.gdf")))
    if not files:
        files = tuple(sorted(session_dir.glob("*.GDF")))
    return files


@dataclass(frozen=True, slots=True)
class MOABBRawRunSelection:
    """One raw run selected by an installed MOABB dataset adapter."""

    subject: int
    raw_subject: int
    original_protocol: str
    session: str
    run: str
    logical_path: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("MOABBRawRunSelection schema_version must be 1")
        if isinstance(self.subject, bool) or not isinstance(self.subject, int):
            raise ValueError("subject must be an integer")
        if isinstance(self.raw_subject, bool) or not isinstance(self.raw_subject, int):
            raise ValueError("raw_subject must be an integer")
        if not 1 <= self.subject <= 18:
            raise ValueError("Kumar2024 subject must lie in 1..18")
        if self.raw_subject <= 0:
            raise ValueError("raw_subject must be positive")
        protocol = _nonempty("original_protocol", self.original_protocol)
        if protocol not in {"GR", "PAR"}:
            raise ValueError("original_protocol must be GR or PAR")
        session = _nonempty("session", self.session)
        if session not in KUMAR2024_EXPECTED_RUNS:
            raise ValueError("unexpected Kumar2024 session")
        run = _nonempty("run", self.run)
        logical_path = _nonempty("logical_path", self.logical_path).replace("\\", "/")
        if logical_path.startswith("/") or "/../" in f"/{logical_path}/":
            raise ValueError("logical_path must be dataset-relative")
        object.__setattr__(self, "original_protocol", protocol)
        object.__setattr__(self, "session", session)
        object.__setattr__(self, "run", run)
        object.__setattr__(self, "logical_path", logical_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "subject": self.subject,
            "raw_subject": self.raw_subject,
            "original_protocol": self.original_protocol,
            "session": self.session,
            "run": self.run,
            "logical_path": self.logical_path,
        }


@dataclass(frozen=True, slots=True)
class MOABBRawMaterializationEvidence:
    """Byte authority plus the subject/session/run paths that produced it."""

    authority: RawMaterializationAuthority
    selections: tuple[MOABBRawRunSelection, ...]
    loader_contract: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("MOABBRawMaterializationEvidence schema_version must be 1")
        if not isinstance(self.authority, RawMaterializationAuthority):
            raise TypeError("authority must be RawMaterializationAuthority")
        selections = tuple(self.selections)
        if not selections:
            raise ValueError("raw materialization evidence requires selected runs")
        if any(not isinstance(item, MOABBRawRunSelection) for item in selections):
            raise TypeError("selections must contain MOABBRawRunSelection objects")
        logical_paths = [item.logical_path for item in selections]
        if len(set(logical_paths)) != len(logical_paths):
            raise ValueError("raw run selections cannot duplicate logical paths")
        authority_paths = [item.logical_path for item in self.authority.files]
        if sorted(logical_paths) != sorted(authority_paths):
            raise ValueError("raw authority files differ from selected MOABB runs")
        object.__setattr__(self, "selections", selections)
        object.__setattr__(self, "loader_contract", _nonempty("loader_contract", self.loader_contract))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "loader_contract": self.loader_contract,
            "raw_materialization_sha256": self.authority.sha256,
            "authority": self.authority.to_dict(),
            "selections": [item.to_dict() for item in self.selections],
        }


def _require_kumar2024_loader_contract(dataset: Any) -> None:
    if dataset.__class__.__name__ != "Kumar2024":
        raise TypeError("Kumar2024 raw materialization requires a Kumar2024 dataset instance")
    for name in (
        "data_path",
        "_MOABB_TO_RAW",
        "_find_online_subject_dir",
        "_find_session_subdir",
    ):
        if not hasattr(dataset, name):
            raise RuntimeError(
                f"installed MOABB Kumar2024 no longer exposes required loader contract {name!r}"
            )


def _session_files_or_fail(
    session_dir: Path | None,
    *,
    subject: int,
    session: str,
) -> tuple[Path, ...]:
    if session_dir is None or not session_dir.is_dir():
        raise FileNotFoundError(
            f"Kumar2024 raw session directory is missing for subject={subject}, session={session}"
        )
    files = _gdf_files(session_dir)
    if not files:
        raise FileNotFoundError(
            f"Kumar2024 raw GDF runs are missing for subject={subject}, session={session}"
        )
    expected = KUMAR2024_EXPECTED_RUNS[session]
    if len(files) != expected:
        raise RuntimeError(
            "Kumar2024 raw run count differs from frozen bar-feedback contract: "
            f"subject={subject}, session={session}, expected={expected}, observed={len(files)}"
        )
    return files


def resolve_kumar2024_raw_materialization(
    dataset: Any,
    *,
    subjects: Sequence[int],
) -> MOABBRawMaterializationEvidence:
    """Resolve and hash exactly the GDF files consumed by MOABB Kumar2024.

    The installed dataset adapter downloads a single archive containing Offline,
    Online, and Race material. Its `_get_single_subject_data` path, however,
    exposes only bar-feedback GDF files under Offline/ and Online/. This function
    mirrors that exact selection and never includes Race files in the scientific
    raw-input authority.
    """

    _require_kumar2024_loader_contract(dataset)
    normalized_subjects = tuple(int(subject) for subject in subjects)
    if not normalized_subjects or len(set(normalized_subjects)) != len(normalized_subjects):
        raise ValueError("subjects must be non-empty and unique")
    if any(subject not in range(1, 19) for subject in normalized_subjects):
        raise ValueError("Kumar2024 subjects must lie in 1..18")

    roots: list[Path] = []
    selections: list[MOABBRawRunSelection] = []
    mapping = dataset._MOABB_TO_RAW

    for subject in normalized_subjects:
        try:
            raw_subject = int(mapping[subject])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"installed Kumar2024 raw subject mapping is unavailable for subject {subject}"
            ) from exc
        root = Path(dataset.data_path(subject)).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"Kumar2024 data_path did not return an extracted directory for subject {subject}"
            )
        roots.append(root)
        protocol = "GR" if raw_subject <= 9 else "PAR"

        offline_subject = root / "Offline" / protocol / f"Subject_{raw_subject:02d}_Offline"
        offline_session = dataset._find_session_subdir(
            offline_subject, raw_subject, 1, "Offline"
        )
        session_files: list[tuple[str, tuple[Path, ...]]] = [
            (
                "0",
                _session_files_or_fail(
                    offline_session,
                    subject=subject,
                    session="0",
                ),
            )
        ]

        online_subject = dataset._find_online_subject_dir(
            root / "Online" / protocol,
            raw_subject,
        )
        if online_subject is None or not Path(online_subject).is_dir():
            raise FileNotFoundError(
                f"Kumar2024 online subject directory is missing for subject {subject}"
            )
        for session_number in range(2, 7):
            moabb_session = str(session_number - 1)
            session_dir = dataset._find_session_subdir(
                Path(online_subject), raw_subject, session_number, "Online"
            )
            session_files.append(
                (
                    moabb_session,
                    _session_files_or_fail(
                        session_dir,
                        subject=subject,
                        session=moabb_session,
                    ),
                )
            )

        for moabb_session, files in session_files:
            for run_index, path in enumerate(files):
                resolved = path.resolve()
                try:
                    logical = resolved.relative_to(root).as_posix()
                except ValueError as exc:
                    raise RuntimeError(
                        "Kumar2024 loader selected a GDF outside its declared extracted root"
                    ) from exc
                selections.append(
                    MOABBRawRunSelection(
                        subject=subject,
                        raw_subject=raw_subject,
                        original_protocol=protocol,
                        session=moabb_session,
                        run=str(run_index),
                        logical_path=logical,
                    )
                )

    first_root = roots[0]
    if any(root != first_root for root in roots[1:]):
        raise RuntimeError(
            "Kumar2024 subjects resolved to different extracted roots; "
            "promoted raw materialization requires one canonical archive materialization"
        )

    try:
        moabb_version = importlib.metadata.version("moabb")
    except importlib.metadata.PackageNotFoundError:
        moabb_version = "unknown"
    authority = hash_raw_materialization(
        dataset_id="moabb-kumar2024",
        root=first_root,
        relative_paths=tuple(item.logical_path for item in selections),
        upstream_identity={
            "dataset_class": "moabb.datasets.Kumar2024",
            "moabb_version": moabb_version,
            "paper_doi": KUMAR2024_PAPER_DOI,
            "data_doi": KUMAR2024_DATA_DOI,
            "repository": KUMAR2024_ZENODO_RECORD,
            "included_task": "bar_feedback_only",
            "excluded_task": "car_racing",
        },
    )
    return MOABBRawMaterializationEvidence(
        authority=authority,
        selections=tuple(selections),
        loader_contract=(
            "Kumar2024.data_path + _MOABB_TO_RAW + _find_online_subject_dir + "
            "_find_session_subdir + sorted *.gdf fallback *.GDF"
        ),
    )


__all__ = [
    "KUMAR2024_EXPECTED_RUNS",
    "MOABBRawMaterializationEvidence",
    "MOABBRawRunSelection",
    "resolve_kumar2024_raw_materialization",
]
