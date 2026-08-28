from __future__ import annotations

import re
from pathlib import Path

import pytest

from neuros.foundation_models.moabb_materialization import (
    KUMAR2024_EXPECTED_RUNS,
    resolve_kumar2024_raw_materialization,
)


class Kumar2024:
    """Filesystem-only fixture mirroring the upstream loader path helpers."""

    _MOABB_TO_RAW = {i: i for i in range(1, 10)}
    _MOABB_TO_RAW.update({i: i + 1 for i in range(10, 19)})

    def __init__(self, root: Path):
        self.root = root

    def data_path(self, subject):
        return self.root

    @staticmethod
    def _find_online_subject_dir(online_group_dir, raw_subj):
        if not online_group_dir.is_dir():
            return None
        for pattern in [
            f"Subject_{raw_subj:02d}_Online",
            f"Subject_{raw_subj:03d}_Online",
            f"Subject_{raw_subj}_Online",
        ]:
            candidate = online_group_dir / pattern
            if candidate.is_dir():
                return candidate
        for child in sorted(online_group_dir.iterdir()):
            if child.is_dir():
                match = re.match(r"Subject_0*(\d+)_Online", child.name)
                if match and int(match.group(1)) == raw_subj:
                    return child
        return None

    @staticmethod
    def _find_session_subdir(parent_dir, raw_subj, sess_num, suffix):
        if parent_dir is None or not parent_dir.is_dir():
            return None
        for subject_format in [
            f"{raw_subj:02d}",
            f"{raw_subj:03d}",
            str(raw_subj),
        ]:
            candidate = (
                parent_dir
                / f"Subject_{subject_format}_Session_{sess_num:03d}_{suffix}"
            )
            if candidate.is_dir():
                return candidate
        for child in sorted(parent_dir.iterdir()):
            if child.is_dir():
                match = re.search(r"Session_0*(\d+)", child.name)
                if match and int(match.group(1)) == sess_num:
                    return child
        return None


def _write_subject(root: Path, *, subject: int) -> None:
    raw_subject = Kumar2024._MOABB_TO_RAW[subject]
    protocol = "GR" if raw_subject <= 9 else "PAR"

    offline_subject = root / "Offline" / protocol / f"Subject_{raw_subject:02d}_Offline"
    offline_session = (
        offline_subject
        / f"Subject_{raw_subject:02d}_Session_001_Offline"
    )
    offline_session.mkdir(parents=True)
    for run in range(KUMAR2024_EXPECTED_RUNS["0"]):
        (offline_session / f"bar_{run:02d}.gdf").write_bytes(
            f"subject={subject};session=0;run={run}".encode()
        )

    # Exercise the upstream 3-digit PAR naming variation for subject 10 -> raw 11.
    subject_width = 3 if raw_subject >= 10 else 2
    online_subject = (
        root
        / "Online"
        / protocol
        / f"Subject_{raw_subject:0{subject_width}d}_Online"
    )
    for session_number in range(2, 7):
        moabb_session = str(session_number - 1)
        session_dir = (
            online_subject
            / f"Subject_{raw_subject:0{subject_width}d}_Session_{session_number:03d}_Online"
        )
        session_dir.mkdir(parents=True)
        extension = "GDF" if moabb_session == "5" else "gdf"
        for run in range(KUMAR2024_EXPECTED_RUNS[moabb_session]):
            (session_dir / f"bar_{run:02d}.{extension}").write_bytes(
                f"subject={subject};session={moabb_session};run={run}".encode()
            )

    # This file exists in the extracted archive but must never enter bar authority.
    race = root / "Race" / protocol / f"Subject_{raw_subject:02d}" / "race.gdf"
    race.parent.mkdir(parents=True)
    race.write_bytes(b"not consumed by MOABB Kumar2024 bar loader")


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "Kumar2024"
    _write_subject(root, subject=1)
    _write_subject(root, subject=10)
    return root


def test_kumar_materialization_selects_only_exact_bar_feedback_runs(tmp_path: Path):
    root = _fixture_root(tmp_path)
    evidence = resolve_kumar2024_raw_materialization(
        Kumar2024(root),
        subjects=(1, 10),
    )

    expected_per_subject = sum(KUMAR2024_EXPECTED_RUNS.values())
    assert len(evidence.selections) == 2 * expected_per_subject
    assert len(evidence.authority.files) == 2 * expected_per_subject
    assert all("Race/" not in item.logical_path for item in evidence.selections)
    assert all("Race/" not in item.logical_path for item in evidence.authority.files)
    assert {item.subject for item in evidence.selections} == {1, 10}
    raw_by_subject = {
        subject: {item.raw_subject for item in evidence.selections if item.subject == subject}
        for subject in (1, 10)
    }
    assert raw_by_subject == {1: {1}, 10: {11}}
    assert {
        (item.subject, item.session): sum(
            1
            for other in evidence.selections
            if other.subject == item.subject and other.session == item.session
        )
        for item in evidence.selections
    } == {
        (subject, session): count
        for subject in (1, 10)
        for session, count in KUMAR2024_EXPECTED_RUNS.items()
    }
    assert evidence.authority.to_dict()["upstream_identity"]["included_task"] == (
        "bar_feedback_only"
    )
    assert evidence.authority.to_dict()["upstream_identity"]["excluded_task"] == (
        "car_racing"
    )


def test_kumar_materialization_is_root_independent_and_byte_sensitive(tmp_path: Path):
    first_root = _fixture_root(tmp_path / "one")
    second_root = _fixture_root(tmp_path / "two")
    first = resolve_kumar2024_raw_materialization(
        Kumar2024(first_root), subjects=(1, 10)
    )
    second = resolve_kumar2024_raw_materialization(
        Kumar2024(second_root), subjects=(1, 10)
    )
    assert first.authority.sha256 == second.authority.sha256

    selected = second_root / first.selections[0].logical_path
    selected.write_bytes(selected.read_bytes() + b"changed")
    changed = resolve_kumar2024_raw_materialization(
        Kumar2024(second_root), subjects=(1, 10)
    )
    assert first.authority.sha256 != changed.authority.sha256


def test_kumar_materialization_fails_if_frozen_bar_run_count_changes(tmp_path: Path):
    root = _fixture_root(tmp_path)
    first_session = next(
        path
        for path in root.rglob("*")
        if path.is_dir() and "Session_001_Offline" in path.name
    )
    (first_session / "unexpected-extra.gdf").write_bytes(b"unexpected")
    with pytest.raises(RuntimeError, match="run count differs"):
        resolve_kumar2024_raw_materialization(Kumar2024(root), subjects=(1,))


def test_kumar_materialization_fails_when_one_consumed_session_is_missing(tmp_path: Path):
    root = _fixture_root(tmp_path)
    target = next(
        path
        for path in root.rglob("*")
        if path.is_dir() and "Session_006_Online" in path.name
    )
    for child in target.iterdir():
        child.unlink()
    target.rmdir()
    with pytest.raises(FileNotFoundError, match="session=5"):
        resolve_kumar2024_raw_materialization(Kumar2024(root), subjects=(1,))


def test_kumar_materialization_rejects_non_kumar_loader(tmp_path: Path):
    class OtherDataset:
        pass

    with pytest.raises(TypeError, match="Kumar2024 dataset instance"):
        resolve_kumar2024_raw_materialization(OtherDataset(), subjects=(1,))
