from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from check_release_dependency_closure import inspect_release_dependency_closure  # noqa: E402


def _wheel(
    root: Path,
    *,
    name: str,
    version: str,
    requires: tuple[str, ...] = (),
) -> Path:
    normalized = name.replace("-", "_")
    path = root / f"{normalized}-{version}-py3-none-any.whl"
    metadata = [
        "Metadata-Version: 2.1",
        f"Name: {name}",
        f"Version: {version}",
    ]
    metadata.extend(f"Requires-Dist: {requirement}" for requirement in requires)
    metadata_text = "\n".join(metadata) + "\n\n"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(f"{normalized}/fixture.py", "VALUE = 1\n")
        archive.writestr(
            f"{normalized}-{version}.dist-info/METADATA",
            metadata_text,
        )
    return path


def test_internal_default_dependencies_must_be_present_and_version_compatible(tmp_path):
    _wheel(tmp_path, name="neuros-core", version="2.0.0")
    _wheel(
        tmp_path,
        name="neuros-drivers",
        version="2.1.0",
        requires=("neuros-core>=2.0.0", "numpy>=1.24"),
    )
    report = inspect_release_dependency_closure(tmp_path)
    assert report["status"] == "pass"
    assert report["distribution_count"] == 2
    assert report["dependencies"] == [
        {
            "from": "neuros-drivers",
            "requirement": "neuros-core>=2.0.0",
            "to": "neuros-core",
            "resolved_version": "2.0.0",
        }
    ]


def test_missing_internal_default_dependency_fails_closed(tmp_path):
    _wheel(
        tmp_path,
        name="neuros",
        version="2.1.0",
        requires=("neuros-models>=2.0.0",),
    )
    with pytest.raises(ValueError, match="unsatisfied internal runtime dependency"):
        inspect_release_dependency_closure(tmp_path)


def test_incompatible_internal_version_fails_closed(tmp_path):
    _wheel(tmp_path, name="neuros-core", version="2.0.0")
    _wheel(
        tmp_path,
        name="neuros-drivers",
        version="2.1.0",
        requires=("neuros-core>=3.0.0",),
    )
    with pytest.raises(ValueError, match="incompatible internal runtime dependency"):
        inspect_release_dependency_closure(tmp_path)


def test_active_internal_extra_requirement_fails_until_extra_closure_is_modeled(tmp_path):
    _wheel(tmp_path, name="neuros-drivers", version="2.1.0")
    _wheel(
        tmp_path,
        name="neuros",
        version="2.1.0",
        requires=("neuros-drivers[eeg]>=2.0.0",),
    )
    with pytest.raises(ValueError, match="requests extras"):
        inspect_release_dependency_closure(tmp_path)


def test_optional_internal_extra_edges_do_not_enter_default_runtime(tmp_path):
    _wheel(tmp_path, name="neuros-core", version="2.0.0")
    _wheel(
        tmp_path,
        name="neuros",
        version="2.1.0",
        requires=(
            "neuros-core>=2.0.0",
            'neuros-foundation>=2.0.0; extra == "research"',
        ),
    )
    report = inspect_release_dependency_closure(tmp_path)
    assert [item["to"] for item in report["dependencies"]] == ["neuros-core"]


def test_duplicate_distribution_fails_closed(tmp_path):
    first = _wheel(tmp_path, name="neuros-core", version="2.0.0")
    second = tmp_path / "alternate.whl"
    second.write_bytes(first.read_bytes())
    with pytest.raises(ValueError, match="duplicate distribution"):
        inspect_release_dependency_closure(tmp_path)
