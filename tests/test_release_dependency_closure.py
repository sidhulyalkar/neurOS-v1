from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from check_release_dependency_closure import (  # noqa: E402
    SUPPORTED_PYTHON_VERSIONS,
    TARGET_ENVIRONMENTS,
    inspect_release_dependency_closure,
)


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
    assert report["python_versions"] == list(SUPPORTED_PYTHON_VERSIONS)
    assert report["target_environments"] == [item[0] for item in TARGET_ENVIRONMENTS]
    assert len(report["dependencies"]) == 1
    dependency = report["dependencies"][0]
    assert dependency["from"] == "neuros-drivers"
    assert dependency["requirement"] == "neuros-core>=2.0.0"
    assert dependency["to"] == "neuros-core"
    assert dependency["resolved_version"] == "2.0.0"
    assert dependency["active_targets"] == report["target_environments"]


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


def test_python_gated_dependency_is_checked_across_supported_versions(tmp_path):
    _wheel(
        tmp_path,
        name="neuros",
        version="2.1.0",
        requires=('neuros-legacy>=1.0; python_version < "3.11"',),
    )
    with pytest.raises(ValueError, match="unsatisfied internal runtime dependency") as exc:
        inspect_release_dependency_closure(tmp_path)
    assert "cp3.10-" in str(exc.value)


def test_platform_gated_dependency_is_checked_beyond_linux_runner(tmp_path):
    _wheel(
        tmp_path,
        name="neuros",
        version="2.1.0",
        requires=('neuros-winbridge>=1.0; sys_platform == "win32"',),
    )
    with pytest.raises(ValueError, match="unsatisfied internal runtime dependency") as exc:
        inspect_release_dependency_closure(tmp_path)
    assert "windows-" in str(exc.value)


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
