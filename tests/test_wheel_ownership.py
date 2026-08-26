from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.check_wheel_ownership import inspect_wheels


def _write_wheel(
    directory: Path,
    filename: str,
    *,
    distribution: str,
    version: str = "1.0.0",
    payloads: tuple[str, ...] = (),
) -> Path:
    wheel = directory / filename
    dist_info = filename.split("-")[0].replace("-", "_")
    with ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"{dist_info}-{version}.dist-info/METADATA",
            "\n".join(
                [
                    "Metadata-Version: 2.1",
                    f"Name: {distribution}",
                    f"Version: {version}",
                    "",
                ]
            ),
        )
        for payload in payloads:
            archive.writestr(payload, b"fixture")
    return wheel


def _sdk(directory: Path) -> Path:
    return _write_wheel(
        directory,
        "neuros-2.1.0-py3-none-any.whl",
        distribution="neuros",
        version="2.1.0",
        payloads=("neuros/__init__.py", "neuros/cli.py"),
    )


def test_distinct_namespace_portions_have_unique_file_ownership(tmp_path: Path):
    _sdk(tmp_path)
    _write_wheel(
        tmp_path,
        "neuros_core-2.0.0-py3-none-any.whl",
        distribution="neuros-core",
        version="2.0.0",
        payloads=("neuros/contracts.py", "neuros/runtime/__init__.py"),
    )
    _write_wheel(
        tmp_path,
        "neuros_drivers-2.0.0-py3-none-any.whl",
        distribution="neuros-drivers",
        version="2.0.0",
        payloads=("neuros/drivers/__init__.py",),
    )

    report = inspect_wheels(tmp_path)

    assert report["status"] == "pass"
    assert report["collisions"] == []
    assert report["sdk_root"] == {
        "path": "neuros/__init__.py",
        "expected_owner": "neuros",
        "owners": ["neuros"],
        "pass": True,
    }


def test_purelib_payload_is_normalized_to_real_install_destination(tmp_path: Path):
    _sdk(tmp_path)
    _write_wheel(
        tmp_path,
        "neuros_core-2.0.0-py3-none-any.whl",
        distribution="neuros-core",
        version="2.0.0",
        payloads=("neuros/contracts.py",),
    )
    _write_wheel(
        tmp_path,
        "external_plugin-1.0.0-py3-none-any.whl",
        distribution="external-plugin",
        payloads=("external_plugin-1.0.0.data/purelib/neuros/contracts.py",),
    )

    report = inspect_wheels(tmp_path)

    assert report["status"] == "fail"
    assert {
        "path": "neuros/contracts.py",
        "owners": ["external-plugin", "neuros-core"],
    } in report["collisions"]


def test_single_wheel_cannot_encode_same_install_target_twice(tmp_path: Path):
    _write_wheel(
        tmp_path,
        "neuros-2.1.0-py3-none-any.whl",
        distribution="neuros",
        version="2.1.0",
        payloads=(
            "neuros/__init__.py",
            "neuros/cli.py",
            "neuros-2.1.0.data/purelib/neuros/cli.py",
        ),
    )

    with pytest.raises(ValueError, match="multiple archive members install to 'neuros/cli.py'"):
        inspect_wheels(tmp_path)


def test_component_cannot_coown_sdk_root_initializer(tmp_path: Path):
    _sdk(tmp_path)
    _write_wheel(
        tmp_path,
        "neuros_models-2.0.0-py3-none-any.whl",
        distribution="neuros-models",
        version="2.0.0",
        payloads=("neuros/__init__.py", "neuros/models/__init__.py"),
    )

    report = inspect_wheels(tmp_path)

    assert report["status"] == "fail"
    assert report["sdk_root"]["owners"] == ["neuros", "neuros-models"]
    assert report["sdk_root"]["pass"] is False
    assert {
        "path": "neuros/__init__.py",
        "owners": ["neuros", "neuros-models"],
    } in report["collisions"]


def test_sdk_root_must_exist_even_when_other_payloads_are_collision_free(tmp_path: Path):
    _write_wheel(
        tmp_path,
        "neuros_core-2.0.0-py3-none-any.whl",
        distribution="neuros-core",
        version="2.0.0",
        payloads=("neuros/contracts.py",),
    )

    report = inspect_wheels(tmp_path)

    assert report["collisions"] == []
    assert report["status"] == "fail"
    assert report["sdk_root"]["owners"] == []


def test_distribution_names_are_canonicalized_before_duplicate_detection(tmp_path: Path):
    _write_wheel(
        tmp_path,
        "a-1.0.0-py3-none-any.whl",
        distribution="NeurOS.Core",
        payloads=("a.py",),
    )
    _write_wheel(
        tmp_path,
        "b-1.0.0-py3-none-any.whl",
        distribution="neuros_core",
        payloads=("b.py",),
    )

    with pytest.raises(ValueError, match="duplicate distribution name"):
        inspect_wheels(tmp_path)


def test_unsafe_parent_traversal_member_fails_closed(tmp_path: Path):
    _sdk(tmp_path)
    _write_wheel(
        tmp_path,
        "bad-1.0.0-py3-none-any.whl",
        distribution="bad",
        payloads=("../neuros/contracts.py",),
    )

    with pytest.raises(ValueError, match="unsafe or ambiguous wheel member path"):
        inspect_wheels(tmp_path)


def test_missing_wheel_directory_content_fails_closed(tmp_path: Path):
    with pytest.raises(ValueError, match="no wheels found"):
        inspect_wheels(tmp_path)
