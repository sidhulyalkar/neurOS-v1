from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
from copy import deepcopy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_script("check_reproducible_wheels.py")
builder = _load_script("build_reproducible_release.py")


def _manifest(
    *, root: str, sha: str = "a" * 64, external_attr: int = 2175008768
):
    return {
        "schema": checker.BUILD_SCHEMA,
        "_root": root,
        "source": {
            "revision": "1" * 40,
            "commit_timestamp_epoch": 1_800_000_000,
            "source_date_epoch": 1_800_000_000,
        },
        "policy": {
            "count": 1,
            "distributions": ["neuros"],
        },
        "build_authority": {
            "python_version": "3.11.9",
            "python_implementation": "CPython",
            "python_hash_seed": "0",
            "toolchain": {
                "pip": "26.2.1",
                "build": "1.6.0",
                "setuptools": "84.0.0",
                "wheel": "0.48.0",
                "packaging": "26.3",
                "pyproject-hooks": "1.2.0",
                "colorama": "absent",
            },
        },
        "builder_environment": {
            "system": "Linux",
            "machine": "x86_64",
        },
        "artifacts": [
            {
                "name": "neuros",
                "canonical_name": "neuros",
                "version": "2.1.0",
                "file": "neuros-2.1.0-py3-none-any.whl",
                "bytes": 123,
                "sha256": sha,
                "zip_entries": [
                    {
                        "name": "neuros/__init__.py",
                        "date_time": [2027, 1, 15, 8, 0, 0],
                        "compress_type": 8,
                        "crc": 1234,
                        "compressed_size": 20,
                        "file_size": 30,
                        "external_attr": external_attr,
                        "create_system": 3,
                    }
                ],
            }
        ],
    }


def _raw_test_wheel(path: Path, *, windows_style: bool) -> None:
    members = {
        "demo/__init__.py": b"VALUE = 1\n",
        "demo-1.0.0.dist-info/METADATA": (
            b"Metadata-Version: 2.1\r\nName: demo\r\nVersion: 1.0.0\r\n\r\n"
            if windows_style
            else b"Metadata-Version: 2.1\nName: demo\nVersion: 1.0.0\n\n"
        ),
        "demo-1.0.0.dist-info/WHEEL": (
            b"Wheel-Version: 1.0\r\nGenerator: test\r\nRoot-Is-Purelib: true\r\n"
            b"Tag: py3-none-any\r\n"
            if windows_style
            else b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
            b"Tag: py3-none-any\n"
        ),
        "demo-1.0.0.dist-info/RECORD": b"stale,sha256=not-authoritative,1\r\n",
    }
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, payload in reversed(list(members.items())):
            info = ZipInfo(name, (2026, 8, 30, 12, 34, 56))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 0 if windows_style else 3
            info.external_attr = (
                0o100666 << 16 if windows_style else 0o100644 << 16
            )
            archive.writestr(info, payload)


def _urlsafe_sha256(payload: bytes) -> str:
    digest = hashlib.sha256(payload).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def test_identical_builds_pass_even_with_platform_only_colorama_difference():
    left = _manifest(root="linux")
    right = deepcopy(left)
    right["_root"] = "windows"
    right["builder_environment"] = {"system": "Windows", "machine": "AMD64"}
    right["build_authority"]["toolchain"]["colorama"] = "0.4.6"

    report = checker.compare_builds([left, right])

    assert report["status"] == "pass"
    assert report["claim_boundary"]["byte_identical_wheels"] is True
    assert report["mismatches"] == []


def test_wheel_hash_drift_fails_and_reports_zip_metadata_cause():
    left = _manifest(root="linux")
    right = _manifest(root="windows", sha="b" * 64, external_attr=123)

    report = checker.compare_builds([left, right])

    assert report["status"] == "fail"
    mismatch = next(
        item for item in report["mismatches"] if item["kind"] == "wheel_identity"
    )
    assert mismatch["distribution"] == "neuros"
    assert "sha256" in mismatch["drift"]
    assert mismatch["zip_metadata_drift"]["neuros/__init__.py"]["external_attr"] == {
        "reference": 2175008768,
        "observed": 123,
    }


def test_source_authority_drift_is_not_masked_by_identical_wheels():
    left = _manifest(root="one")
    right = deepcopy(left)
    right["_root"] = "two"
    right["source"]["source_date_epoch"] += 1

    report = checker.compare_builds([left, right])

    assert report["status"] == "fail"
    assert any(item["kind"] == "source_authority" for item in report["mismatches"])


def test_core_build_toolchain_drift_fails_closed():
    left = _manifest(root="one")
    right = deepcopy(left)
    right["_root"] = "two"
    right["build_authority"]["toolchain"]["setuptools"] = "85.0.0"

    report = checker.compare_builds([left, right])

    assert report["status"] == "fail"
    assert any(
        item["kind"] == "build_toolchain"
        and item["distribution"] == "setuptools"
        for item in report["mismatches"]
    )


def test_builder_canonicalizes_distribution_names():
    assert builder._canonical_name("NeurOS_Core") == "neuros-core"
    assert builder._canonical_name("neuros.models") == "neuros-models"


def test_canonical_wheel_normalizes_windows_and_unix_containers(tmp_path: Path):
    windows = tmp_path / "windows" / "demo-1.0.0-py3-none-any.whl"
    unix = tmp_path / "unix" / "demo-1.0.0-py3-none-any.whl"
    windows.parent.mkdir()
    unix.parent.mkdir()
    _raw_test_wheel(windows, windows_style=True)
    _raw_test_wheel(unix, windows_style=False)

    epoch = 1_800_000_001
    windows_receipt = builder._canonicalize_wheel(
        windows, source_date_epoch=epoch
    )
    unix_receipt = builder._canonicalize_wheel(unix, source_date_epoch=epoch)

    assert windows.read_bytes() == unix.read_bytes()
    assert windows_receipt["sha256_after"] == unix_receipt["sha256_after"]
    assert windows_receipt["sha256_before"] != windows_receipt["sha256_after"]
    assert windows_receipt["zip_datetime_utc"][-1] % 2 == 0

    with ZipFile(windows) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        assert names == sorted(names)
        assert all(item.create_system == 3 for item in infos)
        assert all(item.compress_type == ZIP_STORED for item in infos)
        assert all(item.external_attr == 0o100644 << 16 for item in infos)
        metadata_payload = archive.read("demo-1.0.0.dist-info/METADATA")
        wheel_payload = archive.read("demo-1.0.0.dist-info/WHEEL")
        assert b"\r" not in metadata_payload
        assert b"\r" not in wheel_payload


def test_canonical_wheel_rebuilds_record_from_canonical_bytes(tmp_path: Path):
    wheel = tmp_path / "demo-1.0.0-py3-none-any.whl"
    _raw_test_wheel(wheel, windows_style=True)
    builder._canonicalize_wheel(wheel, source_date_epoch=1_800_000_000)

    record_name = "demo-1.0.0.dist-info/RECORD"
    with ZipFile(wheel) as archive:
        members = {item.filename: archive.read(item.filename) for item in archive.infolist()}
    rows = list(csv.reader(io.StringIO(members[record_name].decode("utf-8"))))
    assert [row[0] for row in rows[:-1]] == sorted(
        name for name in members if name != record_name
    )
    assert rows[-1] == [record_name, "", ""]
    for name, digest, size in rows[:-1]:
        assert digest == _urlsafe_sha256(members[name])
        assert int(size) == len(members[name])
    assert b"\r" not in members[record_name]


def test_canonicalizer_rejects_non_platform_independent_wheel(tmp_path: Path):
    wheel = tmp_path / "demo-1.0.0-cp311-cp311-win_amd64.whl"
    _raw_test_wheel(wheel, windows_style=True)
    try:
        builder._canonicalize_wheel(wheel, source_date_epoch=1_800_000_000)
    except RuntimeError as exc:
        assert "platform-independent" in str(exc)
    else:
        raise AssertionError("platform-specific wheel unexpectedly canonicalized")
