from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path


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


def _manifest(*, root: str, sha: str = "a" * 64, external_attr: int = 2175008768):
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
            "python_version": "3.11.16",
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
    mismatch = next(item for item in report["mismatches"] if item["kind"] == "wheel_identity")
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
        item["kind"] == "build_toolchain" and item["distribution"] == "setuptools"
        for item in report["mismatches"]
    )


def test_builder_canonicalizes_distribution_names():
    assert builder._canonical_name("NeurOS_Core") == "neuros-core"
    assert builder._canonical_name("neuros.models") == "neuros-models"
