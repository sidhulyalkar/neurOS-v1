from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from list_release_packages import POLICY_FILE, release_policy  # noqa: E402
from list_workspace_packages import workspace_members  # noqa: E402


def _payload() -> dict:
    return json.loads(POLICY_FILE.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "package-policy.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_policy_classifies_every_workspace_member_once_and_release_set_is_explicit():
    entries = release_policy()
    assert {item["path"] for item in entries} == set(workspace_members())
    published = {item["distribution"] for item in entries if item["publish_candidate"]}
    assert published == {
        "neuros-core",
        "neuros-drivers",
        "neuros-models",
        "neuros",
    }
    arena = next(item for item in entries if item["distribution"] == "neuros-arena")
    assert arena["release_tier"] == "qualified-integration"
    assert arena["scientific_maturity"] == "synthetic-validation"
    assert arena["publish_candidate"] is False
    assert all(
        (item["release_tier"] == "public-runtime") == item["publish_candidate"]
        for item in entries
    )


def test_policy_fails_if_workspace_member_is_unclassified(tmp_path):
    payload = _payload()
    payload["packages"] = payload["packages"][:-1]
    with pytest.raises(ValueError, match="classify every workspace member exactly once"):
        release_policy(_write(tmp_path, payload))


def test_policy_fails_on_duplicate_path(tmp_path):
    payload = _payload()
    payload["packages"].append(deepcopy(payload["packages"][0]))
    with pytest.raises(ValueError, match="duplicate release package path"):
        release_policy(_write(tmp_path, payload))


def test_policy_fails_on_distribution_metadata_drift(tmp_path):
    payload = _payload()
    payload["packages"][0]["distribution"] = "not-the-project-name"
    with pytest.raises(ValueError, match="distribution mismatch"):
        release_policy(_write(tmp_path, payload))


def test_policy_fails_if_research_package_enters_default_release_set(tmp_path):
    payload = _payload()
    research = next(
        item for item in payload["packages"] if item["release_tier"] == "research-extension"
    )
    research["publish_candidate"] = True
    with pytest.raises(ValueError, match="non-public-runtime package cannot enter"):
        release_policy(_write(tmp_path, payload))


def test_policy_fails_if_qualified_integration_enters_default_release_set(tmp_path):
    payload = _payload()
    arena = next(
        item for item in payload["packages"] if item["distribution"] == "neuros-arena"
    )
    arena["publish_candidate"] = True
    with pytest.raises(ValueError, match="non-public-runtime package cannot enter"):
        release_policy(_write(tmp_path, payload))


def test_policy_fails_if_public_runtime_is_silently_removed_from_release(tmp_path):
    payload = _payload()
    runtime = next(
        item for item in payload["packages"] if item["release_tier"] == "public-runtime"
    )
    runtime["publish_candidate"] = False
    with pytest.raises(ValueError, match="public-runtime package must be a publish candidate"):
        release_policy(_write(tmp_path, payload))


def test_policy_requires_sdk_dependency_closure(tmp_path):
    payload = _payload()
    drivers = next(
        item for item in payload["packages"] if item["path"] == "packages/neuros-drivers"
    )
    drivers["release_tier"] = "qualified-integration"
    drivers["publish_candidate"] = False
    with pytest.raises(ValueError, match="SDK dependency closure"):
        release_policy(_write(tmp_path, payload))
