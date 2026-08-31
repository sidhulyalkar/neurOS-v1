from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_full_package_qualification import (  # noqa: E402
    REQUIRED_DISTRIBUTIONS,
    _append_tamper,
    _assert_clean_runtime,
    _canonical_name,
    _flip_one_byte,
    _policy_selected_packages,
    _runtime_semantics,
)


def test_canonical_distribution_names_follow_packaging_equivalence():
    assert _canonical_name("neuros_core") == "neuros-core"
    assert _canonical_name("NeurOS.Core") == "neuros-core"


def test_recording_tamper_changes_exactly_the_selected_payload(tmp_path: Path):
    target = tmp_path / "frame.npy"
    original = bytes(range(64))
    target.write_bytes(original)

    receipt = _flip_one_byte(target)

    mutated = target.read_bytes()
    assert len(mutated) == len(original)
    assert sum(left != right for left, right in zip(original, mutated)) == 1
    assert receipt["sha256_before"] != receipt["sha256_after"]
    assert receipt["byte_offset"] == len(original) // 2


def test_qualification_tamper_preserves_file_but_changes_identity(tmp_path: Path):
    target = tmp_path / "runtime.json"
    target.write_text(json.dumps({"state": "stopped"}) + "\n", encoding="utf-8")

    receipt = _append_tamper(target)

    assert target.is_file()
    assert receipt["sha256_before"] != receipt["sha256_after"]


def test_runtime_semantics_ignore_timing_but_preserve_failure_counts():
    snapshot = {
        "state": "stopped",
        "runtime_seconds": 99.9,
        "nodes": {
            "source:eeg": {"processed": 7, "failed": 0, "p99_latency_ms": 123.4},
            "decoder:primary": {"processed": 7, "failed": 0, "p99_latency_ms": 0.2},
        },
        "edges": {
            "source:eeg->decoder:primary": {"accepted": 7, "dropped": 0},
        },
    }
    semantics = _runtime_semantics(snapshot)
    assert semantics == {
        "state": "stopped",
        "node_processed": {"decoder:primary": 7, "source:eeg": 7},
        "node_failed": {"decoder:primary": 0, "source:eeg": 0},
        "edge_accepted": {"source:eeg->decoder:primary": 7},
        "edge_dropped": {"source:eeg->decoder:primary": 0},
    }
    _assert_clean_runtime(snapshot, label="fixture")


def test_runtime_gate_rejects_failed_or_dropped_execution():
    failed = {
        "state": "stopped",
        "nodes": {"decoder:primary": {"processed": 1, "failed": 1}},
        "edges": {"edge": {"accepted": 1, "dropped": 0}},
    }
    with pytest.raises(RuntimeError, match="node failures"):
        _assert_clean_runtime(failed, label="failed")

    dropped = {
        "state": "stopped",
        "nodes": {"decoder:primary": {"processed": 1, "failed": 0}},
        "edges": {"edge": {"accepted": 1, "dropped": 1}},
    }
    with pytest.raises(RuntimeError, match="dropped runtime items"):
        _assert_clean_runtime(dropped, label="dropped")


def test_full_package_contract_tracks_exact_default_release_policy():
    selected = _policy_selected_packages(ROOT)
    assert {_canonical_name(item["distribution"]) for item in selected} == {
        _canonical_name(name) for name in REQUIRED_DISTRIBUTIONS
    }
