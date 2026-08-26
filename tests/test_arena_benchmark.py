from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from neuros.arena.benchmark import MetricRule, load_benchmark_pack, run_benchmark_pack


PACK = Path("examples/arena/benchmark_packs/eeg_game_system_v1.json")


def test_portable_eeg_game_system_pack_passes_reference_arena():
    pack = load_benchmark_pack(PACK)
    result = run_benchmark_pack(pack)
    assert result.passed
    assert all(case.passed for case in result.cases)
    payload = result.to_dict()
    assert payload["pack_name"] == "eeg-game-systems"
    assert "does not establish human" in payload["evidence_boundary"]


def test_benchmark_pack_preserves_a_failing_rule_as_evidence():
    pack = load_benchmark_pack(PACK)
    first = pack.cases[0]
    impossible = MetricRule(
        "metrics.transport.packet_drop_fraction",
        ">",
        0.5,
        "clean transport should deliberately fail this injected assertion",
    )
    mutated = replace(pack, cases=(replace(first, rules=first.rules + (impossible,)),) + pack.cases[1:])
    result = run_benchmark_pack(mutated)
    assert not result.passed
    assert not result.cases[0].passed
    assert any("clean transport" in failure for failure in result.cases[0].failures)
