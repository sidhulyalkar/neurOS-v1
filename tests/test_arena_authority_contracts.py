from __future__ import annotations

import pytest

from neuros.arena import (
    PARTICIPANT_RESPONSE_MODEL,
    ArenaScenario,
    StageSpec,
    compile_participant_state_trace,
    ParticipantProfile,
)


def test_typed_target_frequency_rejects_conflicting_generic_metadata():
    scenario = ArenaScenario(
        "conflicting-frequency-authority",
        (
            StageSpec(
                "target",
                1.0,
                target_frequency_hz=10.0,
                attention_gain=0.8,
                target={"frequency_hz": 12.0},
            ),
        ),
    )
    with pytest.raises(ValueError, match="conflicts with authoritative target_frequency_hz"):
        scenario.validate()


def test_matching_frequency_metadata_is_allowed_but_typed_field_remains_authority():
    scenario = ArenaScenario(
        "matching-frequency-authority",
        (
            StageSpec(
                "target",
                0.2,
                target_frequency_hz=10.0,
                attention_gain=0.8,
                target={"frequency_hz": 10.0, "semantic": "sight"},
            ),
        ),
    )
    scenario.validate()
    trace = compile_participant_state_trace(scenario, ParticipantProfile(response_delay_s=0.0), 250.0)
    assert trace.model == PARTICIPANT_RESPONSE_MODEL
    assert trace.target_frequency_hz[0] == 10.0


def test_reserved_frequency_metadata_without_typed_authority_fails_closed():
    scenario = ArenaScenario(
        "untyped-frequency-authority",
        (
            StageSpec(
                "ambiguous-target",
                0.2,
                target_frequency_hz=None,
                attention_gain=1.0,
                target={"frequency_hz": 10.0, "semantic": "ambiguous"},
            ),
        ),
    )
    with pytest.raises(ValueError, match="reserved; set authoritative target_frequency_hz"):
        scenario.validate()


def test_non_frequency_target_metadata_does_not_invent_frequency_participant_dynamics():
    scenario = ArenaScenario(
        "p300-authority-boundary",
        (
            StageSpec(
                "oddball",
                0.2,
                target_frequency_hz=None,
                attention_gain=1.0,
                target={"oddball": True, "symbol": "B"},
            ),
        ),
        metadata={"paradigm": "p300"},
    )
    trace = compile_participant_state_trace(scenario, ParticipantProfile(), 250.0)
    assert trace.to_summary()["scope"] == "frequency_target_visual_attention"
    assert not trace.target_switch.any()
    assert not trace.attention_gain.any()
