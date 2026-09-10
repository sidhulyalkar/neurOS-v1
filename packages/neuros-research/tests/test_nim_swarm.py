import asyncio

from neuros.research.nim_swarm import (
    NvidiaCouncilTransport,
    _user_prompt,
    build_nvidia_council,
    council_configuration_sha256,
)
from neuros.research.swarm import SealedSwarmTask, run_council

_H = "a" * 64


def _task():
    return SealedSwarmTask(
        repository="repo",
        source_revision="sha",
        objective="audit",
        claim_boundary="advisory",
        authority_sha256s=(_H,),
        public_context={"x": 1},
    )


def _finding(finding_id):
    return {
        "finding_id": finding_id,
        "severity": "medium",
        "category": "implementation",
        "claim": "specific concern",
        "evidence": "specific evidence",
        "falsification_test": "specific test",
        "proposed_repair": "",
        "confidence": 0.5,
        "requires_human_judgment": False,
        "reference": "",
    }


class _Client:
    def __init__(self):
        self.calls = []

    def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        role = kwargs["role"].split(":", 1)[1]
        return {"findings": [_finding(role)]}, object()


def test_builds_five_distinct_roles():
    members = build_nvidia_council(("m1", "m2", "m3"))
    assert len(members) == 5
    assert len({member.role for member in members}) == 5


def test_models_round_robin_across_roles():
    members = build_nvidia_council(("m1", "m2", "m3"))
    assert [member.model for member in members] == ["m1", "m2", "m3", "m1", "m2"]


def test_duplicate_models_are_deduplicated():
    members = build_nvidia_council(("m1", "m1", "m2"))
    assert [member.model for member in members][:3] == ["m1", "m2", "m1"]


def test_empty_model_set_rejected():
    try:
        build_nvidia_council(())
    except ValueError:
        pass
    else:
        raise AssertionError("empty model set should fail")


def test_configuration_hash_independent_of_member_input_order():
    members = build_nvidia_council(("m1", "m2"))
    assert council_configuration_sha256(members) == council_configuration_sha256(
        tuple(reversed(members))
    )


def test_prompt_contains_task_hash_boundary_and_schema():
    task = _task()
    prompt = _user_prompt(task)
    assert task.sha256 in prompt
    assert "llm_output_is_authority" in prompt
    assert "falsification_test" in prompt


def test_transport_reuses_existing_chat_json_contract():
    client = _Client()
    member = build_nvidia_council(("m1",))[0]
    output = asyncio.run(NvidiaCouncilTransport(client).review(_task(), member))
    assert output["findings"]
    assert client.calls[0]["temperature"] == 0.1


def test_full_five_member_council_composes_with_core_runner():
    client = _Client()
    members = build_nvidia_council(("m1", "m2", "m3"))
    run = asyncio.run(
        run_council(
            _task(),
            members,
            NvidiaCouncilTransport(client),
        )
    )
    assert len(run.reviews) == 5
    assert len(client.calls) == 5
    assert len(run.findings) == 5


def test_adapter_prompt_contract_contains_no_credential_name():
    prompt = _user_prompt(_task()).lower()
    assert "nvidia_api_key" not in prompt
    assert "authorization" not in prompt


def test_max_tokens_is_bounded():
    try:
        NvidiaCouncilTransport(_Client(), max_tokens=100)
    except ValueError:
        pass
    else:
        raise AssertionError("too-small token budget should fail")
