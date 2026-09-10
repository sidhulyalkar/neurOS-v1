import asyncio
import copy

from neuros.research.swarm import (
    CouncilMember,
    SealedSwarmTask,
    finding_support,
    parse_review_payload,
    run_council,
)

_H = "a" * 64


def _task(**overrides):
    payload = dict(
        repository="sidhulyalkar/neurOS-v1",
        source_revision="5d75e0d",
        objective="Find defects",
        claim_boundary="advisory only",
        allowed_paths=("x.py",),
        authority_sha256s=(_H,),
        forbidden_actions=("merge",),
        public_context={"issue": 178},
    )
    payload.update(overrides)
    return SealedSwarmTask(**payload)


def _member(member_id="a", model="m1"):
    return CouncilMember(
        member_id,
        "scientific_adversary",
        model,
        "Return exact JSON.",
    )


def _finding(finding_id="f1", **overrides):
    payload = dict(
        finding_id=finding_id,
        severity="high",
        category="scientific_validity",
        claim="Leakage possible",
        evidence="split constructed after reveal",
        falsification_test="freeze split first",
        proposed_repair="bind split",
        confidence=0.9,
        requires_human_judgment=False,
        reference="x.py:1",
    )
    payload.update(overrides)
    return payload


class _MockTransport:
    def __init__(self, payloads, fail=()):
        self.payloads = payloads
        self.fail = set(fail)
        self.seen = []

    async def review(self, task, member):
        self.seen.append((task.sha256, member.member_id))
        await asyncio.sleep(0)
        if member.member_id in self.fail:
            raise TimeoutError("secret provider detail")
        return copy.deepcopy(self.payloads[member.member_id])


def test_task_stable_across_context_key_order():
    first = _task(public_context={"b": 2, "a": 1})
    second = _task(public_context={"a": 1, "b": 2})
    assert first.sha256 == second.sha256


def test_task_detaches_context():
    context = {"a": [1]}
    sealed = _task(public_context=context)
    context["a"].append(2)
    assert sealed.to_dict()["public_context"] == {"a": [1]}


def test_task_public_context_is_recursively_immutable():
    sealed = _task(public_context={"nested": {"items": [1]}})
    original_sha = sealed.sha256

    try:
        sealed.public_context["new"] = 1  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("sealed top-level context should be immutable")

    nested = sealed.public_context["nested"]
    try:
        nested["new"] = 2
    except TypeError:
        pass
    else:
        raise AssertionError("sealed nested context should be immutable")

    assert nested["items"] == (1,)
    assert sealed.sha256 == original_sha


def test_task_to_dict_returns_detached_plain_json():
    sealed = _task(public_context={"nested": {"items": [1]}})
    payload = sealed.to_dict()
    assert payload["public_context"] == {"nested": {"items": [1]}}

    payload["public_context"]["nested"]["items"].append(2)
    assert sealed.to_dict()["public_context"] == {"nested": {"items": [1]}}


def test_task_rejects_secret_key_recursively():
    try:
        _task(public_context={"nested": {"api_key": "x"}})
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("secret key should fail")


def test_task_rejects_duplicate_authority_hashes():
    try:
        _task(authority_sha256s=(_H, _H))
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate authority hashes should fail")


def test_finding_requires_exact_schema():
    payload = {"findings": [_finding()]}
    payload["findings"][0]["extra"] = 1
    try:
        parse_review_payload(payload, task=_task(), member=_member())
    except ValueError as exc:
        assert "exactly" in str(exc)
    else:
        raise AssertionError("extra finding field should fail")


def test_finding_confidence_bounded():
    try:
        parse_review_payload(
            {"findings": [_finding(confidence=1.1)]},
            task=_task(),
            member=_member(),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unbounded confidence should fail")


def test_review_binds_task_member_model_and_prompt():
    task = _task()
    member = _member()
    review = parse_review_payload(
        {"findings": [_finding()]},
        task=task,
        member=member,
    )
    assert review.task_sha256 == task.sha256
    assert review.model == "m1"
    assert len(review.prompt_sha256) == 64


def test_duplicate_finding_ids_rejected_per_review():
    try:
        parse_review_payload(
            {"findings": [_finding(), _finding()]},
            task=_task(),
            member=_member(),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate finding IDs should fail")


def test_parallel_council_receives_same_task_identity():
    task = _task()
    members = (_member("a", "m1"), _member("b", "m2"), _member("c", "m3"))
    transport = _MockTransport(
        {member.member_id: {"findings": [_finding(member.member_id)]} for member in members}
    )
    run = asyncio.run(run_council(task, members, transport))
    assert len(set(identity for identity, _ in transport.seen)) == 1
    assert len(run.reviews) == 3


def test_run_identity_independent_of_member_input_order():
    task = _task()
    first = _member("a", "m1")
    second = _member("b", "m2")
    payloads = {
        "a": {"findings": [_finding("x")]},
        "b": {"findings": [_finding("y")]},
    }
    one = asyncio.run(run_council(task, (first, second), _MockTransport(payloads)))
    two = asyncio.run(run_council(task, (second, first), _MockTransport(payloads)))
    assert one.sha256 == two.sha256


def test_require_all_fails_closed_without_provider_message():
    task = _task()
    members = (_member("a"), _member("b"))
    try:
        asyncio.run(
            run_council(
                task,
                members,
                _MockTransport({"a": {"findings": []}}, fail=("b",)),
            )
        )
    except RuntimeError as exc:
        assert "b:TimeoutError" in str(exc)
        assert "secret provider detail" not in str(exc)
    else:
        raise AssertionError("required member failure should fail closed")


def test_partial_mode_preserves_failed_member_identity():
    task = _task()
    members = (_member("a"), _member("b"))
    run = asyncio.run(
        run_council(
            task,
            members,
            _MockTransport({"a": {"findings": []}}, fail=("b",)),
            require_all=False,
        )
    )
    assert run.failed_members == ("b:TimeoutError",)
    assert len(run.reviews) == 1


def test_run_manifest_explicitly_denies_authority():
    task = _task()
    run = asyncio.run(
        run_council(
            task,
            (_member(),),
            _MockTransport({"a": {"findings": []}}),
        )
    )
    payload = run.to_dict()
    assert not payload["merge_authority"]
    assert not payload["provider_execution_authority"]
    assert not payload["scientific_promotion_authority"]


def test_support_preserves_minority_finding():
    task = _task()
    members = (_member("a"), _member("b"), _member("c"))
    payloads = {
        "a": {"findings": [_finding("minority")]},
        "b": {"findings": []},
        "c": {"findings": []},
    }
    run = asyncio.run(run_council(task, members, _MockTransport(payloads)))
    assert finding_support(run) == {"minority": ("a",)}


def test_support_tracks_agreement_without_promoting_it():
    task = _task()
    members = (_member("a"), _member("b"))
    payloads = {key: {"findings": [_finding("same")]} for key in ("a", "b")}
    run = asyncio.run(run_council(task, members, _MockTransport(payloads)))
    assert finding_support(run)["same"] == ("a", "b")
    assert not run.to_dict()["majority_vote_is_authority"]


def test_duplicate_member_ids_rejected_before_transport():
    try:
        asyncio.run(run_council(_task(), (_member("a"), _member("a")), _MockTransport({})))
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate members should fail")
