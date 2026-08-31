from __future__ import annotations

import numpy as np
import pytest

from neuros.runtime import NodeKind, OverflowPolicy, RuntimeEdge, RuntimeGraph, RuntimeNode


class IdentityTransform:
    def transform(self, item):
        return item


class Source:
    async def start(self):
        return None

    async def stop(self):
        return None

    async def frames(self):
        if False:
            yield None


def test_runtime_node_canonicalizes_valid_plain_string_kind():
    node = RuntimeNode("transform", "transform", IdentityTransform())
    assert node.kind is NodeKind.TRANSFORM


def test_runtime_node_string_source_kind_cannot_bypass_source_execution_authority():
    with pytest.raises(ValueError, match="Source nodes currently require executor='inline'"):
        RuntimeNode("source", "source", Source(), executor="thread")


@pytest.mark.parametrize("kind", ["unknown", "", 1, True, object()])
def test_runtime_node_rejects_unknown_or_nonauthoritative_kinds(kind):
    expected = TypeError if not isinstance(kind, str) else ValueError
    with pytest.raises(expected):
        RuntimeNode("node", kind, IdentityTransform())


@pytest.mark.parametrize("node_id", [None, 7, True, object()])
def test_runtime_node_rejects_nonstring_identifiers(node_id):
    with pytest.raises(TypeError, match="node_id must be a string"):
        RuntimeNode(node_id, NodeKind.TRANSFORM, IdentityTransform())


@pytest.mark.parametrize("node_id", ["", " ", "\t\n"])
def test_runtime_node_rejects_blank_identifiers(node_id):
    with pytest.raises(ValueError, match="node_id must be nonblank"):
        RuntimeNode(node_id, NodeKind.TRANSFORM, IdentityTransform())


def test_runtime_node_preserves_nonblank_identifier_bytes_without_silent_stripping():
    node = RuntimeNode(" transform ", NodeKind.TRANSFORM, IdentityTransform())
    assert node.node_id == " transform "


@pytest.mark.parametrize("executor", [None, True, 1, ["inline"]])
def test_runtime_node_executor_requires_explicit_string(executor):
    with pytest.raises(TypeError, match="executor must be a string"):
        RuntimeNode("node", NodeKind.TRANSFORM, IdentityTransform(), executor=executor)


@pytest.mark.parametrize("process_transport", [None, True, 1, ["pickle"]])
def test_runtime_node_process_transport_requires_explicit_string(process_transport):
    with pytest.raises(TypeError, match="process_transport must be a string"):
        RuntimeNode(
            "node",
            NodeKind.TRANSFORM,
            IdentityTransform(),
            process_transport=process_transport,
        )


@pytest.mark.parametrize("field_name", ["source", "target"])
@pytest.mark.parametrize("value", [None, 7, True, object()])
def test_runtime_edge_rejects_nonstring_endpoints(field_name, value):
    kwargs = {"source": "a", "target": "b"}
    kwargs[field_name] = value
    with pytest.raises(TypeError, match=rf"{field_name} must be a string"):
        RuntimeEdge(**kwargs)


@pytest.mark.parametrize("field_name", ["source", "target"])
@pytest.mark.parametrize("value", ["", " ", "\t\n"])
def test_runtime_edge_rejects_blank_endpoints(field_name, value):
    kwargs = {"source": "a", "target": "b"}
    kwargs[field_name] = value
    with pytest.raises(ValueError, match=rf"{field_name} must be nonblank"):
        RuntimeEdge(**kwargs)


@pytest.mark.parametrize("capacity", [True, False, 1.0, 8.0, "8", None, object()])
def test_runtime_edge_rejects_nonintegral_or_bool_capacity(capacity):
    with pytest.raises(TypeError, match="capacity must be an integer"):
        RuntimeEdge("a", "b", capacity=capacity)


@pytest.mark.parametrize("capacity", [0, -1, -10, np.int64(0), np.int64(-4)])
def test_runtime_edge_rejects_nonpositive_integral_capacity(capacity):
    with pytest.raises(ValueError, match="capacity must be positive"):
        RuntimeEdge("a", "b", capacity=capacity)


@pytest.mark.parametrize("capacity", [1, 8, np.int32(3), np.int64(64)])
def test_runtime_edge_normalizes_valid_integral_capacity_to_python_int(capacity):
    edge = RuntimeEdge("a", "b", capacity=capacity)
    assert type(edge.capacity) is int
    assert edge.capacity == int(capacity)


@pytest.mark.parametrize(
    ("overflow", "expected"),
    [
        ("block", "block"),
        (OverflowPolicy.BLOCK, "block"),
        ("drop_oldest", "drop_oldest"),
        (OverflowPolicy.DROP_NEWEST, "drop_newest"),
        (OverflowPolicy.FAIL, "fail"),
    ],
)
def test_runtime_edge_canonicalizes_valid_overflow_declarations(overflow, expected):
    edge = RuntimeEdge("a", "b", overflow=overflow)
    assert type(edge.overflow) is str
    assert edge.overflow == expected


@pytest.mark.parametrize("overflow", ["unknown", "", None, True, 1])
def test_runtime_edge_rejects_invalid_overflow_declarations(overflow):
    with pytest.raises(ValueError, match="Unsupported overflow policy"):
        RuntimeEdge("a", "b", overflow=overflow)


def test_runtime_edge_still_rejects_self_edges_after_endpoint_normalization():
    with pytest.raises(ValueError, match="self edges are not allowed"):
        RuntimeEdge("same", "same")


def test_graph_with_plain_string_node_kinds_validates_with_canonical_topology():
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", "source", Source()))
    graph.add_node(RuntimeNode("transform", "transform", IdentityTransform()))
    graph.add_node(RuntimeNode("sink", "sink", object()))
    graph.connect(
        RuntimeEdge(
            "source",
            "transform",
            capacity=np.int64(2),
            overflow=OverflowPolicy.BLOCK,
        )
    )
    graph.connect(RuntimeEdge("transform", "sink", capacity=2, overflow="block"))

    graph.validate()

    assert graph.nodes["source"].kind is NodeKind.SOURCE
    assert graph.nodes["transform"].kind is NodeKind.TRANSFORM
    assert graph.nodes["sink"].kind is NodeKind.SINK
    assert graph.edges[0].capacity == 2
    assert graph.edges[0].overflow == "block"


def test_unknown_edge_endpoint_still_fails_before_runtime_execution():
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("a", NodeKind.TRANSFORM, IdentityTransform()))
    edge = RuntimeEdge("a", "missing")
    with pytest.raises(ValueError, match="Both edge endpoints must be registered nodes"):
        graph.connect(edge)


def test_graph_mutation_methods_require_typed_node_and_edge_objects():
    graph = RuntimeGraph()
    with pytest.raises(TypeError, match="node must be a RuntimeNode"):
        graph.add_node(object())

    with pytest.raises(TypeError, match="edge must be a RuntimeEdge"):
        graph.connect(object())


def test_graph_validate_rejects_externally_corrupted_node_value():
    graph = RuntimeGraph()
    graph.nodes["corrupt"] = object()
    with pytest.raises(TypeError, match="must be a RuntimeNode"):
        graph.validate()


def test_graph_validate_rejects_node_key_identity_drift():
    graph = RuntimeGraph()
    graph.nodes["alias"] = RuntimeNode("actual", NodeKind.SOURCE, Source())
    with pytest.raises(ValueError, match="does not match node_id"):
        graph.validate()


def test_graph_validate_rejects_externally_corrupted_edge_value():
    graph = RuntimeGraph()
    graph.edges.append(object())
    with pytest.raises(TypeError, match="edges must be RuntimeEdge instances"):
        graph.validate()


def test_graph_validate_rejects_duplicate_edges_even_when_public_list_is_mutated():
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, Source()))
    graph.add_node(RuntimeNode("sink", NodeKind.SINK, object()))
    edge = RuntimeEdge("source", "sink")
    graph.edges.extend([edge, edge])
    with pytest.raises(ValueError, match="Duplicate edge: source -> sink"):
        graph.validate()
