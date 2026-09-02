from __future__ import annotations

import math

import numpy as np
import pytest

from neuros.config import (
    ExecutionConfig,
    PipelineConfig,
    PluginConfig,
    RuntimeConfig,
    StreamConfig,
    resolve_config,
)
from neuros.errors import ConfigurationError
from neuros.plugins import PluginKind, PluginRegistry
from neuros.runtime import ExecutionClass, NodeKind, OverflowPolicy, RuntimeNode


class Source:
    async def start(self):
        return None

    async def stop(self):
        return None

    async def frames(self):
        if False:
            yield None


class Transform:
    def transform(self, item):
        return item


class Decoder:
    def infer(self, item):
        return item


class Sink:
    def write(self, item):
        return None


class Monitor:
    def update(self, payload):
        return None

    def result(self):
        return {}


def _registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register(name="source", kind=PluginKind.SOURCE, factory=Source)
    registry.register(name="transform", kind=PluginKind.TRANSFORM, factory=Transform)
    registry.register(name="decoder", kind=PluginKind.DECODER, factory=Decoder)
    registry.register(name="sink", kind=PluginKind.SINK, factory=Sink)
    registry.register(name="monitor", kind=PluginKind.MONITOR, factory=Monitor)
    return registry


def _raw_config(*, schema_version: int = 2) -> dict:
    return {
        "schema_version": schema_version,
        "metadata": {"fixture": "config-execution-authority"},
        "streams": [
            {
                "id": "eeg",
                "source": {"plugin": "source"},
                "transforms": [{"plugin": "transform"}],
            }
        ],
        "decoder": {"plugin": "decoder"},
        "sinks": [{"plugin": "sink"}],
        "monitors": [{"plugin": "monitor"}],
        "runtime": {"queue_capacity": 8, "overflow_policy": "drop_oldest"},
    }


def test_schema_v1_remains_legacy_inline_after_resolution():
    config = PipelineConfig.from_mapping(_raw_config(schema_version=1))
    resolved = resolve_config(config, registry=_registry())

    assert config.schema_version == 1
    assert config.runtime.queue_capacity == 8
    assert config.runtime.overflow_policy is OverflowPolicy.DROP_OLDEST
    assert all(node.executor == "inline" for node in resolved.graph.nodes.values())
    assert all(node.execution_timeout_s is None for node in resolved.graph.nodes.values())
    assert all(node.process_transport == "pickle" for node in resolved.graph.nodes.values())


def test_schema_v1_rejects_any_serialized_execution_declaration():
    raw = _raw_config(schema_version=1)
    raw["decoder"]["execution"] = {"executor": "inline"}
    with pytest.raises(ConfigurationError, match="schema_version 1 cannot declare"):
        PipelineConfig.from_mapping(raw)


@pytest.mark.parametrize("value", [True, False, "1", 1.0, 0, 3, None])
def test_schema_version_rejects_coercible_or_unsupported_values(value):
    raw = _raw_config(schema_version=2)
    raw["schema_version"] = value
    with pytest.raises(ConfigurationError):
        PipelineConfig.from_mapping(raw)


def test_schema_version_accepts_integral_scalar_and_canonicalizes_to_int():
    raw = _raw_config(schema_version=2)
    raw["schema_version"] = np.int64(2)
    config = PipelineConfig.from_mapping(raw)
    assert type(config.schema_version) is int
    assert config.schema_version == 2


@pytest.mark.parametrize(
    ("location", "key"),
    [
        ("root", "surprise"),
        ("stream", "surprise"),
        ("plugin", "surprise"),
        ("runtime", "surprise"),
        ("execution", "surprise"),
    ],
)
def test_schema_v2_rejects_unknown_authority_keys(location, key):
    raw = _raw_config(schema_version=2)
    if location == "root":
        raw[key] = 1
    elif location == "stream":
        raw["streams"][0][key] = 1
    elif location == "plugin":
        raw["decoder"][key] = 1
    elif location == "runtime":
        raw["runtime"][key] = 1
    else:
        raw["decoder"]["execution"] = {"executor": "inline", key: 1}

    with pytest.raises(ConfigurationError, match="Unknown"):
        PipelineConfig.from_mapping(raw)


def test_schema_v2_unknown_key_reporting_handles_nonstring_yaml_keys():
    raw = _raw_config(schema_version=2)
    raw[1] = "unexpected"
    with pytest.raises(ConfigurationError, match="Unknown configuration keys"):
        PipelineConfig.from_mapping(raw)


@pytest.mark.parametrize("stream_id", [1, True, None, object()])
def test_stream_id_is_never_string_coerced(stream_id):
    raw = _raw_config(schema_version=2)
    raw["streams"][0]["id"] = stream_id
    with pytest.raises(ConfigurationError, match="stream id must be a string"):
        PipelineConfig.from_mapping(raw)


@pytest.mark.parametrize("stream_id", ["", " ", "\t\n"])
def test_stream_id_rejects_blank_identity(stream_id):
    raw = _raw_config(schema_version=2)
    raw["streams"][0]["id"] = stream_id
    with pytest.raises(ConfigurationError, match="stream id must be nonblank"):
        PipelineConfig.from_mapping(raw)


@pytest.mark.parametrize("plugin", [1, True, None, object()])
def test_plugin_identity_is_never_string_coerced(plugin):
    raw = _raw_config(schema_version=2)
    raw["decoder"]["plugin"] = plugin
    with pytest.raises(ConfigurationError, match="plugin must be a string"):
        PipelineConfig.from_mapping(raw)


@pytest.mark.parametrize("plugin", ["", " ", "\n"])
def test_plugin_identity_rejects_blank_strings(plugin):
    raw = _raw_config(schema_version=2)
    raw["decoder"]["plugin"] = plugin
    with pytest.raises(ConfigurationError, match="plugin must be nonblank"):
        PipelineConfig.from_mapping(raw)


def test_runtime_config_reuses_runtime_edge_canonicalization():
    runtime = RuntimeConfig(
        queue_capacity=np.int64(16), overflow_policy="drop_newest"
    )
    assert type(runtime.queue_capacity) is int
    assert runtime.queue_capacity == 16
    assert runtime.overflow_policy is OverflowPolicy.DROP_NEWEST


@pytest.mark.parametrize("capacity", [True, False, 8.0, "8", 0, -1, None])
def test_runtime_config_rejects_ambiguous_queue_capacities(capacity):
    with pytest.raises(ConfigurationError, match="invalid runtime configuration"):
        RuntimeConfig(queue_capacity=capacity)


@pytest.mark.parametrize("overflow", ["unknown", "", None, True, 1])
def test_runtime_config_rejects_invalid_overflow_policy(overflow):
    with pytest.raises(ConfigurationError, match="invalid runtime configuration"):
        RuntimeConfig(overflow_policy=overflow)


def test_execution_config_canonicalizes_thread_and_gpu_intent():
    thread = ExecutionConfig(executor="thread")
    gpu = ExecutionConfig(executor=ExecutionClass.GPU)

    assert thread.executor is ExecutionClass.THREAD
    assert gpu.executor is ExecutionClass.GPU
    assert thread.process_transport == "pickle"
    assert thread.execution_timeout_s is None


def test_execution_config_canonicalizes_process_timeout():
    execution = ExecutionConfig(
        executor="process", execution_timeout_s=np.float64(1.25)
    )
    assert execution.executor is ExecutionClass.PROCESS
    assert type(execution.execution_timeout_s) is float
    assert execution.execution_timeout_s == pytest.approx(1.25)


def test_execution_config_canonicalizes_shared_memory_integral_capacities():
    execution = ExecutionConfig(
        executor="process",
        execution_timeout_s=1.0,
        process_transport="shared_memory",
        process_request_capacity_bytes=np.int32(4096),
        process_response_capacity_bytes=np.int64(8192),
    )
    assert execution.executor is ExecutionClass.PROCESS
    assert execution.process_transport == "shared_memory"
    assert type(execution.process_request_capacity_bytes) is int
    assert type(execution.process_response_capacity_bytes) is int
    assert execution.process_request_capacity_bytes == 4096
    assert execution.process_response_capacity_bytes == 8192


@pytest.mark.parametrize(
    "kwargs",
    [
        {"executor": "process"},
        {"executor": "process", "execution_timeout_s": 0},
        {"executor": "process", "execution_timeout_s": -1},
        {"executor": "process", "execution_timeout_s": True},
        {"executor": "process", "execution_timeout_s": "1"},
        {"executor": "process", "execution_timeout_s": math.nan},
        {"executor": "process", "execution_timeout_s": math.inf},
        {"executor": "inline", "execution_timeout_s": 1.0},
        {"executor": "thread", "process_transport": "shared_memory"},
        {
            "executor": "process",
            "execution_timeout_s": 1.0,
            "process_transport": "shared_memory",
        },
        {
            "executor": "process",
            "execution_timeout_s": 1.0,
            "process_transport": "pickle",
            "process_request_capacity_bytes": 4096,
        },
        {"executor": "unknown"},
    ],
)
def test_execution_config_rejects_runtime_invalid_declarations(kwargs):
    with pytest.raises(ConfigurationError, match="invalid execution configuration"):
        ExecutionConfig(**kwargs)


def test_direct_schema_v1_rejects_nondefault_execution_policy():
    with pytest.raises(ConfigurationError, match="legacy-inline"):
        PipelineConfig(
            schema_version=1,
            streams=(
                StreamConfig(
                    "eeg",
                    PluginConfig("source"),
                    (
                        PluginConfig(
                            "transform",
                            execution=ExecutionConfig(executor="thread"),
                        ),
                    ),
                ),
            ),
            decoder=PluginConfig("decoder"),
        )


def test_schema_v2_compiles_mixed_execution_policy_to_runtime_nodes():
    raw = _raw_config(schema_version=2)
    raw["streams"][0]["transforms"][0]["execution"] = {"executor": "thread"}
    raw["decoder"]["execution"] = {
        "executor": "process",
        "execution_timeout_s": 1.5,
        "process_transport": "pickle",
    }
    raw["sinks"][0]["execution"] = {"executor": "gpu"}
    raw["monitors"][0]["execution"] = {"executor": "thread"}

    config = PipelineConfig.from_mapping(raw)
    resolved = resolve_config(config, registry=_registry())

    source = resolved.graph.nodes["source:eeg"]
    transform = resolved.graph.nodes["transform:eeg:0"]
    decoder = resolved.graph.nodes["decoder:primary"]
    sink = resolved.graph.nodes["sink:0"]
    monitor = resolved.graph.nodes["monitor:0"]

    assert source.executor == "inline"
    assert transform.executor == "thread"
    assert decoder.executor == "process"
    assert decoder.execution_timeout_s == pytest.approx(1.5)
    assert decoder.process_transport == "pickle"
    assert sink.executor == "gpu"
    assert monitor.executor == "thread"


def test_schema_v2_compiles_shared_memory_policy_without_loss():
    config = PipelineConfig(
        schema_version=2,
        streams=(StreamConfig("eeg", PluginConfig("source")),),
        decoder=PluginConfig(
            "decoder",
            execution=ExecutionConfig(
                executor="process",
                execution_timeout_s=2.0,
                process_transport="shared_memory",
                process_request_capacity_bytes=np.int64(65536),
                process_response_capacity_bytes=np.int32(131072),
            ),
        ),
    )
    resolved = resolve_config(config, registry=_registry())
    decoder = resolved.graph.nodes["decoder:primary"]

    assert decoder.executor == "process"
    assert decoder.execution_timeout_s == 2.0
    assert decoder.process_transport == "shared_memory"
    assert decoder.process_request_capacity_bytes == 65536
    assert decoder.process_response_capacity_bytes == 131072
    assert type(decoder.process_request_capacity_bytes) is int
    assert type(decoder.process_response_capacity_bytes) is int


@pytest.mark.parametrize("executor", ["thread", "gpu", "process"])
def test_source_nondefault_execution_fails_at_compilation(executor):
    kwargs = {"executor": executor}
    if executor == "process":
        kwargs["execution_timeout_s"] = 1.0
    raw = _raw_config(schema_version=2)
    raw["streams"][0]["source"]["execution"] = kwargs
    config = PipelineConfig.from_mapping(raw)

    with pytest.raises(ConfigurationError, match="source lifecycle isolation"):
        resolve_config(config, registry=_registry())


@pytest.mark.parametrize(
    ("declaration", "expected_executor", "expected_timeout"),
    [
        ({}, "inline", None),
        ({"executor": "inline"}, "inline", None),
        ({"executor": "thread"}, "thread", None),
        ({"executor": "gpu"}, "gpu", None),
        ({"executor": "process", "execution_timeout_s": 1.25}, "process", 1.25),
    ],
)
def test_serialized_monitor_execution_policy_compiles_losslessly(
    declaration, expected_executor, expected_timeout
):
    raw = _raw_config(schema_version=2)
    raw["monitors"][0]["execution"] = declaration
    config = PipelineConfig.from_mapping(raw)
    resolved = resolve_config(config, registry=_registry())
    monitor = resolved.graph.nodes["monitor:0"]

    assert monitor.executor == expected_executor
    assert monitor.execution_timeout_s == expected_timeout


def test_direct_process_monitor_execution_policy_compiles_losslessly():
    config = PipelineConfig(
        schema_version=2,
        streams=(StreamConfig("eeg", PluginConfig("source")),),
        decoder=PluginConfig("decoder"),
        monitors=(
            PluginConfig(
                "monitor",
                execution=ExecutionConfig(
                    executor="process",
                    execution_timeout_s=2.0,
                    process_transport="shared_memory",
                    process_request_capacity_bytes=np.int64(32768),
                    process_response_capacity_bytes=np.int64(32768),
                ),
            ),
        ),
    )
    resolved = resolve_config(config, registry=_registry())
    monitor = resolved.graph.nodes["monitor:0"]

    assert monitor.executor == "process"
    assert monitor.execution_timeout_s == 2.0
    assert monitor.process_transport == "shared_memory"
    assert monitor.process_request_capacity_bytes == 32768
    assert monitor.process_response_capacity_bytes == 32768


def test_execution_config_and_direct_runtime_node_have_identical_canonical_fields():
    execution = ExecutionConfig(
        executor="process",
        execution_timeout_s=np.float64(1.75),
        process_transport="shared_memory",
        process_request_capacity_bytes=np.int32(4096),
        process_response_capacity_bytes=np.int64(8192),
    )
    node = RuntimeNode(
        "decoder",
        NodeKind.DECODER,
        Decoder(),
        **execution.runtime_kwargs(),
    )

    assert node.executor == execution.executor.value
    assert node.execution_timeout_s == execution.execution_timeout_s
    assert node.process_transport == execution.process_transport
    assert node.process_request_capacity_bytes == execution.process_request_capacity_bytes
    assert node.process_response_capacity_bytes == execution.process_response_capacity_bytes
