"""Versioned, serializable neurOS configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from neuros.errors import ConfigurationError
from neuros.runtime import (
    ExecutionClass,
    NodeKind,
    OverflowPolicy,
    RuntimeEdge,
    RuntimeNode,
)

_SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})
_TOP_LEVEL_V2_KEYS = frozenset(
    {
        "schema_version",
        "streams",
        "decoder",
        "sinks",
        "monitors",
        "runtime",
        "metadata",
    }
)
_STREAM_V2_KEYS = frozenset({"id", "source", "transforms"})
_PLUGIN_V2_KEYS = frozenset({"plugin", "options", "execution"})
_RUNTIME_V2_KEYS = frozenset({"queue_capacity", "overflow_policy"})
_EXECUTION_V2_KEYS = frozenset(
    {
        "executor",
        "execution_timeout_s",
        "process_transport",
        "process_request_capacity_bytes",
        "process_response_capacity_bytes",
    }
)


def _nonblank_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{field_name} must be a string")
    if not value.strip():
        raise ConfigurationError(f"{field_name} must be nonblank")
    return value


def _schema_version(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ConfigurationError("schema_version must be an integer")
    resolved = int(value)
    if resolved not in _SUPPORTED_SCHEMA_VERSIONS:
        expected = ", ".join(str(item) for item in sorted(_SUPPORTED_SCHEMA_VERSIONS))
        raise ConfigurationError(
            f"Unsupported config schema_version={resolved}; expected one of: {expected}"
        )
    return resolved


def _reject_unknown_keys(
    value: Mapping[str, Any], *, allowed: frozenset[str], context: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigurationError(f"Unknown {context} keys: {unknown}")


def _plugin_sequence(value: Any, *, field_name: str) -> tuple[PluginConfig, ...]:
    if not isinstance(value, (list, tuple)):
        raise ConfigurationError(f"{field_name} must be a list or tuple")
    resolved = tuple(value)
    if any(not isinstance(item, PluginConfig) for item in resolved):
        raise ConfigurationError(f"{field_name} must contain PluginConfig values")
    return resolved


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Canonical execution declaration for one runtime-backed plugin.

    The runtime node constructor remains the final authority for execution
    semantics. Configuration reuses that constructor here so serialized and
    programmatic declarations cannot drift into two normalization systems.
    """

    executor: ExecutionClass = ExecutionClass.INLINE
    execution_timeout_s: float | None = None
    process_transport: str = "pickle"
    process_request_capacity_bytes: int | None = None
    process_response_capacity_bytes: int | None = None

    def __post_init__(self) -> None:
        executor = self.executor.value if isinstance(self.executor, ExecutionClass) else self.executor
        try:
            probe = RuntimeNode(
                node_id="__config_execution__",
                kind=NodeKind.TRANSFORM,
                operator=None,
                executor=executor,
                execution_timeout_s=self.execution_timeout_s,
                process_transport=self.process_transport,
                process_request_capacity_bytes=self.process_request_capacity_bytes,
                process_response_capacity_bytes=self.process_response_capacity_bytes,
            )
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"invalid execution configuration: {exc}") from exc

        object.__setattr__(self, "executor", ExecutionClass(probe.executor))
        object.__setattr__(self, "execution_timeout_s", probe.execution_timeout_s)
        object.__setattr__(self, "process_transport", probe.process_transport)
        object.__setattr__(
            self,
            "process_request_capacity_bytes",
            probe.process_request_capacity_bytes,
        )
        object.__setattr__(
            self,
            "process_response_capacity_bytes",
            probe.process_response_capacity_bytes,
        )

    @property
    def is_default(self) -> bool:
        return (
            self.executor is ExecutionClass.INLINE
            and self.execution_timeout_s is None
            and self.process_transport == "pickle"
            and self.process_request_capacity_bytes is None
            and self.process_response_capacity_bytes is None
        )

    def runtime_kwargs(self) -> dict[str, Any]:
        """Return canonical ``RuntimeNode`` keyword arguments."""
        return {
            "executor": self.executor.value,
            "execution_timeout_s": self.execution_timeout_s,
            "process_transport": self.process_transport,
            "process_request_capacity_bytes": self.process_request_capacity_bytes,
            "process_response_capacity_bytes": self.process_response_capacity_bytes,
        }


@dataclass(frozen=True, slots=True)
class PluginConfig:
    """Reference to a plugin plus constructor and execution options."""

    plugin: str
    options: Mapping[str, Any] = field(default_factory=dict)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "plugin", _nonblank_string(self.plugin, field_name="plugin")
        )
        if not isinstance(self.options, Mapping):
            raise ConfigurationError("plugin options must be a mapping")
        if not isinstance(self.execution, ExecutionConfig):
            raise ConfigurationError("plugin execution must be an ExecutionConfig")
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@dataclass(frozen=True, slots=True)
class StreamConfig:
    """One named data stream and its transform chain."""

    stream_id: str
    source: PluginConfig
    transforms: tuple[PluginConfig, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "stream_id", _nonblank_string(self.stream_id, field_name="stream id")
        )
        if not isinstance(self.source, PluginConfig):
            raise ConfigurationError("stream source must be a PluginConfig")
        object.__setattr__(
            self,
            "transforms",
            _plugin_sequence(self.transforms, field_name="stream transforms"),
        )


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    queue_capacity: int = 100
    overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST

    def __post_init__(self) -> None:
        try:
            edge = RuntimeEdge(
                "__config_runtime_source__",
                "__config_runtime_target__",
                capacity=self.queue_capacity,
                overflow=self.overflow_policy,
            )
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"invalid runtime configuration: {exc}") from exc
        object.__setattr__(self, "queue_capacity", edge.capacity)
        object.__setattr__(self, "overflow_policy", OverflowPolicy(edge.overflow))


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Portable configuration for a neurOS runtime graph."""

    schema_version: int
    streams: tuple[StreamConfig, ...]
    decoder: PluginConfig
    sinks: tuple[PluginConfig, ...] = ()
    monitors: tuple[PluginConfig, ...] = ()
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))

        if not isinstance(self.streams, (list, tuple)):
            raise ConfigurationError("streams must be a list or tuple")
        streams = tuple(self.streams)
        if not streams:
            raise ConfigurationError("At least one stream is required")
        if any(not isinstance(stream, StreamConfig) for stream in streams):
            raise ConfigurationError("streams must contain StreamConfig values")
        object.__setattr__(self, "streams", streams)

        if not isinstance(self.decoder, PluginConfig):
            raise ConfigurationError("decoder must be a PluginConfig")
        sinks = _plugin_sequence(self.sinks, field_name="sinks")
        monitors = _plugin_sequence(self.monitors, field_name="monitors")
        object.__setattr__(self, "sinks", sinks)
        object.__setattr__(self, "monitors", monitors)

        if not isinstance(self.runtime, RuntimeConfig):
            raise ConfigurationError("runtime must be a RuntimeConfig")
        if not isinstance(self.metadata, Mapping):
            raise ConfigurationError("metadata must be a mapping")

        ids = [stream.stream_id for stream in streams]
        if len(ids) != len(set(ids)):
            raise ConfigurationError("stream ids must be unique")

        if self.schema_version == 1:
            plugins = [self.decoder, *sinks, *monitors]
            for stream in streams:
                plugins.append(stream.source)
                plugins.extend(stream.transforms)
            if any(not plugin.execution.is_default for plugin in plugins):
                raise ConfigurationError(
                    "schema_version 1 is legacy-inline and cannot carry execution policy"
                )

        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PipelineConfig":
        if not isinstance(raw, Mapping):
            raise ConfigurationError("configuration root must be a mapping")
        schema_version = _schema_version(raw.get("schema_version", 1))
        if schema_version == 2:
            _reject_unknown_keys(
                raw, allowed=_TOP_LEVEL_V2_KEYS, context="configuration"
            )
        if "streams" not in raw or "decoder" not in raw:
            raise ConfigurationError("config requires streams and decoder")
        streams_raw = raw["streams"]
        decoder_raw = raw["decoder"]

        def parse_execution(item: Any) -> ExecutionConfig:
            if not isinstance(item, Mapping):
                raise ConfigurationError("plugin execution must be a mapping")
            _reject_unknown_keys(
                item, allowed=_EXECUTION_V2_KEYS, context="execution"
            )
            return ExecutionConfig(
                executor=item.get("executor", ExecutionClass.INLINE.value),
                execution_timeout_s=item.get("execution_timeout_s"),
                process_transport=item.get("process_transport", "pickle"),
                process_request_capacity_bytes=item.get(
                    "process_request_capacity_bytes"
                ),
                process_response_capacity_bytes=item.get(
                    "process_response_capacity_bytes"
                ),
            )

        def parse_plugin(item: Any) -> PluginConfig:
            if not isinstance(item, Mapping):
                raise ConfigurationError("plugin configuration must be a mapping")
            if schema_version == 1 and "execution" in item:
                raise ConfigurationError(
                    "schema_version 1 cannot declare plugin execution policy; use schema_version 2"
                )
            if schema_version == 2:
                _reject_unknown_keys(
                    item, allowed=_PLUGIN_V2_KEYS, context="plugin configuration"
                )
            plugin = item.get("plugin")
            options = item.get("options", {})
            execution = (
                parse_execution(item.get("execution", {}))
                if schema_version == 2
                else ExecutionConfig()
            )
            return PluginConfig(plugin=plugin, options=options, execution=execution)

        streams: list[StreamConfig] = []
        if not isinstance(streams_raw, list):
            raise ConfigurationError("streams must be a list")
        for stream in streams_raw:
            if not isinstance(stream, Mapping):
                raise ConfigurationError("each stream must be a mapping")
            if schema_version == 2:
                _reject_unknown_keys(
                    stream, allowed=_STREAM_V2_KEYS, context="stream configuration"
                )
            transforms_raw = stream.get("transforms", [])
            if not isinstance(transforms_raw, list):
                raise ConfigurationError("stream transforms must be a list")
            streams.append(
                StreamConfig(
                    stream_id=stream.get("id"),
                    source=parse_plugin(stream.get("source", {})),
                    transforms=tuple(parse_plugin(item) for item in transforms_raw),
                )
            )

        runtime_raw = raw.get("runtime", {})
        if not isinstance(runtime_raw, Mapping):
            raise ConfigurationError("runtime must be a mapping")
        if schema_version == 2:
            _reject_unknown_keys(
                runtime_raw, allowed=_RUNTIME_V2_KEYS, context="runtime configuration"
            )
        runtime = RuntimeConfig(
            queue_capacity=runtime_raw.get("queue_capacity", 100),
            overflow_policy=runtime_raw.get(
                "overflow_policy", OverflowPolicy.DROP_OLDEST.value
            ),
        )

        def parse_many(key: str) -> tuple[PluginConfig, ...]:
            values = raw.get(key, [])
            if not isinstance(values, list):
                raise ConfigurationError(f"{key} must be a list")
            return tuple(parse_plugin(item) for item in values)

        metadata = raw.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ConfigurationError("metadata must be a mapping")

        return cls(
            schema_version=schema_version,
            streams=tuple(streams),
            decoder=parse_plugin(decoder_raw),
            sinks=parse_many("sinks"),
            monitors=parse_many("monitors"),
            runtime=runtime,
            metadata=metadata,
        )


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a YAML neurOS configuration."""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Unable to read configuration: {config_path}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigurationError("configuration root must be a mapping")
    return PipelineConfig.from_mapping(raw)
