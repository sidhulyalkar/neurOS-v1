"""Versioned, serializable neurOS configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from neuros.errors import ConfigurationError
from neuros.runtime import OverflowPolicy


@dataclass(frozen=True, slots=True)
class PluginConfig:
    """Reference to a plugin plus constructor options."""

    plugin: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.plugin:
            raise ConfigurationError("plugin must be non-empty")
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@dataclass(frozen=True, slots=True)
class StreamConfig:
    """One named data stream and its transform chain."""

    stream_id: str
    source: PluginConfig
    transforms: tuple[PluginConfig, ...] = ()

    def __post_init__(self) -> None:
        if not self.stream_id:
            raise ConfigurationError("stream id must be non-empty")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    queue_capacity: int = 100
    overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST

    def __post_init__(self) -> None:
        if self.queue_capacity <= 0:
            raise ConfigurationError("runtime.queue_capacity must be positive")


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
        if self.schema_version != 1:
            raise ConfigurationError(
                f"Unsupported config schema_version={self.schema_version}; expected 1"
            )
        if not self.streams:
            raise ConfigurationError("At least one stream is required")
        ids = [stream.stream_id for stream in self.streams]
        if len(ids) != len(set(ids)):
            raise ConfigurationError("stream ids must be unique")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PipelineConfig":
        try:
            schema_version = int(raw.get("schema_version", 1))
            streams_raw = raw["streams"]
            decoder_raw = raw["decoder"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError("config requires streams and decoder") from exc

        def parse_plugin(item: Mapping[str, Any]) -> PluginConfig:
            if not isinstance(item, Mapping):
                raise ConfigurationError("plugin configuration must be a mapping")
            plugin = item.get("plugin")
            if not isinstance(plugin, str):
                raise ConfigurationError("plugin configuration requires a string 'plugin'")
            options = item.get("options", {})
            if not isinstance(options, Mapping):
                raise ConfigurationError("plugin options must be a mapping")
            return PluginConfig(plugin=plugin, options=options)

        streams: list[StreamConfig] = []
        if not isinstance(streams_raw, list):
            raise ConfigurationError("streams must be a list")
        for stream in streams_raw:
            if not isinstance(stream, Mapping):
                raise ConfigurationError("each stream must be a mapping")
            source = parse_plugin(stream.get("source", {}))
            transforms_raw = stream.get("transforms", [])
            if not isinstance(transforms_raw, list):
                raise ConfigurationError("stream transforms must be a list")
            streams.append(
                StreamConfig(
                    stream_id=str(stream.get("id", "")),
                    source=source,
                    transforms=tuple(parse_plugin(item) for item in transforms_raw),
                )
            )

        runtime_raw = raw.get("runtime", {})
        if not isinstance(runtime_raw, Mapping):
            raise ConfigurationError("runtime must be a mapping")
        try:
            runtime = RuntimeConfig(
                queue_capacity=int(runtime_raw.get("queue_capacity", 100)),
                overflow_policy=OverflowPolicy(
                    runtime_raw.get("overflow_policy", OverflowPolicy.DROP_OLDEST.value)
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("invalid runtime configuration") from exc

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
