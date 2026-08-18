"""Resolve versioned configuration into concrete plugins and a runtime graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from neuros.config.schema import PipelineConfig, PluginConfig
from neuros.errors import ConfigurationError
from neuros.plugins import PluginKind, PluginRegistry, registry as default_registry
from neuros.runtime import NodeKind, RuntimeEdge, RuntimeGraph, RuntimeNode


@dataclass(frozen=True, slots=True)
class ResolvedStream:
    stream_id: str
    source: Any
    transforms: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class ResolvedPipeline:
    config: PipelineConfig
    streams: tuple[ResolvedStream, ...]
    decoder: Any
    sinks: tuple[Any, ...]
    monitors: tuple[Any, ...]
    graph: RuntimeGraph


def _create(
    registry: PluginRegistry,
    kind: PluginKind,
    config: PluginConfig,
) -> Any:
    try:
        return registry.create(kind, config.plugin, **dict(config.options))
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"Unable to instantiate {kind.value} plugin '{config.plugin}': {exc}"
        ) from exc


def resolve_config(
    config: PipelineConfig,
    *,
    registry: PluginRegistry | None = None,
    source_overrides: Mapping[str, Any] | None = None,
) -> ResolvedPipeline:
    """Instantiate configured plugins and compile them to ``RuntimeGraph``.

    ``source_overrides`` is keyed by stream ID and is primarily used for
    deterministic replay. It prevents hardware source construction entirely,
    which means a recorded experiment can be replayed on a machine that does
    not have the original device SDK installed.
    """

    plugin_registry = registry or default_registry
    plugin_registry.discover()
    overrides = dict(source_overrides or {})
    unknown = set(overrides) - {stream.stream_id for stream in config.streams}
    if unknown:
        raise ConfigurationError(f"Unknown source override stream IDs: {sorted(unknown)}")

    graph = RuntimeGraph()
    resolved_streams: list[ResolvedStream] = []
    stream_tails: list[str] = []

    for stream in config.streams:
        source = overrides.get(stream.stream_id)
        if source is None:
            source = _create(plugin_registry, PluginKind.SOURCE, stream.source)
        transforms = tuple(
            _create(plugin_registry, PluginKind.TRANSFORM, transform)
            for transform in stream.transforms
        )
        source_id = f"source:{stream.stream_id}"
        graph.add_node(
            RuntimeNode(
                node_id=source_id,
                kind=NodeKind.SOURCE,
                operator=source,
                metadata={"stream_id": stream.stream_id, "plugin": stream.source.plugin},
            )
        )
        tail = source_id
        for index, (transform_config, transform) in enumerate(
            zip(stream.transforms, transforms)
        ):
            node_id = f"transform:{stream.stream_id}:{index}"
            graph.add_node(
                RuntimeNode(
                    node_id=node_id,
                    kind=NodeKind.TRANSFORM,
                    operator=transform,
                    metadata={"plugin": transform_config.plugin},
                )
            )
            graph.connect(
                RuntimeEdge(
                    source=tail,
                    target=node_id,
                    capacity=config.runtime.queue_capacity,
                    overflow=config.runtime.overflow_policy.value,
                )
            )
            tail = node_id
        stream_tails.append(tail)
        resolved_streams.append(
            ResolvedStream(
                stream_id=stream.stream_id,
                source=source,
                transforms=transforms,
            )
        )

    decoder = _create(plugin_registry, PluginKind.DECODER, config.decoder)
    decoder_id = "decoder:primary"
    graph.add_node(
        RuntimeNode(
            node_id=decoder_id,
            kind=NodeKind.DECODER,
            operator=decoder,
            metadata={"plugin": config.decoder.plugin},
        )
    )

    if len(stream_tails) == 1:
        upstream_id = stream_tails[0]
    else:
        upstream_id = "fusion:primary"
        graph.add_node(
            RuntimeNode(
                node_id=upstream_id,
                kind=NodeKind.FUSION,
                operator=None,
                metadata={"strategy": "concatenate_latest"},
            )
        )
        for tail in stream_tails:
            graph.connect(
                RuntimeEdge(
                    source=tail,
                    target=upstream_id,
                    capacity=config.runtime.queue_capacity,
                    overflow=config.runtime.overflow_policy.value,
                )
            )

    graph.connect(
        RuntimeEdge(
            source=upstream_id,
            target=decoder_id,
            capacity=config.runtime.queue_capacity,
            overflow=config.runtime.overflow_policy.value,
        )
    )

    sinks = tuple(_create(plugin_registry, PluginKind.SINK, item) for item in config.sinks)
    for index, (sink_config, sink) in enumerate(zip(config.sinks, sinks)):
        sink_id = f"sink:{index}"
        graph.add_node(
            RuntimeNode(
                node_id=sink_id,
                kind=NodeKind.SINK,
                operator=sink,
                metadata={"plugin": sink_config.plugin},
            )
        )
        graph.connect(
            RuntimeEdge(
                source=decoder_id,
                target=sink_id,
                capacity=config.runtime.queue_capacity,
                overflow=config.runtime.overflow_policy.value,
            )
        )

    monitors = tuple(
        _create(plugin_registry, PluginKind.MONITOR, item) for item in config.monitors
    )
    for index, (monitor_config, monitor) in enumerate(zip(config.monitors, monitors)):
        graph.add_node(
            RuntimeNode(
                node_id=f"monitor:{index}",
                kind=NodeKind.MONITOR,
                operator=monitor,
                metadata={"plugin": monitor_config.plugin},
            )
        )

    graph.validate()
    return ResolvedPipeline(
        config=config,
        streams=tuple(resolved_streams),
        decoder=decoder,
        sinks=sinks,
        monitors=monitors,
        graph=graph,
    )
