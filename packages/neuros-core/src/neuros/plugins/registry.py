"""Entry-point based plugin discovery for neurOS."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib import metadata
from typing import Any, Callable, Iterable


class PluginKind(str, Enum):
    SOURCE = "source"
    TRANSFORM = "transform"
    TOKENIZER = "tokenizer"
    ENCODER = "encoder"
    DECODER = "decoder"
    SINK = "sink"
    MONITOR = "monitor"
    WORLD_MODEL = "world_model"

    @property
    def entry_point_group(self) -> str:
        return f"neuros.{self.value}s"


@dataclass(frozen=True, slots=True)
class PluginDescriptor:
    name: str
    kind: PluginKind
    factory: Callable[..., Any]
    distribution: str | None = None
    version: str | None = None


class PluginRegistry:
    """In-memory registry backed by Python package entry points."""

    def __init__(self) -> None:
        self._plugins: dict[tuple[PluginKind, str], PluginDescriptor] = {}

    def register(
        self,
        *,
        name: str,
        kind: PluginKind | str,
        factory: Callable[..., Any],
        distribution: str | None = None,
        version: str | None = None,
        replace: bool = False,
    ) -> PluginDescriptor:
        plugin_kind = PluginKind(kind)
        key = (plugin_kind, name)
        if key in self._plugins and not replace:
            raise ValueError(f"Plugin already registered: {plugin_kind.value}:{name}")
        descriptor = PluginDescriptor(
            name=name,
            kind=plugin_kind,
            factory=factory,
            distribution=distribution,
            version=version,
        )
        self._plugins[key] = descriptor
        return descriptor

    def discover(self, kinds: Iterable[PluginKind] | None = None) -> list[PluginDescriptor]:
        selected = list(kinds) if kinds is not None else list(PluginKind)
        discovered: list[PluginDescriptor] = []
        entry_points = metadata.entry_points()
        for kind in selected:
            if hasattr(entry_points, "select"):
                group_entries = entry_points.select(group=kind.entry_point_group)
            else:  # pragma: no cover
                group_entries = entry_points.get(kind.entry_point_group, [])
            for entry_point in group_entries:
                key = (kind, entry_point.name)
                if key in self._plugins:
                    continue
                factory = entry_point.load()
                dist = getattr(entry_point, "dist", None)
                discovered.append(
                    self.register(
                        name=entry_point.name,
                        kind=kind,
                        factory=factory,
                        distribution=getattr(dist, "name", None),
                        version=getattr(dist, "version", None),
                    )
                )
        return discovered

    def get(self, kind: PluginKind | str, name: str) -> PluginDescriptor:
        key = (PluginKind(kind), name)
        if key not in self._plugins:
            self.discover([key[0]])
        try:
            return self._plugins[key]
        except KeyError as exc:
            raise KeyError(f"Unknown plugin: {key[0].value}:{name}") from exc

    def create(self, kind: PluginKind | str, name: str, /, **kwargs: Any) -> Any:
        return self.get(kind, name).factory(**kwargs)

    def list(self, kind: PluginKind | str | None = None) -> list[PluginDescriptor]:
        if kind is None:
            return sorted(self._plugins.values(), key=lambda item: (item.kind.value, item.name))
        plugin_kind = PluginKind(kind)
        return sorted(
            (item for item in self._plugins.values() if item.kind is plugin_kind),
            key=lambda item: item.name,
        )


registry = PluginRegistry()


def load_plugin(name: str, *, kind: PluginKind | str, **kwargs: Any) -> Any:
    """Instantiate a registered or entry-point discovered plugin."""
    return registry.create(kind, name, **kwargs)
