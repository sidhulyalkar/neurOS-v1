"""Diagnostics and plugin-inspection commands for the neurOS CLI."""

from __future__ import annotations

import platform
import sys
from importlib import metadata
from typing import Any

from neuros.plugins import PluginKind, registry


CORE_DISTRIBUTIONS = (
    "neuros",
    "neuros-core",
    "neuros-drivers",
    "neuros-models",
    "neuros-orion",
)


def plugin_inventory() -> list[dict[str, Any]]:
    registry.discover()
    return [
        {
            "kind": descriptor.kind.value,
            "name": descriptor.name,
            "distribution": descriptor.distribution,
            "version": descriptor.version,
        }
        for descriptor in registry.list()
    ]


def devices() -> list[dict[str, Any]]:
    registry.discover([PluginKind.SOURCE])
    return [
        {
            "name": descriptor.name,
            "distribution": descriptor.distribution,
            "version": descriptor.version,
        }
        for descriptor in registry.list(PluginKind.SOURCE)
    ]


def doctor() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for distribution in CORE_DISTRIBUTIONS:
        try:
            packages[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            packages[distribution] = None

    plugins = plugin_inventory()
    by_kind = {
        kind.value: sum(1 for item in plugins if item["kind"] == kind.value)
        for kind in PluginKind
    }
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "supported": sys.version_info >= (3, 10),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "plugins": {"total": len(plugins), "by_kind": by_kind},
        "healthy": bool(
            sys.version_info >= (3, 10)
            and packages["neuros-core"]
            and packages["neuros-drivers"]
            and packages["neuros-models"]
            and by_kind[PluginKind.SOURCE.value] > 0
            and by_kind[PluginKind.DECODER.value] > 0
        ),
    }
