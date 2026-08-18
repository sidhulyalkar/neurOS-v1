"""neurOS extension registry.

Third-party packages can advertise entry points under ``neuros.sources``,
``neuros.transforms``, ``neuros.tokenizers``, ``neuros.encoders``,
``neuros.decoders``, ``neuros.sinks`` and ``neuros.monitors``.
"""

from .registry import (
    PluginDescriptor,
    PluginKind,
    PluginRegistry,
    load_plugin,
    registry,
)

__all__ = [
    "PluginDescriptor",
    "PluginKind",
    "PluginRegistry",
    "load_plugin",
    "registry",
]
