"""
Plugins package
===============

This subpackage contains optional components that extend the core neurOS
framework.  Each plugin resides in its own module or package under
``neuros/plugins`` and exposes a well-defined API.  To register a plugin,
import it via ``neuros.load_plugin`` and instantiate it as needed.

The ``cv`` subpackage provides computer‑vision‑oriented modules including the
DINOv3 backbone and segmentation heads used in the accompanying experiments.
"""

__all__ = ["cv"]
