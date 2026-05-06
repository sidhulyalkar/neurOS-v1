"""
neuros-astro: A glial signal processing layer for neural foundation models.

This package extracts astrocyte calcium events and functional network states from
optical physiology data, converting them into model-ready tokens for multimodal
neural foundation models.

Main modules:
- metadata: Schemas and dataset triage
- io: Data loaders and synthetic generators
- events: Event detection algorithms
- networks: Graph construction and analysis
- tokenization: Model-ready token generation
- export: Format converters
- visualization: Plotting utilities
"""

__version__ = "0.1.0"
__author__ = "neurOS Contributors"

from neuros_astro.metadata.schema import (
    AstroSession,
    AstroRegion,
    AstroEvent,
    AstroGraph,
    TokenizedAstroSequence,
)

__all__ = [
    "AstroSession",
    "AstroRegion",
    "AstroEvent",
    "AstroGraph",
    "TokenizedAstroSequence",
    "__version__",
]
