"""
Federated learning and deployment support for neurOS.

This package contains simple building blocks for federated scenarios
where multiple neurOS instances contribute data or models to a central
coordinator.  The provided classes are illustrative; they may be
extended to support secure aggregation, differential privacy or
multi‑party computation in production environments.

Contents
--------
FederatedAggregator
    Collect metrics and results from multiple neurOS database files
    and compute aggregate statistics.
FederatedClient
    Push run metrics and results to a remote federated aggregator.

These components enable neurOS deployments to cooperate across
independent devices or institutions, facilitating cross‑site analysis
and model evaluation without centralising raw data.
"""

from neuros.federated.aggregator import FederatedAggregator  # noqa: F401
from neuros.federated.client import FederatedClient  # noqa: F401

__all__ = ["FederatedAggregator", "FederatedClient"]