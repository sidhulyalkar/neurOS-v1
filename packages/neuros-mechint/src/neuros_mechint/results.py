"""
Unified Data Structures for Mechanistic Interpretability Results.

This module provides standardized containers for all mech int analyses,
enabling systematic comparison, storage, and workflow optimization.

Key Components:
- MechIntResult: Base class for all analysis results
- Specialized result containers (Circuit, Dynamics, Information, etc.)
- MechIntDatabase: Centralized storage and querying
- Result comparison and aggregation utilities

Design Principles:
1. Interoperability: All results use common interface
2. Provenance: Track what led to each result
3. Efficiency: HDF5/zarr for large arrays, JSON for metadata
4. Extensibility: Easy to add new result types

Author: NeuroS Team
Date: 2025-10-30
"""

import json
import h5py
import numpy as np
import torch
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union, Protocol
from pathlib import Path
from datetime import datetime
import hashlib
import pickle


# ==================== BASE RESULT PROTOCOL ====================

class ResultProtocol(Protocol):
    """Protocol that all result types must implement."""

    method: str
    data: Any
    metadata: Dict[str, Any]
    metrics: Dict[str, float]

    def save(self, path: str) -> None:
        """Save result to disk."""
        ...

    @classmethod
    def load(cls, path: str) -> 'ResultProtocol':
        """Load result from disk."""
        ...

    def compare(self, other: 'ResultProtocol') -> Dict[str, float]:
        """Compare with another result of same type."""
        ...

    def summary(self) -> str:
        """Human-readable summary."""
        ...


# ==================== BASE RESULT CLASS ====================

@dataclass
class MechIntResult:
    """
    Standard container for all mechanistic interpretability results.

    Provides unified interface for storing, loading, comparing, and
    visualizing results from any mech int analysis.

    Attributes:
        method: Name of analysis method (e.g., "SAE", "DMD", "CCA")
        data: Primary result data (arrays, tensors, dicts)
        metadata: Analysis parameters and settings
        metrics: Quantitative evaluation metrics
        visualizations: Plot configurations and data
        provenance: List of parent results that led to this one
        timestamp: When analysis was performed
        content_hash: Hash of data+metadata for caching

    Example:
        >>> result = MechIntResult(
        ...     method="SAE",
        ...     data={"features": features, "reconstructions": recons},
        ...     metadata={"hidden_dim": 1024, "sparsity": 0.01},
        ...     metrics={"mse": 0.05, "sparsity": 0.012}
        ... )
        >>> result.save("results/sae_analysis.h5")
        >>> loaded = MechIntResult.load("results/sae_analysis.h5")
    """

    method: str
    data: Union[np.ndarray, torch.Tensor, Dict[str, Any]]
    metadata: Dict[str, Any]
    metrics: Dict[str, float]
    visualizations: Optional[Dict[str, Any]] = None
    provenance: Optional[List['MechIntResult']] = field(default_factory=list)
    timestamp: Optional[str] = None
    content_hash: Optional[str] = None

    def __post_init__(self):
        """Set timestamp and compute content hash."""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

        if self.content_hash is None:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of data + metadata for caching/deduplication."""
        hasher = hashlib.sha256()

        # Hash metadata (deterministic JSON)
        metadata_str = json.dumps(self.metadata, sort_keys=True)
        hasher.update(metadata_str.encode())

        # Hash data (depends on type)
        if isinstance(self.data, np.ndarray):
            hasher.update(self.data.tobytes())
        elif isinstance(self.data, torch.Tensor):
            hasher.update(self.data.cpu().numpy().tobytes())
        elif isinstance(self.data, dict):
            # Hash dictionary keys and array shapes
            for key in sorted(self.data.keys()):
                hasher.update(key.encode())
                val = self.data[key]
                if isinstance(val, (np.ndarray, torch.Tensor)):
                    shape_str = str(np.array(val).shape)
                    hasher.update(shape_str.encode())

        return hasher.hexdigest()

    def save(self, path: str) -> None:
        """
        Save result to disk.

        Format:
        - HDF5 for arrays/tensors (efficient binary storage)
        - JSON for metadata/metrics (human-readable)
        - Provenance stored as list of paths

        Args:
            path: Save path (will add .h5 extension if not present)
        """
        path = Path(path)
        if path.suffix != '.h5':
            path = path.with_suffix('.h5')

        with h5py.File(path, 'w') as f:
            # Store method and timestamp as attributes
            f.attrs['method'] = self.method
            f.attrs['timestamp'] = self.timestamp
            f.attrs['content_hash'] = self.content_hash

            # Store metadata as JSON string
            f.attrs['metadata'] = json.dumps(self.metadata)
            f.attrs['metrics'] = json.dumps(self.metrics)

            # Store data
            if isinstance(self.data, np.ndarray):
                f.create_dataset('data', data=self.data, compression='gzip')
            elif isinstance(self.data, torch.Tensor):
                f.create_dataset('data', data=self.data.cpu().numpy(), compression='gzip')
            elif isinstance(self.data, dict):
                # Store each array in the dict
                data_group = f.create_group('data')
                for key, value in self.data.items():
                    if isinstance(value, (np.ndarray, torch.Tensor)):
                        arr = value.cpu().numpy() if isinstance(value, torch.Tensor) else value
                        data_group.create_dataset(key, data=arr, compression='gzip')
                    else:
                        # Store as attribute if not array
                        data_group.attrs[key] = str(value)

            # Store visualizations if present
            if self.visualizations:
                f.attrs['visualizations'] = json.dumps(self.visualizations)

            # Store provenance (just the paths to parent results)
            if self.provenance:
                provenance_paths = [str(p.content_hash) for p in self.provenance]
                f.attrs['provenance'] = json.dumps(provenance_paths)

    @classmethod
    def load(cls, path: str) -> 'MechIntResult':
        """
        Load result from disk.

        Args:
            path: Path to saved result

        Returns:
            Loaded MechIntResult
        """
        path = Path(path)

        with h5py.File(path, 'r') as f:
            # Load attributes
            method = f.attrs['method']
            timestamp = f.attrs['timestamp']
            content_hash = f.attrs['content_hash']

            metadata = json.loads(f.attrs['metadata'])
            metrics = json.loads(f.attrs['metrics'])

            # Load data
            if 'data' in f:
                if isinstance(f['data'], h5py.Dataset):
                    data = f['data'][:]
                elif isinstance(f['data'], h5py.Group):
                    # Dictionary of arrays
                    data = {}
                    for key in f['data'].keys():
                        data[key] = f['data'][key][:]
                    # Add non-array attributes
                    for key, value in f['data'].attrs.items():
                        data[key] = value

            # Load visualizations
            visualizations = None
            if 'visualizations' in f.attrs:
                visualizations = json.loads(f.attrs['visualizations'])

            # Note: provenance not loaded to avoid deep recursion
            # Can be loaded separately if needed

            return cls(
                method=method,
                data=data,
                metadata=metadata,
                metrics=metrics,
                visualizations=visualizations,
                provenance=[],  # Empty for now
                timestamp=timestamp,
                content_hash=content_hash
            )

    def compare(self, other: 'MechIntResult') -> Dict[str, float]:
        """
        Compare this result with another result of the same method.

        Args:
            other: Another MechIntResult

        Returns:
            Dictionary of comparison metrics
        """
        if self.method != other.method:
            raise ValueError(f"Cannot compare {self.method} with {other.method}")

        comparison = {}

        # Compare metrics if both have them
        common_metrics = set(self.metrics.keys()) & set(other.metrics.keys())
        for metric in common_metrics:
            diff = abs(self.metrics[metric] - other.metrics[metric])
            comparison[f'{metric}_diff'] = diff
            if other.metrics[metric] != 0:
                rel_diff = diff / abs(other.metrics[metric])
                comparison[f'{metric}_rel_diff'] = rel_diff

        # Compare data if both are arrays
        if isinstance(self.data, np.ndarray) and isinstance(other.data, np.ndarray):
            if self.data.shape == other.data.shape:
                mse = np.mean((self.data - other.data)**2)
                comparison['data_mse'] = mse
                comparison['data_correlation'] = np.corrcoef(
                    self.data.flat, other.data.flat
                )[0, 1]

        return comparison

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"{'='*60}",
            f"MechIntResult: {self.method}",
            f"{'='*60}",
            f"Timestamp: {self.timestamp}",
            f"Hash: {self.content_hash[:16]}...",
            "",
            "Metrics:",
        ]

        for key, value in self.metrics.items():
            lines.append(f"  {key}: {value:.6f}")

        lines.append("")
        lines.append("Metadata:")
        for key, value in self.metadata.items():
            lines.append(f"  {key}: {value}")

        if isinstance(self.data, np.ndarray):
            lines.append("")
            lines.append(f"Data: array of shape {self.data.shape}, dtype {self.data.dtype}")
        elif isinstance(self.data, dict):
            lines.append("")
            lines.append(f"Data: dictionary with {len(self.data)} keys")
            for key in self.data.keys():
                if isinstance(self.data[key], (np.ndarray, torch.Tensor)):
                    shape = self.data[key].shape
                    lines.append(f"  {key}: shape {shape}")

        lines.append(f"{'='*60}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return f"MechIntResult(method='{self.method}', hash={self.content_hash[:8]}..., timestamp={self.timestamp})"


# ==================== SPECIALIZED RESULT CONTAINERS ====================

@dataclass
class CircuitResult(MechIntResult):
    """
    Results from circuit discovery analyses.

    Attributes:
        nodes: List of node names/indices
        edges: List of (source, target, weight) tuples
        circuit_graph: NetworkX graph (stored as dict for serialization)
        circuit_type: Type of circuit (e.g., 'feed-forward', 'recurrent')
    """

    nodes: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str, float]] = field(default_factory=list)
    circuit_type: Optional[str] = None

    def to_networkx(self):
        """Convert to NetworkX DiGraph."""
        import networkx as nx
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        G.add_weighted_edges_from(self.edges)
        return G

    def from_networkx(cls, G, method: str, **kwargs):
        """Create from NetworkX graph."""
        nodes = list(G.nodes())
        edges = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
        return cls(
            method=method,
            nodes=nodes,
            edges=edges,
            **kwargs
        )


@dataclass
class DynamicsResult(MechIntResult):
    """
    Results from dynamical systems analyses.

    Attributes:
        trajectories: Neural trajectories (n_trials, n_timesteps, n_dims)
        eigenvalues: System eigenvalues
        fixed_points: List of fixed point locations
        lyapunov_exponents: Lyapunov exponents
        stability: Overall stability classification
    """

    trajectories: Optional[np.ndarray] = None
    eigenvalues: Optional[np.ndarray] = None
    fixed_points: List[np.ndarray] = field(default_factory=list)
    lyapunov_exponents: Optional[np.ndarray] = None
    stability: Optional[str] = None


@dataclass
class InformationResult(MechIntResult):
    """
    Results from information theory analyses.

    Attributes:
        mutual_information: I(X;Y) estimate
        entropy: H(X) estimate
        conditional_entropy: H(Y|X) estimate
        information_plane: Information plane coordinates
    """

    mutual_information: Optional[float] = None
    entropy: Optional[float] = None
    conditional_entropy: Optional[float] = None
    information_plane: Optional[Dict[str, np.ndarray]] = None


@dataclass
class AlignmentResult(MechIntResult):
    """
    Results from brain alignment analyses.

    Attributes:
        alignment_score: Primary alignment metric
        alignment_type: Type of alignment (CCA, RSA, PLS)
        layer_scores: Per-layer alignment scores
        noise_ceiling: Estimated noise ceiling
    """

    alignment_score: Optional[float] = None
    alignment_type: Optional[str] = None
    layer_scores: Optional[Dict[str, float]] = None
    noise_ceiling: Optional[float] = None


@dataclass
class FractalResult(MechIntResult):
    """
    Results from fractal analysis.

    Attributes:
        fractal_dimension: Primary fractal dimension estimate
        hurst_exponent: Hurst exponent (H)
        spectral_slope: Power law exponent
        scale_free: Whether dynamics are scale-free
    """

    fractal_dimension: Optional[float] = None
    hurst_exponent: Optional[float] = None
    spectral_slope: Optional[float] = None
    scale_free: Optional[bool] = None


# ==================== RESULT COLLECTION ====================

@dataclass
class ResultCollection:
    """
    Collection of related MechIntResults.

    Useful for storing all results from a complete analysis pipeline.

    Attributes:
        results: List of individual results
        name: Name of this collection
        description: Human-readable description
    """

    results: List[MechIntResult] = field(default_factory=list)
    name: Optional[str] = None
    description: Optional[str] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    def add(self, result: MechIntResult):
        """Add a result to collection."""
        self.results.append(result)

    def get_by_method(self, method: str) -> List[MechIntResult]:
        """Get all results of a specific method."""
        return [r for r in self.results if r.method == method]

    def save(self, directory: str):
        """Save all results to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save each result
        for i, result in enumerate(self.results):
            filename = f"{result.method}_{i}.h5"
            result.save(directory / filename)

        # Save collection metadata
        metadata = {
            'name': self.name,
            'description': self.description,
            'timestamp': self.timestamp,
            'n_results': len(self.results),
            'methods': [r.method for r in self.results]
        }
        with open(directory / 'collection_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, directory: str) -> 'ResultCollection':
        """Load collection from directory."""
        directory = Path(directory)

        # Load metadata
        with open(directory / 'collection_metadata.json', 'r') as f:
            metadata = json.load(f)

        # Load all result files
        results = []
        for file in directory.glob('*.h5'):
            if file.name != 'collection_metadata.json':
                result = MechIntResult.load(file)
                results.append(result)

        return cls(
            results=results,
            name=metadata['name'],
            description=metadata['description'],
            timestamp=metadata['timestamp']
        )

    def summary(self) -> str:
        """Generate summary of collection."""
        lines = [
            f"{'='*60}",
            f"ResultCollection: {self.name}",
            f"{'='*60}",
            f"Description: {self.description}",
            f"Timestamp: {self.timestamp}",
            f"Total results: {len(self.results)}",
            "",
            "Results by method:",
        ]

        # Count by method
        method_counts = {}
        for result in self.results:
            method_counts[result.method] = method_counts.get(result.method, 0) + 1

        for method, count in sorted(method_counts.items()):
            lines.append(f"  {method}: {count}")

        lines.append(f"{'='*60}")
        return "\n".join(lines)


__all__ = [
    'ResultProtocol',
    'MechIntResult',
    'CircuitResult',
    'DynamicsResult',
    'InformationResult',
    'AlignmentResult',
    'FractalResult',
    'ResultCollection',
]
