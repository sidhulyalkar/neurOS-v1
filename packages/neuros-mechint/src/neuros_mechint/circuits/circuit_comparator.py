"""
Circuit Comparator for Cross-Model Circuit Analysis.

Compare circuits discovered across different models to identify:
- Common computational motifs
- Model-specific circuits
- Circuit evolution during training
- Architecture-specific patterns

Based on:
- Olah et al. (2020): Zoom In: An Introduction to Circuits
- Elhage et al. (2021): A Mathematical Framework for Transformer Circuits
- Conmy et al. (2023): Towards Automated Circuit Discovery

Example:
    >>> # Compare circuits from two models
    >>> comparator = CircuitComparator()
    >>>
    >>> # Add circuits
    >>> comparator.add_circuit("model_A", circuit_a, metadata={'arch': 'transformer'})
    >>> comparator.add_circuit("model_B", circuit_b, metadata={'arch': 'transformer'})
    >>>
    >>> # Compare
    >>> comparison = comparator.compare_circuits("model_A", "model_B")
    >>> print(f"Similarity: {comparison.similarity_score:.3f}")
    >>> print(f"Common nodes: {len(comparison.common_nodes)}")
    >>>
    >>> # Visualize comparison
    >>> comparison.visualize_comparison(use_bokeh=True)

Author: NeuroS Team
Date: 2025-10-31
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import logging

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from bokeh.plotting import figure, output_file, save
    from bokeh.models import HoverTool, Circle, MultiLine, ColumnDataSource
    from bokeh.layouts import column, row
    from bokeh.palettes import Category10_10
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from neuros_mechint.results import CircuitResult

logger = logging.getLogger(__name__)


@dataclass
class CircuitComparison:
    """Results from comparing two circuits."""

    circuit_a_id: str
    circuit_b_id: str

    # Similarity metrics
    similarity_score: float  # Overall similarity (0-1)
    node_overlap: float  # Jaccard similarity of nodes
    edge_overlap: float  # Jaccard similarity of edges
    structural_similarity: float  # Graph isomorphism score

    # Set operations
    common_nodes: Set[str] = field(default_factory=set)
    unique_a_nodes: Set[str] = field(default_factory=set)
    unique_b_nodes: Set[str] = field(default_factory=set)

    common_edges: Set[Tuple[str, str]] = field(default_factory=set)
    unique_a_edges: Set[Tuple[str, str]] = field(default_factory=set)
    unique_b_edges: Set[Tuple[str, str]] = field(default_factory=set)

    # Alignment
    node_mapping: Dict[str, str] = field(default_factory=dict)  # A -> B mapping

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def visualize_comparison(
        self,
        use_bokeh: bool = True,
        save_path: Optional[str] = None
    ) -> Any:
        """
        Visualize circuit comparison.

        Shows:
        - Common structure (overlap)
        - Unique components in each circuit
        - Alignment mapping

        Args:
            use_bokeh: Use Bokeh for interactive visualization
            save_path: Path to save visualization

        Returns:
            Bokeh figure or matplotlib figure
        """
        if use_bokeh and BOKEH_AVAILABLE:
            return self._visualize_bokeh(save_path)
        else:
            return self._visualize_matplotlib(save_path)

    def _visualize_bokeh(self, save_path: Optional[str]) -> Any:
        """Create interactive Bokeh comparison visualization."""
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX required for visualization")
            return None

        # Create Venn diagram-style visualization
        plots = []

        # Plot 1: Overlap statistics
        categories = ['Nodes', 'Edges']
        common = [len(self.common_nodes), len(self.common_edges)]
        unique_a = [len(self.unique_a_nodes), len(self.unique_a_edges)]
        unique_b = [len(self.unique_b_nodes), len(self.unique_b_edges)]

        p1 = figure(
            x_range=categories,
            title=f'Circuit Comparison: {self.circuit_a_id} vs {self.circuit_b_id}',
            width=600,
            height=400,
            toolbar_location=None
        )

        x = np.arange(len(categories))
        width = 0.25

        p1.vbar(x=x-width, top=common, width=width, color='green',
               legend_label='Common', alpha=0.7)
        p1.vbar(x=x, top=unique_a, width=width, color='blue',
               legend_label=f'Unique to {self.circuit_a_id}', alpha=0.7)
        p1.vbar(x=x+width, top=unique_b, width=width, color='red',
               legend_label=f'Unique to {self.circuit_b_id}', alpha=0.7)

        p1.xaxis.major_label_orientation = 0
        p1.legend.location = "top_left"
        plots.append(p1)

        # Plot 2: Similarity metrics
        metrics = ['Node Overlap', 'Edge Overlap', 'Structural\nSimilarity', 'Overall\nSimilarity']
        values = [
            self.node_overlap,
            self.edge_overlap,
            self.structural_similarity,
            self.similarity_score
        ]

        p2 = figure(
            x_range=metrics,
            title='Similarity Metrics',
            width=600,
            height=400,
            y_range=(0, 1),
            toolbar_location=None
        )

        p2.vbar(x=metrics, top=values, width=0.6, color='teal', alpha=0.7)
        p2.xaxis.major_label_orientation = 0.5
        plots.append(p2)

        layout = column(*plots)

        if save_path:
            output_file(save_path)
            save(layout)

        return layout

    def _visualize_matplotlib(self, save_path: Optional[str]) -> Any:
        """Create matplotlib comparison visualization."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Set overlaps
        ax1 = axes[0]
        categories = ['Nodes', 'Edges']
        common = [len(self.common_nodes), len(self.common_edges)]
        unique_a = [len(self.unique_a_nodes), len(self.unique_a_edges)]
        unique_b = [len(self.unique_b_nodes), len(self.unique_b_edges)]

        x = np.arange(len(categories))
        width = 0.25

        ax1.bar(x - width, common, width, label='Common', color='green', alpha=0.7)
        ax1.bar(x, unique_a, width, label=f'Unique to {self.circuit_a_id}', color='blue', alpha=0.7)
        ax1.bar(x + width, unique_b, width, label=f'Unique to {self.circuit_b_id}', color='red', alpha=0.7)

        ax1.set_ylabel('Count')
        ax1.set_title(f'Circuit Comparison: {self.circuit_a_id} vs {self.circuit_b_id}')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Similarity metrics
        ax2 = axes[1]
        metrics = ['Node\nOverlap', 'Edge\nOverlap', 'Structural\nSimilarity', 'Overall\nSimilarity']
        values = [
            self.node_overlap,
            self.edge_overlap,
            self.structural_similarity,
            self.similarity_score
        ]

        ax2.bar(metrics, values, color='teal', alpha=0.7)
        ax2.set_ylabel('Similarity Score')
        ax2.set_title('Similarity Metrics')
        ax2.set_ylim(0, 1)
        ax2.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


@dataclass
class MultiCircuitAnalysis:
    """Analysis of multiple circuits."""

    circuit_ids: List[str]

    # Pairwise comparisons
    pairwise_similarities: np.ndarray  # (n_circuits, n_circuits)

    # Consensus circuit
    consensus_nodes: Set[str] = field(default_factory=set)
    consensus_edges: Set[Tuple[str, str]] = field(default_factory=set)
    consensus_strength: Dict[str, float] = field(default_factory=dict)  # How many circuits share each component

    # Clustering
    circuit_clusters: Optional[List[List[str]]] = None

    # Evolution (if temporal)
    temporal_order: Optional[List[str]] = None

    def visualize_consensus(
        self,
        use_bokeh: bool = True,
        save_path: Optional[str] = None
    ) -> Any:
        """Visualize consensus circuit across all models."""
        if use_bokeh and BOKEH_AVAILABLE:
            return self._visualize_consensus_bokeh(save_path)
        else:
            return self._visualize_consensus_matplotlib(save_path)

    def _visualize_consensus_bokeh(self, save_path: Optional[str]) -> Any:
        """Bokeh consensus visualization."""
        # Similarity heatmap
        p = figure(
            title='Circuit Similarity Matrix',
            x_range=self.circuit_ids,
            y_range=list(reversed(self.circuit_ids)),
            width=600,
            height=600,
            toolbar_location='above',
            tools='hover,save'
        )

        # Create heatmap data
        colors = []
        alphas = []
        xs = []
        ys = []
        values = []

        for i, id_i in enumerate(self.circuit_ids):
            for j, id_j in enumerate(self.circuit_ids):
                xs.append(id_j)
                ys.append(id_i)
                val = self.pairwise_similarities[i, j]
                values.append(val)
                # Color map: white -> blue
                color_intensity = int(255 * (1 - val))
                colors.append(f'#{color_intensity:02x}{color_intensity:02x}ff')
                alphas.append(0.9)

        source = ColumnDataSource(data={
            'x': xs,
            'y': ys,
            'value': values,
            'color': colors,
            'alpha': alphas
        })

        p.rect('x', 'y', 1, 1, source=source, fill_color='color', fill_alpha='alpha', line_color=None)

        hover = p.select_one(HoverTool)
        hover.tooltips = [('Pair', '@x vs @y'), ('Similarity', '@value{0.000}')]

        p.axis.major_label_orientation = 0.785  # 45 degrees
        p.axis.major_label_text_font_size = '10pt'

        if save_path:
            output_file(save_path)
            save(p)

        return p

    def _visualize_consensus_matplotlib(self, save_path: Optional[str]) -> Any:
        """Matplotlib consensus visualization."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        # Heatmap
        im = ax.imshow(self.pairwise_similarities, cmap='Blues', vmin=0, vmax=1)

        # Labels
        ax.set_xticks(np.arange(len(self.circuit_ids)))
        ax.set_yticks(np.arange(len(self.circuit_ids)))
        ax.set_xticklabels(self.circuit_ids, rotation=45, ha='right')
        ax.set_yticklabels(self.circuit_ids)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Similarity', rotation=270, labelpad=15)

        # Annotations
        for i in range(len(self.circuit_ids)):
            for j in range(len(self.circuit_ids)):
                text = ax.text(j, i, f'{self.pairwise_similarities[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

        ax.set_title('Circuit Similarity Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class CircuitComparator:
    """
    Compare circuits across models, architectures, and training stages.

    Supports:
    - Pairwise circuit comparison
    - Multi-circuit consensus extraction
    - Temporal circuit evolution tracking
    - Architecture-specific pattern identification

    Args:
        similarity_threshold: Threshold for considering circuits similar (default: 0.5)
        use_graph_isomorphism: Use graph isomorphism for structural comparison
        verbose: Enable verbose logging

    Example:
        >>> comparator = CircuitComparator()
        >>> comparator.add_circuit("model1_layer3", circuit1)
        >>> comparator.add_circuit("model2_layer3", circuit2)
        >>> comp = comparator.compare_circuits("model1_layer3", "model2_layer3")
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        use_graph_isomorphism: bool = True,
        verbose: bool = True
    ):
        self.similarity_threshold = similarity_threshold
        self.use_graph_isomorphism = use_graph_isomorphism
        self.verbose = verbose

        # Storage
        self.circuits: Dict[str, CircuitResult] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

        # Cached comparisons
        self._comparison_cache: Dict[Tuple[str, str], CircuitComparison] = {}

        self._log("Initialized CircuitComparator")

    def _log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            logger.info(f"[CircuitComparator] {message}")

    def add_circuit(
        self,
        circuit_id: str,
        circuit: CircuitResult,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a circuit to the comparator.

        Args:
            circuit_id: Unique identifier for this circuit
            circuit: CircuitResult object
            metadata: Optional metadata (model name, layer, epoch, etc.)
        """
        self.circuits[circuit_id] = circuit
        self.metadata[circuit_id] = metadata or {}
        self._log(f"Added circuit: {circuit_id}")

    def compare_circuits(
        self,
        circuit_a_id: str,
        circuit_b_id: str,
        use_cache: bool = True
    ) -> CircuitComparison:
        """
        Compare two circuits.

        Args:
            circuit_a_id: ID of first circuit
            circuit_b_id: ID of second circuit
            use_cache: Use cached comparison if available

        Returns:
            CircuitComparison object
        """
        # Check cache
        cache_key = tuple(sorted([circuit_a_id, circuit_b_id]))
        if use_cache and cache_key in self._comparison_cache:
            self._log(f"Using cached comparison for {circuit_a_id} vs {circuit_b_id}")
            return self._comparison_cache[cache_key]

        self._log(f"Comparing circuits: {circuit_a_id} vs {circuit_b_id}")

        circuit_a = self.circuits[circuit_a_id]
        circuit_b = self.circuits[circuit_b_id]

        # Extract nodes and edges
        nodes_a = set(circuit_a.nodes)
        nodes_b = set(circuit_b.nodes)

        edges_a = set((e[0], e[1]) for e in circuit_a.edges)  # Ignore weights
        edges_b = set((e[0], e[1]) for e in circuit_b.edges)

        # Compute overlaps
        common_nodes = nodes_a & nodes_b
        unique_a_nodes = nodes_a - nodes_b
        unique_b_nodes = nodes_b - nodes_a

        common_edges = edges_a & edges_b
        unique_a_edges = edges_a - edges_b
        unique_b_edges = edges_b - edges_a

        # Jaccard similarities
        node_overlap = len(common_nodes) / len(nodes_a | nodes_b) if (nodes_a | nodes_b) else 0.0
        edge_overlap = len(common_edges) / len(edges_a | edges_b) if (edges_a | edges_b) else 0.0

        # Structural similarity (graph isomorphism)
        structural_similarity = self._compute_structural_similarity(circuit_a, circuit_b)

        # Overall similarity (weighted average)
        similarity_score = (
            0.3 * node_overlap +
            0.3 * edge_overlap +
            0.4 * structural_similarity
        )

        # Node mapping (best alignment)
        node_mapping = self._compute_node_mapping(circuit_a, circuit_b)

        comparison = CircuitComparison(
            circuit_a_id=circuit_a_id,
            circuit_b_id=circuit_b_id,
            similarity_score=similarity_score,
            node_overlap=node_overlap,
            edge_overlap=edge_overlap,
            structural_similarity=structural_similarity,
            common_nodes=common_nodes,
            unique_a_nodes=unique_a_nodes,
            unique_b_nodes=unique_b_nodes,
            common_edges=common_edges,
            unique_a_edges=unique_a_edges,
            unique_b_edges=unique_b_edges,
            node_mapping=node_mapping,
            metadata={
                'circuit_a_metadata': self.metadata.get(circuit_a_id, {}),
                'circuit_b_metadata': self.metadata.get(circuit_b_id, {})
            }
        )

        # Cache
        self._comparison_cache[cache_key] = comparison

        self._log(f"Comparison complete. Similarity: {similarity_score:.3f}")

        return comparison

    def compare_all(self) -> MultiCircuitAnalysis:
        """
        Compare all circuits pairwise and extract consensus.

        Returns:
            MultiCircuitAnalysis with all pairwise comparisons
        """
        self._log(f"Comparing all {len(self.circuits)} circuits...")

        circuit_ids = list(self.circuits.keys())
        n = len(circuit_ids)

        # Pairwise similarity matrix
        similarities = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    similarities[i, j] = 1.0
                elif i < j:
                    comp = self.compare_circuits(circuit_ids[i], circuit_ids[j])
                    similarities[i, j] = comp.similarity_score
                    similarities[j, i] = comp.similarity_score

        # Extract consensus circuit
        consensus_nodes, consensus_edges, consensus_strength = self._extract_consensus()

        # Cluster circuits by similarity
        clusters = self._cluster_circuits(similarities, circuit_ids)

        analysis = MultiCircuitAnalysis(
            circuit_ids=circuit_ids,
            pairwise_similarities=similarities,
            consensus_nodes=consensus_nodes,
            consensus_edges=consensus_edges,
            consensus_strength=consensus_strength,
            circuit_clusters=clusters
        )

        self._log(f"Multi-circuit analysis complete. Found {len(clusters)} clusters")

        return analysis

    def _compute_structural_similarity(
        self,
        circuit_a: CircuitResult,
        circuit_b: CircuitResult
    ) -> float:
        """
        Compute structural similarity using graph metrics.

        If use_graph_isomorphism is True, uses graph edit distance.
        Otherwise, uses simpler graph statistics.
        """
        if not NETWORKX_AVAILABLE:
            # Fallback: simple statistics
            return (circuit_a.metrics.get('sparsity', 0) - circuit_b.metrics.get('sparsity', 0))

        # Build NetworkX graphs
        G_a = nx.DiGraph()
        G_a.add_nodes_from(circuit_a.nodes)
        G_a.add_edges_from([(e[0], e[1]) for e in circuit_a.edges])

        G_b = nx.DiGraph()
        G_b.add_nodes_from(circuit_b.nodes)
        G_b.add_edges_from([(e[0], e[1]) for e in circuit_b.edges])

        if self.use_graph_isomorphism and len(circuit_a.nodes) < 50:
            # Graph edit distance (expensive for large graphs)
            try:
                # Normalized GED
                ged = nx.graph_edit_distance(G_a, G_b, timeout=1.0)
                max_size = max(len(circuit_a.nodes), len(circuit_b.nodes))
                similarity = 1.0 - (ged / max_size if max_size > 0 else 0.0)
                return max(0.0, min(1.0, similarity))
            except:
                pass  # Timeout or error, fall through

        # Use graph statistics
        stats_a = {
            'density': nx.density(G_a),
            'avg_degree': sum(dict(G_a.degree()).values()) / len(G_a.nodes()) if G_a.nodes() else 0
        }

        stats_b = {
            'density': nx.density(G_b),
            'avg_degree': sum(dict(G_b.degree()).values()) / len(G_b.nodes()) if G_b.nodes() else 0
        }

        # Similarity based on statistics
        density_sim = 1.0 - abs(stats_a['density'] - stats_b['density'])
        degree_sim = 1.0 - abs(stats_a['avg_degree'] - stats_b['avg_degree']) / max(stats_a['avg_degree'], stats_b['avg_degree'], 1.0)

        return (density_sim + degree_sim) / 2.0

    def _compute_node_mapping(
        self,
        circuit_a: CircuitResult,
        circuit_b: CircuitResult
    ) -> Dict[str, str]:
        """
        Compute best node alignment between circuits.

        Uses simple name matching for now. Could be extended with
        graph matching algorithms.
        """
        mapping = {}

        # Simple string matching
        for node_a in circuit_a.nodes:
            best_match = None
            best_score = 0.0

            for node_b in circuit_b.nodes:
                # Longest common substring
                score = sum(1 for a, b in zip(node_a, node_b) if a == b) / max(len(node_a), len(node_b))

                if score > best_score and score > 0.5:
                    best_score = score
                    best_match = node_b

            if best_match:
                mapping[node_a] = best_match

        return mapping

    def _extract_consensus(self) -> Tuple[Set[str], Set[Tuple[str, str]], Dict[str, float]]:
        """Extract consensus circuit from all circuits."""
        # Count how many circuits contain each component
        node_counts = defaultdict(int)
        edge_counts = defaultdict(int)

        n_circuits = len(self.circuits)

        for circuit in self.circuits.values():
            for node in circuit.nodes:
                node_counts[node] += 1
            for edge in circuit.edges:
                edge_tuple = (edge[0], edge[1])
                edge_counts[edge_tuple] += 1

        # Consensus: components in > 50% of circuits
        threshold = n_circuits / 2
        consensus_nodes = {node for node, count in node_counts.items() if count > threshold}
        consensus_edges = {edge for edge, count in edge_counts.items() if count > threshold}

        # Strength: fraction of circuits containing each component
        consensus_strength = {
            **{f'node_{node}': count / n_circuits for node, count in node_counts.items()},
            **{f'edge_{edge[0]}_{edge[1]}': count / n_circuits for edge, count in edge_counts.items()}
        }

        return consensus_nodes, consensus_edges, consensus_strength

    def _cluster_circuits(
        self,
        similarities: np.ndarray,
        circuit_ids: List[str]
    ) -> List[List[str]]:
        """Cluster circuits by similarity."""
        # Simple hierarchical clustering
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        # Convert similarity to distance
        distances = 1.0 - similarities

        # Linkage
        try:
            condensed_dist = squareform(distances, checks=False)
            Z = linkage(condensed_dist, method='average')

            # Form clusters
            cluster_labels = fcluster(Z, 1 - self.similarity_threshold, criterion='distance')

            # Group by cluster
            clusters = defaultdict(list)
            for circuit_id, label in zip(circuit_ids, cluster_labels):
                clusters[label].append(circuit_id)

            return list(clusters.values())

        except:
            # Fallback: each circuit is its own cluster
            return [[cid] for cid in circuit_ids]


__all__ = [
    'CircuitComparison',
    'MultiCircuitAnalysis',
    'CircuitComparator',
]
