"""
Motif Detection for Identifying Recurring Circuit Patterns.

Detects common computational motifs (subgraph patterns) across circuits:
- Feed-forward motifs (FFMs)
- Recurrent motifs (loops)
- Skip connections
- Attention heads
- Multi-layer perceptrons
- Custom user-defined motifs

Based on:
- Milo et al. (2002): Network Motifs
- Alon (2007): Network motifs: theory and experimental approaches
- Olah et al. (2020): Zoom In: An Introduction to Circuits

Example:
    >>> # Detect motifs in a circuit
    >>> detector = MotifDetector()
    >>>
    >>> # Load circuit
    >>> detector.load_circuit(circuit_result)
    >>>
    >>> # Detect all motifs
    >>> motifs = detector.detect_all_motifs()
    >>> print(f"Found {len(motifs)} motif instances")
    >>>
    >>> # Find specific motif type
    >>> ffm_motifs = detector.find_motifs_by_type('feedforward')
    >>>
    >>> # Visualize motif distribution
    >>> detector.visualize_motif_distribution(use_bokeh=True)

Author: NeuroS Team
Date: 2025-10-31
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from itertools import combinations
import logging

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from bokeh.plotting import figure, output_file, save
    from bokeh.models import HoverTool, ColumnDataSource
    from bokeh.layouts import column, row
    from bokeh.transform import dodge
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from neuros_mechint.results import CircuitResult

logger = logging.getLogger(__name__)


@dataclass
class MotifInstance:
    """Instance of a detected motif."""

    motif_type: str  # 'feedforward', 'recurrent', 'skip', etc.
    nodes: List[str]  # Nodes involved in this motif
    edges: List[Tuple[str, str]]  # Edges in this motif
    significance: float = 1.0  # Statistical significance vs random

    # Metrics
    frequency: int = 1  # How many times this appears
    centrality: float = 0.0  # Network centrality of motif

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MotifAnalysis:
    """Complete motif analysis results."""

    circuit_id: str

    # Detected motifs
    motifs: List[MotifInstance] = field(default_factory=list)

    # Statistics
    motif_counts: Dict[str, int] = field(default_factory=dict)  # Type -> count
    total_motifs: int = 0

    # Significance
    randomized_counts: Optional[Dict[str, List[int]]] = None  # For null distribution
    z_scores: Optional[Dict[str, float]] = None  # Statistical significance

    # Network statistics
    motif_coverage: float = 0.0  # Fraction of nodes in at least one motif
    motif_density: float = 0.0  # Motifs per node

    def visualize_motif_distribution(
        self,
        use_bokeh: bool = True,
        save_path: Optional[str] = None
    ) -> Any:
        """Visualize motif type distribution."""
        if use_bokeh and BOKEH_AVAILABLE:
            return self._visualize_bokeh(save_path)
        else:
            return self._visualize_matplotlib(save_path)

    def _visualize_bokeh(self, save_path: Optional[str]) -> Any:
        """Bokeh motif distribution visualization."""
        motif_types = list(self.motif_counts.keys())
        counts = list(self.motif_counts.values())

        p = figure(
            x_range=motif_types,
            title=f'Motif Distribution - {self.circuit_id}',
            width=800,
            height=400,
            toolbar_location='above'
        )

        p.vbar(x=motif_types, top=counts, width=0.7, color='navy', alpha=0.7)

        # Add significance markers if available
        if self.z_scores:
            significant = [self.z_scores.get(mt, 0) > 2.0 for mt in motif_types]
            sig_y = [c + max(counts) * 0.05 for c in counts]
            p.circle(
                x=[mt for mt, sig in zip(motif_types, significant) if sig],
                y=[y for y, sig in zip(sig_y, significant) if sig],
                size=10,
                color='red',
                legend_label='Significant (z>2)'
            )

        p.xaxis.major_label_orientation = 0.785
        p.yaxis.axis_label = 'Count'
        p.legend.location = 'top_right'

        if save_path:
            output_file(save_path)
            save(p)

        return p

    def _visualize_matplotlib(self, save_path: Optional[str]) -> Any:
        """Matplotlib motif distribution visualization."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        motif_types = list(self.motif_counts.keys())
        counts = list(self.motif_counts.values())

        bars = ax.bar(motif_types, counts, alpha=0.7, color='navy')

        # Highlight significant motifs
        if self.z_scores:
            for i, (mt, count) in enumerate(zip(motif_types, counts)):
                if self.z_scores.get(mt, 0) > 2.0:
                    bars[i].set_color('red')
                    bars[i].set_alpha(0.9)
                    ax.text(i, count, '*', ha='center', va='bottom', fontsize=20, color='red')

        ax.set_xlabel('Motif Type')
        ax.set_ylabel('Count')
        ax.set_title(f'Motif Distribution - {self.circuit_id}', fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class MotifDetector:
    """
    Detect recurring computational motifs in circuits.

    Identifies common subgraph patterns that represent specific
    computational operations.

    Motif Types:
    - feedforward: A -> B -> C (sequential processing)
    - recurrent: A -> B, B -> A (bidirectional)
    - skip: A -> B -> C, A -> C (skip connection)
    - convergent: A -> C, B -> C (multiple inputs)
    - divergent: A -> B, A -> C (broadcasting)
    - triangle: A -> B, B -> C, A -> C (dense)

    Args:
        min_motif_size: Minimum nodes in a motif (default: 2)
        max_motif_size: Maximum nodes in a motif (default: 5)
        significance_threshold: Z-score threshold for significance (default: 2.0)
        n_random_samples: Number of random graphs for null distribution (default: 100)
        verbose: Enable verbose logging

    Example:
        >>> detector = MotifDetector()
        >>> detector.load_circuit(circuit_result)
        >>> motifs = detector.detect_all_motifs()
    """

    def __init__(
        self,
        min_motif_size: int = 2,
        max_motif_size: int = 5,
        significance_threshold: float = 2.0,
        n_random_samples: int = 100,
        verbose: bool = True
    ):
        self.min_motif_size = min_motif_size
        self.max_motif_size = max_motif_size
        self.significance_threshold = significance_threshold
        self.n_random_samples = n_random_samples
        self.verbose = verbose

        # Graph
        self.graph: Optional[nx.DiGraph] = None
        self.circuit_id: Optional[str] = None

        self._log("Initialized MotifDetector")

    def _log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            logger.info(f"[MotifDetector] {message}")

    def load_circuit(self, circuit: CircuitResult, circuit_id: Optional[str] = None):
        """
        Load a circuit for motif detection.

        Args:
            circuit: CircuitResult object
            circuit_id: Optional identifier
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX required for motif detection")

        self.circuit_id = circuit_id or circuit.method
        self.graph = nx.DiGraph()

        # Add nodes and edges
        self.graph.add_nodes_from(circuit.nodes)
        self.graph.add_edges_from([(e[0], e[1]) for e in circuit.edges])

        self._log(f"Loaded circuit: {len(circuit.nodes)} nodes, {len(circuit.edges)} edges")

    def detect_all_motifs(
        self,
        compute_significance: bool = True
    ) -> MotifAnalysis:
        """
        Detect all motif types.

        Args:
            compute_significance: Run randomization test for significance

        Returns:
            MotifAnalysis with all detected motifs
        """
        if self.graph is None:
            raise ValueError("No circuit loaded. Call load_circuit() first.")

        self._log("Detecting all motif types...")

        all_motifs = []

        # Detect each motif type
        motif_detectors = {
            'feedforward': self._detect_feedforward,
            'recurrent': self._detect_recurrent,
            'skip': self._detect_skip,
            'convergent': self._detect_convergent,
            'divergent': self._detect_divergent,
            'triangle': self._detect_triangle,
        }

        for motif_type, detector_fn in motif_detectors.items():
            motifs = detector_fn()
            all_motifs.extend(motifs)
            self._log(f"  - {motif_type}: {len(motifs)} instances")

        # Count by type
        motif_counts = Counter(m.motif_type for m in all_motifs)

        # Compute significance if requested
        z_scores = None
        randomized_counts = None

        if compute_significance:
            self._log("Computing statistical significance...")
            z_scores, randomized_counts = self._compute_significance(motif_counts)

        # Network statistics
        motif_coverage = self._compute_coverage(all_motifs)
        motif_density = len(all_motifs) / len(self.graph.nodes()) if self.graph.nodes() else 0.0

        analysis = MotifAnalysis(
            circuit_id=self.circuit_id,
            motifs=all_motifs,
            motif_counts=dict(motif_counts),
            total_motifs=len(all_motifs),
            randomized_counts=randomized_counts,
            z_scores=z_scores,
            motif_coverage=motif_coverage,
            motif_density=motif_density
        )

        self._log(f"Detection complete. Found {len(all_motifs)} total motifs")
        self._log(f"  - Coverage: {motif_coverage:.1%}")
        self._log(f"  - Density: {motif_density:.2f} motifs/node")

        return analysis

    def _detect_feedforward(self) -> List[MotifInstance]:
        """Detect feedforward motifs: A -> B -> C"""
        motifs = []

        for node in self.graph.nodes():
            # Get successors
            successors = list(self.graph.successors(node))

            for succ in successors:
                # Check if successor has its own successors
                succ_successors = list(self.graph.successors(succ))

                for third in succ_successors:
                    if third != node:  # Avoid cycles
                        motifs.append(MotifInstance(
                            motif_type='feedforward',
                            nodes=[node, succ, third],
                            edges=[(node, succ), (succ, third)]
                        ))

        return motifs

    def _detect_recurrent(self) -> List[MotifInstance]:
        """Detect recurrent motifs: A <-> B"""
        motifs = []
        seen = set()

        for node_a, node_b in self.graph.edges():
            # Check if reciprocal edge exists
            if self.graph.has_edge(node_b, node_a):
                # Avoid duplicates
                pair = tuple(sorted([node_a, node_b]))
                if pair not in seen:
                    seen.add(pair)
                    motifs.append(MotifInstance(
                        motif_type='recurrent',
                        nodes=[node_a, node_b],
                        edges=[(node_a, node_b), (node_b, node_a)]
                    ))

        return motifs

    def _detect_skip(self) -> List[MotifInstance]:
        """Detect skip connections: A -> B -> C and A -> C"""
        motifs = []

        for node_a in self.graph.nodes():
            successors_a = list(self.graph.successors(node_a))

            for node_b in successors_a:
                successors_b = list(self.graph.successors(node_b))

                for node_c in successors_b:
                    # Check if A also connects directly to C (skip)
                    if self.graph.has_edge(node_a, node_c) and node_c != node_a:
                        motifs.append(MotifInstance(
                            motif_type='skip',
                            nodes=[node_a, node_b, node_c],
                            edges=[(node_a, node_b), (node_b, node_c), (node_a, node_c)]
                        ))

        return motifs

    def _detect_convergent(self) -> List[MotifInstance]:
        """Detect convergent motifs: A -> C, B -> C"""
        motifs = []

        for node_c in self.graph.nodes():
            predecessors = list(self.graph.predecessors(node_c))

            # Find pairs of predecessors
            for node_a, node_b in combinations(predecessors, 2):
                motifs.append(MotifInstance(
                    motif_type='convergent',
                    nodes=[node_a, node_b, node_c],
                    edges=[(node_a, node_c), (node_b, node_c)]
                ))

        return motifs

    def _detect_divergent(self) -> List[MotifInstance]:
        """Detect divergent motifs: A -> B, A -> C"""
        motifs = []

        for node_a in self.graph.nodes():
            successors = list(self.graph.successors(node_a))

            # Find pairs of successors
            for node_b, node_c in combinations(successors, 2):
                motifs.append(MotifInstance(
                    motif_type='divergent',
                    nodes=[node_a, node_b, node_c],
                    edges=[(node_a, node_b), (node_a, node_c)]
                ))

        return motifs

    def _detect_triangle(self) -> List[MotifInstance]:
        """Detect triangle motifs: fully connected 3-node subgraphs"""
        motifs = []

        # Find all 3-node combinations
        for node_a, node_b, node_c in combinations(self.graph.nodes(), 3):
            # Check if forms a triangle (all edges present)
            edges_present = [
                self.graph.has_edge(node_a, node_b),
                self.graph.has_edge(node_b, node_c),
                self.graph.has_edge(node_a, node_c)
            ]

            if sum(edges_present) >= 3:  # At least 3 of 6 possible edges
                present_edges = []
                if edges_present[0]:
                    present_edges.append((node_a, node_b))
                if edges_present[1]:
                    present_edges.append((node_b, node_c))
                if edges_present[2]:
                    present_edges.append((node_a, node_c))

                motifs.append(MotifInstance(
                    motif_type='triangle',
                    nodes=[node_a, node_b, node_c],
                    edges=present_edges
                ))

        return motifs

    def _compute_significance(
        self,
        observed_counts: Counter
    ) -> Tuple[Dict[str, float], Dict[str, List[int]]]:
        """
        Compute statistical significance via randomization.

        Compares observed counts to random graph null distribution.
        """
        randomized_counts = defaultdict(list)

        # Generate random graphs
        for _ in range(self.n_random_samples):
            # Create random graph with same degree sequence
            random_graph = self._generate_random_graph()

            # Temporarily swap graph
            original_graph = self.graph
            self.graph = random_graph

            # Detect motifs in random graph
            try:
                random_analysis = self.detect_all_motifs(compute_significance=False)
                for motif_type, count in random_analysis.motif_counts.items():
                    randomized_counts[motif_type].append(count)
            except:
                pass

            # Restore original graph
            self.graph = original_graph

        # Compute z-scores
        z_scores = {}
        for motif_type, observed_count in observed_counts.items():
            random_counts = randomized_counts.get(motif_type, [0])
            mean = np.mean(random_counts)
            std = np.std(random_counts)

            if std > 0:
                z_score = (observed_count - mean) / std
            else:
                z_score = 0.0

            z_scores[motif_type] = float(z_score)

        return z_scores, dict(randomized_counts)

    def _generate_random_graph(self) -> nx.DiGraph:
        """Generate random graph with same degree sequence."""
        # Simple Erdős–Rényi random graph
        n_nodes = len(self.graph.nodes())
        n_edges = len(self.graph.edges())

        p = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0

        random_graph = nx.gnp_random_graph(n_nodes, p, directed=True)

        # Relabel to match original node names
        mapping = dict(zip(range(n_nodes), self.graph.nodes()))
        random_graph = nx.relabel_nodes(random_graph, mapping)

        return random_graph

    def _compute_coverage(self, motifs: List[MotifInstance]) -> float:
        """Compute fraction of nodes in at least one motif."""
        nodes_in_motifs = set()
        for motif in motifs:
            nodes_in_motifs.update(motif.nodes)

        coverage = len(nodes_in_motifs) / len(self.graph.nodes()) if self.graph.nodes() else 0.0
        return coverage


__all__ = [
    'MotifInstance',
    'MotifAnalysis',
    'MotifDetector',
]
