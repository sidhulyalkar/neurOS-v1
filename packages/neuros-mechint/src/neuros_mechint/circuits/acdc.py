"""
Automated Circuit Discovery (ACDC) for Neural Networks.

Implements the ACDC algorithm from Conmy et al. (2023) for automatically
discovering minimal computational circuits in neural networks.

Key Idea:
    Start with the full model graph and iteratively remove edges that don't
    significantly affect the output. The result is a minimal sufficient circuit
    that implements the computation.

Algorithm:
    1. Start with full graph of all connections
    2. For each edge:
        a. Temporarily ablate (remove) the edge
        b. Measure effect on output
        c. If effect is below threshold, permanently remove edge
    3. Return minimal graph (circuit)

Applications:
    - Understand which connections are actually used
    - Compare circuits across models/tasks
    - Extract and export minimal subnetworks
    - Debug unexpected behaviors

References:
    - Conmy et al. (2023): "Towards Automated Circuit Discovery for Mechanistic Interpretability"
    - Wang et al. (2023): "Interpretability in the Wild: Finding Circuits in Language Models"

Author: NeuroS Team
Date: 2025-10-30
"""

import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Edge:
    """
    Represents an edge (connection) in the computational graph.

    Attributes:
        source: Source node (e.g., "layer2.attention.head3")
        target: Target node (e.g., "layer3.mlp")
        importance: Measured importance score
        ablated: Whether this edge is ablated
    """
    source: str
    target: str
    importance: float = 0.0
    ablated: bool = False

    def __hash__(self):
        return hash((self.source, self.target))

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self.source == other.source and self.target == other.target


@dataclass
class Circuit:
    """
    Minimal computational circuit.

    Attributes:
        edges: Set of edges in the circuit
        nodes: Set of nodes in the circuit
        performance: Circuit performance on task
        sparsity: Fraction of original edges kept
        metadata: Additional circuit information
    """
    edges: Set[Edge] = field(default_factory=set)
    nodes: Set[str] = field(default_factory=set)
    performance: float = 0.0
    sparsity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_edge(self, edge: Edge):
        """Add edge to circuit."""
        self.edges.add(edge)
        self.nodes.add(edge.source)
        self.nodes.add(edge.target)

    def remove_edge(self, edge: Edge):
        """Remove edge from circuit."""
        self.edges.discard(edge)
        # Don't remove nodes - might be connected via other edges

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'edges': [(e.source, e.target, e.importance) for e in self.edges],
            'nodes': list(self.nodes),
            'performance': self.performance,
            'sparsity': self.sparsity,
            'metadata': self.metadata
        }


class AutomatedCircuitDiscovery:
    """
    Automated Circuit Discovery (ACDC) algorithm.

    Automatically finds minimal computational circuits in neural networks
    by iteratively ablating edges and measuring their importance.

    Args:
        model: Neural network to analyze
        threshold: Importance threshold (edges below this are removed)
        metric: Function to measure output quality
        ablation_method: How to ablate edges ('zero', 'mean', 'resample')
        device: Torch device

    Example:
        >>> acdc = AutomatedCircuitDiscovery(model, threshold=0.01)
        >>> circuit = acdc.discover_circuit(inputs, targets)
        >>> print(f"Circuit has {len(circuit.edges)} edges ({circuit.sparsity:.1%} of original)")
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.01,
        metric: Optional[Callable] = None,
        ablation_method: str = 'zero',
        device: Optional[str] = None,
        verbose: bool = True
    ):
        self.model = model
        self.threshold = threshold
        self.metric = metric or self._default_metric
        self.ablation_method = ablation_method
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose

        # Internal state
        self.edge_registry: Dict[Tuple[str, str], Edge] = {}
        self.hooks = []
        self.baseline_output = None

    def _log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            logger.info(f"[ACDC] {message}")

    def _default_metric(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """Default metric: negative MSE (higher is better)."""
        return -torch.mean((output - target) ** 2).item()

    def _build_graph(self) -> Set[Edge]:
        """
        Build complete computational graph of the model.

        Returns:
            Set of all edges in the model
        """
        edges = set()

        # Iterate through named modules to build graph
        modules = dict(self.model.named_modules())

        for name, module in modules.items():
            if name == '':  # Skip root
                continue

            # Identify connections based on module type
            if isinstance(module, nn.Linear):
                # Linear layers connect input to output
                parent = '.'.join(name.split('.')[:-1])
                edge = Edge(source=parent if parent else 'input', target=name)
                edges.add(edge)
                self.edge_registry[(edge.source, edge.target)] = edge

            elif isinstance(module, nn.MultiheadAttention):
                # Attention has multiple heads
                parent = '.'.join(name.split('.')[:-1])
                for head in range(module.num_heads):
                    source = f"{name}.head{head}"
                    target = f"{parent}.output" if parent else 'output'
                    edge = Edge(source=source, target=target)
                    edges.add(edge)
                    self.edge_registry[(edge.source, edge.target)] = edge

        self._log(f"Built graph with {len(edges)} edges")
        return edges

    def _register_hooks(self, edge: Edge):
        """
        Register forward hooks to ablate a specific edge.

        Args:
            edge: Edge to ablate
        """
        # Find the module corresponding to this edge
        source_module = None
        target_module = None

        for name, module in self.model.named_modules():
            if name == edge.source:
                source_module = module
            if name == edge.target:
                target_module = module

        if target_module is None:
            return

        # Hook that ablates the connection
        def ablation_hook(module, input, output):
            if self.ablation_method == 'zero':
                # Zero out the output
                return torch.zeros_like(output)
            elif self.ablation_method == 'mean':
                # Replace with mean activation
                return torch.ones_like(output) * output.mean()
            elif self.ablation_method == 'resample':
                # Resample from distribution
                return torch.randn_like(output) * output.std() + output.mean()
            else:
                return output

        handle = target_module.register_forward_hook(ablation_hook)
        self.hooks.append(handle)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _measure_edge_importance(
        self,
        edge: Edge,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        """
        Measure importance of an edge by ablating it.

        Args:
            edge: Edge to test
            inputs: Input data
            targets: Target outputs

        Returns:
            Importance score (higher = more important)
        """
        # Get baseline output (no ablation)
        if self.baseline_output is None:
            with torch.no_grad():
                self.baseline_output = self.model(inputs)

        baseline_metric = self.metric(self.baseline_output, targets)

        # Register ablation hook
        self._register_hooks(edge)

        # Get output with ablation
        with torch.no_grad():
            ablated_output = self.model(inputs)

        ablated_metric = self.metric(ablated_output, targets)

        # Remove hooks
        self._remove_hooks()

        # Importance = drop in performance when ablated
        importance = baseline_metric - ablated_metric

        return importance

    def discover_circuit(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        max_iterations: Optional[int] = None
    ) -> Circuit:
        """
        Discover minimal computational circuit.

        Args:
            inputs: Input data for evaluation
            targets: Target outputs
            max_iterations: Maximum number of edges to test (None = all)

        Returns:
            Discovered circuit
        """
        self._log("Starting circuit discovery...")

        # Build initial full graph
        all_edges = self._build_graph()
        circuit = Circuit()
        for edge in all_edges:
            circuit.add_edge(edge)

        # Compute baseline performance
        with torch.no_grad():
            self.baseline_output = self.model(inputs)
        baseline_performance = self.metric(self.baseline_output, targets)

        self._log(f"Baseline performance: {baseline_performance:.4f}")

        # Iteratively test and remove unimportant edges
        edges_to_test = list(all_edges)
        n_iterations = max_iterations or len(edges_to_test)

        for i in range(min(n_iterations, len(edges_to_test))):
            edge = edges_to_test[i]

            # Measure importance
            importance = self._measure_edge_importance(edge, inputs, targets)
            edge.importance = importance

            # Remove if below threshold
            if importance < self.threshold:
                circuit.remove_edge(edge)
                edge.ablated = True
                self._log(f"Removed edge {edge.source} → {edge.target} (importance: {importance:.4f})")
            else:
                self._log(f"Kept edge {edge.source} → {edge.target} (importance: {importance:.4f})")

        # Compute final circuit performance and sparsity
        with torch.no_grad():
            final_output = self.model(inputs)
        circuit.performance = self.metric(final_output, targets)
        circuit.sparsity = len(circuit.edges) / len(all_edges) if all_edges else 0.0

        self._log(f"Circuit discovery complete!")
        self._log(f"  Edges: {len(circuit.edges)}/{len(all_edges)} ({circuit.sparsity:.1%})")
        self._log(f"  Performance: {circuit.performance:.4f} (baseline: {baseline_performance:.4f})")

        return circuit

    def compare_circuits(self, circuit1: Circuit, circuit2: Circuit) -> Dict[str, Any]:
        """
        Compare two circuits.

        Args:
            circuit1: First circuit
            circuit2: Second circuit

        Returns:
            Dictionary of comparison metrics
        """
        # Edge overlap
        edges1 = {(e.source, e.target) for e in circuit1.edges}
        edges2 = {(e.source, e.target) for e in circuit2.edges}

        intersection = edges1 & edges2
        union = edges1 | edges2

        overlap = len(intersection) / len(union) if union else 0.0

        # Performance comparison
        perf_diff = abs(circuit1.performance - circuit2.performance)

        # Node overlap
        node_intersection = circuit1.nodes & circuit2.nodes
        node_union = circuit1.nodes | circuit2.nodes
        node_overlap = len(node_intersection) / len(node_union) if node_union else 0.0

        return {
            'edge_overlap': overlap,
            'node_overlap': node_overlap,
            'performance_difference': perf_diff,
            'edges_only_in_1': edges1 - edges2,
            'edges_only_in_2': edges2 - edges1,
            'shared_edges': intersection
        }

    def visualize_circuit(self, circuit: Circuit, save_path: Optional[str] = None):
        """
        Visualize circuit as a graph.

        Args:
            circuit: Circuit to visualize
            save_path: Optional path to save figure
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError:
            self._log("networkx and matplotlib required for visualization")
            return

        # Create directed graph
        G = nx.DiGraph()

        # Add edges with importance weights
        for edge in circuit.edges:
            G.add_edge(edge.source, edge.target, weight=edge.importance)

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Draw
        fig, ax = plt.subplots(figsize=(12, 8))

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                              node_size=500, alpha=0.9, ax=ax)

        # Draw edges (width based on importance)
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1.0

        nx.draw_networkx_edges(
            G, pos,
            width=[3 * w / max_weight for w in weights],
            alpha=0.6,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            ax=ax
        )

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

        ax.set_title(f"Circuit ({len(circuit.edges)} edges, {circuit.sparsity:.1%} sparsity)")
        ax.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self._log(f"Circuit visualization saved to {save_path}")

        return fig


__all__ = [
    'Edge',
    'Circuit',
    'AutomatedCircuitDiscovery',
]
