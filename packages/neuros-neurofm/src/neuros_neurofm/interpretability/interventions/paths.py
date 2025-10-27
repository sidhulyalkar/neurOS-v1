"""
Path Analysis and Information Flow

Tools for analyzing information flow through neural networks, including
path analysis, causal graphs, and gradient-based attribution.

These methods reveal how information propagates from inputs through
intermediate representations to outputs.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Callable, Tuple, Any, Set
from dataclasses import dataclass, field
import logging
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class PathInfo:
    """Information about a computational path through the network.

    Args:
        nodes: List of node names in path (from input to output)
        strength: Strength of information flow along this path
        attribution: How much this path contributes to output
    """
    nodes: List[str]
    strength: float
    attribution: float

    def __repr__(self) -> str:
        path_str = " → ".join(self.nodes)
        return f"Path({path_str}, strength={self.strength:.4f})"


@dataclass
class CausalEdge:
    """Edge in causal graph representing information flow.

    Args:
        source: Source node name
        target: Target node name
        weight: Strength of causal influence
        attribution: Attribution score for this edge
    """
    source: str
    target: str
    weight: float
    attribution: float = 0.0


class InformationFlow:
    """
    Analyze information flow using gradient-based methods.

    Computes how information flows from inputs through the network
    using gradients and integrated gradients.

    Args:
        model: Neural network model
        device: Torch device

    Example:
        >>> flow = InformationFlow(model)
        >>> attributions = flow.compute_flow(
        ...     input_data, target_layer='layer_6', baseline=baseline_input
        ... )
        >>> print(f"Flow to layer 6: {attributions['layer_6'].mean():.3f}")
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
    ):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.activations = {}
        self.gradients = {}
        self.hooks = []

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_flow(
        self,
        input_data: Tensor,
        target_layer: str,
        baseline: Optional[Tensor] = None,
        n_steps: int = 50,
    ) -> Dict[str, Tensor]:
        """
        Compute information flow using integrated gradients.

        Args:
            input_data: Input tensor
            target_layer: Layer to compute flow to
            baseline: Baseline input (default: zeros)
            n_steps: Number of integration steps

        Returns:
            Dictionary mapping layer names to attribution tensors
        """
        if baseline is None:
            baseline = torch.zeros_like(input_data)

        input_data = input_data.to(self.device)
        baseline = baseline.to(self.device)

        # Register hooks to capture activations
        layer_names = []
        for name, module in self.model.named_modules():
            if name and not list(module.children()):  # Leaf modules only
                layer_names.append(name)

        def make_forward_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.activations[name] = output
            return hook

        def make_backward_hook(name):
            def hook(module, grad_input, grad_output):
                if isinstance(grad_output, tuple):
                    grad_output = grad_output[0]
                if grad_output is not None:
                    self.gradients[name] = grad_output
            return hook

        for name, module in self.model.named_modules():
            if name in layer_names:
                self.hooks.append(module.register_forward_hook(make_forward_hook(name)))
                self.hooks.append(module.register_full_backward_hook(make_backward_hook(name)))

        # Integrated gradients
        attributions = defaultdict(lambda: 0)

        for step in range(n_steps):
            # Interpolate between baseline and input
            alpha = (step + 1) / n_steps
            interpolated = baseline + alpha * (input_data - baseline)
            interpolated.requires_grad = True

            # Forward pass
            self.model.eval()
            output = self.model(interpolated)

            # Get target activation
            if target_layer not in self.activations:
                raise ValueError(f"Layer {target_layer} not found in activations")

            target_activation = self.activations[target_layer]

            # Backward pass (sum over all elements in target activation)
            self.model.zero_grad()
            target_activation.sum().backward()

            # Accumulate gradients
            for name in layer_names:
                if name in self.gradients:
                    grad = self.gradients[name]
                    act = self.activations[name]
                    attributions[name] += (grad * act).detach() / n_steps

        self._remove_hooks()

        return dict(attributions)

    def compute_layer_importance(
        self,
        input_data: Tensor,
        output_metric: Callable[[Tensor], Tensor],
    ) -> Dict[str, float]:
        """
        Compute importance of each layer using gradient magnitudes.

        Args:
            input_data: Input tensor
            output_metric: Function that takes model output and returns scalar metric

        Returns:
            Dictionary mapping layer names to importance scores
        """
        input_data = input_data.to(self.device)
        input_data.requires_grad = True

        # Register hooks
        layer_names = []
        for name, module in self.model.named_modules():
            if name and not list(module.children()):
                layer_names.append(name)

        def make_forward_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.activations[name] = output
            return hook

        def make_backward_hook(name):
            def hook(module, grad_input, grad_output):
                if isinstance(grad_output, tuple):
                    grad_output = grad_output[0]
                if grad_output is not None:
                    self.gradients[name] = grad_output
            return hook

        for name, module in self.model.named_modules():
            if name in layer_names:
                self.hooks.append(module.register_forward_hook(make_forward_hook(name)))
                self.hooks.append(module.register_full_backward_hook(make_backward_hook(name)))

        # Forward pass
        self.model.eval()
        output = self.model(input_data)
        metric = output_metric(output)

        # Backward pass
        self.model.zero_grad()
        metric.backward()

        # Compute importance as gradient magnitude
        importance = {}
        for name in layer_names:
            if name in self.gradients:
                grad = self.gradients[name]
                importance[name] = grad.abs().mean().item()

        self._remove_hooks()

        return importance


class PathAnalyzer:
    """
    Analyze computational paths through the network.

    Identifies important paths from inputs to outputs by combining
    activation patterns and gradient information.

    Args:
        model: Neural network model
        device: Torch device

    Example:
        >>> analyzer = PathAnalyzer(model)
        >>> paths = analyzer.find_important_paths(
        ...     input_data, output_metric, top_k=10
        ... )
        >>> for path in paths:
        ...     print(path)
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
    ):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.flow_analyzer = InformationFlow(model, device)

    def find_important_paths(
        self,
        input_data: Tensor,
        output_metric: Callable[[Tensor], Tensor],
        top_k: int = 10,
        min_strength: float = 0.01,
    ) -> List[PathInfo]:
        """
        Find top-k most important computational paths.

        Args:
            input_data: Input data
            output_metric: Function to compute output metric for gradients
            top_k: Number of paths to return
            min_strength: Minimum path strength to consider

        Returns:
            List of PathInfo objects sorted by importance
        """
        # Compute layer importance
        importance = self.flow_analyzer.compute_layer_importance(input_data, output_metric)

        # Build graph of layers
        layer_graph = self._build_layer_graph()

        # Find paths using importance scores
        paths = self._enumerate_paths(layer_graph, importance, min_strength)

        # Sort by strength and return top-k
        paths_sorted = sorted(paths, key=lambda p: p.strength, reverse=True)
        return paths_sorted[:top_k]

    def _build_layer_graph(self) -> Dict[str, List[str]]:
        """Build directed graph of layer connections."""
        graph = defaultdict(list)

        # Get all layer names in order
        layer_names = []
        for name, module in self.model.named_modules():
            if name and not list(module.children()):
                layer_names.append(name)

        # Simple sequential assumption (can be improved for complex architectures)
        for i in range(len(layer_names) - 1):
            graph[layer_names[i]].append(layer_names[i + 1])

        return dict(graph)

    def _enumerate_paths(
        self,
        graph: Dict[str, List[str]],
        importance: Dict[str, float],
        min_strength: float,
        max_length: int = 5,
    ) -> List[PathInfo]:
        """
        Enumerate paths through the graph.

        Args:
            graph: Layer connectivity graph
            importance: Importance scores per layer
            min_strength: Minimum path strength
            max_length: Maximum path length

        Returns:
            List of PathInfo objects
        """
        paths = []

        # Find input and output nodes
        all_nodes = set(graph.keys())
        for targets in graph.values():
            all_nodes.update(targets)

        # Input nodes: no incoming edges
        input_nodes = [n for n in all_nodes if not any(n in targets for targets in graph.values())]
        # Output nodes: no outgoing edges
        output_nodes = [n for n in all_nodes if n not in graph or not graph[n]]

        # BFS to find paths
        for start_node in input_nodes:
            queue = deque([(start_node, [start_node], 1.0)])

            while queue:
                current, path, strength = queue.popleft()

                # Check if path is complete (reached output node)
                if current in output_nodes:
                    if strength >= min_strength:
                        paths.append(PathInfo(
                            nodes=path.copy(),
                            strength=strength,
                            attribution=strength,  # Simplified
                        ))
                    continue

                # Don't extend paths beyond max_length
                if len(path) >= max_length:
                    continue

                # Extend path
                if current in graph:
                    for next_node in graph[current]:
                        # Compute path strength (product of importances)
                        next_strength = strength * importance.get(next_node, 0.0)

                        if next_strength >= min_strength:
                            queue.append((next_node, path + [next_node], next_strength))

        return paths


class CausalGraph:
    """
    Build and analyze causal graph of network computations.

    Constructs a graph where nodes are layers/components and edges
    represent causal information flow.

    Args:
        model: Neural network model
        device: Torch device

    Example:
        >>> graph = CausalGraph(model)
        >>> graph.build_from_data(input_data, output_metric)
        >>> edges = graph.get_important_edges(top_k=20)
        >>> graph.visualize()
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
    ):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.edges: List[CausalEdge] = []
        self.nodes: Set[str] = set()

    def build_from_data(
        self,
        input_data: Tensor,
        output_metric: Callable[[Tensor], Tensor],
    ):
        """
        Build causal graph from data using gradient information.

        Args:
            input_data: Input data for computing gradients
            output_metric: Metric to compute gradients with respect to
        """
        flow_analyzer = InformationFlow(self.model, self.device)
        importance = flow_analyzer.compute_layer_importance(input_data, output_metric)

        # Get layer names
        layer_names = list(importance.keys())
        self.nodes = set(layer_names)

        # Build edges (sequential model assumption)
        for i in range(len(layer_names) - 1):
            source = layer_names[i]
            target = layer_names[i + 1]

            # Edge weight is geometric mean of node importances
            weight = np.sqrt(importance[source] * importance[target])

            edge = CausalEdge(
                source=source,
                target=target,
                weight=weight,
                attribution=weight,
            )
            self.edges.append(edge)

    def get_important_edges(self, top_k: int = 20) -> List[CausalEdge]:
        """Get top-k most important edges."""
        sorted_edges = sorted(self.edges, key=lambda e: e.weight, reverse=True)
        return sorted_edges[:top_k]

    def get_node_importance(self) -> Dict[str, float]:
        """Compute node importance from incoming/outgoing edges."""
        importance = defaultdict(float)

        for edge in self.edges:
            importance[edge.source] += edge.weight
            importance[edge.target] += edge.weight

        return dict(importance)

    def visualize(self, top_k_edges: int = 20):
        """
        Visualize causal graph.

        Args:
            top_k_edges: Number of top edges to show

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            logger.error("NetworkX and Matplotlib required for visualization")
            return None

        # Create NetworkX graph
        G = nx.DiGraph()

        # Add nodes
        for node in self.nodes:
            G.add_node(node)

        # Add top-k edges
        top_edges = self.get_important_edges(top_k_edges)
        edge_weights = []
        for edge in top_edges:
            G.add_edge(edge.source, edge.target)
            edge_weights.append(edge.weight)

        # Normalize edge weights for visualization
        if edge_weights:
            max_weight = max(edge_weights)
            edge_widths = [3 * (w / max_weight) for w in edge_weights]
        else:
            edge_widths = []

        # Layout
        fig, ax = plt.subplots(figsize=(14, 10))
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Draw
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                              node_size=1000, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths,
                              edge_color='gray', arrows=True,
                              arrowsize=15, ax=ax)

        ax.set_title('Causal Graph of Network Computation', fontsize=14)
        ax.axis('off')

        plt.tight_layout()
        return fig

    def summarize(self) -> str:
        """Generate text summary of causal graph."""
        summary = []
        summary.append("=" * 60)
        summary.append("CAUSAL GRAPH SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Nodes: {len(self.nodes)}")
        summary.append(f"Edges: {len(self.edges)}")

        # Node importance
        node_importance = self.get_node_importance()
        sorted_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)

        summary.append("\nMost Important Nodes:")
        for node, importance in sorted_nodes[:10]:
            summary.append(f"  {node}: {importance:.4f}")

        # Important edges
        top_edges = self.get_important_edges(10)
        summary.append("\nMost Important Edges:")
        for edge in top_edges:
            summary.append(f"  {edge.source} → {edge.target}: {edge.weight:.4f}")

        summary.append("=" * 60)
        return "\n".join(summary)
