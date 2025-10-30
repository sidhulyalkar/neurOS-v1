"""
Temporal Causal Graph Estimation and Perturbation Tracing

Tools for building causal graphs from neural activations, tracking how
information flows through time and across layers, and visualizing causal
relationships in neural networks.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class CausalGraph:
    """
    Causal graph representing relationships between neural components.

    Args:
        adjacency_matrix: NxN matrix where [i,j] indicates causal influence from i to j
        node_names: Names of nodes in the graph
        p_values: Statistical significance of edges (optional)
        metadata: Additional information about the graph

    Example:
        >>> adjacency = np.array([[0, 0.8, 0.2], [0, 0, 0.9], [0, 0, 0]])
        >>> names = ['layer_0', 'layer_1', 'layer_2']
        >>> graph = CausalGraph(adjacency, names)
        >>> print(f"Graph has {len(graph.node_names)} nodes")
    """
    adjacency_matrix: np.ndarray
    node_names: List[str]
    p_values: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate graph structure."""
        n = len(self.node_names)
        assert self.adjacency_matrix.shape == (n, n), \
            f"Adjacency matrix shape {self.adjacency_matrix.shape} doesn't match {n} nodes"

        if self.p_values is not None:
            assert self.p_values.shape == (n, n), \
                f"P-values shape {self.p_values.shape} doesn't match adjacency matrix"

    def get_edge_strength(self, source: str, target: str) -> float:
        """Get causal strength from source to target node."""
        i = self.node_names.index(source)
        j = self.node_names.index(target)
        return float(self.adjacency_matrix[i, j])

    def get_parents(self, node: str) -> List[Tuple[str, float]]:
        """Get all parent nodes (causes) of a given node."""
        j = self.node_names.index(node)
        parents = []
        for i, name in enumerate(self.node_names):
            if self.adjacency_matrix[i, j] > 0:
                parents.append((name, float(self.adjacency_matrix[i, j])))
        return sorted(parents, key=lambda x: x[1], reverse=True)

    def get_children(self, node: str) -> List[Tuple[str, float]]:
        """Get all child nodes (effects) of a given node."""
        i = self.node_names.index(node)
        children = []
        for j, name in enumerate(self.node_names):
            if self.adjacency_matrix[i, j] > 0:
                children.append((name, float(self.adjacency_matrix[i, j])))
        return sorted(children, key=lambda x: x[1], reverse=True)

    def prune(self, threshold: float = 0.1) -> 'CausalGraph':
        """Create a pruned version of the graph by removing weak edges."""
        pruned_adj = self.adjacency_matrix.copy()
        pruned_adj[pruned_adj < threshold] = 0

        return CausalGraph(
            adjacency_matrix=pruned_adj,
            node_names=self.node_names.copy(),
            p_values=self.p_values.copy() if self.p_values is not None else None,
            metadata=self.metadata.copy()
        )


@dataclass
class TimeVaryingGraph:
    """
    Time-varying causal graph that changes over time/training.

    Represents how causal relationships evolve during training or across
    different time windows of neural activity.

    Args:
        graphs: List of CausalGraphs at different time points
        timestamps: Timestamps corresponding to each graph
        metadata: Additional information

    Example:
        >>> graphs = [graph_epoch_0, graph_epoch_10, graph_epoch_20]
        >>> timestamps = [0, 10, 20]
        >>> tv_graph = TimeVaryingGraph(graphs, timestamps)
        >>> print(f"Tracked {len(tv_graph.graphs)} time points")
    """
    graphs: List[CausalGraph]
    timestamps: List[Union[int, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate time-varying graph structure."""
        assert len(self.graphs) == len(self.timestamps), \
            "Number of graphs must match number of timestamps"

        # Verify all graphs have same nodes
        if self.graphs:
            node_names = self.graphs[0].node_names
            for graph in self.graphs[1:]:
                assert graph.node_names == node_names, \
                    "All graphs must have the same nodes"

    def get_graph_at_time(self, time: Union[int, float]) -> Optional[CausalGraph]:
        """Get the graph closest to a given time point."""
        if not self.graphs:
            return None

        # Find closest timestamp
        idx = np.argmin([abs(t - time) for t in self.timestamps])
        return self.graphs[idx]

    def get_edge_evolution(self, source: str, target: str) -> np.ndarray:
        """Track how an edge strength evolves over time."""
        strengths = []
        for graph in self.graphs:
            strengths.append(graph.get_edge_strength(source, target))
        return np.array(strengths)

    def get_stability_score(self, source: str, target: str) -> float:
        """
        Compute stability score for an edge (lower variance = more stable).

        Returns:
            Coefficient of variation of edge strength over time
        """
        evolution = self.get_edge_evolution(source, target)
        if evolution.mean() == 0:
            return float('inf')
        return float(evolution.std() / evolution.mean())


@dataclass
class PerturbationEffect:
    """
    Effect of perturbing a neural component.

    Represents the causal impact of intervening on a specific neural component,
    measured by changes in downstream activations or behavior.

    Args:
        perturbed_component: Name of component that was perturbed
        effect_size: Magnitude of effect on target
        downstream_effects: Effects on each downstream component
        p_value: Statistical significance
        perturbation_type: Type of perturbation ('ablation', 'activation_patch', etc.)

    Example:
        >>> effect = PerturbationEffect(
        ...     perturbed_component='layer_6.neuron_42',
        ...     effect_size=0.85,
        ...     downstream_effects={'layer_7': 0.6, 'layer_8': 0.3}
        ... )
        >>> print(f"Perturbing {effect.perturbed_component} has effect size {effect.effect_size}")
    """
    perturbed_component: str
    effect_size: float
    downstream_effects: Dict[str, float] = field(default_factory=dict)
    p_value: Optional[float] = None
    perturbation_type: str = 'ablation'
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_top_affected(self, k: int = 5) -> List[Tuple[str, float]]:
        """Get top k most affected downstream components."""
        sorted_effects = sorted(
            self.downstream_effects.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_effects[:k]

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if perturbation effect is statistically significant."""
        if self.p_value is None:
            logger.warning("No p-value available for significance test")
            return True  # Assume significant if no p-value
        return self.p_value < alpha


@dataclass
class AlignmentScore:
    """
    Alignment score between neural representations.

    Measures how well representations from different models, layers, or
    conditions align with each other or with external data (e.g., brain activity).

    Args:
        score: Alignment score (e.g., correlation, CCA score)
        source_name: Name of source representation
        target_name: Name of target representation
        method: Alignment method used ('correlation', 'cca', 'rsa', etc.)
        p_value: Statistical significance
        component_scores: Per-component alignment scores

    Example:
        >>> alignment = AlignmentScore(
        ...     score=0.87,
        ...     source_name='model_layer_6',
        ...     target_name='brain_v4',
        ...     method='cca'
        ... )
        >>> print(f"Alignment: {alignment.score:.3f}")
    """
    score: float
    source_name: str
    target_name: str
    method: str = 'correlation'
    p_value: Optional[float] = None
    component_scores: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if alignment is statistically significant."""
        if self.p_value is None:
            return True
        return self.p_value < alpha

    def compare_to(self, other: 'AlignmentScore') -> Dict[str, float]:
        """Compare this alignment to another."""
        return {
            'score_difference': self.score - other.score,
            'relative_improvement': (self.score - other.score) / (other.score + 1e-10),
        }


class CausalGraphBuilder:
    """
    Build temporal causal graphs from neural time-series.

    Uses Granger causality to identify directed causal relationships
    between neural components over time.

    Args:
        granularity: Level of analysis ('layer', 'neuron', 'channel')
        regularization: Regularization type ('lasso', 'ridge', 'none')
        alpha: Regularization strength
        max_lag: Maximum time lag to consider

    Example:
        >>> builder = CausalGraphBuilder(granularity='layer', alpha=0.001)
        >>> # latents shape: (batch, time, n_components)
        >>> graph = builder.build_causal_graph(latents)
        >>> print(f"Built graph with {len(graph.node_names)} nodes")
    """

    def __init__(
        self,
        granularity: str = 'layer',
        regularization: str = 'lasso',
        alpha: float = 0.001,
        max_lag: int = 10,
    ):
        self.granularity = granularity
        self.regularization = regularization
        self.alpha = alpha
        self.max_lag = max_lag

    def build_causal_graph(
        self,
        latents: Tensor,
        node_names: Optional[List[str]] = None,
        window_size: int = 256,
    ) -> CausalGraph:
        """
        Estimate causal graph using Granger causality.

        Args:
            latents: Neural activations of shape (batch, time, n_components)
            node_names: Names for each component (defaults to node_0, node_1, ...)
            window_size: Size of time window for estimation

        Returns:
            CausalGraph representing causal relationships
        """
        if latents.dim() == 2:
            # (time, n_components) -> add batch dimension
            latents = latents.unsqueeze(0)

        batch_size, time_steps, n_components = latents.shape

        if node_names is None:
            node_names = [f"node_{i}" for i in range(n_components)]

        # Convert to numpy for statsmodels
        latents_np = latents.detach().cpu().numpy()

        # Average over batch
        latents_avg = latents_np.mean(axis=0)  # (time, n_components)

        # Compute Granger causality matrix
        adjacency = np.zeros((n_components, n_components))
        p_values = np.ones((n_components, n_components))

        try:
            from statsmodels.tsa.stattools import grangercausalitytests

            for i in range(n_components):
                for j in range(n_components):
                    if i == j:
                        continue

                    # Test if component i causes component j
                    data = np.column_stack([latents_avg[:, j], latents_avg[:, i]])

                    try:
                        results = grangercausalitytests(
                            data,
                            maxlag=min(self.max_lag, len(data) // 4),
                            verbose=False
                        )

                        # Get minimum p-value across lags
                        p_vals = [results[lag][0]['ssr_ftest'][1] for lag in results.keys()]
                        min_p = min(p_vals)

                        p_values[i, j] = min_p

                        # Set adjacency based on significance
                        if min_p < 0.05:
                            # Use 1 - p_value as strength (significant -> stronger)
                            adjacency[i, j] = 1.0 - min_p

                    except Exception as e:
                        logger.debug(f"Granger test failed for {i}->{j}: {e}")
                        continue

        except ImportError:
            logger.warning("statsmodels not available, using correlation-based graph")
            # Fallback: use time-lagged correlation
            for lag in range(1, self.max_lag + 1):
                if lag >= time_steps:
                    break

                lagged = latents_avg[:-lag, :]  # earlier time points
                current = latents_avg[lag:, :]  # later time points

                # Correlation matrix
                corr = np.corrcoef(lagged.T, current.T)
                lagged_corr = corr[:n_components, n_components:]

                adjacency = np.maximum(adjacency, np.abs(lagged_corr))

        return CausalGraph(
            adjacency_matrix=adjacency,
            node_names=node_names,
            p_values=p_values,
            metadata={
                'method': 'granger_causality',
                'max_lag': self.max_lag,
                'window_size': window_size,
            }
        )

    def build_time_varying_graph(
        self,
        latents_over_time: List[Tensor],
        timestamps: List[Union[int, float]],
        node_names: Optional[List[str]] = None,
    ) -> TimeVaryingGraph:
        """
        Build time-varying graph from multiple time points.

        Args:
            latents_over_time: List of latent tensors at different times
            timestamps: Corresponding timestamps
            node_names: Node names

        Returns:
            TimeVaryingGraph
        """
        graphs = []
        for latents in latents_over_time:
            graph = self.build_causal_graph(latents, node_names=node_names)
            graphs.append(graph)

        return TimeVaryingGraph(
            graphs=graphs,
            timestamps=timestamps,
            metadata={'builder_config': {
                'granularity': self.granularity,
                'regularization': self.regularization,
                'alpha': self.alpha,
            }}
        )


class CausalGraphVisualizer:
    """
    Visualize causal graphs and their evolution.

    Args:
        graph: CausalGraph or TimeVaryingGraph to visualize
        figsize: Figure size (width, height)

    Example:
        >>> viz = CausalGraphVisualizer(graph)
        >>> fig = viz.plot_graph(threshold=0.3)
        >>> fig.savefig('causal_graph.png')
    """

    def __init__(
        self,
        graph: Union[CausalGraph, TimeVaryingGraph],
        figsize: Tuple[int, int] = (12, 8),
    ):
        self.graph = graph
        self.figsize = figsize

    def plot_graph(
        self,
        threshold: float = 0.1,
        layout: str = 'spring',
        show_weights: bool = True,
    ):
        """
        Plot the causal graph.

        Args:
            threshold: Only show edges above this strength
            layout: Layout algorithm ('spring', 'circular', 'hierarchical')
            show_weights: Whether to show edge weights

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            logger.error("matplotlib and networkx required for visualization")
            return None

        # Get the graph to plot
        if isinstance(self.graph, TimeVaryingGraph):
            # Plot the most recent graph
            causal_graph = self.graph.graphs[-1]
        else:
            causal_graph = self.graph

        # Prune weak edges
        pruned = causal_graph.prune(threshold)

        # Create NetworkX graph
        G = nx.DiGraph()
        for name in pruned.node_names:
            G.add_node(name)

        for i, source in enumerate(pruned.node_names):
            for j, target in enumerate(pruned.node_names):
                weight = pruned.adjacency_matrix[i, j]
                if weight > 0:
                    G.add_edge(source, target, weight=weight)

        # Layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'hierarchical':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)

        # Plot
        fig, ax = plt.subplots(figsize=self.figsize)

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, node_color='lightblue',
            node_size=1500, ax=ax
        )

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

        # Draw edges with varying width based on weight
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1.0
        edge_widths = [3 * (w / max_weight) for w in weights]

        nx.draw_networkx_edges(
            G, pos, width=edge_widths,
            edge_color='gray', arrows=True,
            arrowsize=20, ax=ax
        )

        # Draw edge labels if requested
        if show_weights:
            edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}"
                          for u, v in edges}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

        ax.set_title('Causal Graph', fontsize=14)
        ax.axis('off')

        plt.tight_layout()
        return fig

    def plot_time_evolution(
        self,
        source: str,
        target: str,
    ):
        """
        Plot how an edge evolves over time.

        Args:
            source: Source node name
            target: Target node name

        Returns:
            Matplotlib figure
        """
        if not isinstance(self.graph, TimeVaryingGraph):
            raise ValueError("plot_time_evolution requires TimeVaryingGraph")

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib required for visualization")
            return None

        evolution = self.graph.get_edge_evolution(source, target)
        timestamps = self.graph.timestamps

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(timestamps, evolution, linewidth=2, marker='o')
        ax.set_xlabel('Time')
        ax.set_ylabel('Causal Strength')
        ax.set_title(f'Causal Edge Evolution: {source} → {target}')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def build_and_visualize_graph(
    latents: Tensor,
    node_names: Optional[List[str]] = None,
    threshold: float = 0.2,
    save_path: Optional[str] = None,
) -> Tuple[CausalGraph, Any]:
    """
    Convenience function to build and visualize a causal graph.

    Args:
        latents: Neural activations
        node_names: Node names
        threshold: Edge threshold for visualization
        save_path: Path to save figure (optional)

    Returns:
        Tuple of (CausalGraph, matplotlib figure)

    Example:
        >>> graph, fig = build_and_visualize_graph(latents, threshold=0.3)
        >>> fig.savefig('my_graph.png')
    """
    # Build graph
    builder = CausalGraphBuilder()
    graph = builder.build_causal_graph(latents, node_names=node_names)

    # Visualize
    visualizer = CausalGraphVisualizer(graph)
    fig = visualizer.plot_graph(threshold=threshold)

    # Save if requested
    if save_path and fig is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved graph visualization to {save_path}")

    return graph, fig
