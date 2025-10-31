"""
Path Patching for Causal Circuit Discovery.

Implements activation patching and path patching techniques for identifying
causal paths in neural networks. Based on:
- Wang et al. (2022): Interpretability in the Wild
- Meng et al. (2022): Locating and Editing Factual Associations
- Goldowsky-Dill et al. (2023): Localizing Model Behavior

Path patching reveals which activation paths are causally responsible for
specific model behaviors by systematically replacing activations and measuring
downstream effects.

Key Concepts:
- **Clean Run**: Forward pass on target input
- **Corrupted Run**: Forward pass on baseline/counterfactual input
- **Patching**: Replace corrupted activations with clean activations
- **Causal Effect**: Change in output when patching specific path
- **Path**: Sequence of (layer, position) tuples from input to output

Example:
    >>> # Setup
    >>> patcher = PathPatcher(model, metric_fn=lambda x: x[:, target_token].max())
    >>>
    >>> # Discover important paths
    >>> results = patcher.patch_all_paths(
    ...     clean_input=clean_tokens,
    ...     corrupted_input=corrupted_tokens
    ... )
    >>>
    >>> # Get top causal paths
    >>> top_paths = results.get_top_paths(k=10)
    >>>
    >>> # Visualize causal graph
    >>> viz = results.visualize_causal_graph(use_bokeh=True)

Author: NeuroS Team
Date: 2025-10-31
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Union, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import logging
from tqdm import tqdm

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from bokeh.plotting import figure, output_file, save, show
    from bokeh.models import (
        HoverTool, Circle, MultiLine,
        Plot, Range1d, GraphRenderer,
        StaticLayoutProvider, ColumnDataSource
    )
    from bokeh.palettes import Spectral8
    from bokeh.layouts import column
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

from neuros_mechint.results import CircuitResult

logger = logging.getLogger(__name__)


@dataclass
class PatchEffect:
    """Effect of patching a specific path component."""

    layer_name: str
    position: Optional[int] = None  # None for full layer
    component: str = "residual"  # "residual", "attn", "mlp", etc.

    # Effects
    direct_effect: float = 0.0  # Effect when patching this component alone
    total_effect: float = 0.0   # Effect when patching this + all downstream
    indirect_effect: float = 0.0  # total_effect - direct_effect

    # Metadata
    activation_norm: float = 0.0
    patch_norm: float = 0.0

    def __hash__(self):
        return hash((self.layer_name, self.position, self.component))


@dataclass
class PathPatchingResult:
    """Results from path patching analysis."""

    # All patch effects
    effects: List[PatchEffect] = field(default_factory=list)

    # Graph structure
    causal_graph: Optional[Any] = None  # NetworkX graph if available

    # Metrics
    clean_output: Optional[torch.Tensor] = None
    corrupted_output: Optional[torch.Tensor] = None
    target_metric: float = 0.0
    baseline_metric: float = 0.0

    # Configuration
    layer_names: List[str] = field(default_factory=list)
    metric_name: str = "logit_diff"

    def get_top_paths(self, k: int = 10, by: str = "direct_effect") -> List[PatchEffect]:
        """
        Get top-k most important paths.

        Args:
            k: Number of paths to return
            by: Sort by 'direct_effect', 'total_effect', or 'indirect_effect'

        Returns:
            List of top PatchEffect objects
        """
        sorted_effects = sorted(
            self.effects,
            key=lambda x: abs(getattr(x, by)),
            reverse=True
        )
        return sorted_effects[:k]

    def get_layer_importance(self) -> Dict[str, float]:
        """Aggregate importance by layer."""
        layer_effects = defaultdict(float)
        for effect in self.effects:
            layer_effects[effect.layer_name] += abs(effect.direct_effect)
        return dict(layer_effects)

    def get_component_importance(self) -> Dict[str, float]:
        """Aggregate importance by component type."""
        component_effects = defaultdict(float)
        for effect in self.effects:
            component_effects[effect.component] += abs(effect.direct_effect)
        return dict(component_effects)

    def to_circuit_result(self) -> CircuitResult:
        """Convert to standardized CircuitResult format."""
        # Build edges from causal relationships
        edges = []
        nodes = set()

        for effect in self.effects:
            if abs(effect.direct_effect) > 0.01:  # Threshold for significance
                node_id = f"{effect.layer_name}_{effect.component}"
                nodes.add(node_id)

                # Create edge to output
                edges.append((node_id, "output", abs(effect.direct_effect)))

        return CircuitResult(
            method="PathPatching",
            data={
                'effects': [
                    {
                        'layer': e.layer_name,
                        'component': e.component,
                        'direct_effect': e.direct_effect,
                        'total_effect': e.total_effect
                    }
                    for e in self.effects
                ]
            },
            metadata={
                'metric_name': self.metric_name,
                'n_layers': len(self.layer_names),
                'target_metric': self.target_metric,
                'baseline_metric': self.baseline_metric
            },
            metrics={
                'total_effect': sum(abs(e.direct_effect) for e in self.effects),
                'n_important_paths': len([e for e in self.effects if abs(e.direct_effect) > 0.01])
            },
            nodes=list(nodes),
            edges=edges,
            circuit_type='causal'
        )

    def visualize_causal_graph(
        self,
        use_bokeh: bool = True,
        threshold: float = 0.01,
        save_path: Optional[str] = None
    ) -> Any:
        """
        Visualize the causal graph.

        Args:
            use_bokeh: Use Bokeh for interactive visualization
            threshold: Minimum effect size to include
            save_path: Path to save visualization

        Returns:
            Bokeh figure or matplotlib figure
        """
        if use_bokeh and BOKEH_AVAILABLE:
            return self._visualize_bokeh(threshold, save_path)
        else:
            return self._visualize_matplotlib(threshold, save_path)

    def _visualize_bokeh(self, threshold: float, save_path: Optional[str]) -> Any:
        """Create interactive Bokeh visualization."""
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX required for graph visualization")
            return None

        # Build graph
        G = nx.DiGraph()

        # Add nodes and edges
        significant_effects = [e for e in self.effects if abs(e.direct_effect) > threshold]

        for effect in significant_effects:
            node_id = f"{effect.layer_name}\n{effect.component}"
            G.add_node(
                node_id,
                layer=effect.layer_name,
                component=effect.component,
                effect=effect.direct_effect,
                size=abs(effect.direct_effect) * 100
            )

        # Add edges based on layer ordering
        layer_order = {name: i for i, name in enumerate(self.layer_names)}
        nodes_by_layer = defaultdict(list)

        for effect in significant_effects:
            nodes_by_layer[effect.layer_name].append(effect)

        # Connect sequential layers
        for i in range(len(self.layer_names) - 1):
            curr_layer = self.layer_names[i]
            next_layer = self.layer_names[i + 1]

            for curr_effect in nodes_by_layer[curr_layer]:
                for next_effect in nodes_by_layer[next_layer]:
                    curr_node = f"{curr_effect.layer_name}\n{curr_effect.component}"
                    next_node = f"{next_effect.layer_name}\n{next_effect.component}"

                    # Edge weight based on propagated effect
                    weight = min(abs(curr_effect.direct_effect), abs(next_effect.total_effect))
                    if weight > threshold:
                        G.add_edge(curr_node, next_node, weight=weight)

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Create Bokeh plot
        plot = figure(
            title=f"Path Patching Causal Graph (threshold={threshold})",
            width=1000,
            height=800,
            x_range=Range1d(-1.2, 1.2),
            y_range=Range1d(-1.2, 1.2),
            tools="pan,wheel_zoom,reset,save"
        )

        # Prepare data
        node_indices = list(G.nodes())
        node_data = {
            'index': node_indices,
            'x': [pos[n][0] for n in node_indices],
            'y': [pos[n][1] for n in node_indices],
            'size': [G.nodes[n].get('size', 10) for n in node_indices],
            'effect': [G.nodes[n].get('effect', 0) for n in node_indices],
            'layer': [G.nodes[n].get('layer', '') for n in node_indices],
            'component': [G.nodes[n].get('component', '') for n in node_indices]
        }

        edge_start_x = []
        edge_start_y = []
        edge_end_x = []
        edge_end_y = []
        edge_weights = []

        for edge in G.edges(data=True):
            edge_start_x.append(pos[edge[0]][0])
            edge_start_y.append(pos[edge[0]][1])
            edge_end_x.append(pos[edge[1]][0])
            edge_end_y.append(pos[edge[1]][1])
            edge_weights.append(edge[2].get('weight', 0.1))

        # Plot edges
        edge_source = ColumnDataSource({
            'x0': edge_start_x,
            'y0': edge_start_y,
            'x1': edge_end_x,
            'y1': edge_end_y,
            'weight': edge_weights
        })

        # Color edges by weight
        edge_colors = ['#%02x%02x%02x' % (
            int(255 * (1 - w / max(edge_weights))),
            int(100 * w / max(edge_weights)),
            int(50 * w / max(edge_weights))
        ) for w in edge_weights]

        for i in range(len(edge_start_x)):
            plot.line(
                [edge_start_x[i], edge_end_x[i]],
                [edge_start_y[i], edge_end_y[i]],
                line_width=edge_weights[i] * 5,
                line_alpha=0.6,
                color=edge_colors[i]
            )

        # Plot nodes
        node_source = ColumnDataSource(node_data)

        # Color nodes by effect sign
        colors = ['red' if e < 0 else 'blue' for e in node_data['effect']]

        plot.circle(
            'x', 'y',
            size='size',
            source=node_source,
            fill_color=colors,
            fill_alpha=0.7,
            line_color='black',
            line_width=2
        )

        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Layer", "@layer"),
            ("Component", "@component"),
            ("Effect", "@effect{0.000}")
        ])
        plot.add_tools(hover)

        plot.axis.visible = False
        plot.grid.visible = False

        # Save if path provided
        if save_path:
            output_file(save_path)
            save(plot)

        return plot

    def _visualize_matplotlib(self, threshold: float, save_path: Optional[str]) -> Any:
        """Create matplotlib visualization."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            logger.warning("Matplotlib required for visualization")
            return None

        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX required for graph visualization")
            return None

        # Build graph
        G = nx.DiGraph()
        significant_effects = [e for e in self.effects if abs(e.direct_effect) > threshold]

        for effect in significant_effects:
            node_id = f"{effect.layer_name}_{effect.component}"
            G.add_node(
                node_id,
                effect=effect.direct_effect,
                layer=effect.layer_name
            )

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))

        # Node colors by effect
        node_colors = [
            'red' if G.nodes[n].get('effect', 0) < 0 else 'blue'
            for n in G.nodes()
        ]

        # Node sizes by effect magnitude
        node_sizes = [
            abs(G.nodes[n].get('effect', 0)) * 1000
            for n in G.nodes()
        ]

        # Draw
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.7,
            edgecolors='black',
            linewidths=2
        )

        nx.draw_networkx_labels(
            G, pos, ax=ax,
            font_size=8,
            font_weight='bold'
        )

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            width=2,
            alpha=0.5
        )

        # Legend
        red_patch = mpatches.Patch(color='red', label='Negative Effect')
        blue_patch = mpatches.Patch(color='blue', label='Positive Effect')
        ax.legend(handles=[red_patch, blue_patch], loc='upper right')

        ax.set_title(f"Path Patching Causal Graph (threshold={threshold})",
                    fontsize=16, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class PathPatcher:
    """
    Implements path patching for causal circuit discovery.

    Path patching identifies which activation paths are causally important
    by comparing clean and corrupted forward passes and systematically
    patching activations.

    Args:
        model: Neural network to analyze
        metric_fn: Function that takes model output and returns scalar metric
        layers_to_patch: List of layer names to patch (None = all)
        device: Torch device
        verbose: Enable verbose logging

    Example:
        >>> def metric_fn(output):
        ...     return output[:, target_token] - output[:, baseline_token]
        >>>
        >>> patcher = PathPatcher(model, metric_fn=metric_fn)
        >>> results = patcher.patch_all_paths(clean_input, corrupted_input)
    """

    def __init__(
        self,
        model: nn.Module,
        metric_fn: Callable[[torch.Tensor], float],
        layers_to_patch: Optional[List[str]] = None,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        self.model = model
        self.metric_fn = metric_fn
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose

        # Determine layers to patch
        if layers_to_patch is None:
            # Patch all activations
            self.layers_to_patch = [
                name for name, _ in model.named_modules()
                if isinstance(_, (nn.Linear, nn.Conv2d, nn.MultiheadAttention))
            ]
        else:
            self.layers_to_patch = layers_to_patch

        self._log(f"Initialized PathPatcher with {len(self.layers_to_patch)} layers")

    def _log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            logger.info(f"[PathPatcher] {message}")

    def patch_all_paths(
        self,
        clean_input: torch.Tensor,
        corrupted_input: torch.Tensor,
        components: List[str] = ["residual"],
        batch_size: Optional[int] = None
    ) -> PathPatchingResult:
        """
        Perform path patching across all layers.

        Args:
            clean_input: Target input (what we want to explain)
            corrupted_input: Baseline/counterfactual input
            components: Which components to patch ("residual", "attn", "mlp")
            batch_size: Batch size for processing (None = full batch)

        Returns:
            PathPatchingResult with all patch effects
        """
        self._log("Starting path patching analysis...")

        clean_input = clean_input.to(self.device)
        corrupted_input = corrupted_input.to(self.device)

        # Baseline: measure clean and corrupted outputs
        with torch.no_grad():
            clean_output = self.model(clean_input)
            corrupted_output = self.model(corrupted_input)

            target_metric = self.metric_fn(clean_output)
            baseline_metric = self.metric_fn(corrupted_output)

        self._log(f"Target metric: {target_metric:.4f}")
        self._log(f"Baseline metric: {baseline_metric:.4f}")
        self._log(f"Total effect to explain: {target_metric - baseline_metric:.4f}")

        # Collect clean activations
        clean_cache = self._cache_activations(clean_input)

        # Patch each component and measure effect
        effects = []

        iterator = tqdm(self.layers_to_patch, desc="Patching layers") if self.verbose else self.layers_to_patch

        for layer_name in iterator:
            for component in components:
                effect = self._patch_single_component(
                    layer_name,
                    component,
                    corrupted_input,
                    clean_cache,
                    baseline_metric
                )
                effects.append(effect)

        # Build result
        result = PathPatchingResult(
            effects=effects,
            clean_output=clean_output,
            corrupted_output=corrupted_output,
            target_metric=float(target_metric),
            baseline_metric=float(baseline_metric),
            layer_names=self.layers_to_patch,
            metric_name=getattr(self.metric_fn, '__name__', 'custom_metric')
        )

        self._log(f"Path patching complete. Found {len(effects)} patch effects.")

        return result

    def _cache_activations(self, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Cache all activations from a forward pass."""
        cache = {}

        def make_hook(name):
            def hook(module, input, output):
                cache[name] = output.detach().clone()
            return hook

        # Register hooks
        handles = []
        for name in self.layers_to_patch:
            module = dict(self.model.named_modules())[name]
            handles.append(module.register_forward_hook(make_hook(name)))

        # Forward pass
        with torch.no_grad():
            self.model(input_data)

        # Remove hooks
        for handle in handles:
            handle.remove()

        return cache

    def _patch_single_component(
        self,
        layer_name: str,
        component: str,
        corrupted_input: torch.Tensor,
        clean_cache: Dict[str, torch.Tensor],
        baseline_metric: float
    ) -> PatchEffect:
        """Patch a single component and measure effect."""

        patched_activation = None

        def patch_hook(module, input, output):
            """Hook that replaces output with clean activation."""
            nonlocal patched_activation
            if layer_name in clean_cache:
                patched_activation = clean_cache[layer_name].clone()
                return patched_activation
            return output

        # Register patch hook
        module = dict(self.model.named_modules())[layer_name]
        handle = module.register_forward_hook(patch_hook)

        # Forward pass with patch
        with torch.no_grad():
            patched_output = self.model(corrupted_input)
            patched_metric = self.metric_fn(patched_output)

        handle.remove()

        # Compute effect
        direct_effect = float(patched_metric - baseline_metric)

        # Compute norms
        activation_norm = 0.0
        patch_norm = 0.0

        if patched_activation is not None:
            activation_norm = float(patched_activation.norm())
            if layer_name in clean_cache:
                patch_norm = float(clean_cache[layer_name].norm())

        return PatchEffect(
            layer_name=layer_name,
            component=component,
            direct_effect=direct_effect,
            total_effect=direct_effect,  # Will be computed in post-processing
            activation_norm=activation_norm,
            patch_norm=patch_norm
        )


__all__ = [
    'PatchEffect',
    'PathPatchingResult',
    'PathPatcher',
]
