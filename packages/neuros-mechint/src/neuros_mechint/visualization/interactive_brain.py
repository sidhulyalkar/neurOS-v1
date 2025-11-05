"""
Interactive 3D Brain Visualizations

Real-time interactive visualizations for neural activity, connectivity,
and mechanistic interpretability results using Plotly.

Features:
- 3D brain region mapping with standard atlases
- Real-time activity animation
- Interactive connectivity graphs (force-directed, hierarchical)
- Cross-species comparison views
- Optogenetic stimulation visualization
- Criticality avalanche visualization
- Temporal dynamics animation

References:
- Brain parcellations: Destrieux, AAL, Glasser atlases
- Graph layouts: Fruchterman-Reingold, Kamada-Kawai
- Connectivity visualization: Hagmann et al. (2008)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import pdist, squareform


@dataclass
class BrainRegion:
    """3D brain region specification."""
    name: str
    center: np.ndarray  # (x, y, z) coordinates in mm
    size: float  # Radius for sphere representation
    hemisphere: str  # 'left', 'right', or 'both'
    color: Optional[str] = None
    activation: float = 0.0  # Current activation level


class BrainAtlas:
    """
    Standard brain atlases for visualization.

    Provides coordinates for common parcellations.
    """

    @staticmethod
    def get_visual_system() -> List[BrainRegion]:
        """Get visual system regions (simplified)."""
        return [
            BrainRegion("V1", np.array([-10, -90, 0]), 8.0, "left"),
            BrainRegion("V1", np.array([10, -90, 0]), 8.0, "right"),
            BrainRegion("V2", np.array([-15, -80, 5]), 6.0, "left"),
            BrainRegion("V2", np.array([15, -80, 5]), 6.0, "right"),
            BrainRegion("V4", np.array([-25, -70, -10]), 5.0, "left"),
            BrainRegion("V4", np.array([25, -70, -10]), 5.0, "right"),
            BrainRegion("IT", np.array([-40, -50, -15]), 7.0, "left"),
            BrainRegion("IT", np.array([40, -50, -15]), 7.0, "right"),
            BrainRegion("MT", np.array([-45, -65, 0]), 5.0, "left"),
            BrainRegion("MT", np.array([45, -65, 0]), 5.0, "right"),
        ]

    @staticmethod
    def get_default_mode_network() -> List[BrainRegion]:
        """Get default mode network regions."""
        return [
            BrainRegion("mPFC", np.array([0, 50, 10]), 10.0, "both"),
            BrainRegion("PCC", np.array([0, -50, 30]), 10.0, "both"),
            BrainRegion("IPL_L", np.array([-50, -60, 40]), 8.0, "left"),
            BrainRegion("IPL_R", np.array([50, -60, 40]), 8.0, "right"),
            BrainRegion("MTL_L", np.array([-25, -10, -20]), 7.0, "left"),
            BrainRegion("MTL_R", np.array([25, -10, -20]), 7.0, "right"),
        ]


class Interactive3DBrain:
    """
    Interactive 3D brain visualization.

    Supports multiple visualization modes:
    - Region activation
    - Connectivity
    - Time series animation
    - Cross-species comparison
    """

    def __init__(
        self,
        regions: Optional[List[BrainRegion]] = None,
        background_color: str = 'white'
    ):
        self.regions = regions or BrainAtlas.get_visual_system()
        self.background_color = background_color

    def visualize_activation(
        self,
        activations: np.ndarray,
        colorscale: str = 'Viridis',
        title: str = "Brain Region Activation"
    ) -> go.Figure:
        """
        Visualize activation levels across regions.

        Args:
            activations: Activation values (n_regions,)
            colorscale: Plotly colorscale name
            title: Plot title

        Returns:
            Plotly figure
        """
        # Extract coordinates
        coords = np.array([r.center for r in self.regions])
        sizes = np.array([r.size for r in self.regions])
        names = [r.name for r in self.regions]

        # Normalize activations
        act_norm = (activations - activations.min()) / (activations.max() - activations.min() + 1e-10)

        # Create 3D scatter
        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(
                size=sizes * 2,
                color=act_norm,
                colorscale=colorscale,
                colorbar=dict(title="Activation"),
                showscale=True,
                line=dict(width=0.5, color='black')
            ),
            text=names,
            hovertemplate='<b>%{text}</b><br>Activation: %{marker.color:.3f}<extra></extra>'
        )])

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                bgcolor=self.background_color
            ),
            width=900,
            height=700
        )

        return fig

    def visualize_connectivity(
        self,
        connectivity_matrix: np.ndarray,
        threshold: float = 0.1,
        edge_colorscale: str = 'Blues',
        title: str = "Brain Connectivity"
    ) -> go.Figure:
        """
        Visualize connectivity between regions.

        Args:
            connectivity_matrix: Connection strengths (n_regions, n_regions)
            threshold: Minimum connection strength to display
            edge_colorscale: Color scale for edges
            title: Plot title

        Returns:
            Plotly figure with nodes and edges
        """
        coords = np.array([r.center for r in self.regions])
        n_regions = len(self.regions)

        # Filter connections by threshold
        connections = []
        edge_strengths = []

        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                strength = connectivity_matrix[i, j]
                if abs(strength) > threshold:
                    connections.append((i, j))
                    edge_strengths.append(strength)

        # Create edges
        edge_traces = []
        for (i, j), strength in zip(connections, edge_strengths):
            x_line = [coords[i, 0], coords[j, 0], None]
            y_line = [coords[i, 1], coords[j, 1], None]
            z_line = [coords[i, 2], coords[j, 2], None]

            edge_traces.append(go.Scatter3d(
                x=x_line, y=y_line, z=z_line,
                mode='lines',
                line=dict(
                    color=f'rgba(100, 100, 200, {abs(strength)})',
                    width=2 * abs(strength)
                ),
                hoverinfo='skip',
                showlegend=False
            ))

        # Create nodes
        node_trace = go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers+text',
            marker=dict(
                size=8,
                color='red',
                line=dict(width=1, color='black')
            ),
            text=[r.name for r in self.regions],
            textposition='top center',
            hoverinfo='text'
        )

        # Combine
        fig = go.Figure(data=edge_traces + [node_trace])

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                bgcolor=self.background_color
            ),
            width=900,
            height=700,
            showlegend=False
        )

        return fig

    def animate_temporal_activity(
        self,
        activity_timeseries: np.ndarray,
        fps: int = 10,
        colorscale: str = 'Hot',
        title: str = "Temporal Brain Activity"
    ) -> go.Figure:
        """
        Create animation of brain activity over time.

        Args:
            activity_timeseries: Activity (time, n_regions)
            fps: Frames per second
            colorscale: Color scale
            title: Plot title

        Returns:
            Animated Plotly figure
        """
        coords = np.array([r.center for r in self.regions])
        sizes = np.array([r.size for r in self.regions])
        names = [r.name for r in self.regions]

        n_time, n_regions = activity_timeseries.shape

        # Normalize across all time
        global_min = activity_timeseries.min()
        global_max = activity_timeseries.max()

        # Create frames
        frames = []
        for t in range(n_time):
            act_norm = (activity_timeseries[t] - global_min) / (global_max - global_min + 1e-10)

            frame = go.Frame(
                data=[go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=sizes * 2,
                        color=act_norm,
                        colorscale=colorscale,
                        cmin=0,
                        cmax=1,
                        colorbar=dict(title="Activation"),
                        showscale=True
                    ),
                    text=names,
                    hovertemplate='<b>%{text}</b><br>Activation: %{marker.color:.3f}<extra></extra>'
                )],
                name=str(t)
            )
            frames.append(frame)

        # Initial frame
        fig = go.Figure(data=frames[0].data, frames=frames)

        # Add play/pause buttons
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                bgcolor=self.background_color
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 1000 // fps, 'redraw': True},
                                       'fromcurrent': True}]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                         'mode': 'immediate'}]
                    }
                ]
            }],
            width=900,
            height=700
        )

        return fig


class ForceDirectedGraph:
    """
    Force-directed graph layout for connectivity visualization.

    Uses Fruchterman-Reingold algorithm for aesthetically pleasing layouts.
    """

    def __init__(self, iterations: int = 50, k: Optional[float] = None):
        self.iterations = iterations
        self.k = k  # Optimal distance between nodes

    def layout(
        self,
        connectivity: np.ndarray,
        initial_positions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute force-directed layout.

        Args:
            connectivity: Adjacency matrix (n_nodes, n_nodes)
            initial_positions: Starting positions (n_nodes, 2) or (n_nodes, 3)

        Returns:
            Final positions (n_nodes, 2) or (n_nodes, 3)
        """
        n_nodes = connectivity.shape[0]

        # Initialize positions
        if initial_positions is None:
            dim = 2
            positions = np.random.rand(n_nodes, dim) * 10
        else:
            positions = initial_positions.copy()
            dim = positions.shape[1]

        # Optimal distance
        if self.k is None:
            area = 100.0
            self.k = np.sqrt(area / n_nodes)

        # Temperature schedule
        temperature = 10.0
        cooling_rate = temperature / self.iterations

        for iteration in range(self.iterations):
            # Compute forces
            forces = np.zeros_like(positions)

            # Repulsive forces between all pairs
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        delta = positions[i] - positions[j]
                        distance = np.linalg.norm(delta) + 1e-6

                        # Repulsion
                        force_mag = self.k ** 2 / distance
                        forces[i] += (delta / distance) * force_mag

            # Attractive forces for connected nodes
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if connectivity[i, j] > 0:
                        delta = positions[i] - positions[j]
                        distance = np.linalg.norm(delta) + 1e-6

                        # Attraction proportional to connection strength
                        force_mag = distance ** 2 / self.k * connectivity[i, j]
                        force = (delta / distance) * force_mag

                        forces[i] -= force
                        forces[j] += force

            # Update positions with temperature
            displacement = forces * temperature
            displacement_mag = np.linalg.norm(displacement, axis=1, keepdims=True)
            displacement = displacement / (displacement_mag + 1e-6) * np.minimum(displacement_mag, temperature)

            positions += displacement

            # Cool down
            temperature -= cooling_rate

        return positions


class CriticalityVisualizer:
    """
    Visualize criticality and neuronal avalanches.
    """

    @staticmethod
    def plot_avalanche_raster(
        spike_trains: np.ndarray,
        avalanches: List[Dict],
        title: str = "Neuronal Avalanches"
    ) -> go.Figure:
        """
        Create raster plot with avalanches highlighted.

        Args:
            spike_trains: Binary spike trains (time, neurons)
            avalanches: List of avalanche dictionaries
            title: Plot title

        Returns:
            Plotly figure
        """
        n_time, n_neurons = spike_trains.shape

        # Find spike times and neuron indices
        spike_times, spike_neurons = np.where(spike_trains > 0)

        # Create raster plot
        fig = go.Figure()

        # Add spikes
        fig.add_trace(go.Scatter(
            x=spike_times,
            y=spike_neurons,
            mode='markers',
            marker=dict(size=2, color='black'),
            name='Spikes'
        ))

        # Highlight avalanches
        for i, av in enumerate(avalanches[:10]):  # Show first 10
            times = av['times']
            if len(times) > 0:
                fig.add_vrect(
                    x0=times[0],
                    x1=times[-1],
                    fillcolor='red',
                    opacity=0.2,
                    line_width=0,
                    annotation_text=f"Avalanche {i+1}",
                    annotation_position="top left"
                )

        fig.update_layout(
            title=title,
            xaxis_title="Time (ms)",
            yaxis_title="Neuron ID",
            width=1000,
            height=500,
            showlegend=True
        )

        return fig

    @staticmethod
    def plot_power_law_distribution(
        sizes: np.ndarray,
        exponent: float,
        title: str = "Avalanche Size Distribution"
    ) -> go.Figure:
        """
        Plot avalanche size distribution with power law fit.

        Args:
            sizes: Avalanche sizes
            exponent: Fitted power law exponent
            title: Plot title

        Returns:
            Plotly figure with log-log plot
        """
        # Histogram
        bins = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 20)
        hist, bin_edges = np.histogram(sizes, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Remove zeros
        mask = hist > 0
        bin_centers = bin_centers[mask]
        hist = hist[mask]

        # Power law fit
        x_fit = np.logspace(np.log10(sizes.min()), np.log10(sizes.max()), 100)
        y_fit = x_fit ** (-exponent)
        y_fit = y_fit / y_fit[0] * hist[0]  # Normalize

        fig = go.Figure()

        # Data
        fig.add_trace(go.Scatter(
            x=bin_centers,
            y=hist,
            mode='markers',
            marker=dict(size=8, color='blue'),
            name='Data'
        ))

        # Fit
        fig.add_trace(go.Scatter(
            x=x_fit,
            y=y_fit,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name=f'Power law (α={exponent:.2f})'
        ))

        fig.update_xaxes(type="log", title="Avalanche Size")
        fig.update_yaxes(type="log", title="Count")
        fig.update_layout(
            title=title,
            width=700,
            height=500
        )

        return fig
