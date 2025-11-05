"""
Advanced Visualization Tools for Mechanistic Interpretability

Specialized visualizers for:
- Multifractal spectrum analysis
- Cross-species neural comparison
- Intervention effects (optogenetics, pharmacology, stimulation)
- Temporal dynamics and phase space
- Energy landscapes and information flow

References:
- Multifractal: Kantelhardt et al. (2002) Physica A
- Cross-species: Kriegeskorte et al. (2008) Neuron
- Phase space: Strogatz (1994) Nonlinear Dynamics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import linregress
from scipy.interpolate import griddata


class MultifractalVisualizer:
    """
    Visualize multifractal analysis results.

    Creates comprehensive plots for singularity spectra, scaling exponents,
    and multifractal detrended fluctuation analysis.
    """

    def __init__(self, colorscale: str = 'Viridis'):
        """
        Initialize visualizer.

        Args:
            colorscale: Plotly colorscale name
        """
        self.colorscale = colorscale

    def plot_singularity_spectrum(
        self,
        alpha: np.ndarray,
        f_alpha: np.ndarray,
        title: str = "Singularity Spectrum f(α)"
    ) -> go.Figure:
        """
        Plot singularity spectrum from multifractal analysis.

        The spectrum f(α) vs α characterizes the multifractal nature:
        - Monofractal: Single peak
        - Multifractal: Broad parabolic shape

        Args:
            alpha: Hölder exponents
            f_alpha: Fractal dimensions
            title: Plot title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Main spectrum curve
        fig.add_trace(go.Scatter(
            x=alpha,
            y=f_alpha,
            mode='lines+markers',
            name='f(α)',
            line=dict(width=3, color='blue'),
            marker=dict(size=8)
        ))

        # Find peak
        peak_idx = np.argmax(f_alpha)
        alpha_0 = alpha[peak_idx]
        f_alpha_0 = f_alpha[peak_idx]

        # Mark peak
        fig.add_trace(go.Scatter(
            x=[alpha_0],
            y=[f_alpha_0],
            mode='markers+text',
            name=f'Peak (α₀={alpha_0:.3f})',
            marker=dict(size=15, color='red', symbol='star'),
            text=[f'α₀={alpha_0:.3f}'],
            textposition='top center'
        ))

        # Calculate width (indicator of multifractality strength)
        width = alpha.max() - alpha.min()

        fig.update_layout(
            title=f"{title}<br><sub>Width Δα = {width:.3f} (larger = more multifractal)</sub>",
            xaxis_title='Hölder Exponent α',
            yaxis_title='Fractal Dimension f(α)',
            showlegend=True,
            width=800,
            height=600,
            template='plotly_white'
        )

        return fig

    def plot_scaling_exponents(
        self,
        q_values: np.ndarray,
        tau_q: np.ndarray,
        h_q: Optional[np.ndarray] = None,
        title: str = "Multifractal Scaling Exponents"
    ) -> go.Figure:
        """
        Plot scaling exponents τ(q) and generalized Hurst exponent h(q).

        Args:
            q_values: Moment orders
            tau_q: Scaling exponents τ(q)
            h_q: Generalized Hurst exponents h(q) (optional)
            title: Plot title

        Returns:
            Plotly figure with subplots
        """
        if h_q is not None:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Scaling Exponent τ(q)', 'Generalized Hurst h(q)')
            )

            # τ(q) plot
            fig.add_trace(
                go.Scatter(
                    x=q_values, y=tau_q,
                    mode='lines+markers',
                    name='τ(q)',
                    line=dict(width=2, color='blue'),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )

            # h(q) plot
            fig.add_trace(
                go.Scatter(
                    x=q_values, y=h_q,
                    mode='lines+markers',
                    name='h(q)',
                    line=dict(width=2, color='green'),
                    marker=dict(size=6)
                ),
                row=1, col=2
            )

            # Add h=0.5 reference line (random walk)
            fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                         annotation_text="H=0.5 (random)", row=1, col=2)

            fig.update_xaxes(title_text="Moment q", row=1, col=1)
            fig.update_xaxes(title_text="Moment q", row=1, col=2)
            fig.update_yaxes(title_text="τ(q)", row=1, col=1)
            fig.update_yaxes(title_text="h(q)", row=1, col=2)

        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=q_values, y=tau_q,
                mode='lines+markers',
                name='τ(q)',
                line=dict(width=3, color='blue'),
                marker=dict(size=8)
            ))
            fig.update_xaxes(title_text="Moment q")
            fig.update_yaxes(title_text="τ(q)")

        fig.update_layout(
            title=title,
            width=1000 if h_q is not None else 800,
            height=500,
            showlegend=True,
            template='plotly_white'
        )

        return fig

    def plot_mfdfa_results(
        self,
        scales: np.ndarray,
        fluctuations: np.ndarray,
        q_values: np.ndarray,
        title: str = "MF-DFA Fluctuation Functions"
    ) -> go.Figure:
        """
        Plot MF-DFA fluctuation functions F_q(s) vs scale s.

        Args:
            scales: Scale values
            fluctuations: Fluctuation functions (q_values, scales)
            q_values: Moment orders
            title: Plot title

        Returns:
            Plotly log-log figure
        """
        fig = go.Figure()

        # Plot each q-order
        for i, q in enumerate(q_values):
            # Fit power law for this q
            log_s = np.log10(scales)
            log_F = np.log10(fluctuations[i])

            # Remove inf/nan
            valid = np.isfinite(log_F)
            if valid.sum() > 2:
                slope, intercept, r_value, _, _ = linregress(log_s[valid], log_F[valid])

                fig.add_trace(go.Scatter(
                    x=scales,
                    y=fluctuations[i],
                    mode='markers',
                    name=f'q={q:.1f} (h={slope:.3f})',
                    marker=dict(size=6),
                    showlegend=True
                ))

                # Add fit line
                fit_line = 10**(intercept + slope * log_s)
                fig.add_trace(go.Scatter(
                    x=scales,
                    y=fit_line,
                    mode='lines',
                    line=dict(dash='dash'),
                    showlegend=False,
                    hovertemplate=f'q={q:.1f}<br>h={slope:.3f}<br>R²={r_value**2:.3f}'
                ))

        fig.update_xaxes(type="log", title_text="Scale s")
        fig.update_yaxes(type="log", title_text="Fluctuation F_q(s)")

        fig.update_layout(
            title=title,
            width=900,
            height=600,
            template='plotly_white'
        )

        return fig


class CrossSpeciesVisualizer:
    """
    Visualize cross-species neural comparison results.

    Includes Procrustes alignment, RSA matrices, phylogenetic trees,
    and evolutionary trend analysis.
    """

    def __init__(self):
        """Initialize cross-species visualizer."""
        pass

    def plot_procrustes_alignment(
        self,
        source: np.ndarray,
        target: np.ndarray,
        aligned: np.ndarray,
        species_names: Tuple[str, str] = ("Species A", "Species B"),
        title: str = "Procrustes Alignment"
    ) -> go.Figure:
        """
        Visualize Procrustes alignment of neural representations.

        Args:
            source: Source neural space (n_samples, n_dims)
            target: Target neural space (n_samples, n_dims)
            aligned: Aligned source space (n_samples, n_dims)
            species_names: (source_name, target_name)
            title: Plot title

        Returns:
            3D scatter plot figure
        """
        # Use PCA if dimensionality > 3
        if source.shape[1] > 3:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            source_3d = pca.fit_transform(source)
            target_3d = pca.transform(target)
            aligned_3d = pca.transform(aligned)
        else:
            source_3d = source[:, :3]
            target_3d = target[:, :3]
            aligned_3d = aligned[:, :3]

        fig = go.Figure()

        # Original source (before alignment)
        fig.add_trace(go.Scatter3d(
            x=source_3d[:, 0], y=source_3d[:, 1], z=source_3d[:, 2],
            mode='markers',
            name=f'{species_names[0]} (original)',
            marker=dict(size=6, color='blue', opacity=0.3)
        ))

        # Target
        fig.add_trace(go.Scatter3d(
            x=target_3d[:, 0], y=target_3d[:, 1], z=target_3d[:, 2],
            mode='markers',
            name=species_names[1],
            marker=dict(size=6, color='red', opacity=0.6)
        ))

        # Aligned source
        fig.add_trace(go.Scatter3d(
            x=aligned_3d[:, 0], y=aligned_3d[:, 1], z=aligned_3d[:, 2],
            mode='markers',
            name=f'{species_names[0]} (aligned)',
            marker=dict(size=6, color='green', opacity=0.8, symbol='diamond')
        ))

        # Draw alignment arrows for subset of points
        n_arrows = min(20, len(source_3d))
        indices = np.linspace(0, len(source_3d)-1, n_arrows, dtype=int)

        for idx in indices:
            fig.add_trace(go.Scatter3d(
                x=[source_3d[idx, 0], aligned_3d[idx, 0]],
                y=[source_3d[idx, 1], aligned_3d[idx, 1]],
                z=[source_3d[idx, 2], aligned_3d[idx, 2]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3'
            ),
            width=900,
            height=700,
            showlegend=True
        )

        return fig

    def plot_rsa_matrix(
        self,
        rdm: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Representational Dissimilarity Matrix"
    ) -> go.Figure:
        """
        Plot representational dissimilarity matrix (RDM).

        Args:
            rdm: Dissimilarity matrix (n_stimuli, n_stimuli)
            labels: Stimulus labels
            title: Plot title

        Returns:
            Heatmap figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=rdm,
            x=labels,
            y=labels,
            colorscale='RdBu_r',
            zmid=rdm.mean(),
            text=np.round(rdm, 3),
            texttemplate='%{text}',
            textfont=dict(size=8),
            hovertemplate='Stimulus %{y} vs %{x}<br>Dissimilarity: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Stimuli',
            yaxis_title='Stimuli',
            width=800,
            height=800,
            xaxis=dict(tickangle=-45)
        )

        return fig

    def plot_phylogenetic_comparison(
        self,
        species_names: List[str],
        similarity_matrix: np.ndarray,
        phylo_distances: Optional[np.ndarray] = None,
        title: str = "Cross-Species Neural Similarity"
    ) -> go.Figure:
        """
        Plot cross-species similarity with phylogenetic context.

        Args:
            species_names: List of species names
            similarity_matrix: Neural similarity matrix (n_species, n_species)
            phylo_distances: Phylogenetic distances (optional)
            title: Plot title

        Returns:
            Combined heatmap and scatter plot
        """
        if phylo_distances is not None:
            # Create subplots: heatmap + scatter
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Neural Similarity', 'Similarity vs Phylogenetic Distance'),
                specs=[[{'type': 'heatmap'}, {'type': 'scatter'}]]
            )

            # Heatmap
            fig.add_trace(
                go.Heatmap(
                    z=similarity_matrix,
                    x=species_names,
                    y=species_names,
                    colorscale='Viridis',
                    showscale=True
                ),
                row=1, col=1
            )

            # Extract upper triangle for scatter
            triu_indices = np.triu_indices_from(similarity_matrix, k=1)
            neural_sim = similarity_matrix[triu_indices]
            phylo_dist = phylo_distances[triu_indices]

            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=phylo_dist,
                    y=neural_sim,
                    mode='markers',
                    marker=dict(size=10, color=neural_sim, colorscale='Viridis'),
                    text=[f'{species_names[i]}-{species_names[j]}'
                          for i, j in zip(*triu_indices)],
                    hovertemplate='%{text}<br>Phylo dist: %{x:.3f}<br>Neural sim: %{y:.3f}<extra></extra>'
                ),
                row=1, col=2
            )

            # Add correlation line
            if len(phylo_dist) > 2:
                slope, intercept, r_value, _, _ = linregress(phylo_dist, neural_sim)
                x_line = np.array([phylo_dist.min(), phylo_dist.max()])
                y_line = slope * x_line + intercept

                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode='lines',
                        line=dict(dash='dash', color='red'),
                        name=f'R²={r_value**2:.3f}',
                        showlegend=True
                    ),
                    row=1, col=2
                )

            fig.update_xaxes(title_text="Phylogenetic Distance", row=1, col=2)
            fig.update_yaxes(title_text="Neural Similarity", row=1, col=2)

            width = 1400
        else:
            # Just heatmap
            fig = go.Figure(data=go.Heatmap(
                z=similarity_matrix,
                x=species_names,
                y=species_names,
                colorscale='Viridis'
            ))
            width = 700

        fig.update_layout(
            title=title,
            width=width,
            height=600
        )

        return fig


class InterventionVisualizer:
    """
    Visualize intervention effects (optogenetics, pharmacology, stimulation).

    Shows temporal dynamics, spatial patterns, and dose-response curves.
    """

    def __init__(self):
        """Initialize intervention visualizer."""
        pass

    def plot_optogenetic_response(
        self,
        time: np.ndarray,
        light_intensity: np.ndarray,
        neural_response: np.ndarray,
        opsin_name: str = "ChR2",
        wavelength: float = 470.0,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Plot optogenetic stimulation and neural response.

        Args:
            time: Time vector (ms)
            light_intensity: Light intensity (mW/mm²)
            neural_response: Neural activity (e.g., membrane potential, spike rate)
            opsin_name: Name of opsin
            wavelength: Light wavelength (nm)
            title: Plot title

        Returns:
            Figure with dual y-axes
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Light intensity
        fig.add_trace(
            go.Scatter(
                x=time, y=light_intensity,
                mode='lines',
                name=f'Light ({wavelength:.0f} nm)',
                line=dict(color='blue', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 0, 255, 0.2)'
            ),
            secondary_y=False
        )

        # Neural response
        fig.add_trace(
            go.Scatter(
                x=time, y=neural_response,
                mode='lines',
                name='Neural Response',
                line=dict(color='red', width=2)
            ),
            secondary_y=True
        )

        # Set axis labels
        fig.update_xaxes(title_text="Time (ms)")
        fig.update_yaxes(title_text="Light Intensity (mW/mm²)", secondary_y=False)
        fig.update_yaxes(title_text="Neural Response", secondary_y=True)

        if title is None:
            title = f"{opsin_name} Optogenetic Response ({wavelength:.0f} nm)"

        fig.update_layout(
            title=title,
            width=900,
            height=500,
            hovermode='x unified'
        )

        return fig

    def plot_dose_response_curve(
        self,
        doses: np.ndarray,
        responses: np.ndarray,
        drug_name: str,
        fit_params: Optional[Dict[str, float]] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """
        Plot pharmacological dose-response curve.

        Args:
            doses: Drug concentrations (μM)
            responses: Normalized responses (0-1)
            drug_name: Drug name
            fit_params: Hill equation parameters (EC50, hill_coefficient, Emax)
            title: Plot title

        Returns:
            Semi-log dose-response figure
        """
        fig = go.Figure()

        # Data points
        fig.add_trace(go.Scatter(
            x=doses,
            y=responses,
            mode='markers',
            name='Data',
            marker=dict(size=10, color='blue')
        ))

        # Fitted curve if parameters provided
        if fit_params:
            dose_range = np.logspace(np.log10(doses.min()), np.log10(doses.max()), 100)

            # Hill equation: E = Emax * [D]^n / (EC50^n + [D]^n)
            EC50 = fit_params.get('EC50', np.median(doses))
            n = fit_params.get('hill_coefficient', 1.0)
            Emax = fit_params.get('Emax', 1.0)

            fitted = Emax * (dose_range**n) / (EC50**n + dose_range**n)

            fig.add_trace(go.Scatter(
                x=dose_range,
                y=fitted,
                mode='lines',
                name=f'Hill Fit (EC50={EC50:.2f} μM, n={n:.2f})',
                line=dict(color='red', width=2, dash='dash')
            ))

            # Mark EC50
            EC50_response = Emax * 0.5
            fig.add_trace(go.Scatter(
                x=[EC50],
                y=[EC50_response],
                mode='markers+text',
                name='EC50',
                marker=dict(size=15, color='red', symbol='star'),
                text=[f'EC50={EC50:.2f}'],
                textposition='top center'
            ))

        fig.update_xaxes(type="log", title_text="Dose (μM)")
        fig.update_yaxes(title_text="Normalized Response")

        if title is None:
            title = f"{drug_name} Dose-Response Curve"

        fig.update_layout(
            title=title,
            width=800,
            height=600,
            showlegend=True,
            template='plotly_white'
        )

        return fig

    def plot_stimulation_field(
        self,
        x: np.ndarray,
        y: np.ndarray,
        field_strength: np.ndarray,
        electrode_pos: Optional[Tuple[float, float]] = None,
        title: str = "Stimulation Field Strength"
    ) -> go.Figure:
        """
        Plot spatial distribution of stimulation field (e.g., TMS, DBS, tDCS).

        Args:
            x: X coordinates (mm)
            y: Y coordinates (mm)
            field_strength: Electric field strength (V/m)
            electrode_pos: (x, y) position of electrode/coil
            title: Plot title

        Returns:
            2D heatmap figure
        """
        fig = go.Figure(data=go.Heatmap(
            x=x,
            y=y,
            z=field_strength,
            colorscale='Hot',
            colorbar=dict(title='Field (V/m)')
        ))

        # Mark electrode position
        if electrode_pos is not None:
            fig.add_trace(go.Scatter(
                x=[electrode_pos[0]],
                y=[electrode_pos[1]],
                mode='markers+text',
                marker=dict(size=20, color='blue', symbol='x', line=dict(width=3, color='white')),
                text=['Electrode'],
                textposition='top center',
                name='Electrode',
                showlegend=True
            ))

        fig.update_layout(
            title=title,
            xaxis_title='X Position (mm)',
            yaxis_title='Y Position (mm)',
            width=700,
            height=700,
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )

        return fig

    def plot_multi_intervention_comparison(
        self,
        time: np.ndarray,
        baseline: np.ndarray,
        interventions: Dict[str, np.ndarray],
        title: str = "Intervention Comparison"
    ) -> go.Figure:
        """
        Compare multiple intervention effects over time.

        Args:
            time: Time vector
            baseline: Baseline activity
            interventions: Dict of {intervention_name: activity_trace}
            title: Plot title

        Returns:
            Multi-trace figure
        """
        fig = go.Figure()

        # Baseline
        fig.add_trace(go.Scatter(
            x=time,
            y=baseline,
            mode='lines',
            name='Baseline',
            line=dict(color='gray', width=2, dash='dash')
        ))

        # Interventions
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        for i, (name, activity) in enumerate(interventions.items()):
            fig.add_trace(go.Scatter(
                x=time,
                y=activity,
                mode='lines',
                name=name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Time (ms)',
            yaxis_title='Neural Activity',
            width=900,
            height=600,
            hovermode='x unified',
            showlegend=True
        )

        return fig


class TemporalDynamicsVisualizer:
    """
    Visualize temporal dynamics and phase space trajectories.
    """

    def __init__(self):
        """Initialize temporal dynamics visualizer."""
        pass

    def plot_phase_space(
        self,
        trajectories: np.ndarray,
        fixed_points: Optional[List[np.ndarray]] = None,
        labels: Optional[List[str]] = None,
        title: str = "Neural Phase Space"
    ) -> go.Figure:
        """
        Plot 3D phase space trajectories with fixed points.

        Args:
            trajectories: Trajectories array (n_trajectories, n_timepoints, n_dims)
            fixed_points: List of fixed point coordinates
            labels: Trajectory labels
            title: Plot title

        Returns:
            3D trajectory figure
        """
        # Use first 3 dimensions
        if trajectories.shape[2] > 3:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            traj_flat = trajectories.reshape(-1, trajectories.shape[2])
            traj_3d = pca.fit_transform(traj_flat)
            trajectories = traj_3d.reshape(trajectories.shape[0], trajectories.shape[1], 3)

        fig = go.Figure()

        # Plot trajectories
        n_traj = trajectories.shape[0]
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

        for i in range(n_traj):
            name = labels[i] if labels else f'Trajectory {i+1}'
            fig.add_trace(go.Scatter3d(
                x=trajectories[i, :, 0],
                y=trajectories[i, :, 1],
                z=trajectories[i, :, 2],
                mode='lines',
                name=name,
                line=dict(color=colors[i % len(colors)], width=3),
                opacity=0.7
            ))

            # Mark start point
            fig.add_trace(go.Scatter3d(
                x=[trajectories[i, 0, 0]],
                y=[trajectories[i, 0, 1]],
                z=[trajectories[i, 0, 2]],
                mode='markers',
                marker=dict(size=8, color=colors[i % len(colors)], symbol='circle'),
                showlegend=False,
                hovertext='Start'
            ))

        # Plot fixed points
        if fixed_points:
            fp_array = np.array(fixed_points)
            if fp_array.shape[1] > 3:
                fp_array = fp_array[:, :3]

            fig.add_trace(go.Scatter3d(
                x=fp_array[:, 0],
                y=fp_array[:, 1],
                z=fp_array[:, 2],
                mode='markers',
                name='Fixed Points',
                marker=dict(size=10, color='black', symbol='diamond')
            ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3'
            ),
            width=900,
            height=700,
            showlegend=True
        )

        return fig

    def plot_temporal_heatmap(
        self,
        activity: np.ndarray,
        time: Optional[np.ndarray] = None,
        neuron_labels: Optional[List[str]] = None,
        title: str = "Temporal Activity Pattern"
    ) -> go.Figure:
        """
        Plot temporal activity as heatmap (neurons x time).

        Args:
            activity: Activity matrix (n_neurons, n_timepoints)
            time: Time vector (optional)
            neuron_labels: Neuron labels (optional)
            title: Plot title

        Returns:
            Heatmap figure
        """
        if time is None:
            time = np.arange(activity.shape[1])

        if neuron_labels is None:
            neuron_labels = [f'N{i}' for i in range(activity.shape[0])]

        fig = go.Figure(data=go.Heatmap(
            z=activity,
            x=time,
            y=neuron_labels,
            colorscale='RdBu_r',
            zmid=0,
            colorbar=dict(title='Activity')
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Neurons',
            width=1000,
            height=600
        )

        return fig


__all__ = [
    'MultifractalVisualizer',
    'CrossSpeciesVisualizer',
    'InterventionVisualizer',
    'TemporalDynamicsVisualizer',
]
