"""
SAE Visualization Suite for NeuroFMX

Comprehensive visualization tools for understanding learned SAE features:
- Feature activation heatmaps over time
- Top-activating samples per feature
- Feature co-occurrence matrices
- Modality-specific feature maps
- Feature importance rankings
- Interactive dashboards

Supports both matplotlib and plotly backends.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    # Avoid importing `output_file` which mutates the active Bokeh document and can
    # conflict with `output_notebook()`/`show()`; we'll use `save(..., filename=...)`
    # to write HTML without switching the active document.
    from bokeh.plotting import figure, save
    from bokeh.layouts import column, row
    from bokeh.io import export_png
    BOKEH_AVAILABLE = True
except Exception:
    BOKEH_AVAILABLE = False

from neuros_mechint.sparse_autoencoder import SparseAutoencoder


class SAEVisualizer:
    """
    Visualizer for Sparse Autoencoder features.

    Provides comprehensive visualization and analysis of learned features.
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        layer_name: str = "layer",
        output_dir: Optional[str] = None
    ):
        """
        Initialize visualizer.

        Args:
            sae: Trained sparse autoencoder
            layer_name: Name of layer (for labeling)
            output_dir: Directory to save visualizations
        """
        self.sae = sae
        self.layer_name = layer_name
        self.output_dir = Path(output_dir) if output_dir else Path("./sae_visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cache for feature statistics
        self.feature_stats = None

    def plot_feature_activations(
        self,
        activations: torch.Tensor,
        features: torch.Tensor,
        feature_ids: Optional[List[int]] = None,
        time_axis: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (12, 8),
        backend: str = 'matplotlib'
    ) -> Union[plt.Figure, go.Figure]:
        """
        Plot feature activations over time as heatmap.

        Args:
            activations: (n_samples, latent_dim) input activations
            features: (n_samples, dict_size) feature activations from SAE
            feature_ids: List of specific features to plot (None = top-k active)
            time_axis: Optional time axis for x-axis
            figsize: Figure size
            backend: 'matplotlib' or 'plotly'

        Returns:
            Figure object
        """
        if features.dim() == 3:
            # Has sequence dimension, take mean
            features = features.mean(dim=1)

        features_np = features.cpu().numpy()

        # Select features to plot
        if feature_ids is None:
            # Top 20 most active features
            mean_activations = features_np.mean(axis=0)
            feature_ids = np.argsort(mean_activations)[-20:][::-1]

        # Extract selected features
        selected_features = features_np[:, feature_ids]

        if backend == 'matplotlib':
            fig, ax = plt.subplots(figsize=figsize)

            im = ax.imshow(
                selected_features.T,
                aspect='auto',
                cmap='hot',
                interpolation='nearest'
            )

            ax.set_xlabel('Time / Sample Index')
            ax.set_ylabel('Feature ID')
            ax.set_yticks(range(len(feature_ids)))
            ax.set_yticklabels([f"F{fid}" for fid in feature_ids])
            ax.set_title(f'Feature Activations - {self.layer_name}')

            plt.colorbar(im, ax=ax, label='Activation')
            plt.tight_layout()

            # Save
            save_path = self.output_dir / f"{self.layer_name}_feature_activations.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

            return fig

        elif backend == 'plotly' and PLOTLY_AVAILABLE:
            fig = go.Figure(data=go.Heatmap(
                z=selected_features.T,
                x=time_axis if time_axis is not None else list(range(len(features_np))),
                y=[f"F{fid}" for fid in feature_ids],
                colorscale='Hot',
                colorbar=dict(title='Activation')
            ))

            fig.update_layout(
                title=f'Feature Activations - {self.layer_name}',
                xaxis_title='Time / Sample Index',
                yaxis_title='Feature ID',
                width=1000,
                height=600
            )

            # Save
            save_path = self.output_dir / f"{self.layer_name}_feature_activations.html"
            fig.write_html(str(save_path))
            print(f"Saved to {save_path}")

            return fig

        else:
            raise ValueError(f"Unknown backend: {backend}")

    def plot_top_activating_samples(
        self,
        feature_id: int,
        activations: torch.Tensor,
        features: torch.Tensor,
        original_data: Optional[Dict[str, torch.Tensor]] = None,
        top_k: int = 10,
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Visualize top-k samples that maximally activate a feature.

        Args:
            feature_id: ID of feature to analyze
            activations: (n_samples, latent_dim) input activations
            features: (n_samples, dict_size) feature activations
            original_data: Optional dict of original modality data for visualization
            top_k: Number of top samples to show
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Get feature activations
        feature_acts = features[:, feature_id].cpu().numpy()

        # Get top-k indices
        top_indices = np.argsort(feature_acts)[-top_k:][::-1]

        # Create figure
        n_cols = min(5, top_k)
        n_rows = int(np.ceil(top_k / n_cols))

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.3, wspace=0.3)

        for i, idx in enumerate(top_indices):
            ax = fig.add_subplot(gs[i])

            # Plot activation pattern
            if original_data is not None:
                # Try to plot original data
                for modality, data in original_data.items():
                    if data is not None and idx < len(data):
                        sample = data[idx].cpu().numpy()
                        if len(sample.shape) == 2:
                            # Time series
                            ax.plot(sample)
                        elif len(sample.shape) == 1:
                            ax.plot(sample)
                        break
            else:
                # Plot latent activation pattern
                act_pattern = activations[idx].cpu().numpy()
                ax.plot(act_pattern)

            ax.set_title(f'Sample {idx}\nAct: {feature_acts[idx]:.3f}', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f'Top-{top_k} Activating Samples for Feature {feature_id}',
                     fontsize=14, fontweight='bold')

        # Save
        save_path = self.output_dir / f"{self.layer_name}_feature_{feature_id}_top_samples.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

        return fig

    def plot_feature_cooccurrence(
        self,
        features: torch.Tensor,
        threshold: float = 0.1,
        top_k_features: int = 50,
        figsize: Tuple[int, int] = (12, 10),
        backend: str = 'matplotlib'
    ) -> Union[plt.Figure, go.Figure]:
        """
        Plot feature co-occurrence matrix.

        Shows which features tend to activate together.

        Args:
            features: (n_samples, dict_size) feature activations
            threshold: Activation threshold for binarization
            top_k_features: Number of top features to include
            figsize: Figure size
            backend: 'matplotlib' or 'plotly'

        Returns:
            Figure object
        """
        features_np = features.cpu().numpy()

        # Binarize features (active or not)
        binary_features = (features_np > threshold).astype(float)

        # Compute co-occurrence matrix
        cooccur = binary_features.T @ binary_features  # (dict_size, dict_size)

        # Normalize by number of samples
        cooccur = cooccur / len(features_np)

        # Select top-k most active features
        mean_activation = binary_features.mean(axis=0)
        top_indices = np.argsort(mean_activation)[-top_k_features:][::-1]

        cooccur_subset = cooccur[np.ix_(top_indices, top_indices)]

        if backend == 'matplotlib':
            fig, ax = plt.subplots(figsize=figsize)

            im = ax.imshow(cooccur_subset, cmap='viridis', aspect='auto')

            ax.set_xlabel('Feature ID')
            ax.set_ylabel('Feature ID')
            ax.set_title(f'Feature Co-occurrence Matrix - {self.layer_name}')

            # Add colorbar
            plt.colorbar(im, ax=ax, label='Co-occurrence Probability')

            plt.tight_layout()

            # Save
            save_path = self.output_dir / f"{self.layer_name}_cooccurrence.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")

            return fig

        elif backend == 'plotly' and PLOTLY_AVAILABLE:
            fig = go.Figure(data=go.Heatmap(
                z=cooccur_subset,
                x=[f"F{i}" for i in top_indices],
                y=[f"F{i}" for i in top_indices],
                colorscale='Viridis',
                colorbar=dict(title='Co-occurrence')
            ))

            fig.update_layout(
                title=f'Feature Co-occurrence Matrix - {self.layer_name}',
                xaxis_title='Feature ID',
                yaxis_title='Feature ID',
                width=900,
                height=900
            )

            # Save
            save_path = self.output_dir / f"{self.layer_name}_cooccurrence.html"
            fig.write_html(str(save_path))
            print(f"Saved to {save_path}")

            return fig

    def plot_modality_specific_features(
        self,
        features_by_modality: Dict[str, torch.Tensor],
        top_k: int = 20,
        figsize: Tuple[int, int] = (14, 8)
    ) -> plt.Figure:
        """
        Plot which features are most active for each modality.

        Args:
            features_by_modality: Dict mapping modality name to feature activations
            top_k: Number of top features per modality
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        modalities = list(features_by_modality.keys())
        n_modalities = len(modalities)

        fig, axes = plt.subplots(1, n_modalities, figsize=figsize, sharey=True)
        if n_modalities == 1:
            axes = [axes]

        for ax, modality in zip(axes, modalities):
            features = features_by_modality[modality].cpu().numpy()

            # Mean activation per feature
            mean_acts = features.mean(axis=0)

            # Top-k features
            top_indices = np.argsort(mean_acts)[-top_k:]

            # Bar plot
            ax.barh(range(top_k), mean_acts[top_indices], color=f'C{modalities.index(modality)}')
            ax.set_yticks(range(top_k))
            ax.set_yticklabels([f"F{i}" for i in top_indices])
            ax.set_xlabel('Mean Activation')
            ax.set_title(f'{modality.upper()} Features')
            ax.grid(axis='x', alpha=0.3)

        fig.suptitle(f'Modality-Specific Features - {self.layer_name}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save
        save_path = self.output_dir / f"{self.layer_name}_modality_features.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

        return fig

    def plot_feature_statistics(
        self,
        features: torch.Tensor,
        figsize: Tuple[int, int] = (15, 10)
    ) -> plt.Figure:
        """
        Plot comprehensive feature statistics.

        Args:
            features: (n_samples, dict_size) feature activations
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        features_np = features.cpu().numpy()

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Activation distribution
        ax1 = fig.add_subplot(gs[0, 0])
        activations_flat = features_np.flatten()
        activations_nonzero = activations_flat[activations_flat > 0]
        ax1.hist(activations_nonzero, bins=50, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Activation Value')
        ax1.set_ylabel('Count')
        ax1.set_title('Non-zero Activation Distribution')
        ax1.set_yscale('log')

        # 2. Sparsity per sample
        ax2 = fig.add_subplot(gs[0, 1])
        l0_per_sample = (features_np > 0).sum(axis=1)
        ax2.hist(l0_per_sample, bins=30, alpha=0.7, edgecolor='black', color='coral')
        ax2.set_xlabel('Number of Active Features')
        ax2.set_ylabel('Count')
        ax2.set_title('L0 Sparsity per Sample')
        ax2.axvline(l0_per_sample.mean(), color='red', linestyle='--', label=f'Mean: {l0_per_sample.mean():.1f}')
        ax2.legend()

        # 3. Feature activation frequency
        ax3 = fig.add_subplot(gs[0, 2])
        activation_freq = (features_np > 0).mean(axis=0)
        ax3.hist(activation_freq, bins=50, alpha=0.7, edgecolor='black', color='green')
        ax3.set_xlabel('Activation Frequency')
        ax3.set_ylabel('Count')
        ax3.set_title('Feature Activation Frequency')

        # 4. Top active features
        ax4 = fig.add_subplot(gs[1, 0])
        mean_activations = features_np.mean(axis=0)
        top_20_idx = np.argsort(mean_activations)[-20:]
        ax4.barh(range(20), mean_activations[top_20_idx], color='purple', alpha=0.7)
        ax4.set_yticks(range(20))
        ax4.set_yticklabels([f"F{i}" for i in top_20_idx])
        ax4.set_xlabel('Mean Activation')
        ax4.set_title('Top 20 Most Active Features')

        # 5. Dead features
        ax5 = fig.add_subplot(gs[1, 1])
        dead_features = (activation_freq == 0).sum()
        alive_features = len(activation_freq) - dead_features
        ax5.bar(['Alive', 'Dead'], [alive_features, dead_features], color=['green', 'red'], alpha=0.7)
        ax5.set_ylabel('Count')
        ax5.set_title(f'Feature Status ({dead_features} dead)')

        # 6. Reconstruction quality per sample
        ax6 = fig.add_subplot(gs[1, 2])
        # Can't compute without activations, show placeholder
        ax6.text(0.5, 0.5, 'Reconstruction quality\n(requires activations)',
                ha='center', va='center', fontsize=12)
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')

        fig.suptitle(f'Feature Statistics - {self.layer_name}',
                     fontsize=16, fontweight='bold')

        # Save
        save_path = self.output_dir / f"{self.layer_name}_statistics.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

        return fig

    def create_interactive_dashboard(
        self,
        features: torch.Tensor,
        activations: torch.Tensor,
        top_k_features: int = 100
    ) -> go.Figure:
        """
        Create interactive Plotly dashboard with multiple views.

        Args:
            features: (n_samples, dict_size) feature activations
            activations: (n_samples, latent_dim) input activations
            top_k_features: Number of features to include

        Returns:
            Plotly figure with subplots
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive dashboards")

        features_np = features.cpu().numpy()
        activations_np = activations.cpu().numpy()

        # Select top features
        mean_activations = features_np.mean(axis=0)
        top_indices = np.argsort(mean_activations)[-top_k_features:][::-1]

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Feature Activations',
                'Activation Distribution',
                'Feature Co-occurrence',
                'Sparsity Statistics'
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "histogram"}],
                [{"type": "heatmap"}, {"type": "bar"}]
            ]
        )

        # 1. Feature activations heatmap
        fig.add_trace(
            go.Heatmap(
                z=features_np[:, top_indices].T,
                colorscale='Hot',
                name='Activations'
            ),
            row=1, col=1
        )

        # 2. Activation distribution
        acts_nonzero = features_np[features_np > 0]
        fig.add_trace(
            go.Histogram(x=acts_nonzero, nbinsx=50, name='Distribution'),
            row=1, col=2
        )

        # 3. Co-occurrence matrix
        binary_features = (features_np > 0.1).astype(float)
        cooccur = (binary_features.T @ binary_features) / len(features_np)
        cooccur_subset = cooccur[np.ix_(top_indices[:30], top_indices[:30])]

        fig.add_trace(
            go.Heatmap(
                z=cooccur_subset,
                colorscale='Viridis',
                name='Co-occurrence'
            ),
            row=2, col=1
        )

        # 4. Sparsity per sample
        l0_per_sample = (features_np > 0).sum(axis=1)
        fig.add_trace(
            go.Bar(x=list(range(len(l0_per_sample))), y=l0_per_sample, name='L0 Sparsity'),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title_text=f'SAE Feature Dashboard - {self.layer_name}',
            height=1000,
            showlegend=False
        )

        # Save
        save_path = self.output_dir / f"{self.layer_name}_dashboard.html"
        fig.write_html(str(save_path))
        print(f"Saved interactive dashboard to {save_path}")

        return fig


class MultiLayerSAEVisualizer:
    """
    Visualizer for multiple SAE layers.

    Provides comparative visualizations across layers.
    """

    def __init__(
        self,
        saes: Dict[str, SparseAutoencoder],
        output_dir: Optional[str] = None
    ):
        """
        Initialize multi-layer visualizer.

        Args:
            saes: Dict mapping layer names to trained SAEs
            output_dir: Output directory
        """
        self.saes = saes
        self.layer_names = list(saes.keys())
        self.output_dir = Path(output_dir) if output_dir else Path("./sae_visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create individual visualizers
        self.visualizers = {
            name: SAEVisualizer(sae, name, output_dir)
            for name, sae in saes.items()
        }

    def plot_layer_comparison(
        self,
        features_by_layer: Dict[str, torch.Tensor],
        figsize: Tuple[int, int] = (16, 10)
    ) -> plt.Figure:
        """
        Compare feature statistics across layers.

        Args:
            features_by_layer: Dict mapping layer names to feature activations
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        n_layers = len(self.layer_names)

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        # Collect statistics
        stats = {
            'mean_l0': [],
            'mean_activation': [],
            'dead_features': [],
            'max_activation': []
        }

        for layer_name in self.layer_names:
            features = features_by_layer[layer_name].cpu().numpy()

            # L0 sparsity
            l0 = (features > 0).sum(axis=1).mean()
            stats['mean_l0'].append(l0)

            # Mean activation
            mean_act = features.mean()
            stats['mean_activation'].append(mean_act)

            # Dead features
            activation_freq = (features > 0).mean(axis=0)
            dead = (activation_freq == 0).sum()
            stats['dead_features'].append(dead)

            # Max activation
            max_act = features.max()
            stats['max_activation'].append(max_act)

        # Plot comparisons
        layer_labels = [name.split('.')[-1] for name in self.layer_names]

        # 1. L0 Sparsity
        axes[0].bar(range(n_layers), stats['mean_l0'], color='blue', alpha=0.7)
        axes[0].set_xticks(range(n_layers))
        axes[0].set_xticklabels(layer_labels, rotation=45, ha='right')
        axes[0].set_ylabel('Mean L0')
        axes[0].set_title('Mean L0 Sparsity by Layer')
        axes[0].grid(axis='y', alpha=0.3)

        # 2. Mean activation
        axes[1].bar(range(n_layers), stats['mean_activation'], color='green', alpha=0.7)
        axes[1].set_xticks(range(n_layers))
        axes[1].set_xticklabels(layer_labels, rotation=45, ha='right')
        axes[1].set_ylabel('Mean Activation')
        axes[1].set_title('Mean Activation by Layer')
        axes[1].grid(axis='y', alpha=0.3)

        # 3. Dead features
        axes[2].bar(range(n_layers), stats['dead_features'], color='red', alpha=0.7)
        axes[2].set_xticks(range(n_layers))
        axes[2].set_xticklabels(layer_labels, rotation=45, ha='right')
        axes[2].set_ylabel('Dead Features')
        axes[2].set_title('Dead Features by Layer')
        axes[2].grid(axis='y', alpha=0.3)

        # 4. Max activation
        axes[3].bar(range(n_layers), stats['max_activation'], color='purple', alpha=0.7)
        axes[3].set_xticks(range(n_layers))
        axes[3].set_xticklabels(layer_labels, rotation=45, ha='right')
        axes[3].set_ylabel('Max Activation')
        axes[3].set_title('Max Activation by Layer')
        axes[3].grid(axis='y', alpha=0.3)

        # 5. Dictionary size
        dict_sizes = [self.saes[name].dictionary_size for name in self.layer_names]
        axes[4].bar(range(n_layers), dict_sizes, color='orange', alpha=0.7)
        axes[4].set_xticks(range(n_layers))
        axes[4].set_xticklabels(layer_labels, rotation=45, ha='right')
        axes[4].set_ylabel('Dictionary Size')
        axes[4].set_title('SAE Dictionary Size by Layer')
        axes[4].grid(axis='y', alpha=0.3)

        # Hide unused subplot
        axes[5].axis('off')

        fig.suptitle('Layer-wise SAE Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Save
        save_path = self.output_dir / "layer_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

        return fig

    def get_visualizer(self, layer_name: str) -> SAEVisualizer:
        """Get visualizer for specific layer."""
        return self.visualizers[layer_name]

    def _compute_features_from_activations(self, activations_by_layer: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute SAE feature activations for each layer from raw activations.

        Returns dict mapping layer_name -> feature activations (torch.Tensor)
        """
        features_by_layer = {}
        for layer_name, acts in activations_by_layer.items():
            sae = self.saes.get(layer_name)
            if sae is None:
                continue
            try:
                # SAE is expected to return CPU tensors for feature activations
                feats = sae.get_feature_activations(acts)
                features_by_layer[layer_name] = feats
            except Exception:
                # Fallback: move activations to SAE device then compute
                try:
                    device = next(sae.encoder.parameters()).device
                    feats = sae.get_feature_activations(acts.to(device)).cpu()
                    features_by_layer[layer_name] = feats
                except Exception:
                    continue
        return features_by_layer

    def plot_sparsity_comparison(self, activations_by_layer: Dict[str, torch.Tensor], use_bokeh: bool = False):
        """Plot sparsity and related statistics across layers.

        If use_bokeh=True and Bokeh is available, returns a Bokeh layout and saves an HTML. Otherwise uses matplotlib and
        delegates to plot_layer_comparison.
        """
        features_by_layer = self._compute_features_from_activations(activations_by_layer)

        if not use_bokeh or not BOKEH_AVAILABLE:
            # Reuse existing matplotlib-based comparison
            return self.plot_layer_comparison(features_by_layer)

        # Compute stats
        layer_names = list(features_by_layer.keys())
        mean_l0 = []
        mean_activation = []
        dead_features = []
        dict_sizes = []

        for name in layer_names:
            feats = features_by_layer[name].cpu().numpy()
            mean_l0.append((feats > 0).sum(axis=1).mean())
            mean_activation.append(feats.mean())
            activation_freq = (feats > 0).mean(axis=0)
            dead_features.append(int((activation_freq == 0).sum()))
            dict_sizes.append(getattr(self.saes[name], 'dictionary_size', feats.shape[1]))

        # Create Bokeh figures
        figs = []
        p1 = figure(x_range=layer_names, title='Mean L0 Sparsity by Layer', height=300, width=600)
        p1.vbar(x=layer_names, top=mean_l0, width=0.6, color='navy')
        p1.xaxis.major_label_orientation = 1
        figs.append(p1)

        p2 = figure(x_range=layer_names, title='Mean Activation by Layer', height=300, width=600)
        p2.vbar(x=layer_names, top=mean_activation, width=0.6, color='green')
        p2.xaxis.major_label_orientation = 1
        figs.append(p2)

        p3 = figure(x_range=layer_names, title='Dead Features by Layer', height=300, width=600)
        p3.vbar(x=layer_names, top=dead_features, width=0.6, color='red')
        p3.xaxis.major_label_orientation = 1
        figs.append(p3)

        p4 = figure(x_range=layer_names, title='Dictionary Size by Layer', height=300, width=600)
        p4.vbar(x=layer_names, top=dict_sizes, width=0.6, color='orange')
        p4.xaxis.major_label_orientation = 1
        figs.append(p4)

        layout = column(row(figs[0], figs[1]), row(figs[2], figs[3]))

        # Save HTML
        out_path = self.output_dir / 'sparsity_comparison.html'
        try:
                save(layout, filename=str(out_path))
        except Exception:
            # If saving fails, continue (e.g., no webdriver for export)
            pass

        return layout

    def plot_reconstruction_comparison(self, activations_by_layer: Dict[str, torch.Tensor], use_bokeh: bool = False):
        """Compare reconstruction quality (explained variance) across layers.

        activations_by_layer: dict mapping layer name -> activations (torch.Tensor)
        """
        explained = {}
        for name, acts in activations_by_layer.items():
            sae = self.saes.get(name)
            if sae is None:
                continue
            try:
                device = next(sae.encoder.parameters()).device
                acts_device = acts.to(device)
                with torch.no_grad():
                    recon, _ = sae(acts_device)
                    mse = F.mse_loss(recon, acts_device).item()
                    var = acts_device.var().item()
                    explained[name] = float(1 - mse / var) if var > 0 else 0.0
            except Exception:
                explained[name] = 0.0

        layer_names = list(explained.keys())
        values = [explained[n] for n in layer_names]

        if not use_bokeh or not BOKEH_AVAILABLE:
            # Matplotlib bar chart
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(layer_names, values, color='teal', alpha=0.8)
            ax.set_ylabel('Explained Variance')
            ax.set_title('Reconstruction Quality by Layer')
            ax.set_ylim(0, 1)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            save_path = self.output_dir / 'reconstruction_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig

        # Bokeh bar chart
        p = figure(x_range=layer_names, title='Reconstruction Quality by Layer', height=400, width=800)
        p.vbar(x=layer_names, top=values, width=0.6, color='teal')
        p.y_range.start = 0
        p.y_range.end = 1
        p.xaxis.major_label_orientation = 1

        out_path = self.output_dir / 'reconstruction_comparison.html'
        try:
                save(p, filename=str(out_path))
        except Exception:
            pass

        return p

    def plot_feature_frequency_distributions(self, activations_by_layer: Dict[str, torch.Tensor], use_bokeh: bool = False):
        """Plot distribution of feature activation frequencies for each layer."""
        features_by_layer = self._compute_features_from_activations(activations_by_layer)

        if not use_bokeh or not BOKEH_AVAILABLE:
            # Matplotlib: multiple histograms
            fig, ax = plt.subplots(figsize=(10, 5))
            for name, feats in features_by_layer.items():
                freq = (feats.cpu().numpy() > 0).mean(axis=0)
                ax.hist(freq, bins=50, alpha=0.4, label=name)
            ax.set_xlabel('Activation Frequency')
            ax.set_ylabel('Count')
            ax.set_title('Feature Activation Frequency Distributions')
            ax.legend()
            plt.tight_layout()
            save_path = self.output_dir / 'feature_frequency_distributions.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            return fig

        # Bokeh: multiple histogram panels
        panels = []
        for name, feats in features_by_layer.items():
            freq = (feats.cpu().numpy() > 0).mean(axis=0)
            hist, edges = np.histogram(freq, bins=40)
            p = figure(title=f'Feature Frequency - {name}', height=300, width=350)
            p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_alpha=0.6)
            panels.append(p)

        layout = row(*panels)
        out_path = self.output_dir / 'feature_frequency_distributions.html'
        try:
                save(layout, filename=str(out_path))
        except Exception:
            pass

        return layout


# Example usage
if __name__ == '__main__':
    # Create dummy SAE
    sae = SparseAutoencoder(latent_dim=512, dictionary_size=4096)

    # Create visualizer
    viz = SAEVisualizer(sae, layer_name="mamba.block.0", output_dir="./test_viz")

    # Generate dummy data
    n_samples = 1000
    activations = torch.randn(n_samples, 512)
    features = sae.get_feature_activations(activations)

    # Create visualizations
    viz.plot_feature_activations(activations, features)
    viz.plot_feature_statistics(features)
    viz.plot_feature_cooccurrence(features)

    if PLOTLY_AVAILABLE:
        viz.create_interactive_dashboard(features, activations)

    print("Visualizations complete!")
