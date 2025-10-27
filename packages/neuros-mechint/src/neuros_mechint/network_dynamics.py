"""
Network Dynamics Analysis for NeuroFMx

Large-scale network activity tracking, connectivity analysis, and information flow.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import signal, stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS


class NetworkDynamicsAnalyzer:
    """
    Analyze large-scale network dynamics in NeuroFMx.

    Tracks:
    - Population activity patterns
    - Information flow between layers
    - Temporal dynamics and oscillations
    - Functional connectivity
    - Representational geometry evolution
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.activations = defaultdict(list)
        self.gradients = defaultdict(list)
        self.hooks = []

    def register_hooks(self, layer_names: Optional[List[str]] = None):
        """Register hooks to capture activations and gradients."""
        if layer_names is None:
            # Default: capture all major layers
            layer_names = self._get_default_layers()

        for name, module in self.model.named_modules():
            if layer_names and name not in layer_names:
                continue

            # Forward hook for activations
            hook_fwd = module.register_forward_hook(
                lambda m, inp, out, name=name: self._save_activation(name, out)
            )
            self.hooks.append(hook_fwd)

            # Backward hook for gradients
            hook_bwd = module.register_full_backward_hook(
                lambda m, grad_in, grad_out, name=name: self._save_gradient(name, grad_out)
            )
            self.hooks.append(hook_bwd)

    def _get_default_layers(self) -> List[str]:
        """Get default layer names to analyze."""
        default_layers = []
        for name, module in self.model.named_modules():
            if any(layer_type in name for layer_type in ['mamba', 'perceiver', 'popt', 'head']):
                default_layers.append(name)
        return default_layers

    def _save_activation(self, name: str, output):
        """Save activation from forward pass."""
        if isinstance(output, tuple):
            output = output[0]
        self.activations[name].append(output.detach().cpu())

    def _save_gradient(self, name: str, grad_output):
        """Save gradient from backward pass."""
        if isinstance(grad_output, tuple):
            grad_output = grad_output[0]
        if grad_output is not None:
            self.gradients[name].append(grad_output.detach().cpu())

    def clear_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations.clear()
        self.gradients.clear()

    def compute_population_activity(
        self,
        dataset: torch.utils.data.DataLoader,
        layer_name: str
    ) -> Dict[str, np.ndarray]:
        """
        Compute population activity statistics.

        Returns:
            - mean_activity: Mean activation across population
            - std_activity: Std of activations
            - covariance: Population covariance matrix
            - participation_ratio: Effective dimensionality
        """
        self.model.eval()
        all_activations = []

        with torch.no_grad():
            for batch in dataset:
                modality_dict = batch['modality_dict']
                _ = self.model(modality_dict, task='encoder')

                if layer_name in self.activations:
                    act = self.activations[layer_name][-1]
                    # Reshape to (batch * time, features)
                    act = act.reshape(-1, act.shape[-1])
                    all_activations.append(act.numpy())

        all_activations = np.concatenate(all_activations, axis=0)

        # Compute statistics
        mean_activity = all_activations.mean(axis=0)
        std_activity = all_activations.std(axis=0)
        covariance = np.cov(all_activations.T)

        # Participation ratio: (sum of eigenvalues)^2 / sum of eigenvalues^2
        eigvals = np.linalg.eigvalsh(covariance)
        eigvals = eigvals[eigvals > 0]
        participation_ratio = (eigvals.sum() ** 2) / (eigvals ** 2).sum()

        return {
            'mean_activity': mean_activity,
            'std_activity': std_activity,
            'covariance': covariance,
            'participation_ratio': participation_ratio,
            'n_dimensions': len(eigvals),
            'explained_variance': eigvals[::-1]  # Sorted descending
        }

    def compute_information_flow(
        self,
        source_layer: str,
        target_layer: str,
        dataset: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Compute information flow between layers using mutual information.

        Uses binning-based MI estimation for neural activations.
        """
        self.model.eval()
        source_acts = []
        target_acts = []

        with torch.no_grad():
            for batch in dataset:
                modality_dict = batch['modality_dict']
                _ = self.model(modality_dict, task='encoder')

                if source_layer in self.activations:
                    source_acts.append(self.activations[source_layer][-1].numpy())
                if target_layer in self.activations:
                    target_acts.append(self.activations[target_layer][-1].numpy())

        source_acts = np.concatenate(source_acts, axis=0)
        target_acts = np.concatenate(target_acts, axis=0)

        # Flatten spatial/temporal dimensions
        source_acts = source_acts.reshape(source_acts.shape[0], -1)
        target_acts = target_acts.reshape(target_acts.shape[0], -1)

        # PCA to reduce dimensionality for MI estimation
        pca_source = PCA(n_components=min(10, source_acts.shape[1]))
        pca_target = PCA(n_components=min(10, target_acts.shape[1]))

        source_reduced = pca_source.fit_transform(source_acts)
        target_reduced = pca_target.fit_transform(target_acts)

        # Estimate MI using histogram-based method
        mi_total = 0
        for i in range(source_reduced.shape[1]):
            for j in range(target_reduced.shape[1]):
                mi = self._mutual_information(
                    source_reduced[:, i],
                    target_reduced[:, j],
                    bins=20
                )
                mi_total += mi

        # Normalize by number of dimensions
        mi_normalized = mi_total / (source_reduced.shape[1] * target_reduced.shape[1])

        # Compute transfer entropy (directionality)
        te = self._transfer_entropy(source_reduced[:, 0], target_reduced[:, 0])

        return {
            'mutual_information': mi_normalized,
            'transfer_entropy': te,
            'source_explained_var': pca_source.explained_variance_ratio_.sum(),
            'target_explained_var': pca_target.explained_variance_ratio_.sum()
        }

    def _mutual_information(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bins: int = 20
    ) -> float:
        """Compute mutual information using histogram."""
        hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)

        # Convert to probability
        pxy = hist_2d / hist_2d.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)

        # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
        px_py = px[:, None] * py[None, :]

        # Avoid log(0)
        mask = (pxy > 0) & (px_py > 0)
        mi = (pxy[mask] * np.log(pxy[mask] / px_py[mask])).sum()

        return mi

    def _transfer_entropy(
        self,
        source: np.ndarray,
        target: np.ndarray,
        k: int = 1
    ) -> float:
        """
        Compute transfer entropy: TE(X->Y) = H(Y_t | Y_t-1) - H(Y_t | Y_t-1, X_t-1)

        Approximated using conditional mutual information.
        """
        # Create lagged versions
        y_t = target[k:]
        y_t_minus_1 = target[:-k]
        x_t_minus_1 = source[:-k]

        # TE ≈ MI(Y_t ; X_t-1 | Y_t-1)
        # Simplified: MI(Y_t ; X_t-1) - MI(Y_t-1 ; X_t-1)
        mi_yx = self._mutual_information(y_t, x_t_minus_1)
        mi_yy_x = self._mutual_information(y_t_minus_1, x_t_minus_1)

        te = max(0, mi_yx - mi_yy_x)
        return te

    def analyze_temporal_dynamics(
        self,
        layer_name: str,
        dataset: torch.utils.data.DataLoader,
        sampling_rate: float = 100.0
    ) -> Dict[str, np.ndarray]:
        """
        Analyze temporal dynamics: oscillations, autocorrelation, power spectrum.
        """
        self.model.eval()
        all_activations = []

        with torch.no_grad():
            for batch in dataset:
                modality_dict = batch['modality_dict']
                _ = self.model(modality_dict, task='encoder')

                if layer_name in self.activations:
                    act = self.activations[layer_name][-1]
                    # Keep temporal dimension
                    all_activations.append(act.numpy())

        # Concatenate along batch dimension
        all_activations = np.concatenate(all_activations, axis=0)
        # (batch, time, features)

        # Compute power spectral density
        psd_all = []
        freqs = None

        for i in range(all_activations.shape[0]):
            for j in range(min(10, all_activations.shape[2])):  # First 10 features
                f, psd = signal.welch(
                    all_activations[i, :, j],
                    fs=sampling_rate,
                    nperseg=min(256, all_activations.shape[1])
                )
                psd_all.append(psd)
                if freqs is None:
                    freqs = f

        psd_mean = np.mean(psd_all, axis=0)
        psd_std = np.std(psd_all, axis=0)

        # Autocorrelation
        autocorr = self._compute_autocorrelation(
            all_activations[:, :, 0],  # First feature
            max_lag=min(100, all_activations.shape[1] // 2)
        )

        # Identify dominant frequencies
        peak_indices = signal.find_peaks(psd_mean, height=psd_mean.mean())[0]
        dominant_freqs = freqs[peak_indices]

        return {
            'frequencies': freqs,
            'psd_mean': psd_mean,
            'psd_std': psd_std,
            'autocorrelation': autocorr,
            'dominant_frequencies': dominant_freqs,
            'oscillation_power': psd_mean[peak_indices] if len(peak_indices) > 0 else np.array([])
        }

    def _compute_autocorrelation(
        self,
        signals: np.ndarray,
        max_lag: int
    ) -> np.ndarray:
        """Compute autocorrelation for multiple signals."""
        n_signals = signals.shape[0]
        autocorr_all = []

        for i in range(n_signals):
            signal_i = signals[i]
            signal_i = signal_i - signal_i.mean()

            autocorr = np.correlate(signal_i, signal_i, mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # Take positive lags
            autocorr = autocorr / autocorr[0]  # Normalize

            autocorr_all.append(autocorr[:max_lag])

        return np.mean(autocorr_all, axis=0)

    def compute_functional_connectivity(
        self,
        layer_name: str,
        dataset: torch.utils.data.DataLoader,
        method: str = 'correlation'
    ) -> np.ndarray:
        """
        Compute functional connectivity matrix between units in a layer.

        Args:
            method: 'correlation', 'partial_correlation', or 'coherence'
        """
        self.model.eval()
        all_activations = []

        with torch.no_grad():
            for batch in dataset:
                modality_dict = batch['modality_dict']
                _ = self.model(modality_dict, task='encoder')

                if layer_name in self.activations:
                    act = self.activations[layer_name][-1]
                    # Average over time, keep features
                    act = act.mean(dim=1)  # (batch, features)
                    all_activations.append(act.numpy())

        all_activations = np.concatenate(all_activations, axis=0)

        if method == 'correlation':
            connectivity = np.corrcoef(all_activations.T)

        elif method == 'partial_correlation':
            # Compute precision matrix (inverse covariance)
            cov = np.cov(all_activations.T)
            precision = np.linalg.pinv(cov)

            # Partial correlation from precision
            d = np.sqrt(np.diag(precision))
            connectivity = -precision / np.outer(d, d)
            np.fill_diagonal(connectivity, 1.0)

        elif method == 'coherence':
            # Spectral coherence (for temporal signals)
            n_features = all_activations.shape[1]
            connectivity = np.zeros((n_features, n_features))

            for i in range(min(n_features, 50)):  # Limit for computation
                for j in range(i+1, min(n_features, 50)):
                    f, coh = signal.coherence(
                        all_activations[:, i],
                        all_activations[:, j],
                        fs=100.0
                    )
                    # Average coherence across frequencies
                    connectivity[i, j] = coh.mean()
                    connectivity[j, i] = connectivity[i, j]

        else:
            raise ValueError(f"Unknown method: {method}")

        return connectivity

    def track_representational_geometry(
        self,
        layer_name: str,
        dataset: torch.utils.data.DataLoader,
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Track how representational geometry evolves across layers.

        Computes:
        - RDM (Representational Dissimilarity Matrix)
        - Dimensionality (participation ratio)
        - Clustering (silhouette score)
        - Alignment with behavior
        """
        self.model.eval()
        all_activations = []

        with torch.no_grad():
            for batch in dataset:
                modality_dict = batch['modality_dict']
                _ = self.model(modality_dict, task='encoder')

                if layer_name in self.activations:
                    act = self.activations[layer_name][-1]
                    # Average over time
                    act = act.mean(dim=1)
                    all_activations.append(act.numpy())

        all_activations = np.concatenate(all_activations, axis=0)

        # Compute RDM (pairwise distances)
        from scipy.spatial.distance import pdist, squareform
        rdm = squareform(pdist(all_activations, metric='correlation'))

        # Dimensionality via PCA
        pca = PCA()
        pca.fit(all_activations)
        explained_var = pca.explained_variance_ratio_

        participation_ratio = (explained_var.sum() ** 2) / (explained_var ** 2).sum()

        # Clustering quality if labels provided
        silhouette = None
        if labels is not None:
            from sklearn.metrics import silhouette_score
            try:
                silhouette = silhouette_score(all_activations, labels)
            except:
                silhouette = None

        return {
            'rdm': rdm,
            'explained_variance': explained_var,
            'participation_ratio': participation_ratio,
            'n_effective_dims': participation_ratio,
            'silhouette_score': silhouette,
            'activations': all_activations
        }

    def compute_gradient_flow(self) -> Dict[str, float]:
        """
        Analyze gradient flow through the network.

        Helps identify vanishing/exploding gradients.
        """
        gradient_stats = {}

        for name, grads in self.gradients.items():
            if len(grads) == 0:
                continue

            # Concatenate all gradients for this layer
            all_grads = torch.cat([g.flatten() for g in grads])

            gradient_stats[name] = {
                'mean': all_grads.mean().item(),
                'std': all_grads.std().item(),
                'max': all_grads.max().item(),
                'min': all_grads.min().item(),
                'norm': all_grads.norm().item()
            }

        return gradient_stats

    def visualize_network_activity(
        self,
        layer_name: str,
        save_path: Optional[str] = None
    ):
        """Visualize network activity patterns."""
        if layer_name not in self.activations:
            raise ValueError(f"No activations for layer: {layer_name}")

        # Get activations
        act = self.activations[layer_name][0].numpy()  # First batch

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Activation heatmap
        if act.ndim == 3:  # (batch, time, features)
            act_2d = act[0]  # First sample
        else:
            act_2d = act.reshape(-1, act.shape[-1])

        axes[0, 0].imshow(act_2d.T, aspect='auto', cmap='viridis')
        axes[0, 0].set_title('Activation Heatmap')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Units')

        # 2. Activation distribution
        axes[0, 1].hist(act.flatten(), bins=50, alpha=0.7)
        axes[0, 1].set_title('Activation Distribution')
        axes[0, 1].set_xlabel('Activation')
        axes[0, 1].set_ylabel('Count')

        # 3. PCA projection
        act_flat = act.reshape(-1, act.shape[-1])
        if act_flat.shape[0] > 2:
            pca = PCA(n_components=2)
            proj = pca.fit_transform(act_flat)
            axes[1, 0].scatter(proj[:, 0], proj[:, 1], alpha=0.5)
            axes[1, 0].set_title(f'PCA Projection ({pca.explained_variance_ratio_.sum():.2%} var)')
            axes[1, 0].set_xlabel('PC1')
            axes[1, 0].set_ylabel('PC2')

        # 4. Temporal evolution (mean activation)
        if act.ndim == 3:
            mean_over_features = act.mean(axis=-1)  # (batch, time)
            for i in range(min(5, mean_over_features.shape[0])):
                axes[1, 1].plot(mean_over_features[i], alpha=0.7)
            axes[1, 1].set_title('Temporal Evolution')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Mean Activation')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


def analyze_full_network(
    model: nn.Module,
    dataset: torch.utils.data.DataLoader,
    save_dir: str = './network_analysis'
) -> Dict[str, Dict]:
    """
    Comprehensive network analysis across all layers.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    analyzer = NetworkDynamicsAnalyzer(model)
    analyzer.register_hooks()

    results = {}

    # Run one pass through data to collect activations
    print("Collecting activations...")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataset):
            if i >= 10:  # Limit for speed
                break
            modality_dict = batch['modality_dict']
            _ = model(modality_dict, task='multi-task')

    print(f"Captured activations from {len(analyzer.activations)} layers")

    # Analyze each layer
    for layer_name in list(analyzer.activations.keys())[:5]:  # Analyze first 5 layers
        print(f"\nAnalyzing layer: {layer_name}")

        layer_results = {}

        # Population activity
        pop_stats = analyzer.compute_population_activity(dataset, layer_name)
        layer_results['population'] = pop_stats
        print(f"  Participation ratio: {pop_stats['participation_ratio']:.2f}")

        # Functional connectivity
        connectivity = analyzer.compute_functional_connectivity(layer_name, dataset)
        layer_results['connectivity'] = connectivity
        print(f"  Mean connectivity: {connectivity[~np.eye(connectivity.shape[0], dtype=bool)].mean():.3f}")

        # Representational geometry
        geometry = analyzer.track_representational_geometry(layer_name, dataset)
        layer_results['geometry'] = geometry
        print(f"  Effective dimensions: {geometry['n_effective_dims']:.2f}")

        results[layer_name] = layer_results

    analyzer.clear_hooks()

    return results
