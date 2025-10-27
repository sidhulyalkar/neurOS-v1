"""
Feature Analysis Suite for SAE Interpretability

Advanced analysis of learned SAE features:
- Feature attribution (which input channels activate each feature)
- Temporal dynamics of features
- Cross-modality feature alignment
- Causal importance scoring via ablation
- Feature clustering and taxonomy
- Feature steering and intervention

Provides tools for understanding what each feature represents and how it contributes
to model behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage

import matplotlib.pyplot as plt
import seaborn as sns

from neuros_neurofm.interpretability.sparse_autoencoder import SparseAutoencoder


class FeatureAttributionAnalyzer:
    """
    Analyzes which input channels/modalities activate each SAE feature.

    Helps understand what each feature is "looking for" in the input.
    """

    def __init__(self, sae: SparseAutoencoder, device: str = 'cuda'):
        """
        Initialize analyzer.

        Args:
            sae: Trained sparse autoencoder
            device: Device for computation
        """
        self.sae = sae.to(device)
        self.sae.eval()
        self.device = device

    def compute_feature_attribution(
        self,
        activations: torch.Tensor,
        features: torch.Tensor,
        feature_id: int,
        method: str = 'gradient'
    ) -> torch.Tensor:
        """
        Compute attribution of input activations to a specific feature.

        Args:
            activations: (batch, latent_dim) input activations
            features: (batch, dict_size) feature activations
            feature_id: ID of feature to analyze
            method: Attribution method ('gradient', 'correlation', 'mutual_info')

        Returns:
            Attribution scores for each input dimension (latent_dim,)
        """
        activations = activations.to(self.device)
        features = features.to(self.device)

        if method == 'gradient':
            # Gradient-based attribution
            activations.requires_grad = True

            # Forward through encoder to get feature
            pre_activation = self.sae.encoder(activations)
            feature_acts = F.relu(pre_activation)[:, feature_id]

            # Backward to get gradients
            feature_acts.sum().backward()

            # Attribution = gradient * input
            attribution = (activations.grad * activations).abs().mean(dim=0)

            return attribution.cpu()

        elif method == 'correlation':
            # Correlation-based attribution
            activations_np = activations.cpu().numpy()
            feature_acts = features[:, feature_id].cpu().numpy()

            correlations = np.array([
                pearsonr(activations_np[:, i], feature_acts)[0]
                for i in range(activations_np.shape[1])
            ])

            return torch.tensor(np.abs(correlations))

        elif method == 'mutual_info':
            # Compute mutual information (simplified using binning)
            from sklearn.metrics import mutual_info_score

            activations_np = activations.cpu().numpy()
            feature_acts = features[:, feature_id].cpu().numpy()

            # Discretize
            n_bins = 10
            mutual_infos = []

            for i in range(activations_np.shape[1]):
                # Bin activations
                act_binned = np.digitize(activations_np[:, i],
                                        bins=np.linspace(activations_np[:, i].min(),
                                                        activations_np[:, i].max(), n_bins))
                feat_binned = np.digitize(feature_acts,
                                         bins=np.linspace(feature_acts.min(),
                                                         feature_acts.max(), n_bins))

                mi = mutual_info_score(act_binned, feat_binned)
                mutual_infos.append(mi)

            return torch.tensor(mutual_infos)

        else:
            raise ValueError(f"Unknown attribution method: {method}")

    def get_top_attributed_channels(
        self,
        activations: torch.Tensor,
        features: torch.Tensor,
        feature_id: int,
        top_k: int = 10,
        method: str = 'gradient'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get top-k input channels that most strongly drive a feature.

        Args:
            activations: (batch, latent_dim) input activations
            features: (batch, dict_size) feature activations
            feature_id: Feature ID
            top_k: Number of top channels
            method: Attribution method

        Returns:
            Tuple of (channel_indices, attribution_scores)
        """
        attribution = self.compute_feature_attribution(
            activations, features, feature_id, method
        )

        top_indices = torch.argsort(attribution, descending=True)[:top_k]
        top_scores = attribution[top_indices]

        return top_indices.numpy(), top_scores.numpy()

    def compute_modality_attribution(
        self,
        activations_by_modality: Dict[str, torch.Tensor],
        features_by_modality: Dict[str, torch.Tensor],
        feature_id: int,
        method: str = 'correlation'
    ) -> Dict[str, float]:
        """
        Compute which modality most activates a feature.

        Args:
            activations_by_modality: Dict mapping modality to activations
            features_by_modality: Dict mapping modality to feature activations
            feature_id: Feature ID
            method: Attribution method

        Returns:
            Dict mapping modality to attribution score
        """
        modality_scores = {}

        for modality in activations_by_modality.keys():
            activations = activations_by_modality[modality]
            features = features_by_modality[modality]

            attribution = self.compute_feature_attribution(
                activations, features, feature_id, method
            )

            # Aggregate attribution across channels
            modality_scores[modality] = attribution.mean().item()

        return modality_scores


class TemporalDynamicsAnalyzer:
    """
    Analyzes temporal dynamics of feature activations.

    Examines how features evolve over time and their temporal patterns.
    """

    def __init__(self, sae: SparseAutoencoder):
        self.sae = sae

    def compute_temporal_autocorrelation(
        self,
        features: torch.Tensor,
        max_lag: int = 50
    ) -> np.ndarray:
        """
        Compute autocorrelation of feature activations over time.

        Args:
            features: (time_steps, dict_size) feature activations over time
            max_lag: Maximum lag for autocorrelation

        Returns:
            Autocorrelation matrix (max_lag, dict_size)
        """
        features_np = features.cpu().numpy()
        n_features = features_np.shape[1]

        autocorr = np.zeros((max_lag, n_features))

        for feat_idx in range(n_features):
            feat_series = features_np[:, feat_idx]

            for lag in range(max_lag):
                if len(feat_series) > lag:
                    # Compute correlation with lagged version
                    corr = pearsonr(
                        feat_series[:-lag if lag > 0 else None],
                        feat_series[lag:]
                    )[0]
                    autocorr[lag, feat_idx] = corr

        return autocorr

    def identify_temporal_patterns(
        self,
        features: torch.Tensor,
        pattern_types: List[str] = ['sustained', 'transient', 'oscillatory']
    ) -> Dict[str, List[int]]:
        """
        Identify features with different temporal patterns.

        Args:
            features: (time_steps, dict_size) feature activations
            pattern_types: Types of patterns to detect

        Returns:
            Dict mapping pattern type to list of feature IDs
        """
        features_np = features.cpu().numpy()
        patterns = defaultdict(list)

        # Compute autocorrelation
        autocorr = self.compute_temporal_autocorrelation(features, max_lag=50)

        for feat_idx in range(features_np.shape[1]):
            feat_series = features_np[:, feat_idx]
            feat_autocorr = autocorr[:, feat_idx]

            # Sustained: high autocorrelation at long lags
            if 'sustained' in pattern_types:
                if feat_autocorr[10:30].mean() > 0.5:
                    patterns['sustained'].append(feat_idx)

            # Transient: low autocorrelation, high variance
            if 'transient' in pattern_types:
                if feat_autocorr[5:15].mean() < 0.2 and feat_series.std() > 0.1:
                    patterns['transient'].append(feat_idx)

            # Oscillatory: periodic autocorrelation
            if 'oscillatory' in pattern_types:
                # Look for periodic pattern in autocorrelation
                fft = np.fft.fft(feat_autocorr)
                power = np.abs(fft) ** 2
                # Check for dominant frequency
                if power[1:10].max() > 2 * power[1:10].mean():
                    patterns['oscillatory'].append(feat_idx)

        return dict(patterns)

    def compute_feature_onset_offset(
        self,
        features: torch.Tensor,
        threshold: float = 0.1
    ) -> Dict[int, Tuple[float, float]]:
        """
        Compute average onset and offset times for each feature.

        Args:
            features: (time_steps, dict_size) feature activations
            threshold: Activation threshold

        Returns:
            Dict mapping feature ID to (mean_onset, mean_offset) times
        """
        features_np = features.cpu().numpy()
        onset_offset = {}

        for feat_idx in range(features_np.shape[1]):
            feat_series = features_np[:, feat_idx]

            # Find activation events
            is_active = feat_series > threshold

            # Find transitions
            onsets = np.where(np.diff(is_active.astype(int)) == 1)[0]
            offsets = np.where(np.diff(is_active.astype(int)) == -1)[0]

            if len(onsets) > 0:
                mean_onset = onsets.mean()
            else:
                mean_onset = np.nan

            if len(offsets) > 0:
                mean_offset = offsets.mean()
            else:
                mean_offset = np.nan

            onset_offset[feat_idx] = (mean_onset, mean_offset)

        return onset_offset


class CausalImportanceAnalyzer:
    """
    Analyzes causal importance of features through ablation.

    Measures how much model behavior changes when features are ablated.
    """

    def __init__(
        self,
        model: nn.Module,
        sae: SparseAutoencoder,
        device: str = 'cuda'
    ):
        """
        Initialize analyzer.

        Args:
            model: NeuroFMX model
            sae: Trained SAE
            device: Device
        """
        self.model = model.to(device)
        self.sae = sae.to(device)
        self.device = device

        self.model.eval()
        self.sae.eval()

    def ablate_feature(
        self,
        activations: torch.Tensor,
        feature_id: int,
        ablation_value: float = 0.0
    ) -> torch.Tensor:
        """
        Ablate a specific feature and reconstruct activations.

        Args:
            activations: (batch, latent_dim) input activations
            feature_id: Feature to ablate
            ablation_value: Value to set feature to (default: 0)

        Returns:
            Reconstructed activations with feature ablated
        """
        with torch.no_grad():
            # Encode
            features = self.sae.get_feature_activations(activations)

            # Ablate feature
            features_ablated = features.clone()
            features_ablated[:, feature_id] = ablation_value

            # Decode
            if self.sae.tie_weights:
                reconstruction = F.linear(features_ablated, self.sae.encoder.weight.t())
            else:
                reconstruction = self.sae.decoder(features_ablated)

        return reconstruction

    def compute_feature_importance_by_ablation(
        self,
        dataloader: torch.utils.data.DataLoader,
        layer_name: str,
        metric_fn: Callable,
        top_k_features: int = 100
    ) -> Dict[int, float]:
        """
        Compute importance of features via ablation.

        Args:
            dataloader: DataLoader with inputs
            layer_name: Layer to analyze
            metric_fn: Function to compute metric (takes model outputs, returns scalar)
            top_k_features: Number of top features to ablate

        Returns:
            Dict mapping feature ID to importance score
        """
        # Get baseline metric (no ablation)
        baseline_metric = self._compute_metric(dataloader, layer_name, metric_fn, ablate_feature=None)

        # Get most active features
        all_features = self._collect_features(dataloader, layer_name)
        mean_activation = all_features.mean(dim=0)
        top_feature_ids = torch.argsort(mean_activation, descending=True)[:top_k_features]

        # Ablate each feature and measure impact
        importance_scores = {}

        for feat_id in tqdm(top_feature_ids, desc="Ablating features"):
            ablated_metric = self._compute_metric(
                dataloader, layer_name, metric_fn, ablate_feature=feat_id.item()
            )

            # Importance = change in metric
            importance = abs(ablated_metric - baseline_metric)
            importance_scores[feat_id.item()] = importance

        return importance_scores

    def _collect_features(
        self,
        dataloader: torch.utils.data.DataLoader,
        layer_name: str
    ) -> torch.Tensor:
        """Collect all features from a layer."""
        # Placeholder - would need hook system
        return torch.randn(1000, self.sae.dictionary_size)

    def _compute_metric(
        self,
        dataloader: torch.utils.data.DataLoader,
        layer_name: str,
        metric_fn: Callable,
        ablate_feature: Optional[int] = None
    ) -> float:
        """Compute metric with optional feature ablation."""
        # Placeholder - would need full implementation
        return np.random.random()


class FeatureClusteringAnalyzer:
    """
    Clusters features into interpretable groups.

    Creates a taxonomy of learned features.
    """

    def __init__(self, sae: SparseAutoencoder):
        self.sae = sae

    def cluster_features_by_activation(
        self,
        features: torch.Tensor,
        n_clusters: int = 10,
        method: str = 'kmeans'
    ) -> Tuple[np.ndarray, float]:
        """
        Cluster features based on activation patterns.

        Args:
            features: (n_samples, dict_size) feature activations
            n_clusters: Number of clusters
            method: Clustering method ('kmeans', 'hierarchical')

        Returns:
            Tuple of (cluster_labels, silhouette_score)
        """
        # Transpose: cluster features, not samples
        features_np = features.cpu().numpy().T  # (dict_size, n_samples)

        # Normalize
        features_norm = features_np / (np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-8)

        if method == 'kmeans':
            clustering = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clustering.fit_predict(features_norm)

        elif method == 'hierarchical':
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clustering.fit_predict(features_norm)

        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Compute silhouette score
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(features_norm, labels)
        else:
            silhouette = 0.0

        return labels, silhouette

    def compute_feature_similarity_matrix(
        self,
        features: torch.Tensor,
        metric: str = 'cosine'
    ) -> np.ndarray:
        """
        Compute pairwise similarity between features.

        Args:
            features: (n_samples, dict_size) feature activations
            metric: Similarity metric ('cosine', 'correlation')

        Returns:
            Similarity matrix (dict_size, dict_size)
        """
        features_np = features.cpu().numpy().T  # (dict_size, n_samples)

        if metric == 'cosine':
            # Cosine similarity
            norms = np.linalg.norm(features_np, axis=1, keepdims=True)
            normalized = features_np / (norms + 1e-8)
            similarity = normalized @ normalized.T

        elif metric == 'correlation':
            # Pearson correlation
            from sklearn.metrics import pairwise_distances
            similarity = 1 - pairwise_distances(features_np, metric='correlation')

        else:
            raise ValueError(f"Unknown metric: {metric}")

        return similarity

    def create_feature_hierarchy(
        self,
        features: torch.Tensor,
        method: str = 'ward'
    ) -> np.ndarray:
        """
        Create hierarchical clustering of features.

        Args:
            features: (n_samples, dict_size) feature activations
            method: Linkage method ('ward', 'average', 'complete')

        Returns:
            Linkage matrix for dendrogram
        """
        features_np = features.cpu().numpy().T  # (dict_size, n_samples)

        # Compute linkage
        Z = linkage(features_np, method=method)

        return Z

    def plot_feature_dendrogram(
        self,
        features: torch.Tensor,
        figsize: Tuple[int, int] = (12, 8),
        max_features: int = 100
    ) -> plt.Figure:
        """
        Plot hierarchical clustering dendrogram.

        Args:
            features: (n_samples, dict_size) feature activations
            figsize: Figure size
            max_features: Maximum features to show

        Returns:
            Matplotlib figure
        """
        # Subsample features if too many
        if features.shape[1] > max_features:
            # Select most active features
            mean_activation = features.mean(dim=0)
            top_indices = torch.argsort(mean_activation, descending=True)[:max_features]
            features = features[:, top_indices]

        Z = self.create_feature_hierarchy(features)

        fig, ax = plt.subplots(figsize=figsize)
        dendrogram(Z, ax=ax, leaf_font_size=8)
        ax.set_xlabel('Feature ID')
        ax.set_ylabel('Distance')
        ax.set_title('Feature Hierarchy')
        plt.tight_layout()

        return fig


class FeatureSteeringAnalyzer:
    """
    Analyzes effects of steering/amplifying specific features.

    Enables causal interventions on the model.
    """

    def __init__(
        self,
        model: nn.Module,
        sae: SparseAutoencoder,
        device: str = 'cuda'
    ):
        """
        Initialize steering analyzer.

        Args:
            model: NeuroFMX model
            sae: Trained SAE
            device: Device
        """
        self.model = model.to(device)
        self.sae = sae.to(device)
        self.device = device

    def steer_feature(
        self,
        activations: torch.Tensor,
        feature_id: int,
        steering_strength: float = 2.0
    ) -> torch.Tensor:
        """
        Amplify or suppress a feature and reconstruct.

        Args:
            activations: (batch, latent_dim) input activations
            feature_id: Feature to steer
            steering_strength: Multiplier for feature (>1 = amplify, <1 = suppress)

        Returns:
            Steered activations
        """
        with torch.no_grad():
            # Encode
            features = self.sae.get_feature_activations(activations)

            # Steer
            features_steered = features.clone()
            features_steered[:, feature_id] *= steering_strength

            # Decode
            if self.sae.tie_weights:
                reconstruction = F.linear(features_steered, self.sae.encoder.weight.t())
            else:
                reconstruction = self.sae.decoder(features_steered)

        return reconstruction

    def analyze_steering_effects(
        self,
        activations: torch.Tensor,
        feature_id: int,
        steering_range: Tuple[float, float] = (0.0, 3.0),
        n_steps: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Analyze effects of steering a feature across a range of strengths.

        Args:
            activations: Input activations
            feature_id: Feature to steer
            steering_range: Range of steering strengths
            n_steps: Number of steps in range

        Returns:
            Dict with steering results
        """
        strengths = np.linspace(steering_range[0], steering_range[1], n_steps)

        results = {
            'strengths': strengths,
            'reconstructions': [],
            'reconstruction_errors': []
        }

        for strength in strengths:
            steered = self.steer_feature(activations, feature_id, strength)
            results['reconstructions'].append(steered.cpu().numpy())

            # Compute reconstruction error
            error = F.mse_loss(steered, activations).item()
            results['reconstruction_errors'].append(error)

        return results


# Example usage
if __name__ == '__main__':
    # Create dummy SAE
    sae = SparseAutoencoder(latent_dim=512, dictionary_size=4096)

    # Create analyzers
    attribution_analyzer = FeatureAttributionAnalyzer(sae)
    temporal_analyzer = TemporalDynamicsAnalyzer(sae)
    clustering_analyzer = FeatureClusteringAnalyzer(sae)

    # Generate dummy data
    n_samples = 1000
    activations = torch.randn(n_samples, 512)
    features = sae.get_feature_activations(activations)

    # Test attribution
    print("Computing feature attribution...")
    attribution = attribution_analyzer.compute_feature_attribution(
        activations, features, feature_id=0, method='gradient'
    )
    print(f"Attribution shape: {attribution.shape}")

    # Test clustering
    print("\nClustering features...")
    labels, silhouette = clustering_analyzer.cluster_features_by_activation(
        features, n_clusters=10
    )
    print(f"Silhouette score: {silhouette:.3f}")

    # Test temporal dynamics
    print("\nAnalyzing temporal patterns...")
    temporal_features = torch.randn(500, 4096)  # Time series
    patterns = temporal_analyzer.identify_temporal_patterns(temporal_features)
    print(f"Temporal patterns: {[(k, len(v)) for k, v in patterns.items()]}")

    print("\nFeature analysis complete!")
