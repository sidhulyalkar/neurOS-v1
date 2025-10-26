"""
Representational Similarity Analysis (RSA) for model-to-brain alignment.

This module implements RSA methods for comparing representational geometries
between neural networks and brain recordings.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Union, Callable
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, kendalltau
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


class RepresentationalDissimilarityMatrix:
    """
    Compute and manipulate Representational Dissimilarity Matrices (RDMs).

    RDMs capture the pairwise dissimilarity structure of neural representations.

    Examples:
        >>> import torch
        >>> # Neural representations: (n_stimuli, n_features)
        >>> representations = torch.randn(100, 512)
        >>>
        >>> rdm = RepresentationalDissimilarityMatrix(metric='correlation')
        >>> rdm_matrix = rdm.compute(representations)
        >>> print(rdm_matrix.shape)  # (100, 100)
    """

    def __init__(
        self,
        metric: str = 'correlation',
        device: Optional[str] = None
    ):
        """
        Initialize RDM computer.

        Args:
            metric: Distance metric ('correlation', 'euclidean', 'cosine', 'mahalanobis')
            device: Device for computation ('cuda' or 'cpu')
        """
        self.metric = metric
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.rdm_: Optional[torch.Tensor] = None

    def _to_tensor(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert input to tensor."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        return X.to(self.device)

    def compute(
        self,
        representations: Union[np.ndarray, torch.Tensor],
        return_numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Compute RDM from representations.

        Args:
            representations: Neural representations (n_stimuli, n_features)
            return_numpy: Whether to return numpy array

        Returns:
            RDM matrix (n_stimuli, n_stimuli)
        """
        X = self._to_tensor(representations)
        n_stimuli = X.shape[0]

        if self.metric == 'correlation':
            # 1 - Pearson correlation
            X_centered = X - X.mean(dim=1, keepdim=True)
            X_norm = X_centered / (X_centered.norm(dim=1, keepdim=True) + 1e-10)
            similarity = X_norm @ X_norm.T
            rdm = 1 - similarity

        elif self.metric == 'euclidean':
            # Pairwise Euclidean distances
            X_norm = (X ** 2).sum(1).view(-1, 1)
            rdm = X_norm + X_norm.T - 2.0 * X @ X.T
            rdm = torch.sqrt(torch.clamp(rdm, min=0))

        elif self.metric == 'cosine':
            # 1 - cosine similarity
            X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-10)
            similarity = X_norm @ X_norm.T
            rdm = 1 - similarity

        elif self.metric == 'mahalanobis':
            # Mahalanobis distance
            cov = torch.cov(X.T)
            try:
                cov_inv = torch.linalg.inv(cov + 1e-6 * torch.eye(cov.shape[0], device=self.device))
            except RuntimeError:
                warnings.warn("Covariance matrix is singular, using pseudo-inverse")
                cov_inv = torch.linalg.pinv(cov)

            rdm = torch.zeros(n_stimuli, n_stimuli, device=self.device)
            for i in range(n_stimuli):
                diff = X - X[i:i+1]
                rdm[i] = torch.sqrt(torch.sum(diff @ cov_inv * diff, dim=1))

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # Make symmetric and set diagonal to 0
        rdm = (rdm + rdm.T) / 2
        rdm.fill_diagonal_(0)

        self.rdm_ = rdm

        if return_numpy:
            return rdm.cpu().numpy()
        return rdm

    def visualize(
        self,
        rdm: Optional[Union[np.ndarray, torch.Tensor]] = None,
        labels: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (8, 8),
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize RDM as heatmap.

        Args:
            rdm: RDM to visualize (uses self.rdm_ if None)
            labels: Stimulus labels
            figsize: Figure size
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if rdm is None:
            if self.rdm_ is None:
                raise ValueError("No RDM computed. Call compute() first or provide rdm.")
            rdm = self.rdm_

        if isinstance(rdm, torch.Tensor):
            rdm = rdm.cpu().numpy()

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(rdm, cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax, label='Dissimilarity')

        if labels is not None:
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90)
            ax.set_yticklabels(labels)

        ax.set_xlabel('Stimulus')
        ax.set_ylabel('Stimulus')

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'RDM ({self.metric} distance)')

        plt.tight_layout()
        return fig


class RSA:
    """
    Representational Similarity Analysis.

    Compare RDMs between different representations (e.g., model vs. brain).

    Examples:
        >>> # Model representations
        >>> model_reps = torch.randn(100, 512)
        >>> # Brain representations
        >>> brain_reps = torch.randn(100, 200)
        >>>
        >>> rsa = RSA(metric='correlation', comparison='spearman')
        >>> similarity = rsa.compare(model_reps, brain_reps)
        >>> print(f"RSA similarity: {similarity:.4f}")
    """

    def __init__(
        self,
        metric: str = 'correlation',
        comparison: str = 'spearman',
        device: Optional[str] = None
    ):
        """
        Initialize RSA.

        Args:
            metric: Distance metric for RDMs ('correlation', 'euclidean', 'cosine')
            comparison: Method to compare RDMs ('spearman', 'kendall', 'pearson')
            device: Device for computation
        """
        self.metric = metric
        self.comparison = comparison
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.rdm_computer = RepresentationalDissimilarityMatrix(metric=metric, device=device)

    def _get_upper_triangle(self, matrix: torch.Tensor) -> torch.Tensor:
        """Get upper triangle of symmetric matrix (excluding diagonal)."""
        n = matrix.shape[0]
        idx = torch.triu_indices(n, n, offset=1)
        return matrix[idx[0], idx[1]]

    def compare(
        self,
        representations_1: Union[np.ndarray, torch.Tensor],
        representations_2: Union[np.ndarray, torch.Tensor],
        return_pvalue: bool = False
    ) -> Union[float, Tuple[float, float]]:
        """
        Compare two sets of representations via RSA.

        Args:
            representations_1: First set of representations (n_stimuli, n_features_1)
            representations_2: Second set of representations (n_stimuli, n_features_2)
            return_pvalue: Whether to return p-value

        Returns:
            Similarity score (and p-value if requested)
        """
        # Compute RDMs
        rdm_1 = self.rdm_computer.compute(representations_1)
        rdm_2 = self.rdm_computer.compute(representations_2)

        # Get upper triangles
        rdm_1_vec = self._get_upper_triangle(rdm_1).cpu().numpy()
        rdm_2_vec = self._get_upper_triangle(rdm_2).cpu().numpy()

        # Compare RDMs
        if self.comparison == 'spearman':
            corr, pval = spearmanr(rdm_1_vec, rdm_2_vec)
        elif self.comparison == 'kendall':
            corr, pval = kendalltau(rdm_1_vec, rdm_2_vec)
        elif self.comparison == 'pearson':
            corr = np.corrcoef(rdm_1_vec, rdm_2_vec)[0, 1]
            pval = None
        else:
            raise ValueError(f"Unknown comparison method: {self.comparison}")

        if return_pvalue:
            return corr, pval
        return corr

    def compare_rdms(
        self,
        rdm_1: Union[np.ndarray, torch.Tensor],
        rdm_2: Union[np.ndarray, torch.Tensor],
        return_pvalue: bool = False
    ) -> Union[float, Tuple[float, float]]:
        """
        Compare two pre-computed RDMs.

        Args:
            rdm_1: First RDM (n_stimuli, n_stimuli)
            rdm_2: Second RDM (n_stimuli, n_stimuli)
            return_pvalue: Whether to return p-value

        Returns:
            Similarity score (and p-value if requested)
        """
        if isinstance(rdm_1, np.ndarray):
            rdm_1 = torch.from_numpy(rdm_1).float()
        if isinstance(rdm_2, np.ndarray):
            rdm_2 = torch.from_numpy(rdm_2).float()

        rdm_1_vec = self._get_upper_triangle(rdm_1).cpu().numpy()
        rdm_2_vec = self._get_upper_triangle(rdm_2).cpu().numpy()

        if self.comparison == 'spearman':
            corr, pval = spearmanr(rdm_1_vec, rdm_2_vec)
        elif self.comparison == 'kendall':
            corr, pval = kendalltau(rdm_1_vec, rdm_2_vec)
        elif self.comparison == 'pearson':
            corr = np.corrcoef(rdm_1_vec, rdm_2_vec)[0, 1]
            pval = None
        else:
            raise ValueError(f"Unknown comparison method: {self.comparison}")

        if return_pvalue:
            return corr, pval
        return corr


class HierarchicalRSA:
    """
    Hierarchical clustering of RDMs.

    Cluster stimuli based on their representational structure.

    Examples:
        >>> hrsa = HierarchicalRSA(metric='correlation', linkage_method='average')
        >>> clustering = hrsa.fit(representations)
        >>> hrsa.plot_dendrogram(labels=stimulus_labels)
    """

    def __init__(
        self,
        metric: str = 'correlation',
        linkage_method: str = 'average',
        device: Optional[str] = None
    ):
        """
        Initialize HierarchicalRSA.

        Args:
            metric: Distance metric for RDM
            linkage_method: Linkage method ('average', 'complete', 'single', 'ward')
            device: Device for computation
        """
        self.metric = metric
        self.linkage_method = linkage_method
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.rdm_computer = RepresentationalDissimilarityMatrix(metric=metric, device=device)
        self.rdm_: Optional[np.ndarray] = None
        self.linkage_: Optional[np.ndarray] = None

    def fit(
        self,
        representations: Union[np.ndarray, torch.Tensor]
    ) -> 'HierarchicalRSA':
        """
        Fit hierarchical clustering.

        Args:
            representations: Neural representations (n_stimuli, n_features)

        Returns:
            self
        """
        # Compute RDM
        self.rdm_ = self.rdm_computer.compute(representations, return_numpy=True)

        # Convert to condensed distance matrix
        n = self.rdm_.shape[0]
        idx = np.triu_indices(n, k=1)
        condensed_rdm = self.rdm_[idx]

        # Perform hierarchical clustering
        self.linkage_ = linkage(condensed_rdm, method=self.linkage_method)

        return self

    def plot_dendrogram(
        self,
        labels: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 6),
        **kwargs
    ) -> plt.Figure:
        """
        Plot dendrogram of hierarchical clustering.

        Args:
            labels: Stimulus labels
            figsize: Figure size
            **kwargs: Additional arguments for dendrogram

        Returns:
            Matplotlib figure
        """
        if self.linkage_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        fig, ax = plt.subplots(figsize=figsize)

        dendrogram(
            self.linkage_,
            labels=labels,
            ax=ax,
            **kwargs
        )

        ax.set_xlabel('Stimulus')
        ax.set_ylabel('Dissimilarity')
        ax.set_title('Hierarchical Clustering Dendrogram')

        plt.tight_layout()
        return fig


class MDSVisualization:
    """
    Multidimensional Scaling (MDS) visualization of RDMs.

    Project high-dimensional representations to 2D/3D based on dissimilarity.

    Examples:
        >>> mds_vis = MDSVisualization(n_components=2, metric='correlation')
        >>> embedding = mds_vis.fit_transform(representations)
        >>> mds_vis.plot(labels=stimulus_labels, colors=stimulus_categories)
    """

    def __init__(
        self,
        n_components: int = 2,
        metric: str = 'correlation',
        device: Optional[str] = None
    ):
        """
        Initialize MDS visualization.

        Args:
            n_components: Number of dimensions for embedding (2 or 3)
            metric: Distance metric for RDM
            device: Device for computation
        """
        self.n_components = n_components
        self.metric = metric
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.rdm_computer = RepresentationalDissimilarityMatrix(metric=metric, device=device)
        self.mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
        self.embedding_: Optional[np.ndarray] = None
        self.stress_: Optional[float] = None

    def fit_transform(
        self,
        representations: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Fit MDS and transform representations.

        Args:
            representations: Neural representations (n_stimuli, n_features)

        Returns:
            Low-dimensional embedding (n_stimuli, n_components)
        """
        # Compute RDM
        rdm = self.rdm_computer.compute(representations, return_numpy=True)

        # Fit MDS
        self.embedding_ = self.mds.fit_transform(rdm)
        self.stress_ = self.mds.stress_

        return self.embedding_

    def plot(
        self,
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 8),
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot MDS embedding.

        Args:
            labels: Stimulus labels
            colors: Point colors (categories)
            figsize: Figure size
            title: Plot title

        Returns:
            Matplotlib figure
        """
        if self.embedding_ is None:
            raise ValueError("Model not fitted. Call fit_transform() first.")

        if self.n_components == 2:
            fig, ax = plt.subplots(figsize=figsize)

            if colors is not None:
                scatter = ax.scatter(
                    self.embedding_[:, 0],
                    self.embedding_[:, 1],
                    c=colors,
                    cmap='tab10',
                    alpha=0.7,
                    s=100
                )
                plt.colorbar(scatter, ax=ax)
            else:
                ax.scatter(
                    self.embedding_[:, 0],
                    self.embedding_[:, 1],
                    alpha=0.7,
                    s=100
                )

            if labels is not None:
                for i, label in enumerate(labels):
                    ax.annotate(
                        label,
                        (self.embedding_[i, 0], self.embedding_[i, 1]),
                        fontsize=8,
                        alpha=0.7
                    )

            ax.set_xlabel('MDS Dimension 1')
            ax.set_ylabel('MDS Dimension 2')

            if title:
                ax.set_title(title)
            else:
                ax.set_title(f'MDS Visualization (stress={self.stress_:.4f})')

            plt.tight_layout()

        elif self.n_components == 3:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')

            if colors is not None:
                scatter = ax.scatter(
                    self.embedding_[:, 0],
                    self.embedding_[:, 1],
                    self.embedding_[:, 2],
                    c=colors,
                    cmap='tab10',
                    alpha=0.7,
                    s=100
                )
                plt.colorbar(scatter, ax=ax)
            else:
                ax.scatter(
                    self.embedding_[:, 0],
                    self.embedding_[:, 1],
                    self.embedding_[:, 2],
                    alpha=0.7,
                    s=100
                )

            if labels is not None:
                for i, label in enumerate(labels):
                    ax.text(
                        self.embedding_[i, 0],
                        self.embedding_[i, 1],
                        self.embedding_[i, 2],
                        label,
                        fontsize=8,
                        alpha=0.7
                    )

            ax.set_xlabel('MDS Dimension 1')
            ax.set_ylabel('MDS Dimension 2')
            ax.set_zlabel('MDS Dimension 3')

            if title:
                ax.set_title(title)
            else:
                ax.set_title(f'MDS Visualization (stress={self.stress_:.4f})')

        else:
            raise ValueError("n_components must be 2 or 3 for plotting")

        return fig


def compare_multiple_rdms(
    rdms: List[Union[np.ndarray, torch.Tensor]],
    labels: Optional[List[str]] = None,
    comparison: str = 'spearman'
) -> Tuple[np.ndarray, plt.Figure]:
    """
    Compare multiple RDMs pairwise.

    Args:
        rdms: List of RDMs to compare
        labels: Labels for each RDM
        comparison: Comparison method ('spearman', 'kendall', 'pearson')

    Returns:
        similarity_matrix: Pairwise similarity matrix
        fig: Heatmap visualization

    Examples:
        >>> rdms = [rdm_layer1, rdm_layer2, rdm_layer3, rdm_brain]
        >>> labels = ['Layer 1', 'Layer 2', 'Layer 3', 'Brain']
        >>> sim_matrix, fig = compare_multiple_rdms(rdms, labels)
    """
    n_rdms = len(rdms)

    # Convert to tensors
    rdms_tensor = []
    for rdm in rdms:
        if isinstance(rdm, np.ndarray):
            rdm = torch.from_numpy(rdm).float()
        rdms_tensor.append(rdm)

    # Helper to get upper triangle
    def get_upper_triangle(matrix):
        n = matrix.shape[0]
        idx = torch.triu_indices(n, n, offset=1)
        return matrix[idx[0], idx[1]].cpu().numpy()

    # Compute pairwise similarities
    similarity_matrix = np.zeros((n_rdms, n_rdms))

    for i in range(n_rdms):
        for j in range(n_rdms):
            rdm_i_vec = get_upper_triangle(rdms_tensor[i])
            rdm_j_vec = get_upper_triangle(rdms_tensor[j])

            if comparison == 'spearman':
                corr, _ = spearmanr(rdm_i_vec, rdm_j_vec)
            elif comparison == 'kendall':
                corr, _ = kendalltau(rdm_i_vec, rdm_j_vec)
            elif comparison == 'pearson':
                corr = np.corrcoef(rdm_i_vec, rdm_j_vec)[0, 1]
            else:
                raise ValueError(f"Unknown comparison method: {comparison}")

            similarity_matrix[i, j] = corr

    # Visualize
    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='Similarity')

    if labels is not None:
        ax.set_xticks(range(n_rdms))
        ax.set_yticks(range(n_rdms))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)

    # Add correlation values
    for i in range(n_rdms):
        for j in range(n_rdms):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_title(f'RDM Similarity Matrix ({comparison} correlation)')
    plt.tight_layout()

    return similarity_matrix, fig


if __name__ == "__main__":
    # Example usage
    print("Testing RSA implementations...")

    # Generate synthetic data
    n_stimuli = 50
    n_features_model = 100
    n_features_brain = 30

    # Create correlated representations
    shared = torch.randn(n_stimuli, 10)
    model_reps = torch.cat([shared + 0.1 * torch.randn(n_stimuli, 10),
                           torch.randn(n_stimuli, n_features_model - 10)], dim=1)
    brain_reps = torch.cat([shared + 0.1 * torch.randn(n_stimuli, 10),
                           torch.randn(n_stimuli, n_features_brain - 10)], dim=1)

    print("\n1. Computing RDMs")
    rdm_computer = RepresentationalDissimilarityMatrix(metric='correlation')
    model_rdm = rdm_computer.compute(model_reps)
    brain_rdm = rdm_computer.compute(brain_reps)
    print(f"Model RDM shape: {model_rdm.shape}")
    print(f"Brain RDM shape: {brain_rdm.shape}")

    print("\n2. RSA Comparison")
    rsa = RSA(metric='correlation', comparison='spearman')
    similarity, pval = rsa.compare(model_reps, brain_reps, return_pvalue=True)
    print(f"RSA similarity: {similarity:.4f} (p={pval:.4e})")

    print("\n3. Hierarchical Clustering")
    hrsa = HierarchicalRSA(metric='correlation', linkage_method='average')
    hrsa.fit(model_reps)
    print("Hierarchical clustering completed")

    print("\n4. MDS Visualization")
    mds_vis = MDSVisualization(n_components=2, metric='correlation')
    embedding = mds_vis.fit_transform(model_reps)
    print(f"MDS embedding shape: {embedding.shape}")
    print(f"MDS stress: {mds_vis.stress_:.4f}")

    print("\n5. Comparing Multiple RDMs")
    rdms = [model_rdm, brain_rdm]
    labels = ['Model', 'Brain']
    sim_matrix, _ = compare_multiple_rdms(rdms, labels, comparison='spearman')
    print(f"Similarity matrix:\n{sim_matrix}")

    print("\nAll RSA tests passed!")
