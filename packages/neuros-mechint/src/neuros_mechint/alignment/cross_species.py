"""
Cross-Species Alignment

Methods for aligning neural representations across species to understand
evolutionary conserved computations and species-specific adaptations.

References:
- Kriegeskorte & Diedrichsen (2019): Peeling the onion of brain representations
- Hasson et al. (2020): Direct fit to nature: cross-species comparisons
- Yamins & DiCarlo (2016): Using goal-driven deep learning models
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from scipy.spatial import procrustes
from scipy.stats import spearmanr


@dataclass
class CrossSpeciesAlignment:
    """Results from cross-species alignment."""
    alignment_score: float
    transformation_matrix: np.ndarray
    species_a_transformed: np.ndarray
    species_b_transformed: np.ndarray
    conserved_dimensions: Optional[np.ndarray] = None
    species_specific_a: Optional[np.ndarray] = None
    species_specific_b: Optional[np.ndarray] = None


class ProcrustesAlignment(nn.Module):
    """
    Procrustes alignment for cross-species representations.

    Finds optimal rotation, scaling, and translation to align
    two representational spaces.

    Solves: min ||A*T - B|| where T is orthogonal transformation
    """

    def __init__(self, allow_scaling: bool = True):
        super().__init__()
        self.allow_scaling = allow_scaling

        # Learned transformation
        self.register_buffer('T', None)
        self.register_buffer('scale', None)
        self.register_buffer('translation_a', None)
        self.register_buffer('translation_b', None)

    def fit(
        self,
        species_a: torch.Tensor,
        species_b: torch.Tensor
    ) -> CrossSpeciesAlignment:
        """
        Fit Procrustes transformation.

        Args:
            species_a: Representations from species A (n_samples, n_features_a)
            species_b: Representations from species B (n_samples, n_features_b)

        Returns:
            Alignment results
        """
        # Convert to numpy for scipy
        A = species_a.cpu().numpy()
        B = species_b.cpu().numpy()

        # Center data
        self.translation_a = torch.from_numpy(A.mean(0))
        self.translation_b = torch.from_numpy(B.mean(0))

        A_centered = A - A.mean(0)
        B_centered = B - B.mean(0)

        # Procrustes analysis
        mtx1, mtx2, disparity = procrustes(A_centered, B_centered)

        # Extract transformation
        # mtx2 = scale * mtx1 @ T
        # Solve for T
        U, S, Vt = np.linalg.svd(A_centered.T @ B_centered)
        self.T = torch.from_numpy(U @ Vt)

        if self.allow_scaling:
            scale_num = np.trace(A_centered.T @ B_centered @ self.T.numpy())
            scale_denom = np.trace(A_centered.T @ A_centered)
            self.scale = torch.tensor(scale_num / scale_denom)
        else:
            self.scale = torch.tensor(1.0)

        # Transform
        A_transformed = self.transform(species_a, source='a')
        B_transformed = self.transform(species_b, source='b')

        # Alignment score (1 - normalized disparity)
        alignment_score = 1.0 - disparity

        return CrossSpeciesAlignment(
            alignment_score=alignment_score,
            transformation_matrix=self.T.numpy(),
            species_a_transformed=A_transformed.numpy(),
            species_b_transformed=B_transformed.numpy()
        )

    def transform(
        self,
        X: torch.Tensor,
        source: str = 'a'
    ) -> torch.Tensor:
        """
        Apply transformation.

        Args:
            X: Input representations
            source: 'a' or 'b' to indicate which species

        Returns:
            Transformed representations
        """
        if source == 'a':
            # Apply transformation: scale * (X - mean) @ T
            X_centered = X - self.translation_a
            X_transformed = self.scale * (X_centered @ self.T)
        else:
            # For species B, apply inverse
            X_centered = X - self.translation_b
            X_transformed = X_centered

        return X_transformed


class ConservedSpecificDecomposition:
    """
    Decompose representations into conserved and species-specific components.

    Uses CCA to find shared variance (conserved) and residual
    (species-specific) components.
    """

    def __init__(self, n_conserved: int = 10):
        self.n_conserved = n_conserved

    def decompose(
        self,
        species_a: np.ndarray,
        species_b: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Decompose into conserved and specific components.

        Args:
            species_a: Representations from species A
            species_b: Representations from species B

        Returns:
            Dictionary with conserved and specific components
        """
        from sklearn.cross_decomposition import CCA

        # CCA for conserved components
        cca = CCA(n_components=self.n_conserved)
        a_conserved, b_conserved = cca.fit_transform(species_a, species_b)

        # Reconstruct from conserved
        a_reconstructed = cca.inverse_transform(a_conserved, b_conserved)[0]
        b_reconstructed = cca.inverse_transform(a_conserved, b_conserved)[1]

        # Species-specific is residual
        a_specific = species_a - a_reconstructed
        b_specific = species_b - b_reconstructed

        return {
            'conserved_a': a_conserved,
            'conserved_b': b_conserved,
            'specific_a': a_specific,
            'specific_b': b_specific,
            'canonical_correlations': cca.score(species_a, species_b)
        }


class HomologyMapping:
    """
    Map homologous brain regions across species.

    Uses anatomical atlases and functional alignment to map
    corresponding regions.
    """

    def __init__(
        self,
        atlas_a: Optional[Dict[str, int]] = None,
        atlas_b: Optional[Dict[str, int]] = None
    ):
        """
        Args:
            atlas_a: Region name -> index mapping for species A
            atlas_b: Region name -> index mapping for species B
        """
        self.atlas_a = atlas_a or {}
        self.atlas_b = atlas_b or {}

        # Homology mappings (can be loaded from database)
        self.homology_map = self._initialize_homology_map()

    def _initialize_homology_map(self) -> Dict[str, str]:
        """
        Initialize known homologies.

        Returns:
            Mapping from species A region to species B region
        """
        # Example homologies (human -> mouse)
        return {
            'V1': 'V1',
            'MT': 'MT',
            'IT': 'IT',
            'PFC': 'mPFC',
            'hippocampus': 'hippocampus',
            'amygdala': 'amygdala',
            'striatum': 'striatum',
            'cerebellum': 'cerebellum'
        }

    def get_homologous_regions(
        self,
        region_a: str
    ) -> Optional[str]:
        """
        Get homologous region in species B.

        Args:
            region_a: Region name in species A

        Returns:
            Corresponding region in species B, or None if unknown
        """
        return self.homology_map.get(region_a)

    def align_by_homology(
        self,
        activations_a: Dict[str, np.ndarray],
        activations_b: Dict[str, np.ndarray]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Align activations by known homologies.

        Args:
            activations_a: Region -> activations for species A
            activations_b: Region -> activations for species B

        Returns:
            Dictionary of aligned (region_a_acts, region_b_acts) pairs
        """
        aligned = {}

        for region_a, acts_a in activations_a.items():
            region_b = self.get_homologous_regions(region_a)

            if region_b and region_b in activations_b:
                aligned[region_a] = (acts_a, activations_b[region_b])

        return aligned


class PhylogeneticDistance:
    """
    Compute phylogenetic distance and evolutionary relationships.

    Accounts for evolutionary distance when comparing species.
    """

    def __init__(self):
        # Approximate divergence times (million years ago)
        self.divergence_times = {
            ('human', 'chimpanzee'): 6,
            ('human', 'macaque'): 25,
            ('human', 'mouse'): 90,
            ('human', 'rat'): 90,
            ('human', 'zebrafish'): 435,
            ('human', 'drosophila'): 800,
            ('macaque', 'mouse'): 90,
            ('mouse', 'rat'): 12,
        }

    def get_distance(
        self,
        species_a: str,
        species_b: str
    ) -> float:
        """
        Get phylogenetic distance.

        Args:
            species_a: Species name
            species_b: Species name

        Returns:
            Distance (divergence time in million years)
        """
        key = tuple(sorted([species_a.lower(), species_b.lower()]))

        return self.divergence_times.get(key, float('inf'))

    def weighted_alignment_score(
        self,
        raw_score: float,
        species_a: str,
        species_b: str,
        expected_decay_rate: float = 0.01
    ) -> float:
        """
        Adjust alignment score by phylogenetic distance.

        Closer species should have higher expected alignment.

        Args:
            raw_score: Raw alignment score
            species_a: Species name
            species_b: Species name
            expected_decay_rate: How much alignment decreases with distance

        Returns:
            Normalized score accounting for phylogenetic distance
        """
        distance = self.get_distance(species_a, species_b)

        if distance == float('inf'):
            return raw_score

        # Expected alignment based on phylogenetic distance
        expected_alignment = np.exp(-expected_decay_rate * distance)

        # Normalize
        normalized_score = raw_score / expected_alignment

        return normalized_score


class CrossSpeciesRSA:
    """
    Cross-species Representational Similarity Analysis.

    Compare representational geometries across species.
    """

    def __init__(self, metric: str = 'correlation'):
        self.metric = metric

    def compute_rdm(self, X: np.ndarray) -> np.ndarray:
        """
        Compute representational dissimilarity matrix.

        Args:
            X: Representations (n_stimuli, n_features)

        Returns:
            RDM (n_stimuli, n_stimuli)
        """
        from scipy.spatial.distance import pdist, squareform

        if self.metric == 'correlation':
            rdm = squareform(pdist(X, metric='correlation'))
        elif self.metric == 'euclidean':
            rdm = squareform(pdist(X, metric='euclidean'))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        return rdm

    def compare_rdms(
        self,
        rdm_a: np.ndarray,
        rdm_b: np.ndarray,
        method: str = 'spearman'
    ) -> float:
        """
        Compare two RDMs.

        Args:
            rdm_a: RDM from species A
            rdm_b: RDM from species B
            method: 'spearman', 'pearson', or 'kendall'

        Returns:
            Similarity score
        """
        # Upper triangle (excluding diagonal)
        triu_indices = np.triu_indices_from(rdm_a, k=1)

        vec_a = rdm_a[triu_indices]
        vec_b = rdm_b[triu_indices]

        if method == 'spearman':
            corr, _ = spearmanr(vec_a, vec_b)
        elif method == 'pearson':
            corr = np.corrcoef(vec_a, vec_b)[0, 1]
        elif method == 'kendall':
            from scipy.stats import kendalltau
            corr, _ = kendalltau(vec_a, vec_b)
        else:
            raise ValueError(f"Unknown method: {method}")

        return corr

    def cross_species_rsa(
        self,
        species_a: np.ndarray,
        species_b: np.ndarray,
        method: str = 'spearman'
    ) -> float:
        """
        Perform cross-species RSA.

        Args:
            species_a: Representations from species A
            species_b: Representations from species B
            method: Comparison method

        Returns:
            RSA similarity score
        """
        rdm_a = self.compute_rdm(species_a)
        rdm_b = self.compute_rdm(species_b)

        return self.compare_rdms(rdm_a, rdm_b, method)


class EvolutionaryTrendAnalysis:
    """
    Analyze evolutionary trends in neural representations.

    Identify dimensions that show systematic changes across
    the phylogenetic tree.
    """

    def __init__(self):
        pass

    def fit_evolutionary_trend(
        self,
        species_representations: List[Tuple[str, np.ndarray]],
        phylo_distances: PhylogeneticDistance,
        reference_species: str = 'human'
    ) -> Dict[str, np.ndarray]:
        """
        Fit evolutionary trends.

        Args:
            species_representations: List of (species_name, representations)
            phylo_distances: Phylogenetic distance calculator
            reference_species: Reference species for distance

        Returns:
            Evolutionary trend analysis
        """
        species_names = [s[0] for s in species_representations]
        distances = np.array([
            phylo_distances.get_distance(reference_species, s)
            for s in species_names
        ])

        # Compute alignment scores vs distance
        from .procrustes import ProcrustesAlignment

        reference_idx = species_names.index(reference_species)
        reference_repr = species_representations[reference_idx][1]

        alignment_scores = []
        for name, repr in species_representations:
            if name == reference_species:
                alignment_scores.append(1.0)
            else:
                aligner = ProcrustesAlignment()
                result = aligner.fit(
                    torch.from_numpy(reference_repr),
                    torch.from_numpy(repr)
                )
                alignment_scores.append(result.alignment_score)

        alignment_scores = np.array(alignment_scores)

        # Fit decay model: alignment ~ exp(-rate * distance)
        from scipy.optimize import curve_fit

        def decay(d, rate):
            return np.exp(-rate * d)

        popt, _ = curve_fit(decay, distances, alignment_scores)
        decay_rate = popt[0]

        return {
            'distances': distances,
            'alignment_scores': alignment_scores,
            'decay_rate': decay_rate,
            'species_names': species_names
        }
