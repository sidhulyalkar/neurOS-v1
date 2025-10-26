"""
Hierarchical Sparse Autoencoders with Concept Dictionaries
Extends basic SAE to multi-level concept hierarchies with causal probes
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


@dataclass
class ConceptLabel:
    """Semantic label for an SAE feature"""
    feature_id: int
    label: str
    confidence: float
    evidence: Dict[str, float]  # {probe_name: score}
    modalities: List[str]  # Which modalities activate this
    examples: List[int]  # Sample indices with high activation


class HierarchicalSAE(nn.Module):
    """
    Multi-layer SAE hierarchy for concept discovery

    Architecture:
    Level 0 (bottom): Fine-grained features (512 → 4096)
    Level 1 (mid): Mid-level features (4096 → 16384)
    Level 2 (top): Abstract concepts (16384 → 65536)

    Example:
        >>> hsae = HierarchicalSAE(
        ...     layer_sizes=[512, 4096, 16384],
        ...     sparsity_coefficients=[0.01, 0.005, 0.001]
        ... )
        >>> features_all_levels = hsae(activations)
    """

    def __init__(
        self,
        layer_sizes: List[int],  # e.g., [512, 4096, 16384]
        sparsity_coefficients: List[float],
        tie_weights: bool = True,
    ):
        """
        Args:
            layer_sizes: Dictionary sizes for each level
            sparsity_coefficients: L1 penalty per level
            tie_weights: Tie encoder/decoder weights
        """
        super().__init__()
        self.layer_sizes = layer_sizes
        self.sparsity_coefficients = sparsity_coefficients
        self.num_levels = len(layer_sizes)

        # Create SAE for each level
        self.saes = nn.ModuleList()
        for i in range(self.num_levels - 1):
            input_dim = layer_sizes[i]
            hidden_dim = layer_sizes[i + 1]

            sae = nn.ModuleDict({
                'encoder': nn.Linear(input_dim, hidden_dim),
                'decoder': nn.Linear(hidden_dim, input_dim) if not tie_weights else None,
            })
            self.saes.append(sae)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization with column normalization"""
        for sae in self.saes:
            nn.init.xavier_normal_(sae['encoder'].weight)
            if sae['decoder'] is not None:
                nn.init.xavier_normal_(sae['decoder'].weight)

            # Column normalize encoder
            with torch.no_grad():
                sae['encoder'].weight.data = F.normalize(
                    sae['encoder'].weight.data, dim=0
                )

    def encode_level(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """Encode at specific level"""
        features = F.relu(self.saes[level]['encoder'](x))
        return features

    def decode_level(self, features: torch.Tensor, level: int) -> torch.Tensor:
        """Decode from specific level"""
        if self.saes[level]['decoder'] is not None:
            return self.saes[level]['decoder'](features)
        else:
            # Tied weights: transpose encoder
            return F.linear(features, self.saes[level]['encoder'].weight.t())

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Forward pass through all levels

        Args:
            x: Input activations (B, D)

        Returns:
            Dictionary {level: features} for all levels
        """
        features_all = {}
        current = x

        for level in range(self.num_levels - 1):
            # Encode
            features = self.encode_level(current, level)
            features_all[level] = features

            # Use features as input to next level
            current = features

        return features_all

    def reconstruct_from_level(
        self,
        features: torch.Tensor,
        level: int,
        reconstruct_to_input: bool = True
    ) -> torch.Tensor:
        """
        Reconstruct input from features at given level

        Args:
            features: Features at specified level
            level: Which level features are from
            reconstruct_to_input: If True, decode all the way to input

        Returns:
            Reconstruction
        """
        reconstruction = features

        # Decode from this level down to input
        for l in range(level, -1, -1):
            reconstruction = self.decode_level(reconstruction, l)

        return reconstruction

    def compute_loss(
        self,
        x: torch.Tensor,
        features_all: Dict[int, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hierarchical reconstruction + sparsity loss

        Args:
            x: Original input
            features_all: Features from forward pass

        Returns:
            Dictionary of losses
        """
        losses = {}
        total_loss = 0.0

        # Per-level reconstruction + sparsity
        for level in range(self.num_levels - 1):
            features = features_all[level]

            # Reconstruct input
            if level == 0:
                target = x
            else:
                # Reconstruct features from previous level
                target = features_all[level - 1]

            reconstruction = self.decode_level(features, level)

            # Reconstruction loss
            recon_loss = F.mse_loss(reconstruction, target)

            # Sparsity loss (L1)
            sparsity_loss = self.sparsity_coefficients[level] * features.abs().mean()

            # Combined
            level_loss = recon_loss + sparsity_loss
            total_loss += level_loss

            losses[f'level_{level}_recon'] = recon_loss
            losses[f'level_{level}_sparsity'] = sparsity_loss
            losses[f'level_{level}_total'] = level_loss

        losses['total'] = total_loss
        return losses

    def get_concept_tree(self) -> Dict[int, List[Tuple[int, float]]]:
        """
        Build hierarchical concept tree

        Links low-level features to high-level concepts via decoder weights

        Returns:
            Dictionary {parent_feature: [(child_feature, weight), ...]}
        """
        tree = {}

        for level in range(self.num_levels - 2):
            # Get decoder weights: (input_dim, hidden_dim)
            if self.saes[level + 1]['decoder'] is not None:
                weights = self.saes[level + 1]['decoder'].weight.data
            else:
                weights = self.saes[level + 1]['encoder'].weight.data.t()

            # For each high-level feature (parent)
            for parent_idx in range(weights.shape[0]):
                # Find top-k contributing low-level features (children)
                parent_weights = weights[parent_idx]
                top_k = 10
                top_indices = torch.topk(parent_weights.abs(), k=min(top_k, len(parent_weights)))

                children = [
                    (int(idx), float(parent_weights[idx]))
                    for idx in top_indices.indices
                ]

                tree[(level + 1, parent_idx)] = children

        return tree


class ConceptDictionary:
    """
    Feature dictionary with semantic labels

    Labels features based on linear probes and modality attribution
    """

    def __init__(self, hsae: HierarchicalSAE):
        self.hsae = hsae
        self.labels: Dict[Tuple[int, int], ConceptLabel] = {}  # (level, feature_id)

    def build_dictionary(
        self,
        activations: torch.Tensor,
        probe_labels: Dict[str, torch.Tensor],
        modality_data: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Assign semantic labels to features using probes

        Args:
            activations: Model activations (N, D)
            probe_labels: Dictionary of labels for probing
                e.g., {'region': region_labels, 'behavior': behavior_labels}
            modality_data: Optional modality data for attribution
        """
        # Get features at all levels
        with torch.no_grad():
            features_all = self.hsae(activations)

        # For each level
        for level, features in features_all.items():
            # For each feature dimension
            for feat_id in range(features.shape[1]):
                feat_activations = features[:, feat_id]  # (N,)

                # Train probes to predict labels from this feature
                probe_scores = {}
                for probe_name, labels in probe_labels.items():
                    score = self._probe_feature(feat_activations, labels)
                    probe_scores[probe_name] = score

                # Find best probe
                best_probe = max(probe_scores.items(), key=lambda x: x[1])

                # Get modality attribution
                if modality_data is not None:
                    modalities = self._attribute_to_modalities(
                        feat_activations, modality_data
                    )
                else:
                    modalities = []

                # Find top-activating examples
                top_examples = torch.topk(feat_activations, k=10).indices.tolist()

                # Create label
                label = ConceptLabel(
                    feature_id=feat_id,
                    label=f"{best_probe[0]}_{feat_id}",
                    confidence=best_probe[1],
                    evidence=probe_scores,
                    modalities=modalities,
                    examples=top_examples
                )

                self.labels[(level, feat_id)] = label

    def _probe_feature(
        self,
        feature_activations: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        Train linear probe to predict labels from feature activations

        Returns accuracy/R² score
        """
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.model_selection import cross_val_score

        X = feature_activations.cpu().numpy().reshape(-1, 1)
        y = labels.cpu().numpy()

        # Determine if classification or regression
        if len(np.unique(y)) < 10:
            # Classification
            probe = LogisticRegression(max_iter=1000)
            scores = cross_val_score(probe, X, y, cv=5, scoring='accuracy')
        else:
            # Regression
            probe = Ridge()
            scores = cross_val_score(probe, X, y, cv=5, scoring='r2')

        return float(scores.mean())

    def _attribute_to_modalities(
        self,
        feature_activations: torch.Tensor,
        modality_data: Dict[str, torch.Tensor]
    ) -> List[str]:
        """
        Determine which modalities drive this feature

        Returns list of modality names sorted by correlation
        """
        correlations = {}

        for modality_name, modality_tensor in modality_data.items():
            # Compute correlation
            corr = torch.corrcoef(
                torch.stack([
                    feature_activations,
                    modality_tensor.mean(dim=-1)  # Average over channels
                ])
            )[0, 1]

            correlations[modality_name] = float(corr.abs())

        # Sort by correlation
        sorted_modalities = sorted(
            correlations.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top modalities with corr > 0.1
        return [m for m, c in sorted_modalities if c > 0.1]


class CausalSAEProbe:
    """
    Causal interventions using SAE features

    Reinsert or ablate features to measure causal effects
    """

    def __init__(self, hsae: HierarchicalSAE, model: nn.Module):
        self.hsae = hsae
        self.model = model
        self.hooks = []

    def reinsert_feature(
        self,
        input_data: torch.Tensor,
        layer_name: str,
        level: int,
        feature_id: int,
        magnitude: float = 1.0
    ) -> torch.Tensor:
        """
        Reinsert specific feature at given magnitude

        Args:
            input_data: Model input
            layer_name: Which layer to intervene on
            level: SAE level
            feature_id: Feature dimension
            magnitude: Activation magnitude (multiplier)

        Returns:
            Model output with feature reinserted
        """
        intervention_applied = False

        def hook_fn(module, input, output):
            nonlocal intervention_applied
            if intervention_applied:
                return output

            # Get original activations
            original = output

            # Encode with SAE
            features = self.hsae.encode_level(original, level)

            # Modify specific feature
            features[:, feature_id] *= magnitude

            # Decode back
            modified = self.hsae.decode_level(features, level)

            intervention_applied = True
            return modified

        # Register hook
        target_module = dict(self.model.named_modules())[layer_name]
        handle = target_module.register_forward_hook(hook_fn)
        self.hooks.append(handle)

        # Forward pass
        with torch.no_grad():
            output = self.model(input_data)

        # Remove hook
        handle.remove()
        self.hooks.remove(handle)

        return output

    def causal_importance_score(
        self,
        input_data: torch.Tensor,
        target: torch.Tensor,
        layer_name: str,
        level: int,
        metric: Callable[[torch.Tensor, torch.Tensor], float],
        features_to_test: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """
        Rank features by causal importance via ablation

        Args:
            input_data: Model inputs
            target: Target outputs for metric
            layer_name: Layer to intervene on
            level: SAE level
            metric: Function to compute performance (output, target) -> score
            features_to_test: Specific features to test (None = all)

        Returns:
            Dictionary {feature_id: importance_score}
        """
        # Baseline score (no ablation)
        with torch.no_grad():
            baseline_output = self.model(input_data)
        baseline_score = metric(baseline_output, target)

        # Get number of features at this level
        num_features = self.hsae.layer_sizes[level + 1]

        if features_to_test is None:
            features_to_test = range(num_features)

        importance_scores = {}

        # Test each feature
        for feat_id in tqdm(features_to_test, desc="Computing causal importance"):
            # Ablate feature (set to 0)
            ablated_output = self.reinsert_feature(
                input_data, layer_name, level, feat_id, magnitude=0.0
            )

            # Measure performance drop
            ablated_score = metric(ablated_output, target)

            # Importance = performance drop
            importance = baseline_score - ablated_score
            importance_scores[feat_id] = float(importance)

        return importance_scores


# Example usage
if __name__ == "__main__":
    print("Hierarchical SAE Concept Discovery")
    print("=" * 80)

    # Create hierarchical SAE
    hsae = HierarchicalSAE(
        layer_sizes=[512, 4096, 16384],
        sparsity_coefficients=[0.01, 0.005, 0.001]
    )

    # Synthetic activations
    batch_size = 100
    activations = torch.randn(batch_size, 512)

    # Forward pass
    features_all = hsae(activations)

    print(f"\nHierarchical features:")
    for level, features in features_all.items():
        print(f"  Level {level}: {features.shape}")

    # Compute loss
    losses = hsae.compute_loss(activations, features_all)
    print(f"\nLosses:")
    for name, loss in losses.items():
        print(f"  {name}: {loss.item():.4f}")

    # Build concept tree
    tree = hsae.get_concept_tree()
    print(f"\nConcept tree: {len(tree)} parent nodes")

    # Build dictionary
    dictionary = ConceptDictionary(hsae)
    probe_labels = {
        'region': torch.randint(0, 5, (batch_size,)),
        'behavior': torch.rand(batch_size)
    }

    dictionary.build_dictionary(activations, probe_labels)
    print(f"\nLabeled {len(dictionary.labels)} features across all levels")

    # Show sample labels
    for (level, feat_id), label in list(dictionary.labels.items())[:5]:
        print(f"\n  Level {level}, Feature {feat_id}:")
        print(f"    Label: {label.label}")
        print(f"    Confidence: {label.confidence:.3f}")
        print(f"    Evidence: {label.evidence}")
