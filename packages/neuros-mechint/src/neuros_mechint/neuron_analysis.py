"""
Neuron Activation Analysis

Tools for analyzing individual neurons in the model's latent space.
Identifies selective neurons, computes tuning curves, and maps neuron-to-behavior relationships.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
from tqdm import tqdm


class NeuronActivationAnalyzer:
    """
    Analyzes neuron activations and their relationships to external variables.

    Methods for:
    - Computing selectivity indices
    - Finding behavior-predictive neurons
    - Generating tuning curves
    - Identifying monosemantic vs polysemantic units
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Args:
            model: NeuroFMx model to analyze
            device: Device for computation
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Storage for activations
        self.activations = {}
        self.hooks = []

        # Register hooks on key layers
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to cache activations."""

        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook

        # Hook the latent layer (after PopT)
        if hasattr(self.model, 'popt'):
            handle = self.model.popt.register_forward_hook(get_activation('latents'))
            self.hooks.append(handle)

        # Hook Mamba layers
        if hasattr(self.model, 'mamba_backbone'):
            for i, layer in enumerate(self.model.mamba_backbone.layers):
                handle = layer.register_forward_hook(get_activation(f'mamba_layer_{i}'))
                self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_neuron_selectivity(
        self,
        dataset: torch.utils.data.DataLoader,
        neuron_id: int,
        variable_name: str = 'stimulus',
        layer_name: str = 'latents'
    ) -> Tuple[float, Dict]:
        """
        Compute how selectively a neuron responds to a variable.

        Args:
            dataset: DataLoader yielding batches with modality_dict and labels
            neuron_id: ID of neuron to analyze
            variable_name: Name of variable in batch (e.g., 'stimulus', 'behavior')
            layer_name: Which layer to analyze

        Returns:
            selectivity_index: Float in [0, 1], higher = more selective
            tuning_curve: Dict mapping variable_value -> mean_activation
        """
        activations_by_value = defaultdict(list)

        print(f"Computing selectivity for neuron {neuron_id}...")

        with torch.no_grad():
            for batch in tqdm(dataset, desc="Processing batches"):
                # Forward pass
                modality_dict = batch['inputs']

                # Move to device
                for k in modality_dict:
                    modality_dict[k] = modality_dict[k].to(self.device)

                outputs = self.model(modality_dict)

                # Get activations
                if layer_name not in self.activations:
                    print(f"Warning: Layer {layer_name} not found in activations")
                    continue

                layer_act = self.activations[layer_name]  # (B, n_latents, D) or (B, T, D)

                # Extract neuron activation
                if len(layer_act.shape) == 3:
                    # Take mean over sequence/latent dimension
                    neuron_act = layer_act[:, :, neuron_id].mean(dim=1)  # (B,)
                else:
                    neuron_act = layer_act[:, neuron_id]  # (B,)

                # Get variable values
                if variable_name in batch:
                    variable_values = batch[variable_name].cpu().numpy()

                    for i, val in enumerate(variable_values):
                        activations_by_value[val.item()].append(neuron_act[i].cpu().item())

        # Compute tuning curve
        tuning_curve = {}
        for val, acts in activations_by_value.items():
            tuning_curve[val] = float(np.mean(acts))

        # Compute selectivity index
        if len(tuning_curve) > 0:
            activations = list(tuning_curve.values())
            max_act = max(activations)
            mean_act = np.mean(activations)

            # Selectivity = (max - mean) / (max + epsilon)
            selectivity = (max_act - mean_act) / (max_act + 1e-8)
        else:
            selectivity = 0.0

        return selectivity, tuning_curve

    def find_behavior_predictive_neurons(
        self,
        dataset: torch.utils.data.DataLoader,
        behavior_name: str = 'behavior',
        top_k: int = 20,
        layer_name: str = 'latents'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify neurons most predictive of a behavior.

        Uses mutual information between neuron activation and behavior.

        Args:
            dataset: DataLoader
            behavior_name: Name of behavior variable in batch
            top_k: Number of top neurons to return
            layer_name: Which layer to analyze

        Returns:
            top_neurons: Indices of top-k neurons
            mi_scores: Mutual information scores for top neurons
        """
        from sklearn.feature_selection import mutual_info_regression

        all_activations = []
        all_behaviors = []

        print(f"Collecting activations and behaviors...")

        with torch.no_grad():
            for batch in tqdm(dataset, desc="Processing batches"):
                modality_dict = batch['inputs']

                for k in modality_dict:
                    modality_dict[k] = modality_dict[k].to(self.device)

                outputs = self.model(modality_dict)

                if layer_name in self.activations:
                    layer_act = self.activations[layer_name]  # (B, n_latents, D)

                    # Flatten: (B, n_latents*D)
                    if len(layer_act.shape) == 3:
                        layer_act_flat = layer_act.reshape(layer_act.shape[0], -1)
                    else:
                        layer_act_flat = layer_act

                    all_activations.append(layer_act_flat.cpu().numpy())

                    # Get behavior
                    if behavior_name in batch:
                        behavior = batch[behavior_name].cpu().numpy()

                        # If behavior is multi-dimensional, take first dim
                        if len(behavior.shape) > 1:
                            behavior = behavior[:, 0]

                        all_behaviors.append(behavior)

        # Stack all data
        all_activations = np.vstack(all_activations)  # (N, n_neurons)
        all_behaviors = np.concatenate(all_behaviors)  # (N,)

        print(f"Computing mutual information for {all_activations.shape[1]} neurons...")

        # Compute MI for each neuron
        mi_scores = mutual_info_regression(all_activations, all_behaviors, random_state=42)

        # Get top-k
        top_indices = np.argsort(mi_scores)[-top_k:][::-1]

        print(f"Top {top_k} predictive neurons found")

        return top_indices, mi_scores[top_indices]

    def analyze_population_geometry(
        self,
        dataset: torch.utils.data.DataLoader,
        condition_name: str = 'condition',
        layer_name: str = 'latents',
        n_samples: int = 1000
    ) -> Dict:
        """
        Analyze the geometric structure of population activity.

        Computes:
        - Dimensionality (participation ratio)
        - Clustering quality (silhouette score)
        - Inter-condition distances

        Args:
            dataset: DataLoader
            condition_name: Variable defining conditions
            layer_name: Layer to analyze
            n_samples: Max samples to analyze

        Returns:
            metrics: Dict of geometry metrics
        """
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score

        all_activations = []
        all_conditions = []
        n_collected = 0

        print("Collecting activations for geometry analysis...")

        with torch.no_grad():
            for batch in tqdm(dataset, desc="Processing"):
                if n_collected >= n_samples:
                    break

                modality_dict = batch['inputs']
                for k in modality_dict:
                    modality_dict[k] = modality_dict[k].to(self.device)

                outputs = self.model(modality_dict)

                if layer_name in self.activations:
                    layer_act = self.activations[layer_name]

                    # Pool over sequence
                    if len(layer_act.shape) == 3:
                        layer_act = layer_act.mean(dim=1)  # (B, D)

                    all_activations.append(layer_act.cpu().numpy())

                    if condition_name in batch:
                        all_conditions.append(batch[condition_name].cpu().numpy())

                    n_collected += layer_act.shape[0]

        # Stack
        all_activations = np.vstack(all_activations)[:n_samples]
        all_conditions = np.concatenate(all_conditions)[:n_samples]

        # Compute dimensionality (participation ratio)
        pca = PCA()
        pca.fit(all_activations)
        explained_var = pca.explained_variance_

        # Participation ratio: (sum eigenvalues)^2 / sum(eigenvalues^2)
        pr = np.sum(explained_var)**2 / np.sum(explained_var**2)

        # Clustering quality
        if len(np.unique(all_conditions)) > 1:
            sil_score = silhouette_score(all_activations, all_conditions)
        else:
            sil_score = 0.0

        # Inter-condition distances
        unique_conditions = np.unique(all_conditions)
        condition_means = {}
        for cond in unique_conditions:
            mask = all_conditions == cond
            condition_means[cond] = all_activations[mask].mean(axis=0)

        # Pairwise distances
        distances = {}
        for i, cond1 in enumerate(unique_conditions):
            for cond2 in unique_conditions[i+1:]:
                dist = np.linalg.norm(condition_means[cond1] - condition_means[cond2])
                distances[f"{cond1}_vs_{cond2}"] = float(dist)

        metrics = {
            'participation_ratio': float(pr),
            'n_dimensions': all_activations.shape[1],
            'effective_dimensionality': float(pr),
            'silhouette_score': float(sil_score),
            'condition_distances': distances,
            'explained_variance_90': int(np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.9))
        }

        return metrics

    def find_monosemantic_neurons(
        self,
        dataset: torch.utils.data.DataLoader,
        variables: List[str],
        selectivity_threshold: float = 0.7,
        layer_name: str = 'latents'
    ) -> Dict[int, Dict]:
        """
        Find neurons that respond selectively to single concepts (monosemantic).

        Args:
            dataset: DataLoader
            variables: List of variable names to test selectivity for
            selectivity_threshold: Minimum selectivity to be considered monosemantic
            layer_name: Layer to analyze

        Returns:
            monosemantic_neurons: Dict mapping neuron_id -> {variable, selectivity}
        """
        # Get number of neurons
        sample_batch = next(iter(dataset))
        with torch.no_grad():
            modality_dict = sample_batch['inputs']
            for k in modality_dict:
                modality_dict[k] = modality_dict[k].to(self.device)
            self.model(modality_dict)

        if layer_name not in self.activations:
            raise ValueError(f"Layer {layer_name} not found")

        layer_act = self.activations[layer_name]
        if len(layer_act.shape) == 3:
            n_neurons = layer_act.shape[2]
        else:
            n_neurons = layer_act.shape[1]

        print(f"Analyzing {n_neurons} neurons for monosemanticity...")

        monosemantic_neurons = {}

        for neuron_id in tqdm(range(min(n_neurons, 100)), desc="Neurons"):  # Limit to 100 for speed
            selectivities = {}

            for var in variables:
                if var not in sample_batch:
                    continue

                selectivity, _ = self.compute_neuron_selectivity(
                    dataset, neuron_id, var, layer_name
                )
                selectivities[var] = selectivity

            # Check if monosemantic (selective for one variable)
            if len(selectivities) > 0:
                max_var = max(selectivities, key=selectivities.get)
                max_selectivity = selectivities[max_var]

                if max_selectivity > selectivity_threshold:
                    # Check that it's not also selective for others
                    other_selectivities = [s for v, s in selectivities.items() if v != max_var]

                    if len(other_selectivities) == 0 or max(other_selectivities) < 0.5:
                        monosemantic_neurons[neuron_id] = {
                            'variable': max_var,
                            'selectivity': max_selectivity,
                            'all_selectivities': selectivities
                        }

        print(f"Found {len(monosemantic_neurons)} monosemantic neurons")

        return monosemantic_neurons
