"""
Meta-Dynamics: Training-time Representational Trajectories
Track how representations evolve during training and detect feature emergence
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class TrainingPhase:
    """A distinct phase in training dynamics"""
    name: str  # e.g., "warmup", "fitting", "compression", "saturation"
    start_step: int
    end_step: int
    characteristics: Dict[str, float]  # Metrics defining this phase


class RepresentationalTrajectory:
    """
    Track how representations evolve during training

    Loads checkpoints at different training steps and compares
    representations to measure drift, feature emergence, etc.
    """

    def __init__(self, checkpoint_dir: str):
        """
        Args:
            checkpoint_dir: Directory containing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoints = self._discover_checkpoints()

    def _discover_checkpoints(self) -> List[Tuple[int, Path]]:
        """
        Discover all checkpoints and extract training step

        Returns:
            List of (step, path) tuples sorted by step
        """
        checkpoints = []

        for ckpt_path in self.checkpoint_dir.glob("checkpoint_step_*.pt"):
            # Extract step from filename
            step_str = ckpt_path.stem.split("_")[-1]
            try:
                step = int(step_str)
                checkpoints.append((step, ckpt_path))
            except ValueError:
                continue

        # Sort by step
        checkpoints.sort(key=lambda x: x[0])

        print(f"Found {len(checkpoints)} checkpoints from step {checkpoints[0][0]} to {checkpoints[-1][0]}")

        return checkpoints

    def compute_trajectory(
        self,
        model_class: Callable,
        dataset: torch.utils.data.DataLoader,
        layers: List[str],
        max_samples: int = 1000
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Extract representations at each checkpoint

        Args:
            model_class: Function to instantiate model
            dataset: Data to extract representations on
            layers: Which layers to track
            max_samples: Maximum samples to process

        Returns:
            Dictionary {layer_name: [repr_step0, repr_step1000, ...]}
        """
        trajectory = {layer: [] for layer in layers}

        # For each checkpoint
        for step, ckpt_path in tqdm(self.checkpoints, desc="Computing trajectory"):
            # Load checkpoint
            model = model_class()
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            # Extract representations
            layer_reprs = self._extract_representations(
                model, dataset, layers, max_samples
            )

            # Store
            for layer, repr_tensor in layer_reprs.items():
                trajectory[layer].append(repr_tensor)

        return trajectory

    def _extract_representations(
        self,
        model: nn.Module,
        dataset: torch.utils.data.DataLoader,
        layers: List[str],
        max_samples: int
    ) -> Dict[str, torch.Tensor]:
        """Extract representations from specified layers"""
        representations = {layer: [] for layer in layers}

        # Register hooks
        handles = []

        def make_hook(layer_name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                representations[layer_name].append(output.detach().cpu())
            return hook_fn

        for layer_name in layers:
            module = dict(model.named_modules())[layer_name]
            handle = module.register_forward_hook(make_hook(layer_name))
            handles.append(handle)

        # Forward passes
        num_samples = 0
        with torch.no_grad():
            for batch in dataset:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch

                model(inputs)

                num_samples += inputs.size(0)
                if num_samples >= max_samples:
                    break

        # Remove hooks
        for handle in handles:
            handle.remove()

        # Concatenate
        for layer in layers:
            if representations[layer]:
                representations[layer] = torch.cat(representations[layer], dim=0)
                # Flatten (N, ...) to (N, D)
                representations[layer] = representations[layer].view(
                    representations[layer].size(0), -1
                )

        return representations

    def measure_drift(
        self,
        trajectory: List[torch.Tensor],
        metric: str = 'cca'
    ) -> np.ndarray:
        """
        Measure representational drift over training

        Args:
            trajectory: List of representation tensors
            metric: Similarity metric ('cca', 'rsa', 'procrustes')

        Returns:
            Array of drift scores between consecutive checkpoints
        """
        drift_scores = []

        for i in range(len(trajectory) - 1):
            repr1 = trajectory[i]
            repr2 = trajectory[i + 1]

            if metric == 'cca':
                score = self._cca_similarity(repr1, repr2)
            elif metric == 'rsa':
                score = self._rsa_similarity(repr1, repr2)
            elif metric == 'procrustes':
                score = self._procrustes_similarity(repr1, repr2)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            drift_scores.append(score)

        return np.array(drift_scores)

    def _cca_similarity(self, X1: torch.Tensor, X2: torch.Tensor) -> float:
        """CCA similarity (mean canonical correlation)"""
        from sklearn.cross_decomposition import CCA

        n_components = min(10, X1.shape[1], X2.shape[1])

        cca = CCA(n_components=n_components)
        cca.fit(X1.numpy(), X2.numpy())

        X1_c, X2_c = cca.transform(X1.numpy(), X2.numpy())

        # Mean correlation across components
        correlations = [
            np.corrcoef(X1_c[:, i], X2_c[:, i])[0, 1]
            for i in range(n_components)
        ]

        return float(np.mean(correlations))

    def _rsa_similarity(self, X1: torch.Tensor, X2: torch.Tensor) -> float:
        """RSA similarity (correlation of RDMs)"""
        from scipy.stats import spearmanr

        # Compute RDMs
        rdm1 = torch.cdist(X1, X1).numpy().flatten()
        rdm2 = torch.cdist(X2, X2).numpy().flatten()

        # Spearman correlation
        corr, _ = spearmanr(rdm1, rdm2)

        return float(corr)

    def _procrustes_similarity(self, X1: torch.Tensor, X2: torch.Tensor) -> float:
        """Procrustes similarity (1 - normalized distance)"""
        from scipy.spatial import procrustes

        _, _, disparity = procrustes(X1.numpy(), X2.numpy())

        return float(1.0 - disparity)

    def detect_emergence(
        self,
        trajectory: List[torch.Tensor],
        threshold: float = 0.1
    ) -> List[Tuple[int, int]]:
        """
        Detect when features "emerge" during training

        A feature emerges when its variance suddenly increases

        Args:
            trajectory: List of representations
            threshold: Variance increase threshold

        Returns:
            List of (step_index, feature_dim) for emerged features
        """
        emerged_features = []

        # For each consecutive pair
        for i in range(len(trajectory) - 1):
            repr_before = trajectory[i]
            repr_after = trajectory[i + 1]

            # Compute variance per dimension
            var_before = repr_before.var(dim=0)
            var_after = repr_after.var(dim=0)

            # Find dimensions with large variance increase
            var_increase = (var_after - var_before) / (var_before + 1e-8)

            emerged_dims = torch.where(var_increase > threshold)[0]

            for dim in emerged_dims:
                emerged_features.append((i + 1, int(dim)))

        return emerged_features


class GradientAttribution:
    """
    Attribute training dynamics to specific parameters

    Track gradient flow to see which parameters/circuits consolidate
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoints = self._discover_checkpoints()

    def _discover_checkpoints(self) -> List[Tuple[int, Path]]:
        """Same as RepresentationalTrajectory"""
        checkpoints = []
        for ckpt_path in self.checkpoint_dir.glob("checkpoint_step_*.pt"):
            step_str = ckpt_path.stem.split("_")[-1]
            try:
                step = int(step_str)
                checkpoints.append((step, ckpt_path))
            except ValueError:
                continue
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints

    def gradient_flow_over_time(
        self,
        parameter_groups: Dict[str, List[str]]
    ) -> Dict[str, np.ndarray]:
        """
        Track gradient magnitudes over training

        Args:
            parameter_groups: Dictionary {group_name: [param_names]}

        Returns:
            Dictionary {group_name: gradient_norms_over_time}
        """
        gradient_flow = {group: [] for group in parameter_groups}

        # For each checkpoint
        for step, ckpt_path in self.checkpoints:
            checkpoint = torch.load(ckpt_path, map_location='cpu')

            # Get optimizer state (contains historical gradients)
            optimizer_state = checkpoint.get('optimizer_state_dict', {})

            # Compute gradient norms per group
            for group_name, param_names in parameter_groups.items():
                total_norm = 0.0

                for param_name in param_names:
                    # Get parameter gradient norm from optimizer state
                    if param_name in optimizer_state.get('state', {}):
                        param_state = optimizer_state['state'][param_name]

                        # Check for momentum buffer or gradient
                        if 'momentum_buffer' in param_state:
                            grad = param_state['momentum_buffer']
                        elif 'exp_avg' in param_state:  # Adam
                            grad = param_state['exp_avg']
                        else:
                            continue

                        total_norm += grad.norm().item() ** 2

                gradient_flow[group_name].append(np.sqrt(total_norm))

        # Convert to arrays
        for group in gradient_flow:
            gradient_flow[group] = np.array(gradient_flow[group])

        return gradient_flow

    def plasticity_tracking(
        self,
        layer_names: List[str]
    ) -> pd.DataFrame:
        """
        Track per-layer plasticity (weight change rate)

        Args:
            layer_names: Layers to track

        Returns:
            DataFrame with columns: step, layer, weight_change, gradient_norm
        """
        records = []

        # For consecutive checkpoint pairs
        for i in range(len(self.checkpoints) - 1):
            step1, path1 = self.checkpoints[i]
            step2, path2 = self.checkpoints[i + 1]

            ckpt1 = torch.load(path1, map_location='cpu')
            ckpt2 = torch.load(path2, map_location='cpu')

            state1 = ckpt1['model_state_dict']
            state2 = ckpt2['model_state_dict']

            # For each layer
            for layer_name in layer_names:
                # Find parameters for this layer
                layer_params = [k for k in state1.keys() if layer_name in k]

                if not layer_params:
                    continue

                # Compute weight change
                total_change = 0.0
                total_norm = 0.0

                for param_name in layer_params:
                    if param_name in state1 and param_name in state2:
                        w1 = state1[param_name]
                        w2 = state2[param_name]

                        change = (w2 - w1).norm().item()
                        norm = w1.norm().item()

                        total_change += change ** 2
                        total_norm += norm ** 2

                total_change = np.sqrt(total_change)
                total_norm = np.sqrt(total_norm)

                # Relative change
                relative_change = total_change / (total_norm + 1e-8)

                records.append({
                    'step': step2,
                    'layer': layer_name,
                    'weight_change': total_change,
                    'weight_norm': total_norm,
                    'relative_change': relative_change
                })

        return pd.DataFrame(records)


class TrainingPhaseDetection:
    """
    Detect distinct phases in training dynamics

    Phases: warmup, fitting, compression, saturation
    """

    def detect_phases(
        self,
        loss_curve: np.ndarray,
        drift_curve: Optional[np.ndarray] = None,
        learning_rate_curve: Optional[np.ndarray] = None
    ) -> List[TrainingPhase]:
        """
        Identify distinct training phases

        Args:
            loss_curve: Training loss over time
            drift_curve: Representational drift over time
            learning_rate_curve: Learning rate schedule

        Returns:
            List of detected phases
        """
        phases = []

        # Compute loss derivatives
        loss_derivative = np.gradient(loss_curve)
        loss_second_derivative = np.gradient(loss_derivative)

        # Phase 1: Warmup (loss decreasing rapidly)
        warmup_end = self._find_warmup_end(loss_derivative)
        if warmup_end > 0:
            phases.append(TrainingPhase(
                name="warmup",
                start_step=0,
                end_step=warmup_end,
                characteristics={
                    'loss_slope': float(loss_derivative[:warmup_end].mean()),
                    'loss_acceleration': float(loss_second_derivative[:warmup_end].mean())
                }
            ))

        # Phase 2: Fitting (loss decreasing steadily)
        fitting_start = warmup_end
        fitting_end = self._find_fitting_end(loss_curve, fitting_start)
        if fitting_end > fitting_start:
            phases.append(TrainingPhase(
                name="fitting",
                start_step=fitting_start,
                end_step=fitting_end,
                characteristics={
                    'loss_slope': float(loss_derivative[fitting_start:fitting_end].mean()),
                }
            ))

        # Phase 3: Compression (if drift decreases while loss plateaus)
        if drift_curve is not None:
            compression_start = fitting_end
            compression_end = self._find_compression_end(drift_curve, compression_start)

            if compression_end > compression_start:
                phases.append(TrainingPhase(
                    name="compression",
                    start_step=compression_start,
                    end_step=compression_end,
                    characteristics={
                        'drift': float(drift_curve[compression_start:compression_end].mean()),
                        'loss_slope': float(loss_derivative[compression_start:compression_end].mean())
                    }
                ))
        else:
            compression_end = fitting_end

        # Phase 4: Saturation (loss flat)
        if compression_end < len(loss_curve) - 1:
            phases.append(TrainingPhase(
                name="saturation",
                start_step=compression_end,
                end_step=len(loss_curve) - 1,
                characteristics={
                    'loss_slope': float(loss_derivative[compression_end:].mean()),
                    'loss_std': float(loss_curve[compression_end:].std())
                }
            ))

        return phases

    def _find_warmup_end(self, loss_derivative: np.ndarray, threshold: float = -0.01) -> int:
        """Find where loss derivative starts to stabilize"""
        # Warmup ends when derivative becomes less negative
        for i in range(10, len(loss_derivative)):
            if loss_derivative[i] > threshold:
                return i
        return 10

    def _find_fitting_end(self, loss_curve: np.ndarray, start: int) -> int:
        """Find where loss starts to plateau"""
        # Use sliding window to detect plateau
        window = 20
        plateau_threshold = 0.001  # Max change per step

        for i in range(start + window, len(loss_curve) - window):
            window_change = abs(loss_curve[i + window] - loss_curve[i]) / window
            if window_change < plateau_threshold:
                return i

        return len(loss_curve) - 1

    def _find_compression_end(self, drift_curve: np.ndarray, start: int) -> int:
        """Find where drift stabilizes"""
        # Compression ends when drift stops decreasing
        drift_derivative = np.gradient(drift_curve)

        for i in range(start + 10, len(drift_curve) - 10):
            if drift_derivative[i] > 0:  # Drift starts increasing
                return i

        return len(drift_curve) - 1


# Example usage
if __name__ == "__main__":
    print("Meta-Dynamics: Training Trajectory Analysis")
    print("=" * 80)

    # Simulate loss curve
    steps = np.arange(1000)
    loss_curve = 2.0 * np.exp(-steps / 100) + 0.1 * np.random.randn(1000) * 0.01 + 0.1

    # Simulate drift curve
    drift_curve = 0.9 * np.exp(-steps / 200) + 0.1

    # Detect phases
    detector = TrainingPhaseDetection()
    phases = detector.detect_phases(loss_curve, drift_curve)

    print(f"\nDetected {len(phases)} training phases:")
    for phase in phases:
        print(f"\n  {phase.name.upper()}")
        print(f"    Steps: {phase.start_step} - {phase.end_step}")
        print(f"    Characteristics: {phase.characteristics}")
