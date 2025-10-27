"""
Fractal Probes for NeuroFMX

Real-time fractal analysis during training and inference.
Monitor fractal properties of learned representations, probe attention-fractal coupling,
and perform causal scale ablation experiments.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional, List, Tuple, Any
import numpy as np
import logging

from .metrics import HiguchiFractalDimension, SpectralSlope, GraphFractalDimension

logger = logging.getLogger(__name__)


class LatentFDTracker:
    """
    Track fractal dimension of latent representations over training.

    Monitors the evolution of fractal properties in specified layers throughout
    training, providing insights into how model complexity changes.

    Args:
        layers_to_track: List of layer names to monitor
        metric_type: Type of fractal metric ('higuchi', 'spectral', 'graph')
        compute_interval: Compute metrics every N steps (default: 100)
        device: Torch device

    Example:
        >>> tracker = LatentFDTracker(['layer_0', 'layer_6', 'layer_11'])
        >>> # During training:
        >>> metrics = tracker.compute(activations, step=500)
        >>> print(f"Layer 6 FD: {metrics['layer_6']:.3f}")
    """

    def __init__(
        self,
        layers_to_track: List[str],
        metric_type: str = 'higuchi',
        compute_interval: int = 100,
        device: Optional[str] = None,
    ):
        self.layers_to_track = layers_to_track
        self.metric_type = metric_type
        self.compute_interval = compute_interval
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize metric estimator
        if metric_type == 'higuchi':
            self.estimator = HiguchiFractalDimension(device=self.device)
        elif metric_type == 'spectral':
            self.estimator = SpectralSlope(device=self.device)
        elif metric_type == 'graph':
            self.estimator = GraphFractalDimension(device=self.device)
        else:
            raise ValueError(f"Unknown metric_type: {metric_type}")

        # History: {layer_name: [(step, value), ...]}
        self.history: Dict[str, List[Tuple[int, float]]] = {
            layer: [] for layer in layers_to_track
        }

    def compute(
        self,
        activations: Dict[str, Tensor],
        step: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute fractal dimension for tracked layers.

        Args:
            activations: Dictionary mapping layer names to activation tensors
                        Expected shape: [batch, seq_len, features] or [batch, seq_len]
            step: Current training step (for history tracking)

        Returns:
            Dictionary mapping layer names to fractal dimension values
        """
        results = {}

        for layer_name in self.layers_to_track:
            if layer_name not in activations:
                logger.warning(f"Layer {layer_name} not found in activations")
                continue

            act = activations[layer_name]

            # Compute metric based on type
            if self.metric_type == 'higuchi':
                # Expects [batch, seq_len]
                if act.dim() == 3:
                    act = act.mean(dim=2)  # Average over features
                fd = self.estimator.compute(act).mean().item()

            elif self.metric_type == 'spectral':
                # Expects [batch, seq_len]
                if act.dim() == 3:
                    act = act.mean(dim=2)
                beta, _, _ = self.estimator.compute(act)
                fd = beta.mean().item()

            elif self.metric_type == 'graph':
                # Expects adjacency matrix [batch, n, n]
                # Treat activation correlations as graph
                if act.dim() == 3:
                    # Compute correlation matrix
                    batch_size, seq_len, n_features = act.shape
                    # Reshape to [batch * seq_len, n_features]
                    act_2d = act.reshape(-1, n_features)
                    # Correlation matrix
                    corr = torch.corrcoef(act_2d.T)  # [n_features, n_features]
                    corr = corr.unsqueeze(0)  # [1, n_features, n_features]
                    fd = self.estimator.compute(corr).mean().item()
                else:
                    logger.warning(f"Graph metric requires 3D activations for {layer_name}")
                    continue

            results[layer_name] = fd

            # Update history
            if step is not None:
                self.history[layer_name].append((step, fd))

        return results

    def get_history(self, layer_name: str) -> List[Tuple[int, float]]:
        """Get fractal dimension history for a specific layer."""
        return self.history.get(layer_name, [])

    def get_all_history(self) -> Dict[str, List[Tuple[int, float]]]:
        """Get complete tracking history."""
        return self.history

    def plot_evolution(self, layer_name: Optional[str] = None):
        """
        Plot fractal dimension evolution over training.

        Args:
            layer_name: Specific layer to plot, or None for all layers

        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Matplotlib not available for plotting")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        if layer_name is not None:
            history = self.history[layer_name]
            steps, values = zip(*history) if history else ([], [])
            ax.plot(steps, values, marker='o', label=layer_name)
        else:
            for layer, history in self.history.items():
                if history:
                    steps, values = zip(*history)
                    ax.plot(steps, values, marker='o', label=layer)

        ax.set_xlabel('Training Step')
        ax.set_ylabel(f'Fractal Dimension ({self.metric_type})')
        ax.set_title('Fractal Dimension Evolution During Training')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig


class AttentionFractalCoupling:
    """
    Analyze coupling between attention patterns and fractal properties.

    Investigates whether fractal structure in activations correlates with
    attention weight patterns, revealing scale-dependent processing.

    Args:
        device: Torch device

    Example:
        >>> coupling = AttentionFractalCoupling()
        >>> results = coupling.compute(attention_weights, activations)
        >>> print(f"Correlation: {results['correlation']:.3f}")
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.fd_estimator = HiguchiFractalDimension(device=self.device)
        self.graph_fd_estimator = GraphFractalDimension(device=self.device)

    def compute(
        self,
        attention_weights: Tensor,
        activations: Tensor,
    ) -> Dict[str, Any]:
        """
        Compute attention-fractal coupling metrics.

        Args:
            attention_weights: Attention weights [batch, n_heads, seq_len, seq_len]
            activations: Layer activations [batch, seq_len, features]

        Returns:
            Dictionary with:
                - attention_fd: Graph FD of attention patterns
                - activation_fd: Temporal FD of activations
                - correlation: Correlation between attention entropy and activation FD
                - head_specific: Per-head analysis
        """
        batch_size = attention_weights.size(0)
        n_heads = attention_weights.size(1)

        # 1. Compute graph FD of attention patterns (average over heads)
        attn_avg = attention_weights.mean(dim=1)  # [batch, seq_len, seq_len]
        attention_fd = self.graph_fd_estimator.compute(attn_avg)

        # 2. Compute temporal FD of activations
        if activations.dim() == 3:
            act_1d = activations.mean(dim=2)  # [batch, seq_len]
        else:
            act_1d = activations
        activation_fd = self.fd_estimator.compute(act_1d)

        # 3. Compute attention entropy (measure of attention spread)
        attn_entropy = -(attention_weights * (attention_weights + 1e-10).log()).sum(dim=-1)
        attn_entropy = attn_entropy.mean(dim=(1, 2))  # [batch]

        # 4. Correlation between attention entropy and activation FD
        correlation = torch.corrcoef(torch.stack([attn_entropy, activation_fd]))[0, 1]

        # 5. Per-head analysis
        head_specific = []
        for h in range(n_heads):
            head_attn = attention_weights[:, h, :, :]  # [batch, seq_len, seq_len]
            head_fd = self.graph_fd_estimator.compute(head_attn)
            head_entropy = attn_entropy[:, h] if attn_entropy.dim() > 1 else attn_entropy
            head_specific.append({
                'head': h,
                'mean_fd': head_fd.mean().item(),
                'mean_entropy': head_entropy.mean().item() if isinstance(head_entropy, Tensor) else attn_entropy.mean().item(),
            })

        return {
            'attention_fd': attention_fd.mean().item(),
            'activation_fd': activation_fd.mean().item(),
            'correlation': correlation.item(),
            'head_specific': head_specific,
        }


class CausalScaleAblation:
    """
    Ablate specific fractal scales to test causal importance.

    Selectively removes frequency components to test which temporal scales
    are causally important for model performance.

    Args:
        device: Torch device

    Example:
        >>> ablator = CausalScaleAblation()
        >>> # Remove high-frequency components (keep low frequencies)
        >>> ablated = ablator.ablate_scale(signal, scale_range=(10.0, 100.0))
        >>> # Test model performance with ablated signal
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def ablate_scale(
        self,
        signal: Tensor,
        scale_range: Tuple[float, float],
        sampling_rate: float = 1.0,
        mode: str = 'remove',
    ) -> Tensor:
        """
        Remove or isolate spectral components in a specific scale range.

        Args:
            signal: Input signal [batch, seq_len] or [batch, seq_len, features]
            scale_range: (f_min, f_max) frequency range in Hz
            sampling_rate: Sampling rate in Hz
            mode: 'remove' (bandstop) or 'keep' (bandpass)

        Returns:
            Ablated signal with same shape as input
        """
        original_shape = signal.shape
        if signal.dim() == 3:
            # Apply to each feature independently
            batch_size, seq_len, n_features = signal.shape
            signal = signal.permute(0, 2, 1).reshape(batch_size * n_features, seq_len)
            was_3d = True
        else:
            was_3d = False

        signal = signal.to(self.device)

        # FFT
        signal_fft = torch.fft.rfft(signal, dim=1)
        freqs = torch.fft.rfftfreq(signal.size(1), d=1.0/sampling_rate).to(self.device)

        # Create filter
        f_min, f_max = scale_range
        if mode == 'remove':
            # Bandstop filter: remove frequencies in range
            mask = (freqs < f_min) | (freqs > f_max)
        elif mode == 'keep':
            # Bandpass filter: keep only frequencies in range
            mask = (freqs >= f_min) & (freqs <= f_max)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Apply filter
        filtered_fft = signal_fft * mask.unsqueeze(0)

        # Inverse FFT
        ablated = torch.fft.irfft(filtered_fft, n=signal.size(1), dim=1)

        # Reshape back if needed
        if was_3d:
            ablated = ablated.reshape(original_shape[0], original_shape[2], original_shape[1])
            ablated = ablated.permute(0, 2, 1)

        return ablated

    def multi_scale_ablation(
        self,
        signal: Tensor,
        scale_ranges: List[Tuple[float, float]],
        sampling_rate: float = 1.0,
    ) -> Dict[str, Tensor]:
        """
        Perform ablation at multiple scales simultaneously.

        Args:
            signal: Input signal
            scale_ranges: List of (f_min, f_max) tuples
            sampling_rate: Sampling rate

        Returns:
            Dictionary mapping scale descriptors to ablated signals
        """
        results = {}

        for f_min, f_max in scale_ranges:
            desc = f"{f_min:.1f}-{f_max:.1f}Hz_removed"
            results[desc] = self.ablate_scale(
                signal, (f_min, f_max),
                sampling_rate=sampling_rate,
                mode='remove',
            )

        return results

    def compute_ablation_impact(
        self,
        model: nn.Module,
        signal: Tensor,
        scale_range: Tuple[float, float],
        target: Tensor,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """
        Compute impact of scale ablation on model performance.

        Args:
            model: Neural network model
            signal: Input signal
            scale_range: Frequency range to ablate
            target: Ground truth targets
            criterion: Loss function

        Returns:
            Dictionary with baseline and ablated performance metrics
        """
        model.eval()

        with torch.no_grad():
            # Baseline performance
            baseline_output = model(signal)
            baseline_loss = criterion(baseline_output, target).item()

            # Ablated performance
            ablated_signal = self.ablate_scale(signal, scale_range, mode='remove')
            ablated_output = model(ablated_signal)
            ablated_loss = criterion(ablated_output, target).item()

        return {
            'baseline_loss': baseline_loss,
            'ablated_loss': ablated_loss,
            'delta_loss': ablated_loss - baseline_loss,
            'relative_impact': (ablated_loss - baseline_loss) / (baseline_loss + 1e-10),
        }
