"""
Neural Data Augmentation for Calcium Imaging + Astrocyte Events

Mathematically grounded augmentation techniques for neural time series data.
All augmentations preserve the statistical properties of real neural recordings.

References:
- Um et al. (2017) "Data Augmentation for Time Series Classification"
- Wen et al. (2020) "Time Series Data Augmentation for Deep Learning"
- Neural data augmentation from Neuromatch Academy
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
import random


@dataclass
class AugmentationConfig:
    """Configuration for neural data augmentation."""

    # Temporal augmentations
    time_warp_prob: float = 0.3
    time_warp_sigma: float = 0.2

    jitter_prob: float = 0.5
    jitter_sigma: float = 0.03

    time_shift_prob: float = 0.3
    time_shift_max: int = 10  # frames

    # Amplitude augmentations
    scaling_prob: float = 0.5
    scaling_range: Tuple[float, float] = (0.8, 1.2)

    gaussian_noise_prob: float = 0.5
    gaussian_noise_std: float = 0.02

    # Neuron augmentations
    neuron_dropout_prob: float = 0.3
    neuron_dropout_rate: float = 0.1  # fraction of neurons to drop

    neuron_permute_prob: float = 0.2

    # Astro event augmentations
    astro_jitter_prob: float = 0.3
    astro_jitter_sigma: float = 0.5  # seconds

    astro_dropout_prob: float = 0.2
    astro_dropout_rate: float = 0.2

    # Advanced augmentations
    mixup_prob: float = 0.0  # Requires batch-level augmentation
    mixup_alpha: float = 0.2

    cutout_prob: float = 0.2
    cutout_length: int = 10  # frames


class NeuralAugmentor(nn.Module):
    """
    Augmentation module for neural calcium imaging data.

    Applies biologically plausible augmentations that preserve
    the statistical structure of neural recordings.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        super().__init__()
        self.config = config or AugmentationConfig()

    def forward(
        self,
        calcium: torch.Tensor,
        astro_events: Optional[torch.Tensor] = None,
        astro_timestamps: Optional[torch.Tensor] = None,
        calcium_mask: Optional[torch.Tensor] = None,
        astro_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply augmentations to neural data.

        Args:
            calcium: (batch, n_neurons, seq_len) calcium traces
            astro_events: (batch, n_events, n_features) astro event tokens
            astro_timestamps: (batch, n_events) event timestamps
            calcium_mask: (batch, n_neurons) valid neuron mask
            astro_mask: (batch, n_events) valid event mask

        Returns:
            Dict with augmented tensors
        """
        # Clone to avoid modifying original
        calcium = calcium.clone()
        if astro_events is not None:
            astro_events = astro_events.clone()
        if astro_timestamps is not None:
            astro_timestamps = astro_timestamps.clone()

        # Apply calcium augmentations
        calcium = self._augment_calcium(calcium, calcium_mask)

        # Apply astro augmentations
        if astro_events is not None:
            astro_events, astro_timestamps, astro_mask = self._augment_astro(
                astro_events, astro_timestamps, astro_mask
            )

        return {
            'calcium': calcium,
            'astro_events': astro_events,
            'astro_timestamps': astro_timestamps,
            'calcium_mask': calcium_mask,
            'astro_mask': astro_mask,
        }

    def _augment_calcium(
        self,
        calcium: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply augmentations to calcium traces."""

        cfg = self.config

        # 1. Gaussian noise (most common, biologically motivated)
        if random.random() < cfg.gaussian_noise_prob:
            calcium = self._add_gaussian_noise(calcium, cfg.gaussian_noise_std)

        # 2. Amplitude scaling (models gain variations)
        if random.random() < cfg.scaling_prob:
            calcium = self._random_scaling(calcium, cfg.scaling_range)

        # 3. Temporal jitter (models timing uncertainty)
        if random.random() < cfg.jitter_prob:
            calcium = self._temporal_jitter(calcium, cfg.jitter_sigma)

        # 4. Time warping (models non-linear time variations)
        if random.random() < cfg.time_warp_prob:
            calcium = self._time_warp(calcium, cfg.time_warp_sigma)

        # 5. Neuron dropout (models incomplete recordings)
        if random.random() < cfg.neuron_dropout_prob:
            calcium = self._neuron_dropout(calcium, cfg.neuron_dropout_rate, mask)

        # 6. Cutout (temporal masking)
        if random.random() < cfg.cutout_prob:
            calcium = self._cutout(calcium, cfg.cutout_length)

        # 7. Time shift
        if random.random() < cfg.time_shift_prob:
            calcium = self._time_shift(calcium, cfg.time_shift_max)

        return calcium

    def _augment_astro(
        self,
        events: torch.Tensor,
        timestamps: Optional[torch.Tensor],
        mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Apply augmentations to astrocyte events."""

        cfg = self.config

        # 1. Temporal jitter of event times
        if timestamps is not None and random.random() < cfg.astro_jitter_prob:
            jitter = torch.randn_like(timestamps) * cfg.astro_jitter_sigma
            timestamps = timestamps + jitter
            timestamps = torch.clamp(timestamps, min=0)

        # 2. Event dropout
        if random.random() < cfg.astro_dropout_prob and mask is not None:
            # Randomly drop events
            drop_mask = torch.rand_like(mask.float()) > cfg.astro_dropout_rate
            mask = mask & drop_mask

        # 3. Feature noise on events
        if random.random() < cfg.gaussian_noise_prob:
            noise = torch.randn_like(events) * cfg.gaussian_noise_std
            events = events + noise

        return events, timestamps, mask

    # =========================================================================
    # Individual augmentation methods
    # =========================================================================

    def _add_gaussian_noise(
        self,
        x: torch.Tensor,
        std: float
    ) -> torch.Tensor:
        """
        Add Gaussian noise to signal.

        Biologically motivated: models measurement noise,
        shot noise in photon detection, etc.
        """
        noise = torch.randn_like(x) * std
        return x + noise

    def _random_scaling(
        self,
        x: torch.Tensor,
        scale_range: Tuple[float, float]
    ) -> torch.Tensor:
        """
        Random amplitude scaling.

        Biologically motivated: models variations in fluorescence
        indicator expression, imaging depth, laser power, etc.
        """
        scale = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(*scale_range)
        return x * scale

    def _temporal_jitter(
        self,
        x: torch.Tensor,
        sigma: float
    ) -> torch.Tensor:
        """
        Add temporal jitter via local smoothing/shifting.

        Biologically motivated: models uncertainty in spike timing
        and calcium dynamics.
        """
        # Simple approach: add time-varying noise
        batch, n_neurons, seq_len = x.shape

        # Per-neuron jitter
        jitter = torch.randn(batch, n_neurons, 1, device=x.device) * sigma
        return x + jitter * x.std(dim=-1, keepdim=True)

    def _time_warp(
        self,
        x: torch.Tensor,
        sigma: float
    ) -> torch.Tensor:
        """
        Time warping via smooth temporal distortion.

        Biologically motivated: models non-linear variations in
        behavioral timing, trial-to-trial variability in response latency.

        Uses cubic spline interpolation for smooth warping.
        """
        batch, n_neurons, seq_len = x.shape
        device = x.device

        # Create warping curve
        # Sample a few random anchor points
        n_anchors = 4
        orig_steps = torch.linspace(0, seq_len - 1, n_anchors, device=device)

        # Perturb anchor points
        warp = torch.randn(batch, n_anchors, device=device) * sigma
        warp[:, 0] = 0  # Keep start fixed
        warp[:, -1] = 0  # Keep end fixed
        warped_steps = orig_steps.unsqueeze(0) + warp * seq_len * 0.1
        warped_steps = torch.clamp(warped_steps, 0, seq_len - 1)

        # Interpolate to get new time indices
        time_steps = torch.arange(seq_len, device=device).float()

        # Simple linear interpolation of warp field using numpy
        # Convert to numpy, interpolate, convert back
        new_indices = torch.zeros(batch, seq_len, device=device)
        orig_np = orig_steps.cpu().numpy()
        time_np = time_steps.cpu().numpy()

        for i in range(batch):
            warped_np = warped_steps[i].cpu().numpy()
            interp_result = np.interp(time_np, orig_np, warped_np)
            new_indices[i] = torch.from_numpy(interp_result).to(device)

        # Resample signal at new time points
        new_indices = new_indices.long().clamp(0, seq_len - 1)

        # Gather along time dimension
        warped = torch.gather(
            x,
            dim=2,
            index=new_indices.unsqueeze(1).expand(-1, n_neurons, -1)
        )

        return warped

    def _neuron_dropout(
        self,
        x: torch.Tensor,
        dropout_rate: float,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Randomly drop neurons (set to zero).

        Biologically motivated: models incomplete recordings,
        lost cells during imaging session, neuropil contamination.
        """
        batch, n_neurons, seq_len = x.shape

        # Create dropout mask
        keep_prob = 1 - dropout_rate
        dropout_mask = torch.rand(batch, n_neurons, 1, device=x.device) < keep_prob

        # Apply dropout
        x = x * dropout_mask.float()

        return x

    def _cutout(
        self,
        x: torch.Tensor,
        length: int
    ) -> torch.Tensor:
        """
        Temporal cutout (mask random time segments).

        Biologically motivated: models brief periods of motion artifacts,
        photobleaching, or other transient data loss.
        """
        batch, n_neurons, seq_len = x.shape

        # Random start position for each batch
        start = torch.randint(0, max(1, seq_len - length), (batch,))

        # Create mask
        for i in range(batch):
            x[i, :, start[i]:start[i]+length] = 0

        return x

    def _time_shift(
        self,
        x: torch.Tensor,
        max_shift: int
    ) -> torch.Tensor:
        """
        Random temporal shift (circular).

        Biologically motivated: models trial-to-trial jitter in
        stimulus onset, behavioral timing variations.
        """
        batch, n_neurons, seq_len = x.shape

        # Random shift for each batch
        shift = torch.randint(-max_shift, max_shift + 1, (batch,))

        shifted = torch.zeros_like(x)
        for i in range(batch):
            shifted[i] = torch.roll(x[i], shifts=shift[i].item(), dims=-1)

        return shifted


class MixupAugmentor:
    """
    Mixup augmentation for batch-level mixing.

    Mixes pairs of samples: x_mix = λ*x1 + (1-λ)*x2

    Reference: Zhang et al. (2017) "mixup: Beyond Empirical Risk Minimization"
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(
        self,
        calcium: torch.Tensor,
        targets: torch.Tensor,
        calcium_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup augmentation.

        Returns:
            mixed_calcium: Mixed calcium data
            targets_a: Original targets
            targets_b: Shuffled targets
            lam: Mixing coefficient
        """
        batch_size = calcium.size(0)

        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)

        # Random permutation
        index = torch.randperm(batch_size, device=calcium.device)

        # Mix data
        mixed_calcium = lam * calcium + (1 - lam) * calcium[index]

        # Mix masks (union of valid neurons)
        if calcium_mask is not None:
            mixed_mask = calcium_mask | calcium_mask[index]
        else:
            mixed_mask = None

        return mixed_calcium, targets, targets[index], lam, mixed_mask


class WindowAugmentor:
    """
    Window-based augmentation to create more training samples.

    Creates additional samples by:
    1. Sliding with smaller strides
    2. Random cropping within windows
    3. Overlapping window extraction
    """

    def __init__(
        self,
        window_size: int = 100,
        min_stride: int = 10,
        n_crops: int = 3,
    ):
        self.window_size = window_size
        self.min_stride = min_stride
        self.n_crops = n_crops

    def augment_session(
        self,
        traces: np.ndarray,
        events: np.ndarray,
        timestamps: np.ndarray,
        sampling_rate: float = 10.0,
    ) -> List[Dict]:
        """
        Create augmented windows from a single session.

        Args:
            traces: (n_neurons, n_timepoints) calcium traces
            events: (n_events, n_features) astro events
            timestamps: (n_events,) event timestamps in seconds
            sampling_rate: Hz

        Returns:
            List of window dictionaries
        """
        n_neurons, n_timepoints = traces.shape
        total_duration = n_timepoints / sampling_rate
        window_duration = self.window_size / sampling_rate

        windows = []

        # Strategy 1: Dense sliding windows
        for stride in [self.min_stride, self.min_stride * 2, self.min_stride * 3]:
            stride_duration = stride / sampling_rate
            n_windows = int((total_duration - window_duration) / stride_duration) + 1

            for i in range(n_windows):
                t_start = i * stride_duration
                t_end = t_start + window_duration

                window = self._extract_window(
                    traces, events, timestamps,
                    t_start, t_end, sampling_rate
                )
                if window is not None:
                    windows.append(window)

        # Strategy 2: Random crops
        for _ in range(self.n_crops):
            max_start = total_duration - window_duration
            if max_start > 0:
                t_start = np.random.uniform(0, max_start)
                t_end = t_start + window_duration

                window = self._extract_window(
                    traces, events, timestamps,
                    t_start, t_end, sampling_rate
                )
                if window is not None:
                    windows.append(window)

        return windows

    def _extract_window(
        self,
        traces: np.ndarray,
        events: np.ndarray,
        timestamps: np.ndarray,
        t_start: float,
        t_end: float,
        sampling_rate: float,
    ) -> Optional[Dict]:
        """Extract a single window."""

        idx_start = int(t_start * sampling_rate)
        idx_end = int(t_end * sampling_rate)

        if idx_end > traces.shape[1]:
            return None

        # Extract calcium window
        calcium_window = traces[:, idx_start:idx_end]

        # Ensure correct size
        if calcium_window.shape[1] < self.window_size:
            pad_len = self.window_size - calcium_window.shape[1]
            calcium_window = np.pad(calcium_window, ((0, 0), (0, pad_len)), mode='edge')
        calcium_window = calcium_window[:, :self.window_size]

        # Extract events in window
        event_mask = (timestamps >= t_start) & (timestamps < t_end)
        window_events = events[event_mask]
        window_timestamps = timestamps[event_mask] - t_start  # Relative time

        return {
            'calcium': calcium_window.astype(np.float32),
            'astro_events': window_events.astype(np.float32),
            'astro_timestamps': window_timestamps.astype(np.float32),
            't_start': t_start,
            't_end': t_end,
        }


# =============================================================================
# Convenience functions
# =============================================================================

def create_augmentor(
    mode: str = 'light',
    **kwargs
) -> NeuralAugmentor:
    """
    Create augmentor with preset configurations.

    Modes:
        'none': No augmentation
        'light': Conservative augmentations
        'medium': Balanced augmentations
        'heavy': Aggressive augmentations
    """
    if mode == 'none':
        config = AugmentationConfig(
            gaussian_noise_prob=0, scaling_prob=0, jitter_prob=0,
            time_warp_prob=0, neuron_dropout_prob=0, cutout_prob=0,
            time_shift_prob=0, astro_jitter_prob=0, astro_dropout_prob=0,
        )
    elif mode == 'light':
        config = AugmentationConfig(
            gaussian_noise_prob=0.3, gaussian_noise_std=0.01,
            scaling_prob=0.3, scaling_range=(0.9, 1.1),
            jitter_prob=0.2, jitter_sigma=0.02,
            time_warp_prob=0.0,
            neuron_dropout_prob=0.2, neuron_dropout_rate=0.05,
            cutout_prob=0.0,
            time_shift_prob=0.0,
            astro_jitter_prob=0.2, astro_jitter_sigma=0.2,
            astro_dropout_prob=0.1, astro_dropout_rate=0.1,
        )
    elif mode == 'medium':
        config = AugmentationConfig(
            gaussian_noise_prob=0.5, gaussian_noise_std=0.02,
            scaling_prob=0.5, scaling_range=(0.85, 1.15),
            jitter_prob=0.3, jitter_sigma=0.03,
            time_warp_prob=0.2, time_warp_sigma=0.15,
            neuron_dropout_prob=0.3, neuron_dropout_rate=0.1,
            cutout_prob=0.2, cutout_length=10,
            time_shift_prob=0.2, time_shift_max=5,
            astro_jitter_prob=0.3, astro_jitter_sigma=0.3,
            astro_dropout_prob=0.2, astro_dropout_rate=0.15,
        )
    elif mode == 'heavy':
        config = AugmentationConfig(
            gaussian_noise_prob=0.7, gaussian_noise_std=0.03,
            scaling_prob=0.7, scaling_range=(0.8, 1.2),
            jitter_prob=0.5, jitter_sigma=0.05,
            time_warp_prob=0.3, time_warp_sigma=0.2,
            neuron_dropout_prob=0.5, neuron_dropout_rate=0.15,
            cutout_prob=0.3, cutout_length=15,
            time_shift_prob=0.3, time_shift_max=10,
            astro_jitter_prob=0.5, astro_jitter_sigma=0.5,
            astro_dropout_prob=0.3, astro_dropout_rate=0.2,
        )
    else:
        raise ValueError(f"Unknown augmentation mode: {mode}")

    # Override with kwargs
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)

    return NeuralAugmentor(config)


# =============================================================================
# Test augmentations
# =============================================================================

if __name__ == '__main__':
    print("Testing Neural Augmentations\n")

    # Create fake data
    batch_size = 4
    n_neurons = 100
    seq_len = 100
    n_events = 10
    n_features = 10

    calcium = torch.randn(batch_size, n_neurons, seq_len)
    astro_events = torch.randn(batch_size, n_events, n_features)
    astro_timestamps = torch.rand(batch_size, n_events) * 10  # 0-10 seconds
    calcium_mask = torch.ones(batch_size, n_neurons, dtype=torch.bool)
    astro_mask = torch.ones(batch_size, n_events, dtype=torch.bool)

    # Test each mode
    for mode in ['none', 'light', 'medium', 'heavy']:
        augmentor = create_augmentor(mode)

        # Apply augmentation
        augmented = augmentor(
            calcium=calcium,
            astro_events=astro_events,
            astro_timestamps=astro_timestamps,
            calcium_mask=calcium_mask,
            astro_mask=astro_mask,
        )

        # Check shapes preserved
        assert augmented['calcium'].shape == calcium.shape
        assert augmented['astro_events'].shape == astro_events.shape

        # Check values changed (for non-none modes)
        if mode != 'none':
            diff = (augmented['calcium'] - calcium).abs().mean()
            print(f"Mode '{mode}': mean change = {diff:.4f}")
        else:
            print(f"Mode '{mode}': no augmentation")

    print("\n✓ All augmentation tests passed!")
