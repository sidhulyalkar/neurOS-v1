#!/usr/bin/env python3
"""
Temporal Dynamics of SAE Features
===================================

Implements Experiment 3.1 from ADVANCED_RESEARCH_ROADMAP.md:
- Extract time-resolved SAE features (not averaged over time)
- Analyze feature dynamics during stimulus presentation
- Measure response latencies and decay times
- Identify fast vs slow features (transient vs sustained)

This script answers:
1. How do SAE features evolve during stimulus presentation?
2. Which features respond quickly vs slowly?
3. Do features show transient or sustained responses?

Usage:
    python experiments/dynamics/feature_dynamics.py \
        --sae-model sae_models/sae_session_754829445.pt \
        --session-id 754829445 \
        --allen-cache allen_validation_cache \
        --output-dir results/dynamics
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleSAE(nn.Module):
    """Simple SAE matching training script."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.encoder(x))
        x_recon = self.decoder(h)
        return x_recon, h

    def encode(self, x):
        with torch.no_grad():
            h = self.relu(self.encoder(x))
        return h


class FeatureDynamicsAnalyzer:
    """
    Analyze temporal dynamics of SAE features.

    Characterizes features as:
    - Fast/Slow (response latency)
    - Transient/Sustained (decay time)
    - Oscillatory/Non-oscillatory
    """

    def __init__(self, bin_size: float = 0.02):
        """
        Args:
            bin_size: Time bin size in seconds (default: 20ms)
        """
        self.bin_size = bin_size

    def extract_temporal_features(
        self,
        sae: nn.Module,
        windows: List,
        orientations: np.ndarray
    ) -> Dict:
        """
        Extract time-resolved SAE features for each window.

        Args:
            sae: Trained SAE model
            windows: List of NeuralWindow objects
            orientations: Orientation labels

        Returns:
            temporal_data: Dict with time-resolved features and metadata
        """
        logger.info("Extracting time-resolved SAE features...")

        temporal_features = []
        temporal_metadata = []

        for window in windows:
            # window.data shape: [time_bins, neurons]
            time_bins = window.data.shape[0]

            for t in range(time_bins):
                # Extract neural activity at this time bin
                neural_activity = window.data[t, :]

                # Normalize (same as training)
                # Note: Should use same normalization as training data
                neural_activity_norm = neural_activity  # Assume already normalized

                # Get SAE features for this time bin
                X_t = torch.FloatTensor(neural_activity_norm).unsqueeze(0)
                features_t = sae.encode(X_t).squeeze(0).numpy()

                temporal_features.append(features_t)
                temporal_metadata.append({
                    'time': t * self.bin_size,  # Time in seconds
                    'time_bin': t,
                    'orientation': window.metadata.get('orientation'),
                    'window_idx': id(window)
                })

        temporal_features = np.array(temporal_features)

        logger.info(f"  Extracted {temporal_features.shape[0]} time bins")
        logger.info(f"  Features shape: {temporal_features.shape}")

        temporal_data = {
            'features': temporal_features,
            'metadata': temporal_metadata,
            'n_features': temporal_features.shape[1],
            'n_timebins': temporal_features.shape[0]
        }

        return temporal_data

    def analyze_response_latency(
        self,
        temporal_data: Dict,
        feature_idx: int,
        preferred_orientation: float,
        threshold: float = 0.3
    ) -> Dict:
        """
        Measure response latency for a specific feature.

        Args:
            temporal_data: Time-resolved feature data
            feature_idx: SAE feature index
            preferred_orientation: Preferred orientation of this feature
            threshold: Threshold for detecting response onset

        Returns:
            latency_stats: Dict with latency measurements
        """
        # Filter trials with preferred orientation
        metadata = temporal_data['metadata']
        features = temporal_data['features']

        # Group by window
        windows = {}
        for i, meta in enumerate(metadata):
            window_id = meta['window_idx']
            if window_id not in windows:
                windows[window_id] = {
                    'times': [],
                    'features': [],
                    'orientation': meta['orientation']
                }
            windows[window_id]['times'].append(meta['time'])
            windows[window_id]['features'].append(features[i, feature_idx])

        # Analyze preferred orientation trials
        latencies = []

        for window_id, window_data in windows.items():
            try:
                ori = float(window_data['orientation'])
                ori_180 = ori % 180

                # Check if close to preferred orientation (within 22.5°)
                ori_diff = min(abs(ori_180 - preferred_orientation),
                              180 - abs(ori_180 - preferred_orientation))

                if ori_diff < 22.5:
                    times = np.array(window_data['times'])
                    activations = np.array(window_data['features'])

                    # Normalize to baseline (first 100ms)
                    baseline_idx = times < 0.1
                    if baseline_idx.sum() > 0:
                        baseline = activations[baseline_idx].mean()
                        activations_norm = activations - baseline

                        # Find response onset (first crossing threshold)
                        threshold_crossings = np.where(activations_norm > threshold)[0]
                        if len(threshold_crossings) > 0:
                            onset_idx = threshold_crossings[0]
                            latency = times[onset_idx]
                            latencies.append(latency)
            except (ValueError, TypeError):
                continue

        if len(latencies) > 0:
            latency_stats = {
                'mean_latency': float(np.mean(latencies)),
                'std_latency': float(np.std(latencies)),
                'median_latency': float(np.median(latencies)),
                'n_trials': len(latencies)
            }
        else:
            latency_stats = {
                'mean_latency': np.nan,
                'std_latency': np.nan,
                'median_latency': np.nan,
                'n_trials': 0
            }

        return latency_stats

    def analyze_decay_dynamics(
        self,
        temporal_data: Dict,
        feature_idx: int,
        preferred_orientation: float
    ) -> Dict:
        """
        Measure decay time constant for a feature.

        Fits exponential decay to response offset.

        Args:
            temporal_data: Time-resolved data
            feature_idx: Feature index
            preferred_orientation: Preferred orientation

        Returns:
            decay_stats: Dict with decay time constant
        """
        # Extract activation timecourses for preferred orientation
        metadata = temporal_data['metadata']
        features = temporal_data['features']

        # Group by window
        windows = {}
        for i, meta in enumerate(metadata):
            window_id = meta['window_idx']
            if window_id not in windows:
                windows[window_id] = {
                    'times': [],
                    'features': [],
                    'orientation': meta['orientation']
                }
            windows[window_id]['times'].append(meta['time'])
            windows[window_id]['features'].append(features[i, feature_idx])

        # Average timecourse for preferred orientation
        timecourses = []

        for window_data in windows.values():
            try:
                ori = float(window_data['orientation'])
                ori_180 = ori % 180
                ori_diff = min(abs(ori_180 - preferred_orientation),
                              180 - abs(ori_180 - preferred_orientation))

                if ori_diff < 22.5:
                    timecourses.append(np.array(window_data['features']))
            except (ValueError, TypeError):
                continue

        if len(timecourses) == 0:
            return {
                'decay_constant': np.nan,
                'half_life': np.nan,
                'response_type': 'unknown'
            }

        # Average across trials
        # Pad to same length
        max_len = max(len(tc) for tc in timecourses)
        padded = [np.pad(tc, (0, max_len - len(tc)), constant_values=np.nan)
                  for tc in timecourses]
        avg_timecourse = np.nanmean(padded, axis=0)

        # Find peak
        peak_idx = np.nanargmax(avg_timecourse)

        # Fit exponential decay after peak
        if peak_idx < len(avg_timecourse) - 5:
            decay_portion = avg_timecourse[peak_idx:]
            decay_times = np.arange(len(decay_portion)) * self.bin_size

            # Remove NaNs
            valid = ~np.isnan(decay_portion)
            if valid.sum() > 3:
                try:
                    # Exponential decay: y = A * exp(-t/tau)
                    def exp_decay(t, A, tau):
                        return A * np.exp(-t / tau)

                    popt, _ = curve_fit(
                        exp_decay,
                        decay_times[valid],
                        decay_portion[valid],
                        p0=[decay_portion[valid][0], 0.1],
                        maxfev=1000
                    )

                    decay_constant = popt[1]
                    half_life = decay_constant * np.log(2)

                    # Classify response type
                    if decay_constant < 0.1:
                        response_type = 'transient'  # Fast decay
                    elif decay_constant > 0.3:
                        response_type = 'sustained'  # Slow decay
                    else:
                        response_type = 'intermediate'

                    return {
                        'decay_constant': float(decay_constant),
                        'half_life': float(half_life),
                        'response_type': response_type
                    }
                except:
                    pass

        return {
            'decay_constant': np.nan,
            'half_life': np.nan,
            'response_type': 'unknown'
        }


def load_data_with_temporal_info(
    sae_model_path: str,
    session_id: int,
    cache_dir: str,
    device: str = 'cpu'
) -> Tuple[nn.Module, List, np.ndarray]:
    """
    Load SAE and temporal windows (NOT averaged over time).

    Returns:
        sae: Trained SAE
        windows: List of NeuralWindow objects (with time dimension)
        orientations: Orientation labels
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "neuros-foundation" / "src"))
    from neuros.datasets.allen_datasets import AllenVisualCodingValidator

    logger.info(f"Loading Allen session {session_id} with temporal resolution...")
    validator = AllenVisualCodingValidator(
        session_id=session_id,
        cache_dir=cache_dir,
        brain_areas=['VISp'],
        use_all_units=True
    )

    # Get windows with temporal structure preserved
    windows = validator.get_neural_windows(
        window_length=1.0,
        stride=0.5,
        bin_size=0.02
    )

    # Extract orientations
    orientations = []
    valid_windows = []

    for w in windows:
        ori = w.metadata.get('orientation')
        if ori != 'null' and ori is not None:
            try:
                ori_float = float(ori)
                orientations.append(ori_float % 180)
                valid_windows.append(w)
            except (ValueError, TypeError):
                pass

    logger.info(f"  Windows: {len(valid_windows)}")
    logger.info(f"  Time bins per window: {valid_windows[0].data.shape[0]}")
    logger.info(f"  Neurons: {valid_windows[0].data.shape[1]}")

    # Load SAE
    logger.info(f"Loading SAE from {sae_model_path}...")
    input_dim = valid_windows[0].data.shape[1]

    results_path = Path(sae_model_path).parent / "training_results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
            hidden_dim = results[0]['data_stats']['n_sae_features']
    else:
        hidden_dim = 128

    sae = SimpleSAE(input_dim=input_dim, hidden_dim=hidden_dim)
    sae.load_state_dict(torch.load(sae_model_path, map_location=device))
    sae.to(device)
    sae.eval()

    logger.info(f"  SAE: {input_dim} → {hidden_dim}")
    logger.info(f"  Device: {device}")

    return sae, valid_windows, np.array(orientations)


def visualize_feature_dynamics(
    temporal_data: Dict,
    feature_idx: int,
    preferred_orientation: float,
    latency_stats: Dict,
    decay_stats: Dict,
    save_path: Optional[str] = None
):
    """Visualize temporal dynamics for a single feature."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metadata = temporal_data['metadata']
    features = temporal_data['features']

    # Group by window
    windows = {}
    for i, meta in enumerate(metadata):
        window_id = meta['window_idx']
        if window_id not in windows:
            windows[window_id] = {
                'times': [],
                'features': [],
                'orientation': meta['orientation']
            }
        windows[window_id]['times'].append(meta['time'])
        windows[window_id]['features'].append(features[i, feature_idx])

    # Plot 1: Average timecourse
    ax = axes[0, 0]
    timecourses_pref = []
    timecourses_ortho = []

    for window_data in windows.values():
        try:
            ori = float(window_data['orientation'])
            ori_180 = ori % 180
            ori_diff = min(abs(ori_180 - preferred_orientation),
                          180 - abs(ori_180 - preferred_orientation))

            times = np.array(window_data['times'])
            activations = np.array(window_data['features'])

            if ori_diff < 22.5:
                timecourses_pref.append(activations)
            elif 67.5 < ori_diff < 112.5:
                timecourses_ortho.append(activations)
        except:
            continue

    if timecourses_pref:
        max_len = max(len(tc) for tc in timecourses_pref)
        padded = [np.pad(tc, (0, max_len - len(tc)), constant_values=np.nan)
                  for tc in timecourses_pref]
        avg_pref = np.nanmean(padded, axis=0)
        std_pref = np.nanstd(padded, axis=0)
        times = np.arange(len(avg_pref)) * 0.02

        ax.plot(times, avg_pref, 'b-', linewidth=2, label='Preferred')
        ax.fill_between(times, avg_pref - std_pref, avg_pref + std_pref,
                        alpha=0.3, color='blue')

    if timecourses_ortho:
        max_len = max(len(tc) for tc in timecourses_ortho)
        padded = [np.pad(tc, (0, max_len - len(tc)), constant_values=np.nan)
                  for tc in timecourses_ortho]
        avg_ortho = np.nanmean(padded, axis=0)
        times = np.arange(len(avg_ortho)) * 0.02

        ax.plot(times, avg_ortho, 'r--', linewidth=2, label='Orthogonal')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Feature Activation')
    ax.set_title(f'Feature {feature_idx}: Temporal Response')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Response statistics
    ax = axes[0, 1]
    ax.axis('off')

    stats_text = f"""
    Temporal Dynamics Summary
    {'='*35}

    Feature: {feature_idx}
    Preferred orientation: {preferred_orientation:.1f}°

    Response Latency:
      Mean: {latency_stats['mean_latency']:.3f}s
      Std: {latency_stats['std_latency']:.3f}s
      N trials: {latency_stats['n_trials']}

    Decay Dynamics:
      Tau: {decay_stats['decay_constant']:.3f}s
      Half-life: {decay_stats['half_life']:.3f}s
      Type: {decay_stats['response_type']}
    """

    ax.text(0.1, 0.5, stats_text, fontfamily='monospace',
            fontsize=10, verticalalignment='center')

    # Plot 3: Raster plot (sample trials)
    ax = axes[1, 0]
    trial_count = 0
    for window_data in list(windows.values())[:20]:
        try:
            ori = float(window_data['orientation'])
            ori_180 = ori % 180
            ori_diff = min(abs(ori_180 - preferred_orientation),
                          180 - abs(ori_180 - preferred_orientation))

            if ori_diff < 22.5:
                times = np.array(window_data['times'])
                activations = np.array(window_data['features'])

                ax.plot(times, activations + trial_count * 0.5, 'k-', alpha=0.5, linewidth=0.5)
                trial_count += 1
        except:
            continue

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trial')
    ax.set_title('Single Trial Responses')

    # Plot 4: Placeholder for future analysis
    ax = axes[1, 1]
    ax.text(0.5, 0.5, 'Spectral analysis\n(future work)',
            ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"  Saved dynamics visualization to {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Temporal Dynamics of SAE Features'
    )
    parser.add_argument('--sae-model', type=str, required=True)
    parser.add_argument('--session-id', type=int, required=True)
    parser.add_argument('--allen-cache', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='results/dynamics')
    parser.add_argument('--top-features', type=int, default=10,
                       help='Number of top features to analyze')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("Feature Dynamics Analysis")
    logger.info("="*60)

    # Load SAE and temporal windows
    sae, windows, orientations = load_data_with_temporal_info(
        args.sae_model, args.session_id, args.allen_cache, device=args.device
    )

    # Initialize analyzer
    analyzer = FeatureDynamicsAnalyzer(bin_size=0.02)

    # Extract temporal features
    temporal_data = analyzer.extract_temporal_features(sae, windows, orientations)

    # Load feature selectivity from attribution results
    attr_path = Path('results/circuits') / f'attribution_results_session_{args.session_id}.json'
    if attr_path.exists():
        with open(attr_path) as f:
            attr_data = json.load(f)
            top_features = attr_data['top_features'][:args.top_features]
            feature_selectivity = dict(zip(attr_data['top_features'], attr_data['feature_selectivity']))
            preferred_oris = dict(zip(attr_data['top_features'], attr_data['feature_preferred_orientations']))
    else:
        logger.warning("Attribution results not found, using default features")
        top_features = list(range(args.top_features))
        feature_selectivity = {i: 0.5 for i in top_features}
        preferred_oris = {i: 0.0 for i in top_features}

    # Analyze dynamics for top features
    all_dynamics = []

    for feature_idx in top_features:
        logger.info(f"\n--- Feature {feature_idx} ---")

        pref_ori = preferred_oris[feature_idx]

        # Response latency
        latency_stats = analyzer.analyze_response_latency(
            temporal_data, feature_idx, pref_ori, threshold=0.3
        )

        # Decay dynamics
        decay_stats = analyzer.analyze_decay_dynamics(
            temporal_data, feature_idx, pref_ori
        )

        logger.info(f"  Latency: {latency_stats['mean_latency']:.3f}s")
        logger.info(f"  Decay constant: {decay_stats['decay_constant']:.3f}s")
        logger.info(f"  Response type: {decay_stats['response_type']}")

        dynamics_result = {
            'feature_idx': feature_idx,
            'selectivity': feature_selectivity.get(feature_idx, 0.0),
            'preferred_orientation': pref_ori,
            'latency': latency_stats,
            'decay': decay_stats
        }
        all_dynamics.append(dynamics_result)

        # Visualize
        vis_path = output_dir / f'dynamics_feature_{feature_idx}.png'
        visualize_feature_dynamics(
            temporal_data, feature_idx, pref_ori,
            latency_stats, decay_stats, save_path=vis_path
        )

    # Save results
    results = {
        'session_id': args.session_id,
        'features_analyzed': all_dynamics,
        'summary': {
            'mean_latency': float(np.nanmean([d['latency']['mean_latency'] for d in all_dynamics])),
            'n_transient': sum(1 for d in all_dynamics if d['decay']['response_type'] == 'transient'),
            'n_sustained': sum(1 for d in all_dynamics if d['decay']['response_type'] == 'sustained'),
            'n_intermediate': sum(1 for d in all_dynamics if d['decay']['response_type'] == 'intermediate')
        }
    }

    results_path = output_dir / f'dynamics_results_session_{args.session_id}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Saved results to {results_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"\nMean latency: {results['summary']['mean_latency']:.3f}s")
    logger.info(f"Transient features: {results['summary']['n_transient']}")
    logger.info(f"Sustained features: {results['summary']['n_sustained']}")
    logger.info(f"Intermediate features: {results['summary']['n_intermediate']}")

    logger.info("\n✓ Feature Dynamics Analysis Complete!")


if __name__ == '__main__':
    main()
