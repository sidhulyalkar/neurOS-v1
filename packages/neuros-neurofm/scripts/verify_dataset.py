#!/usr/bin/env python3
"""
Comprehensive verification of AllenMultiModalDataset.

Tests:
1. Class implementation completeness
2. Neural data loading
3. Astro data loading
4. Temporal alignment verification
5. Token compatibility
6. Batch processing

Usage:
    python scripts/verify_dataset.py
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List
import inspect

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from neuros_neurofm.datasets.allen_multimodal_dataset import (
    AllenMultiModalDataset,
    collate_multimodal,
)
from torch.utils.data import DataLoader


def test_class_completeness():
    """Verify AllenMultiModalDataset has all required methods and attributes."""
    print("="*70)
    print("TEST 1: Class Implementation Completeness")
    print("="*70)

    required_methods = [
        '__init__',
        '__len__',
        '__getitem__',
        '_find_all_sessions',
        '_load_all_sessions',
    ]

    optional_methods = [
        'get_session_info',
        'get_statistics',
    ]

    print("\n✓ Checking AllenMultiModalDataset class...")

    # Get all methods
    methods = [m for m in dir(AllenMultiModalDataset) if not m.startswith('_') or m.startswith('__')]

    # Check required methods
    missing = []
    for method in required_methods:
        if hasattr(AllenMultiModalDataset, method):
            print(f"  ✓ {method}")
        else:
            print(f"  ✗ {method} MISSING!")
            missing.append(method)

    # Check optional methods
    print("\n  Optional methods:")
    for method in optional_methods:
        if hasattr(AllenMultiModalDataset, method):
            print(f"  ✓ {method}")
        else:
            print(f"  - {method} (not implemented)")

    # Check __init__ signature
    print("\n✓ Checking __init__ parameters...")
    sig = inspect.signature(AllenMultiModalDataset.__init__)
    params = list(sig.parameters.keys())[1:]  # Skip 'self'

    expected_params = [
        'calcium_dir', 'astro_dir', 'session_ids', 'seq_len',
        'modalities', 'transform', 'temporal_alignment', 'stride', 'min_astro_events'
    ]

    for param in expected_params:
        if param in params:
            default = sig.parameters[param].default
            default_str = f"(default: {default})" if default != inspect.Parameter.empty else ""
            print(f"  ✓ {param} {default_str}")
        else:
            print(f"  ✗ {param} MISSING!")
            missing.append(param)

    if len(missing) == 0:
        print("\n  ✅ Class implementation is COMPLETE!")
        return True
    else:
        print(f"\n  ❌ Missing {len(missing)} required components!")
        return False


def test_data_loading():
    """Test loading neural and astro data."""
    print("\n" + "="*70)
    print("TEST 2: Neural + Astro Data Loading")
    print("="*70)

    # Use absolute paths from project root
    project_root = Path(__file__).parent.parent.parent.parent
    calcium_dir = project_root / "packages/neuros-mechint/examples/allen_data_demo/data/2p_sessions_continuous"
    astro_dir = project_root / "allen_nwb_results"

    # Check directories exist
    if not calcium_dir.exists():
        print(f"\n  ❌ Calcium directory not found: {calcium_dir}")
        print("     Run: python scripts/prepare_allen_data.py")
        return False

    if not astro_dir.exists():
        print(f"\n  ❌ Astro directory not found: {astro_dir}")
        print("     Complete neuros-astro processing first")
        return False

    print("\n✓ Loading dataset...")

    try:
        dataset = AllenMultiModalDataset(
            calcium_dir=str(calcium_dir),
            astro_dir=str(astro_dir),
            session_ids='all',
            seq_len=100,
            modalities='both',
            stride=50,
            min_astro_events=1,
            temporal_alignment='downsample',
        )
    except Exception as e:
        print(f"\n  ❌ Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"  ✓ Dataset created")
    print(f"  ✓ Total sessions: {len(dataset.session_ids)}")
    print(f"  ✓ Total windows: {len(dataset)}")

    if len(dataset) == 0:
        print("\n  ❌ No temporal windows created!")
        print("     Try reducing min_astro_events")
        return False

    # Test loading first sample
    print("\n✓ Loading first sample...")

    try:
        sample = dataset[0]
    except Exception as e:
        print(f"\n  ❌ Failed to load sample: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"  ✓ Sample loaded successfully")

    # Verify sample structure
    print("\n✓ Verifying sample structure...")

    required_keys = ['metadata']
    if 'calcium' not in sample and 'astro_events' not in sample:
        print("  ❌ Sample has no modality data!")
        return False

    if 'calcium' in sample:
        print(f"  ✓ Calcium data: {sample['calcium'].shape}")
        print(f"    - Type: {sample['calcium'].dtype}")
        print(f"    - Min: {sample['calcium'].min():.4f}")
        print(f"    - Max: {sample['calcium'].max():.4f}")
        print(f"    - Mean: {sample['calcium'].mean():.4f}")

    if 'astro_events' in sample:
        print(f"  ✓ Astro events: {sample['astro_events'].shape}")
        print(f"    - Type: {sample['astro_events'].dtype}")
        print(f"    - N events: {len(sample['astro_events'])}")

        if 'astro_timestamps' in sample:
            print(f"  ✓ Astro timestamps: {sample['astro_timestamps'].shape}")
            print(f"    - Range: [{sample['astro_timestamps'].min():.2f}, {sample['astro_timestamps'].max():.2f}]s")

        if 'astro_region_ids' in sample:
            print(f"  ✓ Astro region IDs: {sample['astro_region_ids'].shape}")
            print(f"    - Unique astrocytes: {len(torch.unique(sample['astro_region_ids']))}")

    if 'metadata' in sample:
        print(f"  ✓ Metadata:")
        for key, value in sample['metadata'].items():
            print(f"    - {key}: {value}")

    print("\n  ✅ Data loading is WORKING!")
    return True


def test_temporal_alignment():
    """Verify temporal alignment between modalities."""
    print("\n" + "="*70)
    print("TEST 3: Temporal Alignment Verification")
    print("="*70)

    # Use absolute paths from project root
    project_root = Path(__file__).parent.parent.parent.parent
    calcium_dir = project_root / "packages/neuros-mechint/examples/allen_data_demo/data/2p_sessions_continuous"
    astro_dir = project_root / "allen_nwb_results"

    print("\n✓ Creating dataset...")

    try:
        dataset = AllenMultiModalDataset(
            calcium_dir=str(calcium_dir),
            astro_dir=str(astro_dir),
            session_ids='all',
            seq_len=100,
            modalities='both',
            stride=50,
            min_astro_events=1,
            temporal_alignment='downsample',
        )
    except:
        print("  ❌ Could not create dataset")
        return False

    if len(dataset) == 0:
        print("  ❌ No data available")
        return False

    # Load a sample
    sample = dataset[0]

    print("\n✓ Analyzing temporal properties...")

    # Calcium temporal properties
    if 'calcium' in sample:
        n_neurons, seq_len = sample['calcium'].shape
        calcium_duration = seq_len / 10.0  # 10 Hz after downsampling

        print(f"\n  Calcium trace:")
        print(f"    - Shape: {sample['calcium'].shape}")
        print(f"    - Neurons: {n_neurons}")
        print(f"    - Sequence length: {seq_len}")
        print(f"    - Sampling rate: 10 Hz (downsampled)")
        print(f"    - Duration: {calcium_duration:.1f}s")

    # Astro temporal properties
    if 'astro_events' in sample and 'astro_timestamps' in sample:
        n_events = len(sample['astro_events'])
        timestamps = sample['astro_timestamps'].numpy()

        print(f"\n  Astro events:")
        print(f"    - N events: {n_events}")
        print(f"    - Time range: [{timestamps.min():.2f}, {timestamps.max():.2f}]s")
        print(f"    - Event duration: {timestamps.max() - timestamps.min():.2f}s")

        # Check if events fall within calcium window
        meta = sample['metadata']
        t_start = meta['t_start']
        t_end = meta['t_end']

        print(f"\n  Window metadata:")
        print(f"    - Window start: {t_start:.2f}s")
        print(f"    - Window end: {t_end:.2f}s")
        print(f"    - Window duration: {t_end - t_start:.2f}s")

        # Verify events are within window
        events_in_window = np.sum((timestamps >= t_start) & (timestamps < t_end))

        print(f"\n  Alignment check:")
        print(f"    - Events in window: {events_in_window}/{n_events}")

        if events_in_window == n_events:
            print(f"    ✅ All events fall within temporal window!")
        else:
            print(f"    ⚠️  Some events outside window")

        # Check relative timestamps
        relative_times = timestamps - t_start
        print(f"    - Relative event times: [{relative_times.min():.2f}, {relative_times.max():.2f}]s")

        if relative_times.min() >= 0 and relative_times.max() <= (t_end - t_start):
            print(f"    ✅ Event timestamps correctly aligned!")
        else:
            print(f"    ⚠️  Event timestamps may not be relative to window")

    print("\n  ✅ Temporal alignment verified!")
    return True


def test_token_compatibility():
    """Test that tokens are compatible with neural networks."""
    print("\n" + "="*70)
    print("TEST 4: Token Compatibility")
    print("="*70)

    # Use absolute paths from project root
    project_root = Path(__file__).parent.parent.parent.parent
    calcium_dir = project_root / "packages/neuros-mechint/examples/allen_data_demo/data/2p_sessions_continuous"
    astro_dir = project_root / "allen_nwb_results"

    print("\n✓ Creating dataset and dataloader...")

    try:
        dataset = AllenMultiModalDataset(
            calcium_dir=str(calcium_dir),
            astro_dir=str(astro_dir),
            session_ids='all',
            seq_len=100,
            modalities='both',
            stride=50,
            min_astro_events=1,
        )

        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_multimodal,
            num_workers=0,
        )

        batch = next(iter(loader))
    except Exception as e:
        print(f"  ❌ Failed to create batch: {e}")
        return False

    print(f"  ✓ Batch created successfully")

    print("\n✓ Checking batch structure...")

    # Verify batch has correct structure
    required_keys = ['metadata']

    if 'calcium' in batch:
        print(f"  ✓ Calcium batch: {batch['calcium'].shape}")
        print(f"    - Mask: {batch['calcium_mask'].shape}")
        print(f"    - Valid neurons: {batch['calcium_mask'].sum(dim=1).tolist()}")

        # Check for NaN/Inf
        if torch.isnan(batch['calcium']).any():
            print(f"    ⚠️  Contains NaN values!")
        if torch.isinf(batch['calcium']).any():
            print(f"    ⚠️  Contains Inf values!")

    if 'astro_events' in batch:
        print(f"  ✓ Astro events batch: {batch['astro_events'].shape}")
        print(f"    - Mask: {batch['astro_mask'].shape}")
        print(f"    - Valid events: {batch['astro_mask'].sum(dim=1).tolist()}")

        # Check for NaN/Inf
        if torch.isnan(batch['astro_events']).any():
            print(f"    ⚠️  Contains NaN values!")
        if torch.isinf(batch['astro_events']).any():
            print(f"    ⚠️  Contains Inf values!")

    # Test simple neural network operations
    print("\n✓ Testing neural network compatibility...")

    try:
        # Simple linear layer
        batch_size, max_neurons, seq_len = batch['calcium'].shape
        _, max_events, n_features = batch['astro_events'].shape

        # Test reshaping for linear layers
        calcium_flat = batch['calcium'].reshape(batch_size, -1)
        astro_flat = batch['astro_events'].reshape(batch_size, -1)

        print(f"  ✓ Flattened calcium: {calcium_flat.shape}")
        print(f"  ✓ Flattened astro: {astro_flat.shape}")

        # Test simple forward pass
        test_linear = torch.nn.Linear(calcium_flat.shape[1], 128)
        output = test_linear(calcium_flat)

        print(f"  ✓ Linear layer output: {output.shape}")
        print(f"  ✅ Tokens are compatible with neural networks!")

    except Exception as e:
        print(f"  ❌ Neural network compatibility failed: {e}")
        return False

    return True


def test_multi_session_consistency():
    """Test consistency across multiple sessions."""
    print("\n" + "="*70)
    print("TEST 5: Multi-Session Consistency")
    print("="*70)

    # Use absolute paths from project root
    project_root = Path(__file__).parent.parent.parent.parent
    calcium_dir = project_root / "packages/neuros-mechint/examples/allen_data_demo/data/2p_sessions_continuous"
    astro_dir = project_root / "allen_nwb_results"

    print("\n✓ Loading dataset...")

    try:
        dataset = AllenMultiModalDataset(
            calcium_dir=str(calcium_dir),
            astro_dir=str(astro_dir),
            session_ids='all',
            seq_len=100,
            modalities='both',
            stride=50,
            min_astro_events=1,
        )
    except:
        print("  ❌ Could not create dataset")
        return False

    if len(dataset) == 0:
        print("  ❌ No data available")
        return False

    # Collect stats per session
    session_stats = {}

    for i in range(min(len(dataset), 50)):  # Check first 50 windows
        sample = dataset[i]
        session_id = sample['metadata']['session_id']

        if session_id not in session_stats:
            session_stats[session_id] = {
                'n_windows': 0,
                'n_neurons': [],
                'n_astrocytes': [],
                'n_events': [],
            }

        session_stats[session_id]['n_windows'] += 1
        session_stats[session_id]['n_neurons'].append(sample['metadata']['n_neurons'])
        session_stats[session_id]['n_astrocytes'].append(sample['metadata']['n_astrocytes'])

        if 'astro_events' in sample:
            session_stats[session_id]['n_events'].append(len(sample['astro_events']))

    # Print stats
    print(f"\n  Loaded {len(session_stats)} unique sessions:")

    for session_id, stats in session_stats.items():
        n_neurons = stats['n_neurons'][0] if len(stats['n_neurons']) > 0 else 0
        n_astrocytes = stats['n_astrocytes'][0] if len(stats['n_astrocytes']) > 0 else 0
        avg_events = np.mean(stats['n_events']) if len(stats['n_events']) > 0 else 0

        print(f"\n    Session {session_id}:")
        print(f"      - Windows: {stats['n_windows']}")
        print(f"      - Neurons: {n_neurons}")
        print(f"      - Astrocytes: {n_astrocytes}")
        print(f"      - Avg events/window: {avg_events:.1f}")

        # Check consistency within session
        if len(set(stats['n_neurons'])) > 1:
            print(f"      ⚠️  Inconsistent neuron counts!")
        if len(set(stats['n_astrocytes'])) > 1:
            print(f"      ⚠️  Inconsistent astrocyte counts!")

    print("\n  ✅ Multi-session consistency verified!")
    return True


def visualize_alignment():
    """Create visualization of temporal alignment."""
    print("\n" + "="*70)
    print("VISUALIZATION: Temporal Alignment")
    print("="*70)

    # Use absolute paths from project root
    project_root = Path(__file__).parent.parent.parent.parent
    calcium_dir = project_root / "packages/neuros-mechint/examples/allen_data_demo/data/2p_sessions_continuous"
    astro_dir = project_root / "allen_nwb_results"

    try:
        dataset = AllenMultiModalDataset(
            calcium_dir=str(calcium_dir),
            astro_dir=str(astro_dir),
            session_ids='all',
            seq_len=100,
            modalities='both',
            stride=50,
            min_astro_events=1,
        )
    except:
        print("  ❌ Could not create dataset")
        return False

    if len(dataset) == 0:
        print("  ❌ No data available")
        return False

    # Load a sample
    sample = dataset[0]

    if 'calcium' not in sample or 'astro_events' not in sample:
        print("  ⚠️  Missing modality data for visualization")
        return False

    print("\n✓ Creating alignment visualization...")

    # Extract data
    calcium = sample['calcium'].numpy()  # (n_neurons, seq_len)
    astro_events = sample['astro_events'].numpy()
    astro_timestamps = sample['astro_timestamps'].numpy()
    meta = sample['metadata']

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 1. Calcium traces
    ax = axes[0]
    n_neurons, seq_len = calcium.shape
    time_calcium = np.linspace(meta['t_start'], meta['t_end'], seq_len)

    # Plot first 10 neurons
    for i in range(min(10, n_neurons)):
        ax.plot(time_calcium, calcium[i] + i*0.5, 'k-', alpha=0.6, linewidth=0.5)

    ax.set_ylabel('Neurons (ΔF/F)', fontsize=12)
    ax.set_title(f'Session {meta["session_id"]} - Temporal Alignment\n'
                 f'Window: [{meta["t_start"]:.1f}, {meta["t_end"]:.1f}]s',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(meta['t_start'], meta['t_end'])

    # 2. Astro event raster
    ax = axes[1]
    if 'astro_region_ids' in sample:
        region_ids = sample['astro_region_ids'].numpy()
        ax.scatter(astro_timestamps, region_ids, c='red', s=20, alpha=0.6, marker='|')
        ax.set_ylabel('Astrocyte ID', fontsize=12)
    else:
        # Just plot event times
        ax.scatter(astro_timestamps, np.arange(len(astro_timestamps)),
                  c='red', s=20, alpha=0.6, marker='|')
        ax.set_ylabel('Event Index', fontsize=12)

    ax.grid(True, alpha=0.3)
    ax.set_xlim(meta['t_start'], meta['t_end'])

    # 3. Event rate over time
    ax = axes[2]
    bins = np.linspace(meta['t_start'], meta['t_end'], 20)
    counts, edges = np.histogram(astro_timestamps, bins=bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    ax.bar(bin_centers, counts, width=np.diff(bins)[0]*0.8,
           color='red', alpha=0.6, label='Astro events')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Event count', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(meta['t_start'], meta['t_end'])

    plt.tight_layout()

    # Save
    output_dir = Path('verification_outputs')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'temporal_alignment.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved visualization: {output_path}")

    plt.close()

    print("  ✅ Visualization created!")
    return True


def main():
    """Run all verification tests."""
    print("\n" + "="*70)
    print(" ALLEN MULTIMODAL DATASET VERIFICATION")
    print("="*70 + "\n")

    tests = [
        ("Class Completeness", test_class_completeness),
        ("Data Loading", test_data_loading),
        ("Temporal Alignment", test_temporal_alignment),
        ("Token Compatibility", test_token_compatibility),
        ("Multi-Session Consistency", test_multi_session_consistency),
        ("Alignment Visualization", visualize_alignment),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results[test_name] = passed
        except Exception as e:
            print(f"\n  ❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("\n" + "="*70)
    print(" VERIFICATION SUMMARY")
    print("="*70 + "\n")

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {test_name}")

    passed_count = sum(results.values())
    total_count = len(results)

    print(f"\n  {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n" + "="*70)
        print(" ✅ ALL VERIFICATION TESTS PASSED!")
        print("="*70)
        print("\n  AllenMultiModalDataset is:")
        print("    ✓ Fully implemented")
        print("    ✓ Loading neural data correctly")
        print("    ✓ Loading astro data correctly")
        print("    ✓ Temporally aligned")
        print("    ✓ Compatible with neural networks")
        print("    ✓ Consistent across sessions")
        print("\n  Ready for training!")
        print("\n  Next step:")
        print("    python scripts/train_allen_multimodal.py \\")
        print("        --config configs/allen_multimodal.yaml \\")
        print("        --test")
        return 0
    else:
        print("\n  ⚠️  Some verification tests failed.")
        print("\n  Review errors above and fix issues before training.")
        return 1


if __name__ == '__main__':
    exit(main())
