"""
Simple Astrocyte Integration Demo (No Mamba Required)

This script demonstrates how to:
1. Load astrocyte event tokens from neuros-astro
2. Use the AstroTokenizer to prepare them for neuroFMx
3. Run basic processing without full model

Usage:
    python examples/astro_integration_demo_simple.py --astro-tokens /path/to/astro_tokens.npz
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import json

from neuros_neurofm.tokenizers import AstroTokenizer


def load_neuros_astro_tokens(token_file: Path):
    """
    Load astrocyte event tokens from neuros-astro.

    Args:
        token_file: Path to astro_tokens.npz from neuros-astro

    Returns:
        Dict with loaded data
    """
    print(f"Loading astro tokens from {token_file}")
    data = np.load(token_file, allow_pickle=True)

    # Parse metadata if available
    metadata = {}
    if 'metadata_json' in data:
        metadata = json.loads(str(data['metadata_json']))

    # Convert region_ids from strings to integers
    region_ids_raw = data.get('region_ids', data.get('astrocyte_ids'))
    if region_ids_raw.dtype == object:
        # String IDs like 'roi_002' -> map to integers
        unique_regions = np.unique(region_ids_raw)
        region_to_idx = {region: idx for idx, region in enumerate(unique_regions)}
        region_ids = np.array([region_to_idx[r] for r in region_ids_raw], dtype=np.int64)
    else:
        region_ids = region_ids_raw.astype(np.int64)

    tokens = {
        'event_tokens': data['tokens'],  # (n_events, n_features)
        'timestamps': data['timestamps_s'],      # (n_events,)
        'region_ids': region_ids,  # (n_events,) - integer indices
        'metadata': metadata,
        'feature_names': data.get('feature_names', None),
        'session_id': data.get('session_id', None)
    }

    print(f"  Loaded {len(tokens['timestamps'])} events")
    print(f"  From {len(np.unique(tokens['region_ids']))} regions")
    if tokens['feature_names'] is not None:
        print(f"  Features: {', '.join(str(f) for f in tokens['feature_names'])}")
    print(f"  Time range: {tokens['timestamps'].min():.2f} - {tokens['timestamps'].max():.2f}s")

    return tokens


def run_simple_demo(args):
    """Run simplified astro integration demo."""

    print("="*60)
    print("Astrocyte Tokenization Demo (Simplified)")
    print("="*60)
    print()

    # 1. Load astro tokens from neuros-astro
    tokens_dict = load_neuros_astro_tokens(args.astro_tokens)
    n_astrocytes = len(np.unique(tokens_dict['region_ids']))

    # 2. Create AstroTokenizer
    print("\nCreating AstroTokenizer...")

    tokenizer = AstroTokenizer(
        n_astrocytes=n_astrocytes,
        d_model=512,
        seq_len=100,
        sampling_rate=10.0,
        use_events=True
    )

    print(f"  Tokenizer config:")
    print(f"    n_astrocytes: {n_astrocytes}")
    print(f"    d_model: 512")
    print(f"    seq_len: 100")

    # 3. Prepare batch
    print("\nPreparing astro data batch...")

    max_events = min(args.max_events, len(tokens_dict['timestamps']))
    event_tokens = torch.from_numpy(tokens_dict['event_tokens'][:max_events]).float()
    timestamps = torch.from_numpy(tokens_dict['timestamps'][:max_events]).float()
    region_ids = torch.from_numpy(tokens_dict['region_ids'][:max_events]).long()

    print(f"  Event tokens shape: {event_tokens.shape}")
    print(f"  Timestamps shape: {timestamps.shape}")
    print(f"  Region IDs shape: {region_ids.shape}")

    # 4. Tokenize events
    print("\nTokenizing astro events...")

    # Convert neuros-astro format to model input
    event_tensor, timestamp_tensor = tokenizer.from_neuros_astro_tokens(
        event_tokens=event_tokens,
        timestamps=timestamps,
        max_events=max_events
    )

    print(f"  Event tensor shape: {event_tensor.shape}")
    print(f"  Timestamp tensor shape: {timestamp_tensor.shape}")
    print(f"  Data type: {event_tensor.dtype}")
    print(f"  Value range: [{event_tensor.min().item():.3f}, {event_tensor.max().item():.3f}]")

    # Now pass through tokenizer forward pass to get model input
    print("\nGenerating model input...")
    astro_input, mask = tokenizer(event_tensor, return_sequence=False)

    print(f"  Model input shape: {astro_input.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Expected: (batch=1, seq_len={astro_input.shape[1]}, d_model={tokenizer.d_model})")

    # 5. Analyze token distribution
    print("\nToken Statistics:")
    print(f"  Mean: {astro_input.mean().item():.4f}")
    print(f"  Std: {astro_input.std().item():.4f}")
    print(f"  Non-zero elements: {(astro_input != 0).sum().item()} / {astro_input.numel()}")
    print(f"  Active positions (mask=True): {mask.sum().item()} / {mask.numel()}")

    # 6. Check event features
    if tokens_dict['feature_names'] is not None:
        print("\nEvent Features:")
        for i, fname in enumerate(tokens_dict['feature_names']):
            feat_vals = tokens_dict['event_tokens'][:max_events, i]
            print(f"  {fname:20s}: range=[{feat_vals.min():.3f}, {feat_vals.max():.3f}], mean={feat_vals.mean():.3f}")

    print("\n" + "="*60)
    print("Demo Complete! ✨")
    print("="*60)
    print("\nWhat was demonstrated:")
    print("  ✓ Loading astro tokens from neuros-astro export")
    print("  ✓ Creating AstroTokenizer with event-based processing")
    print("  ✓ Converting neuros-astro format to neuroFMx format")
    print("  ✓ Validating token shapes and distributions")
    print()
    print("Next steps:")
    print("  1. Install mamba-ssm: pip install mamba-ssm")
    print("  2. Run full demo: python examples/astro_integration_demo.py")
    print("  3. Add neural modality for multimodal fusion")
    print("  4. Train on prediction tasks")
    print()


def main():
    parser = argparse.ArgumentParser(description="Simple Astrocyte Tokenization Demo")

    parser.add_argument(
        '--astro-tokens',
        type=Path,
        required=True,
        help='Path to astro_tokens.npz from neuros-astro'
    )

    parser.add_argument(
        '--max-events',
        type=int,
        default=512,
        help='Maximum number of events to use (default: 512)'
    )

    args = parser.parse_args()

    # Validate input
    if not args.astro_tokens.exists():
        raise FileNotFoundError(f"Astro tokens not found: {args.astro_tokens}")

    run_simple_demo(args)


if __name__ == '__main__':
    main()
