"""
Demo: Integrating Astrocyte Tokens with neuroFMx

This script demonstrates how to:
1. Load astrocyte event tokens from neuros-astro
2. Feed them into the multimodal neuroFMx model
3. Run inference and extract representations

Usage:
    python examples/astro_integration_demo.py --astro-tokens /path/to/astro_tokens.npz
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from neuros_neurofm.models.multimodal_neurofmx import MultiModalNeuroFMX
from neuros_neurofm.tokenizers import AstroTokenizer


def load_neuros_astro_tokens(token_file: Path):
    """
    Load astrocyte event tokens from neuros-astro.

    Expected format from neuros-astro export:
    - tokens: (n_events, n_features) - event features
    - timestamps_s: (n_events,) - event times in seconds
    - region_ids: (n_events,) - which astrocyte
    - metadata_json: JSON string with session info
    - feature_names: names of features
    - session_id: session identifier

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
        import json
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


def prepare_astro_batch(tokens_dict, n_astrocytes=None, max_events=512):
    """
    Convert neuros-astro tokens to model-ready batch format.

    Args:
        tokens_dict: From load_neuros_astro_tokens()
        n_astrocytes: Number of astrocytes (inferred if None)
        max_events: Max events to include

    Returns:
        Dict ready for AstroTokenizer.from_neuros_astro_tokens()
    """
    event_tokens = tokens_dict['event_tokens'][:max_events]
    timestamps = tokens_dict['timestamps'][:max_events]
    region_ids = tokens_dict['region_ids'][:max_events]

    if n_astrocytes is None:
        n_astrocytes = int(region_ids.max()) + 1

    # Convert to tensors
    batch = {
        'event_tokens': torch.from_numpy(event_tokens).float(),
        'timestamps': torch.from_numpy(timestamps).float(),
        'region_ids': torch.from_numpy(region_ids).long(),
        'n_astrocytes': n_astrocytes
    }

    return batch


def run_demo(args):
    """Run astro integration demo."""

    print("="*60)
    print("Astrocyte + neuroFMx Integration Demo")
    print("="*60)
    print()

    # 1. Load astro tokens from neuros-astro
    tokens_dict = load_neuros_astro_tokens(args.astro_tokens)
    n_astrocytes = len(np.unique(tokens_dict['region_ids']))

    # 2. Create multimodal model with astro support
    print("\nCreating MultiModalNeuroFMX with astrocyte support...")

    modality_config = {
        'astro': {
            'n_astrocytes': n_astrocytes,
            'seq_len': 100,
            'sampling_rate': 10.0,  # 10 Hz from neuros-astro
            'use_events': True,     # Using event-based tokens
        }
    }

    model = MultiModalNeuroFMX(
        d_model=512,
        n_mamba_blocks=4,
        n_latents=64,
        latent_dim=512,
        modality_config=modality_config,
        use_domain_adversarial=False
    )

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Supported modalities: {model.get_modality_names()}")

    # 3. Prepare batch
    print("\nPreparing astro data batch...")
    batch = prepare_astro_batch(
        tokens_dict,
        n_astrocytes=n_astrocytes,
        max_events=args.max_events
    )

    print(f"  Event tokens shape: {batch['event_tokens'].shape}")
    print(f"  Timestamps shape: {batch['timestamps'].shape}")
    print(f"  Number of astrocytes: {batch['n_astrocytes']}")

    # 4. Get tokenizer and convert to model input
    print("\nTokenizing astro events...")
    astro_tokenizer = model.tokenizers['astro']

    # Convert neuros-astro format to tensors
    event_tensor, timestamp_tensor = astro_tokenizer.from_neuros_astro_tokens(
        event_tokens=batch['event_tokens'],
        timestamps=batch['timestamps'],
        max_events=args.max_events
    )

    print(f"  Event tensor shape: {event_tensor.shape}")
    print(f"  Timestamp tensor shape: {timestamp_tensor.shape}")

    # Pass through tokenizer to get model input
    astro_input, mask = astro_tokenizer(event_tensor, return_sequence=False)
    print(f"  Model input shape: {astro_input.shape}")
    print(f"  Mask shape: {mask.shape}")

    # 5. Add batch dimension if needed
    if astro_input.dim() == 2:
        astro_input = astro_input.unsqueeze(0)  # (1, S, D)

    # 6. Run through model
    print("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(
            modality_dict={'astro': astro_input},
            task='multi-task'
        )

    # 7. Inspect outputs
    print("\n" + "="*60)
    print("Model Outputs:")
    print("="*60)

    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:20s}: {str(value.shape):20s} {value.dtype}")

            # Show some stats
            if value.numel() > 0:
                print(f"    └─ range: [{value.min().item():.3f}, {value.max().item():.3f}]")

    # 8. Extract astrocyte representations
    if 'latents' in outputs:
        latents = outputs['latents']  # (B, n_latents, latent_dim)
        print(f"\nExtracted astrocyte latent representations:")
        print(f"  Shape: {latents.shape}")
        print(f"  Can be used for downstream tasks!")

    print("\n" + "="*60)
    print("Demo Complete! ✨")
    print("="*60)
    print("\nNext steps:")
    print("  1. Add neural modality (spike/calcium) for multimodal fusion")
    print("  2. Train on prediction tasks")
    print("  3. Run ablation: with vs without astro")
    print("  4. Analyze astro contribution to model performance")
    print()


def main():
    parser = argparse.ArgumentParser(description="Astrocyte + neuroFMx Integration Demo")

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

    run_demo(args)


if __name__ == '__main__':
    main()
