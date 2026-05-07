#!/usr/bin/env python
"""
ENGRAM-FMx Example with Allen Visual Coding Data.

Demonstrates how to use ENGRAM-FMx with real neural data:
1. Load Allen Visual Coding sessions
2. Tokenize with CalciumTokenizer
3. Train ENGRAM-FMx with forecasting objective
4. Visualize diagnostics

Usage:
    python examples/engram_fmx_allen_example.py
    python examples/engram_fmx_allen_example.py --sessions 5 --max_steps 500
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ENGRAM imports
from neuros_neurofm.backbones.engram_fmx import ENGRAMBackbone, ENGRAMFMxConfig
from neuros_neurofm.backbones.engram_fmx.diagnostics import (
    MemoryTracker,
    LatentTracker,
    create_diagnostic_dashboard,
)

# Tokenizer imports
try:
    from neuros_neurofm.tokenizers import CalciumTokenizer, BinnedTokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_allen_data(
    num_sessions: int = 5,
    neurons_per_session: int = 100,
    timesteps: int = 1000,
    bin_size_ms: int = 50,
) -> Dict[str, torch.Tensor]:
    """Create synthetic data mimicking Allen Visual Coding format.

    This is for demonstration - replace with actual Allen data loading.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary with 'dff' (calcium traces), 'behavior', 'stimulus'.
    """
    logger.info(f"Creating synthetic Allen-like data: {num_sessions} sessions")

    # Simulate calcium imaging data (dF/F)
    # Shape: [sessions, timesteps, neurons]
    dff = torch.randn(num_sessions, timesteps, neurons_per_session) * 0.5
    dff = dff.abs()  # dF/F is typically positive

    # Add some temporal structure (simulated neural dynamics)
    for s in range(num_sessions):
        # Add slow oscillations
        t = torch.linspace(0, 10, timesteps)
        for n in range(neurons_per_session):
            freq = 0.1 + 0.5 * torch.rand(1).item()
            dff[s, :, n] += 0.3 * torch.sin(2 * 3.14159 * freq * t)

    # Simulate behavior (running speed)
    behavior = torch.randn(num_sessions, timesteps, 1).abs() * 10

    # Simulate stimulus (orientation, 8 directions)
    stimulus = torch.randint(0, 8, (num_sessions, timesteps, 1)).float()

    return {
        "dff": dff,
        "behavior": behavior,
        "stimulus": stimulus,
        "num_sessions": num_sessions,
        "timesteps": timesteps,
        "bin_size_ms": bin_size_ms,
    }


def tokenize_data(
    data: Dict[str, torch.Tensor],
    hidden_dim: int = 128,
    context_length: int = 64,
) -> torch.Tensor:
    """Tokenize neural data for ENGRAM-FMx.

    Parameters
    ----------
    data : Dict[str, torch.Tensor]
        Raw data dictionary.
    hidden_dim : int
        Target embedding dimension.
    context_length : int
        Sequence length for each sample.

    Returns
    -------
    torch.Tensor
        Tokenized sequences [num_samples, context_length, hidden_dim].
    """
    dff = data["dff"]  # [sessions, timesteps, neurons]
    num_sessions, timesteps, neurons = dff.shape

    # Simple tokenization: project neural activity to hidden_dim
    # In practice, use CalciumTokenizer from neuros_neurofm.tokenizers
    projection = nn.Linear(neurons, hidden_dim)

    # Extract sliding windows
    samples = []
    for s in range(num_sessions):
        for t in range(0, timesteps - context_length, context_length // 2):
            window = dff[s, t:t+context_length, :]  # [context_length, neurons]
            tokenized = projection(window)  # [context_length, hidden_dim]
            samples.append(tokenized)

    tokens = torch.stack(samples, dim=0)  # [num_samples, context_length, hidden_dim]

    logger.info(f"Tokenized {len(samples)} samples, shape: {tokens.shape}")

    return tokens.detach()


def create_forecasting_targets(
    tokens: torch.Tensor,
    forecast_horizon: int = 8,
) -> tuple:
    """Create forecasting targets.

    Parameters
    ----------
    tokens : torch.Tensor
        Tokenized sequences [B, T, D].
    forecast_horizon : int
        Steps ahead to forecast.

    Returns
    -------
    tuple
        (inputs, targets, mask)
    """
    B, T, D = tokens.shape

    inputs = tokens.clone()
    targets = torch.zeros_like(tokens)
    mask = torch.zeros(B, T, dtype=torch.bool)

    # Target: predict tokens[t+horizon] from tokens[t]
    for t in range(T - forecast_horizon):
        targets[:, t, :] = tokens[:, t + forecast_horizon, :]
        mask[:, t] = True

    return inputs, targets, mask


def train_engram_allen(
    config: ENGRAMFMxConfig,
    data: Dict[str, torch.Tensor],
    max_steps: int = 500,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str = "auto",
    output_dir: str = "runs/engram_allen",
) -> Dict[str, any]:
    """Train ENGRAM-FMx on Allen-like data.

    Parameters
    ----------
    config : ENGRAMFMxConfig
        Model configuration.
    data : Dict[str, torch.Tensor]
        Data dictionary from create_synthetic_allen_data.
    max_steps : int
        Maximum training steps.
    batch_size : int
        Batch size.
    learning_rate : float
        Learning rate.
    device : str
        Device to train on.
    output_dir : str
        Output directory.

    Returns
    -------
    Dict[str, any]
        Training results.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training on device: {device}")

    # Tokenize data
    tokens = tokenize_data(data, hidden_dim=config.hidden_dim)

    # Create forecasting dataset
    inputs, targets, mask = create_forecasting_targets(tokens, forecast_horizon=8)

    dataset = TensorDataset(inputs, targets, mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = ENGRAMBackbone(config)
    model = model.to(device)

    logger.info(f"Model parameters: {model.get_num_params():,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Trackers
    memory_tracker = MemoryTracker()
    latent_tracker = LatentTracker()

    # Training loop
    model.train()
    step = 0
    losses = []

    pbar = tqdm(total=max_steps, desc="Training")

    while step < max_steps:
        for batch_inputs, batch_targets, batch_mask in dataloader:
            if step >= max_steps:
                break

            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_mask = batch_mask.to(device)

            # Forward
            output = model(batch_inputs)

            # Compute loss (masked MSE)
            pred = output.sequence_output
            diff = ((pred - batch_targets) ** 2).mean(dim=-1)
            mask_float = batch_mask.float()
            loss = (diff * mask_float).sum() / (mask_float.sum() + 1e-8)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            step += 1
            pbar.update(1)

            # Track diagnostics
            if step % 50 == 0:
                diagnostics = output.diagnostics

                # Extract memory weights from diagnostics
                for key in diagnostics:
                    if "memory_weights" in key:
                        memory_tracker.update(step, diagnostics[key])
                        break

                # Track latents
                latent_tracker.update(step, output.latent_output.detach())

                avg_loss = sum(losses[-50:]) / 50
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                logger.info(f"Step {step}: loss={avg_loss:.4f}")

    pbar.close()

    # Save model
    torch.save(model.state_dict(), output_path / "model.pt")

    # Save training curves
    results = {
        "final_loss": sum(losses[-100:]) / min(100, len(losses)),
        "losses": losses,
        "num_params": model.get_num_params(),
        "steps": step,
    }

    with open(output_path / "results.json", "w") as f:
        json.dump({k: v for k, v in results.items() if k != "losses"}, f, indent=2)

    # Create diagnostic dashboard
    try:
        # Get final diagnostics
        with torch.no_grad():
            sample_input = inputs[:4].to(device)
            final_output = model(sample_input)

        dashboard_diagnostics = {
            **final_output.diagnostics,
            "sequence_length": inputs.shape[1],
        }

        # Find memory weights in diagnostics
        for key, value in final_output.diagnostics.items():
            if "memory_weights" in key and isinstance(value, torch.Tensor):
                dashboard_diagnostics["memory_weights"] = value
                break

        fig = create_diagnostic_dashboard(
            dashboard_diagnostics,
            save_path=str(output_path / "diagnostics.png"),
        )
        logger.info(f"Saved diagnostic dashboard to {output_path / 'diagnostics.png'}")
    except Exception as e:
        logger.warning(f"Could not create diagnostic dashboard: {e}")

    logger.info(f"Training complete. Final loss: {results['final_loss']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="ENGRAM-FMx Allen Example")

    parser.add_argument("--sessions", type=int, default=5, help="Number of sessions")
    parser.add_argument("--max_steps", type=int, default=500, help="Max training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--model_size", type=str, default="tiny",
                        choices=["tiny", "small", "medium"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="runs/engram_allen")

    args = parser.parse_args()

    # Create config
    if args.model_size == "tiny":
        config = ENGRAMFMxConfig.tiny()
    elif args.model_size == "small":
        config = ENGRAMFMxConfig.small()
    else:
        config = ENGRAMFMxConfig.medium()

    # Create synthetic Allen-like data
    data = create_synthetic_allen_data(num_sessions=args.sessions)

    # Train
    results = train_engram_allen(
        config=config,
        data=data,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
    )

    print(f"\nTraining Results:")
    print(f"  Final Loss: {results['final_loss']:.4f}")
    print(f"  Parameters: {results['num_params']:,}")
    print(f"  Steps: {results['steps']}")


if __name__ == "__main__":
    main()
