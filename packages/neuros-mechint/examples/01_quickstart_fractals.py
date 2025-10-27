"""
Quickstart: Fractal Analysis

This example demonstrates how to:
1. Compute fractal dimension of neural signals
2. Use fractal regularization during training
3. Track fractal dimension evolution

Requirements:
    pip install neuros-mechint torch
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from neuros_mechint.fractals import (
    HiguchiFractalDimension,
    SpectralPrior,
    LatentFDTracker,
    FractionalBrownianMotion,
)


def example_1_compute_fractal_dimension():
    """Example 1: Compute Higuchi Fractal Dimension"""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Computing Fractal Dimension")
    print("=" * 60)

    # Create fractal dimension estimator
    fd = HiguchiFractalDimension(k_max=10, device='cpu')

    # Generate some signals
    signal = torch.randn(32, 1000)  # 32 signals, 1000 timesteps each

    # Compute fractal dimension
    fractal_dims = fd.compute(signal)

    print(f"\nFractal Dimension Statistics:")
    print(f"  Mean: {fractal_dims.mean():.3f}")
    print(f"  Std:  {fractal_dims.std():.3f}")
    print(f"  Min:  {fractal_dims.min():.3f}")
    print(f"  Max:  {fractal_dims.max():.3f}")

    # Plot distribution
    plt.figure(figsize=(10, 4))
    plt.hist(fractal_dims.numpy(), bins=20, edgecolor='black')
    plt.xlabel('Fractal Dimension')
    plt.ylabel('Count')
    plt.title('Distribution of Fractal Dimensions')
    plt.axvline(1.5, color='r', linestyle='--', label='Expected (white noise)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fractal_distribution.png', dpi=150)
    print("\nSaved fractal_distribution.png")


def example_2_fractal_regularization():
    """Example 2: Train with Fractal Regularization"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Fractal Regularization During Training")
    print("=" * 60)

    # Simple model
    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    # Create fractal regularizer
    fractal_reg = SpectralPrior(
        target_beta=1.0,  # Target 1/f spectrum
        weight=0.01,      # Regularization strength
    )

    # Dummy data
    x = torch.randn(32, 100)
    y = torch.randint(0, 10, (32,))

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    losses = []
    fractal_losses = []

    for epoch in range(50):
        optimizer.zero_grad()

        # Forward pass
        logits = model(x)
        task_loss = criterion(logits, y)

        # Get intermediate activations
        activations = []
        def hook_fn(module, input, output):
            activations.append(output)

        hooks = []
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                hooks.append(layer.register_forward_hook(hook_fn))

        _ = model(x)  # Run again to capture activations

        # Compute fractal regularization
        frac_loss = fractal_reg(activations[1])  # Middle layer

        # Total loss
        total_loss = task_loss + frac_loss

        # Backward
        total_loss.backward()
        optimizer.step()

        # Remove hooks
        for hook in hooks:
            hook.remove()

        losses.append(task_loss.item())
        fractal_losses.append(frac_loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50: Task Loss = {task_loss:.4f}, "
                  f"Fractal Loss = {frac_loss:.4f}")

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Task Loss')
    plt.title('Task Loss During Training')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(fractal_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Fractal Regularization Loss')
    plt.title('Fractal Loss During Training')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fractal_training.png', dpi=150)
    print("\nSaved fractal_training.png")


def example_3_track_fd_evolution():
    """Example 3: Track FD Evolution During Training"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Tracking FD Evolution")
    print("=" * 60)

    # Simple model
    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    # Create FD tracker
    tracker = LatentFDTracker(
        layer_names=['1', '3'],  # Track middle layers
        k_max=10,
        log_interval=10,
    )

    # Dummy data
    x = torch.randn(32, 100)
    y = torch.randint(0, 10, (32,))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for step in range(100):
        optimizer.zero_grad()

        # Capture activations
        activations = {}
        def make_hook(name):
            def hook_fn(module, input, output):
                activations[name] = output.detach()
            return hook_fn

        hooks = []
        for name, layer in model.named_modules():
            if name in ['1', '3']:
                hooks.append(layer.register_forward_hook(make_hook(name)))

        # Forward
        logits = model(x)
        loss = criterion(logits, y)

        # Track FD
        tracker.compute(activations, step)

        # Backward
        loss.backward()
        optimizer.step()

        # Remove hooks
        for hook in hooks:
            hook.remove()

        if (step + 1) % 20 == 0:
            print(f"Step {step+1}/100: Loss = {loss:.4f}")

    # Plot FD evolution
    tracker.plot_history()
    plt.savefig('fd_evolution.png', dpi=150)
    print("\nSaved fd_evolution.png")

    # Get final stats
    final_fd = tracker.get_current_fd()
    print(f"\nFinal Fractal Dimensions:")
    for layer, fd_value in final_fd.items():
        print(f"  Layer {layer}: {fd_value:.3f}")


def example_4_generate_fractal_stimuli():
    """Example 4: Generate Fractal Stimuli"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Generating Fractal Stimuli")
    print("=" * 60)

    # Generate fractional Brownian motion with different Hurst exponents
    fbm = FractionalBrownianMotion(n_samples=1000, hurst=0.7)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    for i, hurst in enumerate([0.3, 0.5, 0.7]):
        fbm.hurst = hurst
        signal = fbm.generate(batch_size=1)

        axes[i].plot(signal[0].numpy())
        axes[i].set_title(f'fBm with Hurst = {hurst}')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Amplitude')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fractal_stimuli.png', dpi=150)
    print("\nSaved fractal_stimuli.png")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NEUROS-MECHINT: Fractal Analysis Quickstart")
    print("=" * 60)

    example_1_compute_fractal_dimension()
    example_2_fractal_regularization()
    example_3_track_fd_evolution()
    example_4_generate_fractal_stimuli()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("Check the generated PNG files for visualizations.")
    print("=" * 60 + "\n")
