"""
SAE Training Suite for NeuroFMX Mechanistic Interpretability

Trains Sparse Autoencoders on frozen NeuroFMX activations to decompose
polysemantic neurons into interpretable features.

Features:
- Multi-layer training (train SAEs on multiple layers simultaneously)
- Efficient batched training with GPU support
- Checkpoint management and resumption
- Activation hook system for extracting layer activations
- Support for both Transformer and Mamba backbones
- Advanced sparsity techniques (L1, TopK, L0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from neuros_mechint.sparse_autoencoder import SparseAutoencoder


class ActivationCache:
    """
    Manages extraction and caching of activations from NeuroFMX model.

    Uses hooks to efficiently capture layer activations during forward passes.
    """

    def __init__(self):
        self.activations = defaultdict(list)
        self.hooks = []

    def register_hooks(
        self,
        model: nn.Module,
        layer_names: List[str]
    ) -> None:
        """
        Register forward hooks on specified layers.

        Args:
            model: NeuroFMX model
            layer_names: List of layer names to hook (e.g., ['mamba_backbone.blocks.0', 'popt'])
        """
        self.clear()

        def get_activation(name):
            def hook(module, input, output):
                # Handle different output types
                if isinstance(output, torch.Tensor):
                    act = output.detach()
                elif isinstance(output, tuple):
                    act = output[0].detach()
                elif isinstance(output, dict):
                    act = output.get('hidden_states', output.get('latents')).detach()
                else:
                    return

                self.activations[name].append(act.cpu())
            return hook

        # Register hooks
        for name, module in model.named_modules():
            if name in layer_names:
                handle = module.register_forward_hook(get_activation(name))
                self.hooks.append(handle)
                print(f"Registered hook on: {name}")

    def clear(self):
        """Clear cached activations and remove hooks."""
        self.activations.clear()
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_activations(self, layer_name: str) -> torch.Tensor:
        """
        Get stacked activations for a layer.

        Args:
            layer_name: Name of layer

        Returns:
            Tensor of shape (n_samples, ...) with all cached activations
        """
        if layer_name not in self.activations:
            raise ValueError(f"No activations cached for layer: {layer_name}")

        acts = self.activations[layer_name]

        # Stack and flatten
        stacked = torch.cat(acts, dim=0)  # (N, ..., D)

        # Flatten to (N, D) if needed
        if len(stacked.shape) > 2:
            # Has sequence dimension, flatten it
            batch_size = stacked.shape[0]
            dim = stacked.shape[-1]
            stacked = stacked.reshape(-1, dim)

        return stacked

    def get_all_layer_names(self) -> List[str]:
        """Get names of all layers with cached activations."""
        return list(self.activations.keys())


class MultiLayerSAETrainer:
    """
    Trains Sparse Autoencoders on multiple NeuroFMX layers simultaneously.

    Features:
    - Efficient multi-layer training
    - Checkpointing and resumption
    - Progress tracking
    - Automatic layer discovery
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None,
        sae_config: Optional[Dict] = None,
        device: str = 'cuda',
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize multi-layer SAE trainer.

        Args:
            model: NeuroFMX model (will be frozen)
            layer_names: List of layer names to train SAEs on. If None, auto-detect.
            sae_config: Configuration for SAE architecture
            device: Device for training
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False

        # Auto-detect layers if not specified
        if layer_names is None:
            layer_names = self._auto_detect_layers()

        self.layer_names = layer_names

        # Default SAE config
        if sae_config is None:
            sae_config = {
                'expansion_factor': 8,  # dictionary_size = expansion_factor * latent_dim
                'sparsity_coefficient': 0.01,
                'tie_weights': False
            }
        self.sae_config = sae_config

        # Activation cache
        self.activation_cache = ActivationCache()

        # SAEs (will be initialized after seeing activations)
        self.saes: Dict[str, SparseAutoencoder] = {}

        # Checkpoint management
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training stats
        self.training_stats = defaultdict(list)

    def _auto_detect_layers(self) -> List[str]:
        """
        Auto-detect interesting layers to train SAEs on.

        Returns:
            List of layer names
        """
        layers = []

        # Look for Mamba blocks
        for name, module in self.model.named_modules():
            if 'mamba' in name.lower() and 'block' in name.lower():
                layers.append(name)

        # Look for Perceiver/fusion layers
        for name, module in self.model.named_modules():
            if 'perceiver' in name.lower() or 'fusion' in name.lower():
                if isinstance(module, nn.Module) and len(list(module.children())) == 0:
                    layers.append(name)

        # Look for PopT/aggregator
        for name, module in self.model.named_modules():
            if 'popt' in name.lower():
                layers.append(name)
                break

        # Take first few Mamba blocks + final aggregation
        if len(layers) > 5:
            layers = layers[:3] + layers[-2:]

        return layers

    def cache_activations(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_samples: Optional[int] = None
    ) -> None:
        """
        Cache activations from all layers.

        Args:
            dataloader: DataLoader with NeuroFMX inputs
            max_samples: Maximum number of samples to cache (None = all)
        """
        print(f"\nCaching activations from {len(self.layer_names)} layers...")

        # Register hooks
        self.activation_cache.register_hooks(self.model, self.layer_names)

        # Collect activations
        n_samples = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Caching")):
                # Handle different input formats
                if isinstance(batch, dict):
                    if 'inputs' in batch:
                        inputs = batch['inputs']
                    else:
                        inputs = batch
                else:
                    inputs = batch

                # Move to device
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.device) if torch.is_tensor(v) else v
                             for k, v in inputs.items()}
                else:
                    inputs = inputs.to(self.device)

                # Forward pass (hooks will capture activations)
                try:
                    _ = self.model(inputs)
                except Exception as e:
                    print(f"Warning: Forward pass failed: {e}")
                    continue

                # Check sample limit
                if isinstance(inputs, dict):
                    batch_size = next(iter(inputs.values())).shape[0]
                else:
                    batch_size = inputs.shape[0]

                n_samples += batch_size
                if max_samples and n_samples >= max_samples:
                    break

        print(f"Cached {n_samples} samples from {len(self.activation_cache.get_all_layer_names())} layers")

    def initialize_saes(self) -> None:
        """
        Initialize SAEs based on cached activation dimensions.
        """
        print("\nInitializing SAEs...")

        for layer_name in self.layer_names:
            # Get activation shape
            acts = self.activation_cache.get_activations(layer_name)
            latent_dim = acts.shape[-1]

            # Create SAE
            dictionary_size = int(latent_dim * self.sae_config['expansion_factor'])

            sae = SparseAutoencoder(
                latent_dim=latent_dim,
                dictionary_size=dictionary_size,
                sparsity_coefficient=self.sae_config['sparsity_coefficient'],
                tie_weights=self.sae_config['tie_weights']
            )

            sae = sae.to(self.device)
            self.saes[layer_name] = sae

            print(f"  {layer_name}: latent_dim={latent_dim}, dict_size={dictionary_size}")

    def train(
        self,
        num_epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        save_every: int = 10,
        log_every: int = 1
    ) -> Dict[str, List[float]]:
        """
        Train SAEs on all layers.

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            save_every: Save checkpoint every N epochs
            log_every: Log stats every N epochs

        Returns:
            Dictionary of training statistics
        """
        print(f"\nTraining SAEs for {num_epochs} epochs...")

        # Create optimizers
        optimizers = {
            name: torch.optim.Adam(sae.parameters(), lr=learning_rate)
            for name, sae in self.saes.items()
        }

        # Create dataloaders for each layer
        dataloaders = {}
        for layer_name in self.layer_names:
            acts = self.activation_cache.get_activations(layer_name)
            dataset = torch.utils.data.TensorDataset(acts)
            dataloaders[layer_name] = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, drop_last=True
            )

        # Training loop
        for epoch in range(num_epochs):
            epoch_stats = defaultdict(dict)

            # Train each layer
            for layer_name in self.layer_names:
                sae = self.saes[layer_name]
                optimizer = optimizers[layer_name]
                dataloader = dataloaders[layer_name]

                # Epoch stats
                total_loss = 0.0
                total_recon_loss = 0.0
                total_sparsity_loss = 0.0
                total_l0 = 0.0
                n_batches = 0

                for (batch_acts,) in dataloader:
                    batch_acts = batch_acts.to(self.device)

                    # Forward
                    recon, features, loss = sae(batch_acts, return_features=True)

                    # Compute losses
                    recon_loss = F.mse_loss(recon, batch_acts)
                    sparsity_loss = sae.sparsity_coef * torch.abs(features).mean()

                    # L0 sparsity (fraction of active features)
                    l0 = (features > 0).float().mean()

                    # Backward
                    optimizer.zero_grad()
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)

                    optimizer.step()

                    # Stats
                    total_loss += loss.item()
                    total_recon_loss += recon_loss.item()
                    total_sparsity_loss += sparsity_loss.item()
                    total_l0 += l0.item()
                    n_batches += 1

                # Epoch averages
                epoch_stats[layer_name] = {
                    'loss': total_loss / n_batches,
                    'recon_loss': total_recon_loss / n_batches,
                    'sparsity_loss': total_sparsity_loss / n_batches,
                    'l0_sparsity': total_l0 / n_batches
                }

            # Store stats
            for layer_name, stats in epoch_stats.items():
                for key, value in stats.items():
                    self.training_stats[f"{layer_name}/{key}"].append(value)

            # Logging
            if epoch % log_every == 0:
                print(f"\nEpoch {epoch}/{num_epochs}")
                for layer_name, stats in epoch_stats.items():
                    print(f"  {layer_name}:")
                    print(f"    Loss: {stats['loss']:.6f} | "
                          f"Recon: {stats['recon_loss']:.6f} | "
                          f"L0: {stats['l0_sparsity']:.3f}")

            # Checkpointing
            if self.checkpoint_dir and (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch)

        print("\nTraining complete!")
        return dict(self.training_stats)

    def save_checkpoint(self, epoch: int) -> None:
        """Save checkpoint for all SAEs."""
        if not self.checkpoint_dir:
            return

        checkpoint = {
            'epoch': epoch,
            'layer_names': self.layer_names,
            'sae_config': self.sae_config,
            'training_stats': dict(self.training_stats),
            'saes': {}
        }

        for layer_name, sae in self.saes.items():
            checkpoint['saes'][layer_name] = sae.state_dict()

        save_path = self.checkpoint_dir / f"sae_checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load checkpoint and resume training.

        Returns:
            Epoch number to resume from
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load SAEs
        for layer_name, state_dict in checkpoint['saes'].items():
            if layer_name in self.saes:
                self.saes[layer_name].load_state_dict(state_dict)

        # Load stats
        self.training_stats = defaultdict(list, checkpoint['training_stats'])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']

    def get_sae(self, layer_name: str) -> SparseAutoencoder:
        """Get trained SAE for a specific layer."""
        if layer_name not in self.saes:
            raise ValueError(f"No SAE for layer: {layer_name}")
        return self.saes[layer_name]

    def get_all_saes(self) -> Dict[str, SparseAutoencoder]:
        """Get all trained SAEs."""
        return self.saes


class SAETrainingPipeline:
    """
    End-to-end pipeline for SAE training on NeuroFMX.

    Usage:
        pipeline = SAETrainingPipeline(model, dataloader)
        pipeline.run(num_epochs=100)
        saes = pipeline.get_saes()
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        layer_names: Optional[List[str]] = None,
        sae_config: Optional[Dict] = None,
        device: str = 'cuda',
        checkpoint_dir: str = './sae_checkpoints'
    ):
        """
        Initialize training pipeline.

        Args:
            model: NeuroFMX model
            dataloader: DataLoader with training data
            layer_names: Layers to train SAEs on
            sae_config: SAE configuration
            device: Training device
            checkpoint_dir: Checkpoint directory
        """
        self.dataloader = dataloader

        self.trainer = MultiLayerSAETrainer(
            model=model,
            layer_names=layer_names,
            sae_config=sae_config,
            device=device,
            checkpoint_dir=checkpoint_dir
        )

    def run(
        self,
        num_epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        max_cache_samples: Optional[int] = None,
        save_every: int = 10
    ) -> Dict[str, List[float]]:
        """
        Run full training pipeline.

        Args:
            num_epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            max_cache_samples: Max samples to cache (None = all)
            save_every: Save frequency

        Returns:
            Training statistics
        """
        # Cache activations
        self.trainer.cache_activations(
            self.dataloader,
            max_samples=max_cache_samples
        )

        # Initialize SAEs
        self.trainer.initialize_saes()

        # Train
        stats = self.trainer.train(
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            save_every=save_every
        )

        return stats

    def get_saes(self) -> Dict[str, SparseAutoencoder]:
        """Get trained SAEs."""
        return self.trainer.get_all_saes()

    def get_trainer(self) -> MultiLayerSAETrainer:
        """Get trainer instance."""
        return self.trainer


# Example usage
if __name__ == '__main__':
    from neuros_neurofm.models.multimodal_neurofmx import MultiModalNeuroFMX
    from neuros_neurofm.datasets.synthetic import create_synthetic_dataloader

    # Create model
    model = MultiModalNeuroFMX(
        d_model=512,
        n_mamba_blocks=4,
        n_latents=64,
        latent_dim=512
    )

    # Create synthetic data
    dataloader = create_synthetic_dataloader(
        batch_size=8,
        n_batches=20,
        modalities=['spike', 'eeg']
    )

    # Run SAE training pipeline
    pipeline = SAETrainingPipeline(
        model=model,
        dataloader=dataloader,
        sae_config={
            'expansion_factor': 8,
            'sparsity_coefficient': 0.01,
            'tie_weights': False
        },
        checkpoint_dir='./sae_checkpoints'
    )

    # Train
    stats = pipeline.run(
        num_epochs=50,
        batch_size=128,
        learning_rate=1e-3,
        max_cache_samples=1000
    )

    # Get trained SAEs
    saes = pipeline.get_saes()
    print(f"\nTrained {len(saes)} SAEs:")
    for layer_name, sae in saes.items():
        print(f"  {layer_name}: {sae.get_feature_statistics()}")
