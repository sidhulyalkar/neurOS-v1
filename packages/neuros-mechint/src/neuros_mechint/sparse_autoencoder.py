"""
Sparse Autoencoder for Decomposing Polysemantic Neurons

Inspired by Anthropic's "Towards Monosemanticity" work.

Learns an overcomplete dictionary of interpretable features
from polysemantic neuron activations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm
import numpy as np


class SparseAutoencoder(nn.Module):
    """
    Sparse autoencoder for decomposing polysemantic features.

    Learns an overcomplete basis where each basis vector represents
    an interpretable feature.

    Args:
        latent_dim: Input dimensionality (model latent dimension)
        dictionary_size: Size of feature dictionary (> latent_dim for overcompleteness)
        sparsity_coefficient: L1 penalty strength
        tie_weights: Whether to tie encoder and decoder weights
    """

    def __init__(
        self,
        latent_dim: int = 512,
        dictionary_size: int = 4096,
        sparsity_coefficient: float = 0.01,
        tie_weights: bool = False
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.dictionary_size = dictionary_size
        self.sparsity_coef = sparsity_coefficient
        self.tie_weights = tie_weights

        # Encoder: latent -> features
        self.encoder = nn.Linear(latent_dim, dictionary_size, bias=True)

        # Decoder: features -> latent
        if tie_weights:
            # Tied weights: decoder = encoder.T
            self.decoder = None
        else:
            self.decoder = nn.Linear(dictionary_size, latent_dim, bias=True)

        # Initialize with small values
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        # Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.encoder.weight)
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)

        if self.decoder is not None:
            nn.init.xavier_uniform_(self.decoder.weight)
            if self.decoder.bias is not None:
                nn.init.zeros_(self.decoder.bias)

        # Optionally: normalize columns to unit norm
        with torch.no_grad():
            self.encoder.weight.div_(
                self.encoder.weight.norm(dim=0, keepdim=True) + 1e-8
            )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (batch, latent_dim) activations from model
            return_features: Whether to return feature activations

        Returns:
            reconstruction: (batch, latent_dim) reconstructed activations
            features: (batch, dictionary_size) feature activations (if return_features)
            loss: Total loss (reconstruction + sparsity)
        """
        # Encode to features
        pre_activation = self.encoder(x)
        features = F.relu(pre_activation)  # Enforce non-negativity

        # Decode to reconstruction
        if self.tie_weights:
            # Use transposed encoder weights
            reconstruction = F.linear(features, self.encoder.weight.t())
            if self.encoder.bias is not None:
                reconstruction = reconstruction + self.encoder.bias
        else:
            reconstruction = self.decoder(features)

        # Compute losses
        recon_loss = F.mse_loss(reconstruction, x)
        sparsity_loss = self.sparsity_coef * torch.abs(features).mean()

        total_loss = recon_loss + sparsity_loss

        if return_features:
            return reconstruction, features, total_loss
        else:
            return reconstruction, total_loss

    def train_on_raw_activations(
        self,
        activations: torch.Tensor,
        num_epochs: int = 100,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        device: str = 'cuda'
    ) -> List[float]:
        """
        Train SAE directly on provided activation vectors.

        Args:
            activations: Tensor of activation vectors (N, latent_dim)
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            device: Device to train on

        Returns:
            losses: List of average losses per epoch
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(activations)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        losses = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for (batch,) in dataloader:  # Note comma to unpack the tuple
                batch = batch.to(device)
                
                # Forward pass
                reconstruction, loss = self(batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs}: Loss = {avg_loss:.6f}")
        
        return losses

    def train_on_activations(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        device: str = 'cuda'
    ):
        """
        Train SAE on cached activations from main model.

        Args:
            model: Main NeuroFMx model
            dataloader: DataLoader with input data
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            device: Device
        """
        self.to(device)
        model.to(device)
        model.eval()

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Phase 1: Cache activations
        print("Phase 1: Caching activations from model...")
        all_activations = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Caching"):
                # Move inputs to device
                modality_dict = {}
                for k, v in batch['inputs'].items():
                    modality_dict[k] = v.to(device)

                # Forward through main model
                outputs = model(modality_dict)

                # Extract latents (pooled)
                if 'latents' in outputs:
                    latents = outputs['latents']

                    # Pool over sequence: (B, n_latents, D) -> (B, D)
                    if len(latents.shape) == 3:
                        latents = latents.mean(dim=1)

                    all_activations.append(latents.cpu())

        # Stack all activations
        all_activations = torch.cat(all_activations, dim=0)  # (N, latent_dim)
        print(f"Cached {len(all_activations)} activation vectors")

        # Phase 2: Train SAE
        print(f"Phase 2: Training SAE for {num_epochs} epochs...")

        dataset = torch.utils.data.TensorDataset(all_activations)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=True
        )

        for epoch in range(num_epochs):
            epoch_losses = []

            for (batch_acts,) in train_loader:
                batch_acts = batch_acts.to(device)

                # Forward
                recon, loss = self(batch_acts)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs}: Loss = {avg_loss:.6f}")

        print("SAE training complete!")

    def get_feature_activations(
        self,
        activations: torch.Tensor
    ) -> torch.Tensor:
        """
        Get sparse feature activations for given inputs.

        Args:
            activations: (batch, latent_dim) model activations

        Returns:
            features: (batch, dictionary_size) feature activations
        """
        # Ensure activations are on the same device as the encoder weights
        # and return features on CPU by default to avoid device-mismatch
        # when the caller keeps activations on CPU (common in analysis code).
        with torch.no_grad():
            # Determine encoder device (works if model has parameters)
            try:
                encoder_device = next(self.encoder.parameters()).device
            except StopIteration:
                # Fallback to current default device
                encoder_device = torch.device('cpu')

            # Move inputs to encoder device if needed
            activations_device = activations.device
            if activations_device != encoder_device:
                activations = activations.to(encoder_device)

            pre_act = self.encoder(activations)
            features = F.relu(pre_act)

            # Move features to CPU for downstream analysis/plotting convenience
            features_cpu = features.cpu()

        return features_cpu

    def interpret_feature(
        self,
        feature_id: int,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        top_k: int = 10,
        device: str = 'cuda'
    ) -> Dict:
        """
        Interpret a learned feature by finding inputs that maximally activate it.

        Args:
            feature_id: ID of feature to interpret
            model: Main NeuroFMx model
            dataloader: DataLoader with inputs
            top_k: Number of top activating examples to return
            device: Device

        Returns:
            interpretation: Dict with top activating inputs and metadata
        """
        self.eval()
        model.eval()

        max_activations = []
        max_inputs = []
        max_metadata = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Interpreting feature {feature_id}"):
                modality_dict = {}
                for k, v in batch['inputs'].items():
                    modality_dict[k] = v.to(device)

                # Forward through model
                outputs = model(modality_dict)
                latents = outputs['latents']

                if len(latents.shape) == 3:
                    latents = latents.mean(dim=1)

                # Get feature activations
                features = self.get_feature_activations(latents)

                # Get activations for this feature
                feature_acts = features[:, feature_id]

                # Store top-k
                for i, act in enumerate(feature_acts):
                    max_activations.append(act.item())
                    max_inputs.append({k: v[i].cpu() for k, v in modality_dict.items()})

                    if 'metadata' in batch:
                        max_metadata.append(batch['metadata'][i])

        # Get top-k overall
        if len(max_activations) > top_k:
            top_indices = np.argsort(max_activations)[-top_k:][::-1]
        else:
            top_indices = list(range(len(max_activations)))

        interpretation = {
            'feature_id': feature_id,
            'top_activations': [max_activations[i] for i in top_indices],
            'top_inputs': [max_inputs[i] for i in top_indices],
            'top_metadata': [max_metadata[i] for i in top_indices] if max_metadata else None,
            'mean_activation': np.mean(max_activations),
            'max_activation': np.max(max_activations)
        }

        return interpretation

    def get_feature_statistics(self) -> Dict:
        """Get statistics about learned features."""
        # Feature sparsity
        with torch.no_grad():
            encoder_norms = self.encoder.weight.norm(dim=0)

            stats = {
                'dictionary_size': self.dictionary_size,
                'latent_dim': self.latent_dim,
                'overcompleteness': self.dictionary_size / self.latent_dim,
                'mean_feature_norm': encoder_norms.mean().item(),
                'max_feature_norm': encoder_norms.max().item(),
                'min_feature_norm': encoder_norms.min().item()
            }

        return stats


# Example usage
if __name__ == '__main__':
    # Create SAE
    sae = SparseAutoencoder(
        latent_dim=512,
        dictionary_size=4096,
        sparsity_coefficient=0.01
    )

    print(f"SAE parameters: {sum(p.numel() for p in sae.parameters()):,}")
    print(f"Feature statistics: {sae.get_feature_statistics()}")

    # Test forward pass
    dummy_activations = torch.randn(16, 512)
    reconstruction, features, loss = sae(dummy_activations, return_features=True)

    print(f"\nInput shape: {dummy_activations.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"Loss: {loss.item():.6f}")
    print(f"Feature sparsity: {(features == 0).float().mean().item():.3f}")
