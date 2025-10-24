"""
MultiModalNeuroFMX: Multimodal Foundation Model

Extends NeuroFMX to handle multiple neural data modalities simultaneously:
- Electrophysiology: Spikes, LFP
- Calcium imaging
- EEG
- fMRI
- ECoG
- EMG

Implements cross-modal fusion, domain adversarial training, and mechanistic interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Union, Tuple

from neuros_neurofm.models.mamba_backbone import MambaBackbone
from neuros_neurofm.models.popt import PopT, PopTWithLatents
from neuros_neurofm.fusion.perceiver import PerceiverIO
from neuros_neurofm.models.heads import MultiTaskHeads
from neuros_neurofm.tokenizers import (
    SpikeTokenizer, BinnedTokenizer, LFPTokenizer, CalciumTokenizer
)
from neuros_neurofm.tokenizers.eeg_tokenizer import EEGTokenizer
from neuros_neurofm.tokenizers.fmri_tokenizer import fMRITokenizer


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for domain adversarial training.

    During forward pass, acts as identity.
    During backward pass, reverses gradients by multiplying by -lambda.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class DomainDiscriminator(nn.Module):
    """
    Discriminator for domain/species classification.

    Used in domain adversarial training to learn domain-invariant features.
    """

    def __init__(self, input_dim: int, n_domains: int, hidden_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_domains)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) features

        Returns:
            logits: (batch, n_domains) domain classification logits
        """
        return self.network(x)


class MultiModalNeuroFMX(nn.Module):
    """
    Multimodal Neural Foundation Model.

    Architecture:
    1. Modality-specific tokenizers
    2. Learned modality embeddings
    3. Cross-modal Perceiver fusion
    4. Mamba SSM backbone for temporal modeling
    5. PopT aggregation
    6. Multi-task heads
    7. Domain discriminator (optional)

    Args:
        d_model: Model dimension
        n_mamba_blocks: Number of Mamba blocks
        n_latents: Number of Perceiver latent vectors
        latent_dim: Latent dimension
        modality_config: Dict of modality-specific configurations
        use_domain_adversarial: Enable domain adversarial training
        n_domains: Number of domains/species
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 512,
        n_mamba_blocks: int = 8,
        n_latents: int = 64,
        latent_dim: int = 512,
        modality_config: Optional[Dict] = None,
        use_domain_adversarial: bool = False,
        n_domains: int = 3,  # e.g., mouse, monkey, human
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.d_model = d_model
        self.n_latents = n_latents
        self.latent_dim = latent_dim
        self.use_domain_adversarial = use_domain_adversarial

        # Default modality configurations
        if modality_config is None:
            modality_config = self._get_default_modality_config()

        self.modality_config = modality_config

        # Create modality-specific tokenizers
        self.tokenizers = nn.ModuleDict()
        self._initialize_tokenizers()

        # Learned modality embeddings
        self.modality_embeddings = nn.ParameterDict()
        for modality in self.tokenizers.keys():
            self.modality_embeddings[modality] = nn.Parameter(
                torch.randn(1, 1, d_model) * 0.02
            )

        # Cross-modal Perceiver fusion
        self.perceiver_fusion = PerceiverIO(
            num_latents=n_latents,
            latent_dim=d_model,  # Match d_model for compatibility
            input_dim=d_model,
            num_cross_attention_layers=4,
            num_self_attention_layers=2,
            cross_attention_widening_factor=1,
            self_attention_widening_factor=1
        )

        # Mamba backbone for temporal modeling
        self.mamba_backbone = MambaBackbone(
            d_model=d_model,
            n_layers=n_mamba_blocks,
            use_multi_rate=True,
            downsample_rates=[1, 4],  # Multi-rate temporal
            dropout=dropout
        )

        # PopT aggregator for population-level features
        self.popt = PopTWithLatents(
            d_model=d_model,
            n_latents=n_latents,
            latent_dim=latent_dim,
            n_layers=2,
            n_heads=8,
            dropout=dropout
        )

        # Multi-task heads
        self.heads = MultiTaskHeads(
            latent_dim=latent_dim,
            decoder_output_dim=kwargs.get('decoder_output_dim', 10),
            encoder_output_dim=kwargs.get('encoder_output_dim', 384),
            contrastive_dim=kwargs.get('contrastive_dim', 128),
            enable_decoder=True,
            enable_encoder=True,
            enable_contrastive=True,
            enable_forecast=kwargs.get('enable_forecast', False)
        )

        # Domain adversarial discriminator
        if use_domain_adversarial:
            self.domain_discriminator = DomainDiscriminator(
                input_dim=latent_dim,
                n_domains=n_domains
            )

    def _get_default_modality_config(self) -> Dict:
        """Get default configuration for all modalities."""
        return {
            'spike': {'n_units': 384, 'seq_len': 100},
            'lfp': {'n_channels': 32, 'seq_len': 100},
            'calcium': {'n_cells': 200, 'seq_len': 100},
            'eeg': {'n_channels': 64, 'seq_len': 100, 'sfreq': 128.0},
            'fmri': {'n_rois': 400, 'seq_len': 50, 'tr': 0.72},
            'ecog': {'n_channels': 128, 'seq_len': 100},
            'emg': {'n_channels': 16, 'seq_len': 100}
        }

    def _initialize_tokenizers(self):
        """Initialize all modality-specific tokenizers."""

        # Spike tokenizer
        if 'spike' in self.modality_config:
            cfg = self.modality_config['spike']
            self.tokenizers['spike'] = SpikeTokenizer(
                n_units=cfg.get('n_units', 384),
                d_model=self.d_model,
                seq_len=cfg.get('seq_len', 100)
            )

        # LFP tokenizer
        if 'lfp' in self.modality_config:
            cfg = self.modality_config['lfp']
            self.tokenizers['lfp'] = LFPTokenizer(
                n_channels=cfg.get('n_channels', 32),
                d_model=self.d_model,
                seq_len=cfg.get('seq_len', 100)
            )

        # Calcium tokenizer
        if 'calcium' in self.modality_config:
            cfg = self.modality_config['calcium']
            self.tokenizers['calcium'] = CalciumTokenizer(
                n_cells=cfg.get('n_cells', 200),
                d_model=self.d_model,
                seq_len=cfg.get('seq_len', 100)
            )

        # EEG tokenizer
        if 'eeg' in self.modality_config:
            cfg = self.modality_config['eeg']
            self.tokenizers['eeg'] = EEGTokenizer(
                n_channels=cfg.get('n_channels', 64),
                d_model=self.d_model,
                seq_len=cfg.get('seq_len', 100),
                sfreq=cfg.get('sfreq', 128.0)
            )

        # fMRI tokenizer
        if 'fmri' in self.modality_config:
            cfg = self.modality_config['fmri']
            self.tokenizers['fmri'] = fMRITokenizer(
                n_rois=cfg.get('n_rois', 400),
                d_model=self.d_model,
                seq_len=cfg.get('seq_len', 50),
                tr=cfg.get('tr', 0.72)
            )

        # ECoG (similar to EEG)
        if 'ecog' in self.modality_config:
            cfg = self.modality_config['ecog']
            self.tokenizers['ecog'] = EEGTokenizer(  # Reuse EEG tokenizer
                n_channels=cfg.get('n_channels', 128),
                d_model=self.d_model,
                seq_len=cfg.get('seq_len', 100),
                sfreq=cfg.get('sfreq', 500.0),  # Higher sampling for ECoG
                use_spectral=True
            )

        # EMG (temporal patterns)
        if 'emg' in self.modality_config:
            cfg = self.modality_config['emg']
            self.tokenizers['emg'] = LFPTokenizer(  # Reuse LFP tokenizer
                n_channels=cfg.get('n_channels', 16),
                d_model=self.d_model,
                seq_len=cfg.get('seq_len', 100)
            )

    def forward(
        self,
        modality_dict: Dict[str, torch.Tensor],
        task: str = 'multi-task',
        species_labels: Optional[torch.Tensor] = None,
        grl_lambda: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multimodal model.

        Args:
            modality_dict: Dict mapping modality names to tensors
                Example: {'spike': (B, T, N), 'eeg': (B, T, C)}
            task: Task mode ('multi-task', 'decode', 'encode', etc.)
            species_labels: (batch,) species/domain labels for adversarial training
            grl_lambda: Gradient reversal strength

        Returns:
            outputs: Dict with task-specific outputs
        """
        # 1. Tokenize each modality
        token_list = []
        modality_ids = []

        for modality, data in modality_dict.items():
            if modality not in self.tokenizers:
                print(f"Warning: Unknown modality '{modality}', skipping")
                continue

            # Tokenize
            tokens = self.tokenizers[modality](data)  # (B, S, D)

            # Add modality embedding
            tokens = tokens + self.modality_embeddings[modality]

            token_list.append(tokens)
            modality_ids.append(modality)

        if len(token_list) == 0:
            raise ValueError("No valid modalities provided")

        # 2. Cross-modal fusion via Perceiver
        if len(token_list) == 1:
            # Single modality, skip fusion
            fused_tokens = token_list[0]
        else:
            # Concatenate all modality tokens
            # Perceiver can handle variable-length sequences
            max_seq_len = max(t.shape[1] for t in token_list)

            # Pad to same length
            padded_tokens = []
            for tokens in token_list:
                if tokens.shape[1] < max_seq_len:
                    pad_len = max_seq_len - tokens.shape[1]
                    tokens = F.pad(tokens, (0, 0, 0, pad_len))  # Pad sequence dim
                padded_tokens.append(tokens)

            # Stack for Perceiver
            concat_tokens = torch.cat(padded_tokens, dim=1)  # (B, M*S, D)

            # Perceiver cross-attention
            fused_latents = self.perceiver_fusion(concat_tokens)  # (B, n_latents, D)

            # Use latents as tokens for backbone
            fused_tokens = fused_latents

        # 3. Mamba backbone for temporal modeling
        backbone_out = self.mamba_backbone(fused_tokens)  # (B, S, D)

        # 4. PopT aggregation to fixed-size latents
        latents = self.popt(backbone_out)  # (B, n_latents, latent_dim)

        # 5. Multi-task heads
        outputs = self.heads(latents, task=task)
        outputs['latents'] = latents

        # 6. Domain adversarial (if enabled)
        if self.use_domain_adversarial and species_labels is not None:
            # Pool latents
            pooled_latents = latents.mean(dim=1)  # (B, latent_dim)

            # Gradient reversal
            reversed_latents = GradientReversalLayer.apply(pooled_latents, grl_lambda)

            # Domain classification
            domain_logits = self.domain_discriminator(reversed_latents)
            outputs['domain_logits'] = domain_logits

        return outputs

    def get_modality_names(self) -> List[str]:
        """Get list of supported modalities."""
        return list(self.tokenizers.keys())

    def freeze_backbone(self):
        """Freeze backbone for transfer learning."""
        for param in self.mamba_backbone.parameters():
            param.requires_grad = False
        for param in self.popt.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone."""
        for param in self.mamba_backbone.parameters():
            param.requires_grad = True
        for param in self.popt.parameters():
            param.requires_grad = True


# Example usage
if __name__ == '__main__':
    # Create model
    model = MultiModalNeuroFMX(
        d_model=512,
        n_mamba_blocks=4,
        n_latents=64,
        latent_dim=512,
        use_domain_adversarial=True,
        n_domains=3
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Supported modalities: {model.get_modality_names()}")

    # Test with multiple modalities
    batch_size = 2

    modality_inputs = {
        'spike': torch.randn(batch_size, 100, 384),  # (B, T, N_units)
        'eeg': torch.randn(batch_size, 256, 64),      # (B, T, N_channels)
    }

    species = torch.tensor([0, 1])  # Mouse=0, Human=1

    # Forward pass
    outputs = model(
        modality_inputs,
        task='multi-task',
        species_labels=species
    )

    print("\nOutputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
