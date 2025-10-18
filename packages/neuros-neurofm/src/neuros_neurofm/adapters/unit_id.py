"""
Unit-ID adapter for NeuroFM-X.

Implements the POYO-style adapter that learns unit-specific embeddings
while keeping the backbone frozen, enabling few-shot transfer to new
neural populations.

Reference:
    Inspired by POYO (Population-level neural decoding with transferable representations)
"""

from typing import Optional

import torch
import torch.nn as nn


class UnitIDAdapter(nn.Module):
    """Unit-ID adapter for transfer learning.

    Adds learnable per-unit embeddings that are combined with the frozen
    backbone features, allowing the model to adapt to new neural populations
    with minimal training.

    Parameters
    ----------
    backbone_dim : int
        Dimension of backbone features (d_model).
    n_units : int
        Number of units in the target population.
    bottleneck_dim : int, optional
        Dimension of bottleneck layer (for parameter efficiency).
        Default: 128.
    freeze_backbone : bool, optional
        Whether to freeze the backbone during training.
        Default: True.
    adapter_position : str, optional
        Where to insert adapter: 'pre' (before backbone), 'post' (after backbone).
        Default: 'post'.
    dropout : float, optional
        Dropout probability.
        Default: 0.1.
    """

    def __init__(
        self,
        backbone_dim: int,
        n_units: int,
        bottleneck_dim: int = 128,
        freeze_backbone: bool = True,
        adapter_position: str = 'post',
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone_dim = backbone_dim
        self.n_units = n_units
        self.bottleneck_dim = bottleneck_dim
        self.freeze_backbone = freeze_backbone
        self.adapter_position = adapter_position

        # Learnable unit-specific embeddings
        self.unit_embeddings = nn.Embedding(n_units, backbone_dim)

        # Adapter layers (down-project → nonlinearity → up-project)
        self.adapter = nn.Sequential(
            nn.Linear(backbone_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, backbone_dim),
            nn.Dropout(dropout),
        )

        # Layer norm
        self.norm = nn.LayerNorm(backbone_dim)

        # Initialize adapter weights to near-zero for residual learning
        self._init_weights()

    def _init_weights(self):
        """Initialize adapter weights to small values."""
        for module in self.adapter.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        unit_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Apply Unit-ID adapter.

        Parameters
        ----------
        x : torch.Tensor
            Backbone features, shape (batch, n_units, backbone_dim).
        unit_indices : torch.Tensor
            Unit indices, shape (batch, n_units).

        Returns
        -------
        torch.Tensor
            Adapted features, shape (batch, n_units, backbone_dim).
        """
        # Get unit-specific embeddings
        unit_embs = self.unit_embeddings(unit_indices)
        # unit_embs: (batch, n_units, backbone_dim)

        # Combine with backbone features
        combined = x + unit_embs

        # Apply adapter (residual connection)
        adapted = combined + self.adapter(self.norm(combined))

        return adapted


class SessionStitcher(nn.Module):
    """Session/region stitching adapter.

    Aligns features across different recording sessions or brain regions
    using learnable affine transformations.

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_sessions : int, optional
        Number of sessions/regions to handle.
        Default: 10.
    use_affine : bool, optional
        Use affine transformation (scale + shift), else just shift.
        Default: True.
    """

    def __init__(
        self,
        d_model: int,
        n_sessions: int = 10,
        use_affine: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_sessions = n_sessions
        self.use_affine = use_affine

        # Session-specific shift (bias) parameters
        self.session_shift = nn.Embedding(n_sessions, d_model)

        # Session-specific scale parameters (if affine)
        if use_affine:
            self.session_scale = nn.Embedding(n_sessions, d_model)
            # Initialize scale to 1
            nn.init.ones_(self.session_scale.weight)
        else:
            self.session_scale = None

        # Initialize shift to 0
        nn.init.zeros_(self.session_shift.weight)

    def forward(
        self,
        x: torch.Tensor,
        session_id: torch.Tensor,
    ) -> torch.Tensor:
        """Apply session stitching.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (batch, ..., d_model).
        session_id : torch.Tensor
            Session ID for each batch element, shape (batch,).

        Returns
        -------
        torch.Tensor
            Stitched features, same shape as input.
        """
        # Get session-specific parameters
        shift = self.session_shift(session_id)
        # shift: (batch, d_model)

        # Reshape for broadcasting: (batch, 1, ..., 1, d_model)
        shift = shift.view(shift.shape[0], *([1] * (x.dim() - 2)), shift.shape[-1])

        if self.use_affine:
            scale = self.session_scale(session_id)
            scale = scale.view(scale.shape[0], *([1] * (x.dim() - 2)), scale.shape[-1])
            # Apply affine: x_new = scale * x + shift
            stitched = scale * x + shift
        else:
            # Apply shift only
            stitched = x + shift

        return stitched


class LoRAAdapter(nn.Module):
    """Low-Rank Adaptation (LoRA) for efficient fine-tuning.

    Adds low-rank decomposed matrices to linear layers, allowing
    parameter-efficient fine-tuning while keeping original weights frozen.

    Parameters
    ----------
    in_features : int
        Input dimension of the linear layer.
    out_features : int
        Output dimension of the linear layer.
    rank : int, optional
        Rank of the low-rank decomposition.
        Default: 8.
    alpha : float, optional
        Scaling factor (typically 2 * rank).
        Default: 16.
    dropout : float, optional
        Dropout probability on the LoRA path.
        Default: 0.1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices: A (in -> rank), B (rank -> out)
        # Output: B @ A @ x
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=0.0)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adapter.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape (..., in_features).

        Returns
        -------
        torch.Tensor
            LoRA output to add to frozen layer, shape (..., out_features).
        """
        # Low-rank path: B @ A @ x
        lora_out = self.lora_B(self.lora_A(self.dropout(x)))

        # Scale by alpha / rank
        lora_out = lora_out * self.scaling

        return lora_out


class LoRALinear(nn.Module):
    """Linear layer with LoRA adapter.

    Wraps a frozen linear layer with a LoRA adapter for efficient fine-tuning.

    Parameters
    ----------
    linear_layer : nn.Linear
        Original linear layer to adapt (will be frozen).
    rank : int, optional
        LoRA rank.
        Default: 8.
    alpha : float, optional
        LoRA alpha scaling.
        Default: 16.
    dropout : float, optional
        Dropout probability.
        Default: 0.1.
    freeze_original : bool, optional
        Whether to freeze the original layer.
        Default: True.
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.1,
        freeze_original: bool = True,
    ):
        super().__init__()

        # Store original layer
        self.linear = linear_layer
        if freeze_original:
            for param in self.linear.parameters():
                param.requires_grad = False

        # Add LoRA adapter
        self.lora = LoRAAdapter(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through frozen linear + LoRA.

        Parameters
        ----------
        x : torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor
            Output features.
        """
        # Original frozen path
        original_out = self.linear(x)

        # LoRA adapter path
        lora_out = self.lora(x)

        # Combine
        return original_out + lora_out


def apply_lora_to_model(
    model: nn.Module,
    target_modules: list = ['q_proj', 'v_proj'],
    rank: int = 8,
    alpha: float = 16,
    dropout: float = 0.1,
) -> None:
    """Apply LoRA to specific modules in a model.

    Parameters
    ----------
    model : nn.Module
        Model to modify.
    target_modules : list, optional
        List of module name patterns to apply LoRA to.
        Default: ['q_proj', 'v_proj'] (attention query and value projections).
    rank : int, optional
        LoRA rank.
    alpha : float, optional
        LoRA alpha.
    dropout : float, optional
        Dropout probability.
    """
    for name, module in model.named_modules():
        # Check if module name matches target patterns
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Get parent module and attribute name
                *parent_path, attr_name = name.split('.')
                parent = model
                for p in parent_path:
                    parent = getattr(parent, p)

                # Replace with LoRA version
                lora_linear = LoRALinear(
                    linear_layer=module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    freeze_original=True,
                )
                setattr(parent, attr_name, lora_linear)

                print(f"Applied LoRA to {name}")
