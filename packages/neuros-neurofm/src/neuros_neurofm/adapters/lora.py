"""
LoRA (Low-Rank Adaptation) for NeuroFMx

Efficient fine-tuning by injecting low-rank matrices into model layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
import math


class LoRALayer(nn.Module):
    """
    LoRA layer that wraps an existing linear layer.

    Instead of fine-tuning W, we learn:
        W' = W + BA
    where B is (d_out, r) and A is (r, d_in) with r << min(d_in, d_out)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: Rank of low-rank matrices (r)
            alpha: Scaling factor (typically 2*rank or rank)
            dropout: Dropout probability
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Scaling factor
        self.scaling = alpha / rank

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            base_output: Output from the base (frozen) layer

        Returns:
            output: base_output + LoRA adaptation
        """
        # LoRA path: x @ A^T @ B^T
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()

        # Scale and add to base output
        output = base_output + lora_out * self.scaling

        return output


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Wraps a frozen base linear layer and adds trainable LoRA weights.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()

        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)  # Freeze base weights

        self.lora = LoRALayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output (frozen)
        base_out = self.base_layer(x)

        # Add LoRA adaptation
        output = self.lora(x, base_out)

        return output


class LoRAMultiheadAttention(nn.Module):
    """
    MultiheadAttention with LoRA applied to Q, K, V projections.
    """

    def __init__(
        self,
        base_attn: nn.MultiheadAttention,
        rank: int = 8,
        alpha: float = 16.0,
        adapt_qkv: tuple = (True, True, True)
    ):
        """
        Args:
            base_attn: Base attention module to adapt
            rank: LoRA rank
            alpha: LoRA alpha
            adapt_qkv: Tuple indicating which of (Q, K, V) to adapt
        """
        super().__init__()

        self.base_attn = base_attn
        self.base_attn.requires_grad_(False)

        embed_dim = base_attn.embed_dim

        # LoRA for Q, K, V projections
        self.adapt_q, self.adapt_k, self.adapt_v = adapt_qkv

        if self.adapt_q:
            self.lora_q = LoRALayer(embed_dim, embed_dim, rank, alpha)
        if self.adapt_k:
            self.lora_k = LoRALayer(embed_dim, embed_dim, rank, alpha)
        if self.adapt_v:
            self.lora_v = LoRALayer(embed_dim, embed_dim, rank, alpha)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs
    ):
        """Forward with LoRA adaptations on QKV."""

        # Apply LoRA to Q, K, V if enabled
        if self.adapt_q:
            # Get base Q projection
            q_base = F.linear(query, self.base_attn.in_proj_weight[:self.base_attn.embed_dim])
            query_adapted = self.lora_q(query, q_base)
        else:
            query_adapted = query

        if self.adapt_k:
            k_start = self.base_attn.embed_dim
            k_end = 2 * self.base_attn.embed_dim
            k_base = F.linear(key, self.base_attn.in_proj_weight[k_start:k_end])
            key_adapted = self.lora_k(key, k_base)
        else:
            key_adapted = key

        if self.adapt_v:
            v_start = 2 * self.base_attn.embed_dim
            v_base = F.linear(value, self.base_attn.in_proj_weight[v_start:])
            value_adapted = self.lora_v(value, v_base)
        else:
            value_adapted = value

        # Use base attention with adapted QKV
        # Note: This is simplified; in practice need to handle the full attention computation
        return self.base_attn(query_adapted, key_adapted, value_adapted, **kwargs)


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Optional[List[str]] = None,
    dropout: float = 0.0
) -> Dict[str, nn.Module]:
    """
    Inject LoRA into specified modules of a model.

    Args:
        model: Model to adapt
        rank: LoRA rank
        alpha: LoRA alpha
        target_modules: List of module name patterns to target
                       (e.g., ['attn', 'mlp', 'proj'])
        dropout: LoRA dropout

    Returns:
        lora_modules: Dict of injected LoRA modules
    """
    if target_modules is None:
        target_modules = ['linear', 'attn']

    lora_modules = {}

    for name, module in model.named_modules():
        # Check if module name matches any target pattern
        should_adapt = any(pattern in name.lower() for pattern in target_modules)

        if not should_adapt:
            continue

        # Adapt Linear layers
        if isinstance(module, nn.Linear):
            # Get parent module
            parent_name = '.'.join(name.split('.')[:-1])
            module_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            # Replace with LoRALinear
            lora_linear = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, module_name, lora_linear)

            lora_modules[name] = lora_linear
            print(f"Injected LoRA into {name}")

        # Adapt MultiheadAttention
        elif isinstance(module, nn.MultiheadAttention):
            parent_name = '.'.join(name.split('.')[:-1])
            module_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            lora_attn = LoRAMultiheadAttention(module, rank=rank, alpha=alpha)
            setattr(parent, module_name, lora_attn)

            lora_modules[name] = lora_attn
            print(f"Injected LoRA into {name}")

    print(f"\nTotal LoRA modules injected: {len(lora_modules)}")

    # Count trainable parameters
    lora_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {lora_params:,} / {total_params:,} ({lora_params/total_params*100:.2f}%)")

    return lora_modules


def merge_lora(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights into base model for inference.

    After merging, the adapted model has the same architecture
    as the original but with updated weights.
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Merge LoRA weights into base linear
            with torch.no_grad():
                lora_weight = module.lora.lora_B @ module.lora.lora_A * module.lora.scaling
                module.base_layer.weight.data += lora_weight

            # Replace LoRALinear with base layer
            parent_name = '.'.join(name.split('.')[:-1])
            module_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            # Make base layer trainable again
            module.base_layer.requires_grad_(True)
            setattr(parent, module_name, module.base_layer)

            print(f"Merged LoRA weights in {name}")

    return model


def save_lora_weights(model: nn.Module, path: str):
    """Save only LoRA weights (not the full model)."""
    lora_state_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAMultiheadAttention)):
            lora_state_dict[name] = {
                'lora_A': module.lora.lora_A,
                'lora_B': module.lora.lora_B
            }

    torch.save(lora_state_dict, path)
    print(f"Saved LoRA weights to {path}")


def load_lora_weights(model: nn.Module, path: str):
    """Load LoRA weights into a model with LoRA layers."""
    lora_state_dict = torch.load(path)

    for name, module in model.named_modules():
        if name in lora_state_dict:
            if isinstance(module, (LoRALinear, LoRAMultiheadAttention)):
                module.lora.lora_A.data = lora_state_dict[name]['lora_A']
                module.lora.lora_B.data = lora_state_dict[name]['lora_B']

    print(f"Loaded LoRA weights from {path}")


class AdapterLayer(nn.Module):
    """
    Adapter layer (Houlsby et al., 2019).

    Bottleneck architecture: down-project → activation → up-project
    """

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int = 64,
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        super().__init__()

        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, d_model)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.dropout = nn.Dropout(dropout)

        # Initialize near-identity (small weights for stability)
        nn.init.normal_(self.down_proj.weight, std=1e-3)
        nn.init.normal_(self.up_proj.weight, std=1e-3)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        # Residual connection
        residual = x

        # Adapter path
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)

        # Add residual
        output = residual + x

        return output


def inject_adapters(
    model: nn.Module,
    bottleneck_dim: int = 64,
    target_layers: Optional[List[str]] = None
) -> Dict[str, nn.Module]:
    """
    Inject adapter layers after attention/FFN blocks.

    Args:
        model: Model to adapt
        bottleneck_dim: Adapter bottleneck dimension
        target_layers: List of layer names to add adapters after

    Returns:
        adapter_modules: Dict of injected adapters
    """
    if target_layers is None:
        # Default: add after each transformer/mamba block
        target_layers = ['mamba', 'transformer', 'perceiver']

    adapter_modules = {}

    # This is a simplified version - in practice you'd need to
    # hook into the specific architecture
    print(f"Adapter injection requires architecture-specific implementation")
    print(f"Target layers: {target_layers}")
    print(f"Bottleneck dim: {bottleneck_dim}")

    return adapter_modules
