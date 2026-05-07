"""
ENGRAM-FMx Configuration.

Defines the configuration dataclass for the ENGRAM-FMx architecture
with all hyperparameters and ablation flags.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ENGRAMFMxConfig:
    """Configuration for ENGRAM-FMx backbone.

    Parameters
    ----------
    input_dim : int
        Input dimension (from tokenizer). Default: 256.
    hidden_dim : int
        Hidden/model dimension. Default: 256.
    output_dim : int, optional
        Output dimension. If None, uses hidden_dim. Default: None.

    num_layers : int
        Number of ENGRAM blocks to stack. Default: 4.
    num_latents : int
        Number of latent workspace slots (K). Default: 64.
    memory_slots : int
        Number of attractor memory slots (M). Default: 256.
    num_heads : int
        Number of attention heads. Default: 4.

    ssm_state_dim : int
        State dimension for selective SSM. Default: 128.
    operator_modes : int
        Number of spectral modes for operator dynamics. Default: 16.
    sparse_top_k : int
        Number of tokens to select for sparse anchor attention. Default: 128.
    local_conv_width : int
        Kernel width for local convolution. Default: 7.

    dropout : float
        Dropout probability. Default: 0.1.
    layer_norm_eps : float
        LayerNorm epsilon. Default: 1e-5.

    use_local_processing : bool
        Enable local processing block. Default: True.
    use_ssm : bool
        Enable selective SSM. Default: True.
    use_latent_workspace : bool
        Enable latent workspace compression. Default: True.
    use_attractor_memory : bool
        Enable Hopfield-style attractor memory. Default: True.
    use_operator_dynamics : bool
        Enable spectral operator dynamics. Default: True.
    use_sparse_anchor_attention : bool
        Enable sparse anchor attention. Default: True.
    use_controller : bool
        Enable controller gates. Default: True.

    memory_beta : float
        Temperature for Hopfield memory retrieval. Default: 8.0.
    memory_residual_alpha : float
        Residual mixing coefficient for memory. Default: 0.5.
    memory_entropy_reg : float
        Memory entropy regularization weight. Default: 0.0.
    memory_usage_reg : float
        Memory usage regularization weight. Default: 0.0.

    return_diagnostics : bool
        Return diagnostic information. Default: True.
    """

    # Dimensions
    input_dim: int = 256
    hidden_dim: int = 256
    output_dim: Optional[int] = None

    # Architecture
    num_layers: int = 4
    num_latents: int = 64
    memory_slots: int = 256
    num_heads: int = 4

    # Module-specific
    ssm_state_dim: int = 128
    operator_modes: int = 16
    sparse_top_k: int = 128
    local_conv_width: int = 7

    # Regularization
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # Ablation flags - enable/disable components
    use_local_processing: bool = True
    use_ssm: bool = True
    use_latent_workspace: bool = True
    use_attractor_memory: bool = True
    use_operator_dynamics: bool = True
    use_sparse_anchor_attention: bool = True
    use_controller: bool = True

    # Memory hyperparameters
    memory_beta: float = 8.0
    memory_residual_alpha: float = 0.5
    memory_entropy_reg: float = 0.0
    memory_usage_reg: float = 0.0

    # Output options
    return_diagnostics: bool = True

    def __post_init__(self):
        """Set derived parameters."""
        if self.output_dim is None:
            self.output_dim = self.hidden_dim

    @classmethod
    def tiny(cls) -> "ENGRAMFMxConfig":
        """Create a tiny config for testing and development."""
        return cls(
            input_dim=128,
            hidden_dim=128,
            num_layers=2,
            num_latents=32,
            memory_slots=128,
            num_heads=4,
            ssm_state_dim=64,
            operator_modes=8,
            sparse_top_k=32,
        )

    @classmethod
    def small(cls) -> "ENGRAMFMxConfig":
        """Create a small config for local GPU training."""
        return cls(
            input_dim=256,
            hidden_dim=256,
            num_layers=4,
            num_latents=64,
            memory_slots=256,
            num_heads=4,
            ssm_state_dim=128,
            operator_modes=16,
            sparse_top_k=64,
        )

    @classmethod
    def medium(cls) -> "ENGRAMFMxConfig":
        """Create a medium config for larger experiments."""
        return cls(
            input_dim=384,
            hidden_dim=384,
            num_layers=6,
            num_latents=96,
            memory_slots=384,
            num_heads=6,
            ssm_state_dim=192,
            operator_modes=24,
            sparse_top_k=96,
        )

    @classmethod
    def large(cls) -> "ENGRAMFMxConfig":
        """Create a large config for cloud training."""
        return cls(
            input_dim=512,
            hidden_dim=512,
            num_layers=8,
            num_latents=128,
            memory_slots=512,
            num_heads=8,
            ssm_state_dim=256,
            operator_modes=32,
            sparse_top_k=128,
        )
