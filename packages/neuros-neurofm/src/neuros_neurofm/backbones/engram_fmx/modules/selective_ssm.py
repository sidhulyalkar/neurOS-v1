"""
Selective State-Space Model Block for ENGRAM-FMx.

Implements efficient input-conditioned recurrent state update
following the Mamba-style selective SSM approach.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SelectiveSSMBlock(nn.Module):
    """Selective State-Space Model block with input-dependent dynamics.

    Implements a simple gated diagonal SSM:
        a_t = sigmoid(W_a @ x_t)  # forget gate
        b_t = sigmoid(W_b @ x_t)  # input gate
        c_t = tanh(W_c @ x_t)     # output gate
        state = a_t * state + b_t * x_t
        y_t = c_t * state + D @ x_t

    This is a simplified MVP that can be replaced with Mamba/Mamba-2 later.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension (D).
    state_dim : int
        State dimension for recurrent state. Default: 128.
    dropout : float
        Dropout probability. Default: 0.1.
    layer_norm_eps : float
        LayerNorm epsilon. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 128,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        # Pre-normalization
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)

        # Input projection to state dimension
        self.input_proj = nn.Linear(hidden_dim, state_dim)

        # Gate projections (computed from input)
        self.W_a = nn.Linear(state_dim, state_dim)  # forget gate
        self.W_b = nn.Linear(state_dim, state_dim)  # input gate
        self.W_c = nn.Linear(state_dim, state_dim)  # output gate

        # Skip connection (D matrix in SSM notation)
        self.D_proj = nn.Linear(hidden_dim, state_dim)

        # Output projection back to hidden_dim
        self.output_proj = nn.Linear(state_dim, hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Forward pass through selective SSM.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape [B, T, D].
        initial_state : torch.Tensor, optional
            Initial recurrent state, shape [B, state_dim].
            If None, initializes to zeros.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, dict]
            - Output tensor [B, T, D]
            - Final state [B, state_dim]
            - Diagnostics dict
        """
        B, T, D = x.shape
        residual = x
        x = self.norm(x)

        # Project to state dimension
        x_state = self.input_proj(x)  # [B, T, state_dim]

        # Initialize state
        if initial_state is None:
            state = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)
        else:
            state = initial_state

        # Collect outputs and diagnostics
        outputs = []
        forget_gates = []
        input_gates = []

        # Sequential scan (MVP implementation - can be parallelized with scan ops later)
        for t in range(T):
            x_t = x_state[:, t, :]  # [B, state_dim]

            # Compute gates
            a_t = torch.sigmoid(self.W_a(x_t))  # forget gate
            b_t = torch.sigmoid(self.W_b(x_t))  # input gate
            c_t = torch.tanh(self.W_c(x_t))     # output gate

            # State update
            state = a_t * state + b_t * x_t

            # Output with skip connection
            y_t = c_t * state + self.D_proj(x[:, t, :])

            outputs.append(y_t)
            forget_gates.append(a_t.mean().item())
            input_gates.append(b_t.mean().item())

        # Stack outputs: [B, T, state_dim]
        y = torch.stack(outputs, dim=1)

        # Project back to hidden dimension
        y = self.output_proj(y)  # [B, T, D]

        # Dropout and residual
        y = self.dropout(y)
        y = y + residual

        # Diagnostics
        diagnostics = {
            "ssm_mean_forget_gate": sum(forget_gates) / len(forget_gates),
            "ssm_mean_input_gate": sum(input_gates) / len(input_gates),
            "ssm_state_norm": state.norm(dim=-1).mean().item(),
        }

        return y, state, diagnostics
