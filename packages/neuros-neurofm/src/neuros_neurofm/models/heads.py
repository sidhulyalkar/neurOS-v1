"""
Multi-task heads for NeuroFM-X.

Implements various task-specific heads for behavioral decoding,
neural encoding, contrastive learning, and forecasting.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderHead(nn.Module):
    """Behavioral decoding head.

    Predicts behavioral variables (position, velocity, choice, etc.)
    from neural population activity.

    Parameters
    ----------
    input_dim : int
        Input dimension (latent dim from Perceiver/PopT).
    output_dim : int
        Output dimension (number of behavioral variables).
    hidden_dims : List[int], optional
        Hidden layer dimensions.
        Default: [256, 128].
    dropout : float, optional
        Dropout probability.
        Default: 0.1.
    activation : str, optional
        Activation function ('relu', 'gelu', 'tanh').
        Default: 'gelu'.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
        activation: str = 'gelu',
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Activation function
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build MLP
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                act_fn(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode behavioral variables from latents.

        Parameters
        ----------
        x : torch.Tensor
            Latent features, shape (batch, input_dim) or (batch, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted behavioral variables, same shape with output_dim.
        """
        return self.mlp(x)


class EncoderHead(nn.Module):
    """Neural encoding head (Reconstruction).

    Reconstructs neural activity sequences from latent representations.
    For NeuroFM-X, this reconstructs (B, S, N) from pooled latents (B, latent_dim).

    Parameters
    ----------
    input_dim : int
        Input dimension (latent dimension from PopT/Perceiver).
    output_dim : int
        Output dimension (neural activity dimension = n_units).
    sequence_length : int, optional
        Sequence length to reconstruct. If None, outputs single timestep.
        Default: None.
    hidden_dims : List[int], optional
        Hidden layer dimensions.
        Default: [256, 128].
    dropout : float, optional
        Dropout probability.
        Default: 0.1.
    output_activation : str, optional
        Output activation ('softplus' for firing rates, 'sigmoid', None).
        Default: 'softplus'.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sequence_length: Optional[int] = None,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1,
        output_activation: Optional[str] = 'softplus',
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length

        # Build MLP
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Output layer: project to (S * N) if sequence_length is specified
        if sequence_length is not None:
            output_size = sequence_length * output_dim
        else:
            output_size = output_dim

        layers.append(nn.Linear(prev_dim, output_size))

        # Output activation
        if output_activation == 'softplus':
            layers.append(nn.Softplus())
        elif output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif output_activation is not None:
            raise ValueError(f"Unknown output activation: {output_activation}")

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode/reconstruct neural activity.

        Parameters
        ----------
        x : torch.Tensor
            Latent features, shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed neural activity.
            If sequence_length is set: (batch, sequence_length, output_dim)
            Otherwise: (batch, output_dim)
        """
        output = self.mlp(x)  # (batch, S*N) or (batch, N)

        # Reshape to sequence if needed
        if self.sequence_length is not None:
            batch_size = x.shape[0]
            output = output.view(batch_size, self.sequence_length, self.output_dim)

        return output


class ContrastiveHead(nn.Module):
    """Contrastive learning head (CEBRA-style).

    Projects latents to a space where behavior-aligned samples are close
    and temporally distant samples are far apart.

    Parameters
    ----------
    input_dim : int
        Input dimension (latent dim).
    projection_dim : int, optional
        Projection dimension for contrastive space.
        Default: 256.
    temperature : float, optional
        Temperature parameter for InfoNCE loss.
        Default: 0.07.
    use_temporal_contrast : bool, optional
        Use temporal contrastive learning (time-contrastive).
        Default: True.
    temporal_window : int, optional
        Temporal window for positive pairs (in time steps).
        Default: 50.
    """

    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 256,
        temperature: float = 0.07,
        use_temporal_contrast: bool = True,
        temporal_window: int = 50,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.use_temporal_contrast = use_temporal_contrast
        self.temporal_window = temporal_window

        # Projection head (2-layer MLP)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

        # L2 normalization
        self.normalize = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project latents to contrastive space.

        Parameters
        ----------
        x : torch.Tensor
            Latent features, shape (batch, input_dim) or (batch, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Projected features, L2-normalized, same shape with projection_dim.
        """
        # Project
        z = self.projection(x)

        # L2 normalize
        if self.normalize:
            z = F.normalize(z, p=2, dim=-1)

        return z

    def compute_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute InfoNCE contrastive loss.

        Parameters
        ----------
        z1 : torch.Tensor
            First set of projections, shape (batch, projection_dim).
        z2 : torch.Tensor
            Second set of projections (positives), shape (batch, projection_dim).
        labels : torch.Tensor, optional
            Behavior labels for alignment, shape (batch,).

        Returns
        -------
        torch.Tensor
            Contrastive loss (scalar).
        """
        batch_size = z1.shape[0]

        # Compute similarity matrix
        # sim[i, j] = cosine similarity between z1[i] and z2[j]
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature
        # sim_matrix: (batch, batch)

        # Positive pairs are on the diagonal
        # Negative pairs are off-diagonal

        # Labels for cross-entropy: positives are at index i for row i
        targets = torch.arange(batch_size, device=z1.device)

        # InfoNCE loss
        loss = F.cross_entropy(sim_matrix, targets)

        return loss


class ForecastHead(nn.Module):
    """Forecasting head for predicting future neural activity.

    Parameters
    ----------
    input_dim : int
        Input dimension (latent dim).
    output_dim : int
        Output dimension (neural activity dimension).
    forecast_steps : int, optional
        Number of future steps to predict.
        Default: 10.
    hidden_dim : int, optional
        Hidden dimension for RNN/MLP.
        Default: 256.
    use_rnn : bool, optional
        Use RNN for sequential forecasting.
        Default: False (use MLP for parallel forecasting).
    dropout : float, optional
        Dropout probability.
        Default: 0.1.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        forecast_steps: int = 10,
        hidden_dim: int = 256,
        use_rnn: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.forecast_steps = forecast_steps
        self.use_rnn = use_rnn

        if use_rnn:
            # Use GRU for autoregressive forecasting
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                dropout=dropout if dropout > 0 else 0,
                batch_first=True,
            )
            self.output_proj = nn.Linear(hidden_dim, output_dim)
        else:
            # Use MLP for parallel forecasting (predict all steps at once)
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim * forecast_steps),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast future neural activity.

        Parameters
        ----------
        x : torch.Tensor
            Current latent state, shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Forecasted activity, shape (batch, forecast_steps, output_dim).
        """
        batch_size = x.shape[0]

        if self.use_rnn:
            # Autoregressive forecasting
            # Initialize hidden state
            hidden = None
            forecasts = []

            # Start with current state
            current = x.unsqueeze(1)  # (batch, 1, input_dim)

            for _ in range(self.forecast_steps):
                output, hidden = self.rnn(current, hidden)
                forecast = self.output_proj(output)
                forecasts.append(forecast)
                # Use forecast as input for next step
                current = forecast

            # Stack forecasts
            forecasted = torch.cat(forecasts, dim=1)
            # forecasted: (batch, forecast_steps, output_dim)

        else:
            # Parallel forecasting
            output = self.mlp(x)
            # output: (batch, output_dim * forecast_steps)

            # Reshape to (batch, forecast_steps, output_dim)
            forecasted = output.view(
                batch_size,
                self.forecast_steps,
                self.output_dim,
            )

        return forecasted


class MultiTaskHeads(nn.Module):
    """Container for multiple task-specific heads.

    Parameters
    ----------
    input_dim : int
        Input dimension (from Perceiver/PopT latents).
    decoder_output_dim : int, optional
        Decoder output dimension (behavioral variables).
    encoder_output_dim : int, optional
        Encoder output dimension (neural activity = n_units).
    sequence_length : int, optional
        Sequence length for reconstruction (encoder head).
        Default: None.
    enable_decoder : bool, optional
        Enable decoder head.
        Default: True.
    enable_encoder : bool, optional
        Enable encoder head.
        Default: True.
    enable_contrastive : bool, optional
        Enable contrastive head.
        Default: True.
    enable_forecast : bool, optional
        Enable forecast head.
        Default: False.
    dropout : float, optional
        Dropout probability.
        Default: 0.1.
    """

    def __init__(
        self,
        input_dim: int,
        decoder_output_dim: Optional[int] = None,
        encoder_output_dim: Optional[int] = None,
        sequence_length: Optional[int] = None,
        enable_decoder: bool = True,
        enable_encoder: bool = True,
        enable_contrastive: bool = True,
        enable_forecast: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.heads = nn.ModuleDict()

        if enable_decoder and decoder_output_dim is not None:
            self.heads['decoder'] = DecoderHead(
                input_dim=input_dim,
                output_dim=decoder_output_dim,
                dropout=dropout,
            )

        if enable_encoder and encoder_output_dim is not None:
            self.heads['encoder'] = EncoderHead(
                input_dim=input_dim,
                output_dim=encoder_output_dim,
                sequence_length=sequence_length,
                dropout=dropout,
            )

        if enable_contrastive:
            self.heads['contrastive'] = ContrastiveHead(
                input_dim=input_dim,
                projection_dim=256,
            )

        if enable_forecast and encoder_output_dim is not None:
            self.heads['forecast'] = ForecastHead(
                input_dim=input_dim,
                output_dim=encoder_output_dim,
                forecast_steps=10,
                dropout=dropout,
            )

    def forward(
        self,
        x: torch.Tensor,
        task: str,
    ) -> torch.Tensor:
        """Forward pass through specific task head.

        Parameters
        ----------
        x : torch.Tensor
            Input latent features.
        task : str
            Task name ('decoder', 'encoder', 'contrastive', 'forecast').

        Returns
        -------
        torch.Tensor
            Task-specific output.
        """
        if task not in self.heads:
            raise ValueError(
                f"Task '{task}' not enabled. Available: {list(self.heads.keys())}"
            )

        return self.heads[task](x)

    def get_available_tasks(self) -> List[str]:
        """Get list of enabled tasks."""
        return list(self.heads.keys())
