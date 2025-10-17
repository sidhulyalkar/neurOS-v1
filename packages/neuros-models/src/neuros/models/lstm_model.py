"""
LSTM model for temporal sequence classification in neurOS.

This module provides an LSTM-based classifier for BCI applications that
require modeling temporal dependencies in neural signals. LSTMs excel at
capturing long-range dependencies and temporal patterns in time-series data.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
from .base_model import BaseModel


class LSTMModel(BaseModel):
    """LSTM-based classifier for temporal EEG sequences.

    This model uses Long Short-Term Memory (LSTM) networks to capture
    temporal dependencies in neural signals. It's particularly effective for:
    - Event-related potentials (ERPs) like P300
    - Motor imagery with temporal dynamics
    - Continuous signal decoding
    - Online adaptation scenarios

    The architecture consists of:
    1. LSTM layers to capture temporal patterns
    2. Dropout for regularization
    3. Dense layer for classification

    Parameters
    ----------
    n_channels : int
        Number of input channels (EEG electrodes).
    n_timepoints : int
        Number of time points in each trial.
    n_classes : int, optional
        Number of output classes. Default is 2.
    lstm_units : int, optional
        Number of units in each LSTM layer. Default is 64.
    n_lstm_layers : int, optional
        Number of stacked LSTM layers. Default is 2.
    dropout : float, optional
        Dropout rate for regularization. Default is 0.5.
    learning_rate : float, optional
        Learning rate for Adam optimizer. Default is 0.001.
    n_epochs : int, optional
        Number of training epochs. Default is 100.
    batch_size : int, optional
        Batch size for training. Default is 32.

    Examples
    --------
    >>> from neuros.models import LSTMModel
    >>> import numpy as np

    >>> # Create model for 8 channels, 250 time points, 3 classes
    >>> model = LSTMModel(
    ...     n_channels=8,
    ...     n_timepoints=250,
    ...     n_classes=3,
    ...     lstm_units=128,
    ...     n_lstm_layers=2,
    ... )

    >>> # Generate synthetic data (samples, channels, timepoints)
    >>> X_train = np.random.randn(100, 8, 250)
    >>> y_train = np.random.randint(0, 3, 100)

    >>> # Train model
    >>> model.train(X_train, y_train)

    >>> # Predict
    >>> X_test = np.random.randn(20, 8, 250)
    >>> predictions = model.predict(X_test)
    >>> print(predictions.shape)  # (20,)

    Notes
    -----
    This implementation uses PyTorch for the LSTM layers and training loop.
    If PyTorch is not available, it will raise an ImportError with
    installation instructions.
    """

    def __init__(
        self,
        n_channels: int,
        n_timepoints: int,
        n_classes: int = 2,
        lstm_units: int = 64,
        n_lstm_layers: int = 2,
        dropout: float = 0.5,
        learning_rate: float = 0.001,
        n_epochs: int = 100,
        batch_size: int = 32,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.n_classes = n_classes
        self.lstm_units = lstm_units
        self.n_lstm_layers = n_lstm_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Will be initialized during training
        self.model: Optional[object] = None
        self.device: Optional[object] = None

    def _build_model(self):
        """Build the PyTorch LSTM model."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError(
                "PyTorch is required for LSTMModel. "
                "Install with: pip install torch"
            )

        class LSTMNet(nn.Module):
            """LSTM network architecture."""

            def __init__(
                self,
                n_channels,
                n_timepoints,
                n_classes,
                lstm_units,
                n_lstm_layers,
                dropout,
            ):
                super().__init__()

                self.lstm = nn.LSTM(
                    input_size=n_channels,
                    hidden_size=lstm_units,
                    num_layers=n_lstm_layers,
                    batch_first=True,
                    dropout=dropout if n_lstm_layers > 1 else 0,
                )

                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(lstm_units, n_classes)

            def forward(self, x):
                # x shape: (batch, timepoints, channels)
                # LSTM expects: (batch, seq_len, input_size)

                # LSTM output
                lstm_out, (hidden, cell) = self.lstm(x)

                # Use last time step's output
                last_output = lstm_out[:, -1, :]

                # Dropout
                dropped = self.dropout(last_output)

                # Classification
                logits = self.fc(dropped)
                return logits

        self.model = LSTMNet(
            self.n_channels,
            self.n_timepoints,
            self.n_classes,
            self.lstm_units,
            self.n_lstm_layers,
            self.dropout,
        )

        # Set device
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the LSTM model.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_channels, n_timepoints).
        y : np.ndarray
            Training labels of shape (n_samples,).

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        ValueError
            If input shapes don't match expected dimensions.
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader
        except ImportError:
            raise ImportError(
                "PyTorch is required for LSTMModel. "
                "Install with: pip install torch"
            )

        # Validate input shapes
        if X.ndim != 3:
            raise ValueError(
                f"Expected X to have 3 dimensions (samples, channels, timepoints), "
                f"got {X.ndim} dimensions"
            )

        if X.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {X.shape[1]}"
            )

        if X.shape[2] != self.n_timepoints:
            raise ValueError(
                f"Expected {self.n_timepoints} timepoints, got {X.shape[2]}"
            )

        # Build model
        self._build_model()

        # Transpose to (samples, timepoints, channels) for LSTM
        X_transposed = np.transpose(X, (0, 2, 1))

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_transposed).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in loader:
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track metrics
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                accuracy = 100 * correct / total
                print(
                    f"Epoch [{epoch+1}/{self.n_epochs}], "
                    f"Loss: {avg_loss:.4f}, "
                    f"Accuracy: {accuracy:.2f}%"
                )

        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for input data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_channels, n_timepoints).

        Returns
        -------
        np.ndarray
            Predicted labels of shape (n_samples,).

        Raises
        ------
        ValueError
            If model is not trained or input shapes are invalid.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for LSTMModel")

        # Validate input
        if X.ndim != 3:
            raise ValueError(
                f"Expected X to have 3 dimensions (samples, channels, timepoints), "
                f"got {X.ndim} dimensions"
            )

        # Transpose to (samples, timepoints, channels)
        X_transposed = np.transpose(X, (0, 2, 1))

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_transposed).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_channels, n_timepoints).

        Returns
        -------
        np.ndarray
            Class probabilities of shape (n_samples, n_classes).
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("PyTorch is required for LSTMModel")

        # Transpose
        X_transposed = np.transpose(X, (0, 2, 1))

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_transposed).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probas = F.softmax(outputs, dim=1)

        return probas.cpu().numpy()

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Incrementally update the model with new samples.

        This allows for online learning scenarios where new data arrives
        over time.

        Parameters
        ----------
        X : np.ndarray
            New training data of shape (n_samples, n_channels, n_timepoints).
        y : np.ndarray
            New training labels of shape (n_samples,).
        """
        if not self.is_trained:
            # First time: full training
            self.train(X, y)
            return

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            raise ImportError("PyTorch is required for LSTMModel")

        # Transpose
        X_transposed = np.transpose(X, (0, 2, 1))

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_transposed).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        # Single update step
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        outputs = self.model(X_tensor)
        loss = criterion(outputs, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
