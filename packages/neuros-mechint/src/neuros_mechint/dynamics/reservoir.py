"""
Reservoir Computing

This module provides echo state networks and reservoir computing
methods for learning and predicting dynamics.

Key capabilities:
- Echo State Networks (ESN)
- Liquid State Machines
- Reservoir construction and optimization
- Next-generation reservoir computing (NGRC)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReservoirResult:
    """Results from reservoir computing."""

    # Predictions
    predictions: np.ndarray  # Predicted time series
    prediction_error: float  # Mean squared error

    # Reservoir properties
    reservoir_states: np.ndarray  # Internal reservoir states
    spectral_radius: float  # Spectral radius of reservoir
    effective_dimension: Optional[float] = None  # Effective dimension of dynamics


class ReservoirComputing:
    """
    Reservoir computing for learning and predicting dynamics.

    Reservoir computing uses a fixed random nonlinear dynamical system
    (the reservoir) to transform inputs into a high-dimensional space,
    where simple linear regression can learn complex functions.
    """

    def __init__(
        self,
        n_reservoir: int = 500,
        spectral_radius: float = 0.9,
        input_scaling: float = 1.0,
        leak_rate: float = 1.0,
        ridge_param: float = 1e-6,
        seed: int = 42,
        verbose: bool = True
    ):
        """
        Initialize reservoir computer.

        Args:
            n_reservoir: Number of reservoir neurons
            spectral_radius: Spectral radius of reservoir matrix
            input_scaling: Input weight scaling
            leak_rate: Leak rate for leaky integrator neurons
            ridge_param: Ridge regression regularization
            seed: Random seed
            verbose: Whether to log information
        """
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.ridge_param = ridge_param
        self.seed = seed
        self.verbose = verbose

        self.W_reservoir = None  # Reservoir weight matrix
        self.W_input = None  # Input weight matrix
        self.W_output = None  # Output weight matrix (learned)

        self._initialize_reservoir()

    def _initialize_reservoir(self):
        """Initialize reservoir weights."""
        np.random.seed(self.seed)

        # Create sparse random reservoir matrix
        density = 0.1  # Connection density
        W = sparse.random(
            self.n_reservoir,
            self.n_reservoir,
            density=density,
            data_rvs=lambda n: np.random.randn(n)
        ).toarray()

        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W)
        current_spectral_radius = np.max(np.abs(eigenvalues))

        if current_spectral_radius > 0:
            W = W * (self.spectral_radius / current_spectral_radius)

        self.W_reservoir = W

    def train(
        self,
        X_train: np.ndarray,
        Y_train: Optional[np.ndarray] = None,
        warmup_steps: int = 100
    ) -> None:
        """
        Train reservoir on data.

        Args:
            X_train: Input training data (n_timesteps, n_features)
            Y_train: Target training data (n_timesteps, n_outputs)
                    If None, uses X_train[1:] (one-step prediction)
            warmup_steps: Number of warmup steps to discard
        """
        n_timesteps, n_input = X_train.shape

        # Initialize input weights if needed
        if self.W_input is None or self.W_input.shape[1] != n_input:
            self.W_input = np.random.randn(self.n_reservoir, n_input) * self.input_scaling

        # Default target: next-step prediction
        if Y_train is None:
            Y_train = X_train[1:]
            X_train = X_train[:-1]
            n_timesteps -= 1

        n_output = Y_train.shape[1]

        # Collect reservoir states
        reservoir_states = np.zeros((n_timesteps, self.n_reservoir))
        r = np.zeros(self.n_reservoir)

        for t in range(n_timesteps):
            u = X_train[t]

            # Reservoir update: r[t+1] = (1-α)r[t] + α·tanh(W_res·r[t] + W_in·u[t])
            r = (1 - self.leak_rate) * r + self.leak_rate * np.tanh(
                self.W_reservoir @ r + self.W_input @ u
            )

            reservoir_states[t] = r

        # Discard warmup
        reservoir_states = reservoir_states[warmup_steps:]
        Y_train = Y_train[warmup_steps:]

        # Train output weights using ridge regression
        self.W_output = Ridge(alpha=self.ridge_param, fit_intercept=True)
        self.W_output.fit(reservoir_states, Y_train)

    def predict(
        self,
        X_test: Optional[np.ndarray] = None,
        n_steps: int = 100,
        initial_state: Optional[np.ndarray] = None
    ) -> ReservoirResult:
        """
        Make predictions using trained reservoir.

        Args:
            X_test: Test inputs (n_timesteps, n_features)
                   If None, autonomous prediction from initial_state
            n_steps: Number of steps to predict (for autonomous mode)
            initial_state: Initial reservoir state

        Returns:
            ReservoirResult
        """
        if X_test is not None:
            # Teacher-forced prediction
            return self._predict_teacher_forced(X_test, initial_state)
        else:
            # Autonomous prediction
            return self._predict_autonomous(n_steps, initial_state)

    def _predict_teacher_forced(
        self,
        X_test: np.ndarray,
        initial_state: Optional[np.ndarray]
    ) -> ReservoirResult:
        """Prediction with external input."""
        n_timesteps = len(X_test)

        # Initialize reservoir state
        if initial_state is None:
            r = np.zeros(self.n_reservoir)
        else:
            r = initial_state

        # Collect states and predictions
        reservoir_states = np.zeros((n_timesteps, self.n_reservoir))
        predictions = np.zeros((n_timesteps, X_test.shape[1]))

        for t in range(n_timesteps):
            u = X_test[t]

            # Update reservoir
            r = (1 - self.leak_rate) * r + self.leak_rate * np.tanh(
                self.W_reservoir @ r + self.W_input @ u
            )

            reservoir_states[t] = r

            # Predict output
            predictions[t] = self.W_output.predict(r.reshape(1, -1))[0]

        # Compute error (one-step ahead)
        if n_timesteps > 1:
            prediction_error = np.mean((predictions[:-1] - X_test[1:]) ** 2)
        else:
            prediction_error = 0.0

        return ReservoirResult(
            predictions=predictions,
            prediction_error=prediction_error,
            reservoir_states=reservoir_states,
            spectral_radius=self.spectral_radius
        )

    def _predict_autonomous(
        self,
        n_steps: int,
        initial_state: Optional[np.ndarray]
    ) -> ReservoirResult:
        """Autonomous prediction (closed-loop)."""
        if initial_state is None:
            r = np.zeros(self.n_reservoir)
        else:
            r = initial_state

        # Predict output dimension from W_output
        n_output = self.W_output.coef_.shape[0] if hasattr(self.W_output, 'coef_') else 1

        reservoir_states = np.zeros((n_steps, self.n_reservoir))
        predictions = np.zeros((n_steps, n_output))

        # Initial input
        u = np.zeros(self.W_input.shape[1])

        for t in range(n_steps):
            # Update reservoir
            r = (1 - self.leak_rate) * r + self.leak_rate * np.tanh(
                self.W_reservoir @ r + self.W_input @ u
            )

            reservoir_states[t] = r

            # Predict output
            y = self.W_output.predict(r.reshape(1, -1))[0]
            predictions[t] = y

            # Feedback: use prediction as next input
            u = y

        return ReservoirResult(
            predictions=predictions,
            prediction_error=0.0,  # No ground truth
            reservoir_states=reservoir_states,
            spectral_radius=self.spectral_radius
        )

    def compute_memory_capacity(
        self,
        input_sequence: np.ndarray,
        max_delay: int = 50
    ) -> np.ndarray:
        """
        Compute memory capacity of reservoir.

        Memory capacity measures how well the reservoir can recall
        past inputs at different delays.

        Args:
            input_sequence: Random input sequence (n_timesteps,)
            max_delay: Maximum delay to test

        Returns:
            Memory capacity at each delay (max_delay,)
        """
        n_timesteps = len(input_sequence)

        # Expand to 2D if needed
        if input_sequence.ndim == 1:
            input_sequence = input_sequence.reshape(-1, 1)

        # Initialize reservoir
        if self.W_input is None:
            self.W_input = np.random.randn(self.n_reservoir, 1) * self.input_scaling

        # Collect reservoir states
        reservoir_states = []
        r = np.zeros(self.n_reservoir)

        for t in range(n_timesteps):
            u = input_sequence[t]

            r = (1 - self.leak_rate) * r + self.leak_rate * np.tanh(
                self.W_reservoir @ r + self.W_input @ u
            )

            reservoir_states.append(r.copy())

        reservoir_states = np.array(reservoir_states)

        # Test memory at each delay
        memory_capacities = np.zeros(max_delay)

        for delay in range(1, max_delay + 1):
            # Target: input delayed by 'delay' steps
            X = reservoir_states[delay:]
            y = input_sequence[:-delay].flatten()

            if len(X) < 10:
                break

            # Train readout
            readout = Ridge(alpha=self.ridge_param)
            readout.fit(X, y)

            # Predict
            y_pred = readout.predict(X)

            # Memory capacity: correlation^2
            correlation = np.corrcoef(y, y_pred)[0, 1]
            memory_capacities[delay - 1] = correlation ** 2 if not np.isnan(correlation) else 0

        return memory_capacities

    def next_generation_rc(
        self,
        X_train: np.ndarray,
        Y_train: Optional[np.ndarray] = None,
        nonlinearity_order: int = 2
    ) -> None:
        """
        Next-generation reservoir computing (NGRC).

        NGRC uses time-delay embeddings and polynomial features
        instead of a random reservoir.

        Args:
            X_train: Training input
            Y_train: Training target
            nonlinearity_order: Order of polynomial features
        """
        if Y_train is None:
            Y_train = X_train[1:]
            X_train = X_train[:-1]

        # Create time-delay embedding
        delay = 1
        n_delays = 10

        n_samples = len(X_train) - n_delays * delay
        n_input = X_train.shape[1]

        # Build feature matrix
        features = []

        for t in range(n_delays * delay, len(X_train)):
            # Time delays
            delays = []
            for d in range(n_delays):
                delays.append(X_train[t - d * delay])

            delay_vector = np.concatenate(delays)

            # Polynomial features
            poly_features = [delay_vector]

            if nonlinearity_order >= 2:
                # Quadratic terms
                quad = []
                for i in range(len(delay_vector)):
                    for j in range(i, len(delay_vector)):
                        quad.append(delay_vector[i] * delay_vector[j])
                poly_features.append(np.array(quad))

            feature_vector = np.concatenate(poly_features)
            features.append(feature_vector)

        features = np.array(features)
        targets = Y_train[n_delays * delay:]

        # Train readout
        self.W_output = Ridge(alpha=self.ridge_param)
        self.W_output.fit(features, targets)

        # Store for prediction
        self._ngrc_features = features
        self._ngrc_delay = delay
        self._ngrc_n_delays = n_delays
        self._ngrc_nonlinearity_order = nonlinearity_order
