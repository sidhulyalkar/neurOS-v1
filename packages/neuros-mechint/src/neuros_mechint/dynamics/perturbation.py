"""
Perturbation Analysis

This module provides tools for analyzing system response to perturbations,
including sensitivity analysis, response functions, and robustness metrics.

Key capabilities:
- Perturbation response analysis
- Sensitivity to initial conditions
- Input-output response functions
- Robustness metrics
- Perturbation propagation tracking
"""

from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
import numpy as np
from scipy.linalg import expm
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerturbationResponse:
    """Response to a perturbation."""

    # Perturbation info
    perturbation_time: float  # Time of perturbation
    perturbation_direction: np.ndarray  # Direction of perturbation
    perturbation_magnitude: float  # Magnitude of perturbation

    # Response metrics
    max_response: float  # Maximum deviation from unperturbed trajectory
    response_time: float  # Time to reach maximum response
    recovery_time: float  # Time to recover to baseline
    amplification_factor: float  # max_response / perturbation_magnitude

    # Full response trajectory
    perturbed_trajectory: Optional[np.ndarray] = None
    unperturbed_trajectory: Optional[np.ndarray] = None
    response_trajectory: Optional[np.ndarray] = None  # Difference


@dataclass
class SensitivityResult:
    """Results from sensitivity analysis."""

    # Sensitivity matrices
    sensitivity_matrix: np.ndarray  # dX/dX0 (state sensitivity)

    # Summary metrics
    max_sensitivity: float  # Maximum sensitivity
    mean_sensitivity: float  # Mean sensitivity
    condition_number: float  # Condition number of sensitivity matrix

    # Directional sensitivities
    most_sensitive_direction: np.ndarray  # Direction of maximum sensitivity
    least_sensitive_direction: np.ndarray  # Direction of minimum sensitivity

    # Optional parameter sensitivity
    parameter_sensitivity: Optional[np.ndarray] = None  # dX/dp (parameter sensitivity)


@dataclass
class RobustnessResult:
    """Results from robustness analysis."""

    # Robustness metrics
    structural_robustness: float  # Resistance to structural changes
    dynamical_robustness: float  # Resistance to dynamical perturbations
    noise_robustness: float  # Resistance to noise

    # Perturbation response statistics
    mean_recovery_time: float  # Average recovery time
    max_amplification: float  # Maximum amplification factor

    # Failure modes
    perturbation_threshold: Optional[float] = None  # Threshold for system failure


class PerturbationAnalyzer:
    """
    Analyze system response to perturbations.

    Perturbation analysis reveals how systems respond to external
    disturbances and provides insights into stability and robustness.
    """

    def __init__(
        self,
        dt: float = 0.01,
        device: str = "cpu",
        verbose: bool = True
    ):
        """
        Initialize perturbation analyzer.

        Args:
            dt: Time step
            device: Device for computations
            verbose: Whether to log information
        """
        self.dt = dt
        self.device = device
        self.verbose = verbose

    def analyze_response(
        self,
        system_function: Callable,
        initial_state: np.ndarray,
        perturbation_direction: np.ndarray,
        perturbation_magnitude: float,
        perturbation_time: float = 0.0,
        duration: float = 10.0
    ) -> PerturbationResponse:
        """
        Analyze system response to a single perturbation.

        Args:
            system_function: Function f(x, t) computing dx/dt
            initial_state: Initial state
            perturbation_direction: Direction of perturbation (unit vector)
            perturbation_magnitude: Magnitude of perturbation
            perturbation_time: Time to apply perturbation
            duration: Total simulation duration

        Returns:
            PerturbationResponse object
        """
        n_steps = int(duration / self.dt)
        pert_step = int(perturbation_time / self.dt)

        # Integrate unperturbed trajectory
        unperturbed = self._integrate_trajectory(
            system_function,
            initial_state,
            n_steps
        )

        # Integrate perturbed trajectory
        # Apply perturbation at specified time
        perturbed_initial = initial_state.copy()
        if pert_step == 0:
            perturbed_initial += perturbation_magnitude * perturbation_direction

        perturbed = self._integrate_trajectory(
            system_function,
            perturbed_initial,
            n_steps
        )

        # Apply perturbation mid-trajectory if needed
        if pert_step > 0:
            perturbed[pert_step] += perturbation_magnitude * perturbation_direction

        # Compute response
        response_traj = perturbed - unperturbed
        response_magnitudes = np.linalg.norm(response_traj, axis=1)

        max_response = np.max(response_magnitudes)
        response_time = np.argmax(response_magnitudes) * self.dt

        # Recovery time: when response drops below 10% of max
        recovery_threshold = 0.1 * max_response
        recovery_indices = np.where(
            response_magnitudes[np.argmax(response_magnitudes):] < recovery_threshold
        )[0]

        if len(recovery_indices) > 0:
            recovery_time = (np.argmax(response_magnitudes) + recovery_indices[0]) * self.dt
        else:
            recovery_time = duration

        amplification_factor = max_response / perturbation_magnitude

        return PerturbationResponse(
            perturbation_time=perturbation_time,
            perturbation_direction=perturbation_direction,
            perturbation_magnitude=perturbation_magnitude,
            max_response=max_response,
            response_time=response_time,
            recovery_time=recovery_time,
            amplification_factor=amplification_factor,
            perturbed_trajectory=perturbed,
            unperturbed_trajectory=unperturbed,
            response_trajectory=response_traj
        )

    def compute_sensitivity(
        self,
        system_function: Callable,
        state: np.ndarray,
        duration: float = 1.0,
        method: str = "finite_difference"
    ) -> SensitivityResult:
        """
        Compute sensitivity to initial conditions.

        Args:
            system_function: Function f(x, t) computing dx/dt
            state: State at which to compute sensitivity
            duration: Time duration for sensitivity propagation
            method: Method ("finite_difference", "tangent_linear")

        Returns:
            SensitivityResult
        """
        if method == "finite_difference":
            return self._compute_sensitivity_fd(system_function, state, duration)
        elif method == "tangent_linear":
            return self._compute_sensitivity_tangent(system_function, state, duration)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _compute_sensitivity_fd(
        self,
        system_function: Callable,
        state: np.ndarray,
        duration: float
    ) -> SensitivityResult:
        """
        Compute sensitivity using finite differences.

        Args:
            system_function: System function
            state: State
            duration: Duration

        Returns:
            SensitivityResult
        """
        n_features = len(state)
        n_steps = int(duration / self.dt)

        # Reference trajectory
        reference = self._integrate_trajectory(system_function, state, n_steps)
        final_state = reference[-1]

        # Perturb each direction
        epsilon = 1e-6
        sensitivity_matrix = np.zeros((n_features, n_features))

        for i in range(n_features):
            # Perturbation
            perturbed_state = state.copy()
            perturbed_state[i] += epsilon

            # Integrate
            perturbed_traj = self._integrate_trajectory(
                system_function,
                perturbed_state,
                n_steps
            )
            perturbed_final = perturbed_traj[-1]

            # Sensitivity
            sensitivity_matrix[:, i] = (perturbed_final - final_state) / epsilon

        # Compute metrics
        max_sensitivity = np.max(np.abs(sensitivity_matrix))
        mean_sensitivity = np.mean(np.abs(sensitivity_matrix))

        # Condition number
        try:
            cond_number = np.linalg.cond(sensitivity_matrix)
        except:
            cond_number = np.inf

        # SVD for directional sensitivities
        U, S, Vh = np.linalg.svd(sensitivity_matrix)
        most_sensitive_dir = Vh[0]  # Right singular vector with largest singular value
        least_sensitive_dir = Vh[-1]  # Right singular vector with smallest singular value

        return SensitivityResult(
            sensitivity_matrix=sensitivity_matrix,
            max_sensitivity=max_sensitivity,
            mean_sensitivity=mean_sensitivity,
            condition_number=cond_number,
            most_sensitive_direction=most_sensitive_dir,
            least_sensitive_direction=least_sensitive_dir
        )

    def _compute_sensitivity_tangent(
        self,
        system_function: Callable,
        state: np.ndarray,
        duration: float
    ) -> SensitivityResult:
        """
        Compute sensitivity using tangent linear model.

        Args:
            system_function: System function
            state: State
            duration: Duration

        Returns:
            SensitivityResult
        """
        n_features = len(state)

        # Estimate Jacobian
        J = self._estimate_jacobian(system_function, state)

        # Propagate tangent linear model
        # Φ(t) = exp(J * t)
        sensitivity_matrix = expm(J * duration)

        # Compute metrics
        max_sensitivity = np.max(np.abs(sensitivity_matrix))
        mean_sensitivity = np.mean(np.abs(sensitivity_matrix))

        try:
            cond_number = np.linalg.cond(sensitivity_matrix)
        except:
            cond_number = np.inf

        # SVD
        U, S, Vh = np.linalg.svd(sensitivity_matrix)
        most_sensitive_dir = Vh[0]
        least_sensitive_dir = Vh[-1]

        return SensitivityResult(
            sensitivity_matrix=sensitivity_matrix,
            max_sensitivity=max_sensitivity,
            mean_sensitivity=mean_sensitivity,
            condition_number=cond_number,
            most_sensitive_direction=most_sensitive_dir,
            least_sensitive_direction=least_sensitive_dir
        )

    def analyze_robustness(
        self,
        system_function: Callable,
        initial_state: np.ndarray,
        n_perturbations: int = 100,
        perturbation_magnitude: float = 0.1,
        duration: float = 10.0
    ) -> RobustnessResult:
        """
        Analyze system robustness to multiple random perturbations.

        Args:
            system_function: System function
            initial_state: Initial state
            n_perturbations: Number of random perturbations to test
            perturbation_magnitude: Magnitude of perturbations
            duration: Simulation duration

        Returns:
            RobustnessResult
        """
        n_features = len(initial_state)
        responses = []

        # Test multiple random perturbations
        for _ in range(n_perturbations):
            # Random direction
            direction = np.random.randn(n_features)
            direction /= np.linalg.norm(direction)

            # Analyze response
            response = self.analyze_response(
                system_function,
                initial_state,
                direction,
                perturbation_magnitude,
                duration=duration
            )

            responses.append(response)

        # Compute robustness metrics
        recovery_times = [r.recovery_time for r in responses]
        amplification_factors = [r.amplification_factor for r in responses]

        mean_recovery_time = np.mean(recovery_times)
        max_amplification = np.max(amplification_factors)

        # Dynamical robustness: inverse of mean amplification
        mean_amplification = np.mean(amplification_factors)
        dynamical_robustness = 1.0 / (1.0 + mean_amplification)

        # Structural robustness: consistency of responses
        amplification_std = np.std(amplification_factors)
        structural_robustness = 1.0 / (1.0 + amplification_std)

        # Noise robustness: test with noise
        noise_robustness = self._test_noise_robustness(
            system_function,
            initial_state,
            duration
        )

        return RobustnessResult(
            structural_robustness=structural_robustness,
            dynamical_robustness=dynamical_robustness,
            noise_robustness=noise_robustness,
            mean_recovery_time=mean_recovery_time,
            max_amplification=max_amplification
        )

    def _test_noise_robustness(
        self,
        system_function: Callable,
        initial_state: np.ndarray,
        duration: float,
        noise_level: float = 0.01
    ) -> float:
        """
        Test robustness to continuous noise.

        Args:
            system_function: System function
            initial_state: Initial state
            duration: Duration
            noise_level: Noise standard deviation

        Returns:
            Noise robustness score [0, 1]
        """
        n_steps = int(duration / self.dt)

        # Reference trajectory
        reference = self._integrate_trajectory(system_function, initial_state, n_steps)

        # Noisy trajectory
        noisy = self._integrate_trajectory_with_noise(
            system_function,
            initial_state,
            n_steps,
            noise_level
        )

        # Compute deviation
        deviation = np.linalg.norm(noisy - reference, axis=1)
        mean_deviation = np.mean(deviation)

        # Robustness: inverse of normalized deviation
        noise_robustness = 1.0 / (1.0 + mean_deviation / noise_level)

        return noise_robustness

    def analyze_perturbation_propagation(
        self,
        system_function: Callable,
        initial_state: np.ndarray,
        perturbation_direction: np.ndarray,
        perturbation_magnitude: float,
        duration: float = 10.0
    ) -> dict:
        """
        Analyze how perturbations propagate through the system.

        Args:
            system_function: System function
            initial_state: Initial state
            perturbation_direction: Perturbation direction
            perturbation_magnitude: Perturbation magnitude
            duration: Duration

        Returns:
            Dictionary with propagation analysis
        """
        response = self.analyze_response(
            system_function,
            initial_state,
            perturbation_direction,
            perturbation_magnitude,
            duration=duration
        )

        # Analyze spatial propagation
        response_traj = response.response_trajectory

        # Compute energy in perturbation
        energy = np.sum(response_traj ** 2, axis=1)

        # Find peak energy and decay
        peak_energy_idx = np.argmax(energy)
        peak_energy_time = peak_energy_idx * self.dt

        # Exponential decay rate (after peak)
        if peak_energy_idx < len(energy) - 10:
            post_peak_energy = energy[peak_energy_idx:]
            post_peak_time = np.arange(len(post_peak_energy)) * self.dt

            # Fit exponential: E(t) = E0 * exp(-λt)
            log_energy = np.log(post_peak_energy + 1e-10)
            if len(log_energy) > 2:
                decay_rate = -np.polyfit(post_peak_time, log_energy, 1)[0]
            else:
                decay_rate = 0.0
        else:
            decay_rate = 0.0

        # Spatial spread
        # Compute which dimensions are most affected
        component_responses = np.max(np.abs(response_traj), axis=0)
        dominant_components = np.argsort(component_responses)[::-1][:3]

        return {
            'peak_energy': energy[peak_energy_idx],
            'peak_energy_time': peak_energy_time,
            'decay_rate': decay_rate,
            'dominant_components': dominant_components,
            'component_responses': component_responses,
            'energy_trajectory': energy
        }

    def _integrate_trajectory(
        self,
        system_function: Callable,
        initial_state: np.ndarray,
        n_steps: int
    ) -> np.ndarray:
        """
        Integrate trajectory using Euler method.

        Args:
            system_function: System function f(x, t)
            initial_state: Initial state
            n_steps: Number of steps

        Returns:
            Trajectory (n_steps, n_features)
        """
        n_features = len(initial_state)
        trajectory = np.zeros((n_steps, n_features))
        trajectory[0] = initial_state

        for t in range(1, n_steps):
            dx = system_function(trajectory[t-1], t * self.dt)
            trajectory[t] = trajectory[t-1] + self.dt * dx

        return trajectory

    def _integrate_trajectory_with_noise(
        self,
        system_function: Callable,
        initial_state: np.ndarray,
        n_steps: int,
        noise_level: float
    ) -> np.ndarray:
        """
        Integrate trajectory with added noise.

        Args:
            system_function: System function
            initial_state: Initial state
            n_steps: Number of steps
            noise_level: Noise standard deviation

        Returns:
            Noisy trajectory
        """
        n_features = len(initial_state)
        trajectory = np.zeros((n_steps, n_features))
        trajectory[0] = initial_state

        for t in range(1, n_steps):
            dx = system_function(trajectory[t-1], t * self.dt)
            noise = np.random.randn(n_features) * noise_level
            trajectory[t] = trajectory[t-1] + self.dt * dx + np.sqrt(self.dt) * noise

        return trajectory

    def _estimate_jacobian(
        self,
        system_function: Callable,
        state: np.ndarray,
        epsilon: float = 1e-6
    ) -> np.ndarray:
        """
        Estimate Jacobian matrix using finite differences.

        Args:
            system_function: System function
            state: State
            epsilon: Finite difference step

        Returns:
            Jacobian matrix
        """
        n_features = len(state)
        J = np.zeros((n_features, n_features))

        f0 = system_function(state, 0.0)

        for i in range(n_features):
            state_plus = state.copy()
            state_plus[i] += epsilon

            f_plus = system_function(state_plus, 0.0)

            J[:, i] = (f_plus - f0) / epsilon

        return J

    def test_perturbation_threshold(
        self,
        system_function: Callable,
        initial_state: np.ndarray,
        perturbation_direction: np.ndarray,
        max_magnitude: float = 10.0,
        n_tests: int = 20,
        duration: float = 10.0,
        divergence_threshold: float = 10.0
    ) -> float:
        """
        Find the perturbation magnitude that causes system divergence.

        Args:
            system_function: System function
            initial_state: Initial state
            perturbation_direction: Direction of perturbation
            max_magnitude: Maximum magnitude to test
            n_tests: Number of magnitudes to test
            duration: Simulation duration
            divergence_threshold: Threshold for considering system diverged

        Returns:
            Critical perturbation magnitude
        """
        magnitudes = np.linspace(0, max_magnitude, n_tests)

        for mag in magnitudes:
            response = self.analyze_response(
                system_function,
                initial_state,
                perturbation_direction,
                mag,
                duration=duration
            )

            if response.max_response > divergence_threshold:
                return mag

        return max_magnitude  # System stable for all tested magnitudes
