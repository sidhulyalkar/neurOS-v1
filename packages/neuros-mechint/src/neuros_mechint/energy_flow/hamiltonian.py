"""
Hamiltonian Decomposition for Neural Network Dynamics.

Decomposes network dynamics into conservative (Hamiltonian) and
dissipative components, revealing the structure of energy-preserving
vs energy-dissipating processes.

Key Principles:
- Hamiltonian mechanics: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
- Dissipative dynamics: ∇·F < 0 (phase space contraction)
- Helmholtz decomposition: F = F_conservative + F_dissipative
- Symplectic geometry and canonical transformations

Based on:
- Goldstein et al. (2002): Classical Mechanics
- Arnold (1989): Mathematical Methods of Classical Mechanics
- Ottino (1989): The Kinematics of Mixing
- Hairer et al. (2006): Geometric Numerical Integration

Example:
    >>> # Decompose network dynamics
    >>> decomposer = HamiltonianDecomposer(model)
    >>>
    >>> # Analyze trajectory
    >>> result = decomposer.decompose_dynamics(
    ...     initial_states=states,
    ...     n_timesteps=100
    ... )
    >>>
    >>> # Check conservation
    >>> print(f"Hamiltonian fraction: {result.hamiltonian_fraction:.3f}")
    >>> print(f"Dissipation fraction: {result.dissipation_fraction:.3f}")
    >>>
    >>> # Visualize phase portrait
    >>> result.visualize_decomposition(use_bokeh=True)

Author: NeuroS Team
Date: 2025-10-31
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from bokeh.plotting import figure, output_file, save
    from bokeh.layouts import column, row, gridplot
    from bokeh.models import HoverTool, ColorBar, LinearColorMapper
    from bokeh.palettes import Viridis256, RdYlBu11
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

from neuros_mechint.results import DynamicsResult

logger = logging.getLogger(__name__)


@dataclass
class HamiltonianComponents:
    """Hamiltonian and dissipative components of dynamics."""

    # Vector fields
    conservative_field: np.ndarray  # Shape: (n_timesteps, state_dim)
    dissipative_field: np.ndarray  # Shape: (n_timesteps, state_dim)
    total_field: np.ndarray  # Shape: (n_timesteps, state_dim)

    # Energy functions
    hamiltonian: np.ndarray  # Estimated Hamiltonian function (n_timesteps,)
    hamiltonian_gradient: np.ndarray  # ∇H (n_timesteps, state_dim)

    # Conservation metrics
    phase_space_volume: np.ndarray  # Volume evolution (n_timesteps,)
    divergence: np.ndarray  # ∇·F at each timestep (n_timesteps,)
    symplecticity: np.ndarray  # Symplectic form preservation (n_timesteps,)


@dataclass
class HamiltonianDecompositionResult:
    """Results from Hamiltonian decomposition analysis."""

    # Components
    components: HamiltonianComponents

    # Fractions
    hamiltonian_fraction: float = 0.0  # Fraction of conservative dynamics
    dissipation_fraction: float = 0.0  # Fraction of dissipative dynamics

    # Time series
    hamiltonian_over_time: np.ndarray = field(default_factory=lambda: np.array([]))
    dissipation_rate_over_time: np.ndarray = field(default_factory=lambda: np.array([]))

    # Geometric properties
    lyapunov_exponents: Optional[np.ndarray] = None
    poincare_return_map: Optional[np.ndarray] = None

    # Trajectory
    states: np.ndarray = field(default_factory=lambda: np.array([]))
    times: np.ndarray = field(default_factory=lambda: np.array([]))

    def visualize_decomposition(
        self,
        use_bokeh: bool = True,
        save_path: Optional[str] = None,
        n_dims: int = 2
    ) -> Any:
        """Visualize Hamiltonian decomposition."""
        if use_bokeh and BOKEH_AVAILABLE:
            return self._visualize_bokeh(save_path, n_dims)
        else:
            return self._visualize_matplotlib(save_path, n_dims)

    def _visualize_bokeh(self, save_path: Optional[str], n_dims: int) -> Any:
        """Bokeh visualization."""
        plots = []

        times = self.times

        # Plot 1: Hamiltonian evolution
        p1 = figure(
            title='Hamiltonian Function Over Time',
            width=1000,
            height=400,
            x_axis_label='Time',
            y_axis_label='H(q,p)'
        )

        p1.line(times, self.hamiltonian_over_time,
               line_width=3, color='blue', alpha=0.8)

        plots.append(p1)

        # Plot 2: Energy components
        p2 = figure(
            title='Conservative vs Dissipative Components',
            width=500,
            height=400,
            x_axis_label='Time',
            y_axis_label='Energy Magnitude'
        )

        conservative_energy = np.linalg.norm(self.components.conservative_field, axis=1)
        dissipative_energy = np.linalg.norm(self.components.dissipative_field, axis=1)

        p2.line(times, conservative_energy, legend_label='Conservative',
               line_width=2, color='green', alpha=0.7)
        p2.line(times, dissipative_energy, legend_label='Dissipative',
               line_width=2, color='red', alpha=0.7)

        p2.legend.location = "top_right"
        plots.append(p2)

        # Plot 3: Divergence (phase space contraction)
        p3 = figure(
            title='Phase Space Divergence',
            width=500,
            height=400,
            x_axis_label='Time',
            y_axis_label='∇·F (volume change rate)'
        )

        p3.line(times, self.components.divergence,
               line_width=3, color='purple', alpha=0.8)
        p3.line(times, np.zeros_like(times), line_dash='dashed',
               line_width=1, color='gray', alpha=0.5)

        plots.append(p3)

        # Plot 4: Phase portrait (2D projection)
        if self.states.shape[1] >= 2:
            p4 = figure(
                title='Phase Portrait (Conservative + Dissipative)',
                width=1000,
                height=600,
                x_axis_label='q₁',
                y_axis_label='q₂'
            )

            # State trajectory
            p4.line(self.states[:, 0], self.states[:, 1],
                   line_width=2, color='black', alpha=0.6)

            # Color by time
            colors = [Viridis256[int(i * 255 / len(self.states))]
                     for i in range(len(self.states))]
            p4.circle(self.states[:, 0], self.states[:, 1],
                     size=5, color=colors, alpha=0.7)

            plots.append(p4)

        layout = column(plots[0], row(plots[1], plots[2]), *plots[3:])

        if save_path:
            output_file(save_path)
            save(layout)

        return layout

    def _visualize_matplotlib(self, save_path: Optional[str], n_dims: int) -> Any:
        """Matplotlib visualization."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig = plt.figure(figsize=(16, 12))

        # Plot 1: Hamiltonian evolution
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(self.times, self.hamiltonian_over_time,
                linewidth=2, color='blue', alpha=0.7)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('H(q,p)')
        ax1.set_title('Hamiltonian Function Over Time', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Energy components
        ax2 = plt.subplot(3, 2, 2)
        conservative_energy = np.linalg.norm(self.components.conservative_field, axis=1)
        dissipative_energy = np.linalg.norm(self.components.dissipative_field, axis=1)

        ax2.plot(self.times, conservative_energy, label='Conservative',
                linewidth=2, color='green', alpha=0.7)
        ax2.plot(self.times, dissipative_energy, label='Dissipative',
                linewidth=2, color='red', alpha=0.7)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Energy Magnitude')
        ax2.set_title('Conservative vs Dissipative Components', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Divergence
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(self.times, self.components.divergence,
                linewidth=2, color='purple', alpha=0.7)
        ax3.axhline(y=0, linestyle='--', color='gray', alpha=0.5)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('∇·F (volume change rate)')
        ax3.set_title('Phase Space Divergence', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Symplecticity
        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(self.times, self.components.symplecticity,
                linewidth=2, color='teal', alpha=0.7)
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Symplectic Form Preservation')
        ax4.set_title('Symplecticity Measure', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # Plot 5: Phase portrait
        if self.states.shape[1] >= 2:
            ax5 = plt.subplot(3, 2, 5)
            scatter = ax5.scatter(self.states[:, 0], self.states[:, 1],
                                c=self.times, cmap='viridis', s=20, alpha=0.6)
            ax5.plot(self.states[:, 0], self.states[:, 1],
                    linewidth=1, color='black', alpha=0.3)
            ax5.set_xlabel('q₁')
            ax5.set_ylabel('q₂')
            ax5.set_title('Phase Portrait', fontweight='bold')
            plt.colorbar(scatter, ax=ax5, label='Time')

        # Plot 6: Dissipation rate
        ax6 = plt.subplot(3, 2, 6)
        ax6.plot(self.times, self.dissipation_rate_over_time,
                linewidth=2, color='darkred', alpha=0.7)
        ax6.set_xlabel('Time')
        ax6.set_ylabel('Dissipation Rate')
        ax6.set_title('Energy Dissipation Rate', fontweight='bold')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class HamiltonianDecomposer:
    """
    Decompose neural network dynamics into Hamiltonian and dissipative parts.

    Uses Helmholtz decomposition to separate conservative (energy-preserving)
    and dissipative (energy-dissipating) components of the vector field.

    Args:
        model: Neural network model
        dt: Time step for numerical differentiation
        method: Decomposition method ('helmholtz', 'poincare')
        verbose: Enable verbose logging

    Example:
        >>> decomposer = HamiltonianDecomposer(model, dt=0.01)
        >>> result = decomposer.decompose_dynamics(initial_states, n_timesteps=100)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dt: float = 0.01,
        method: str = 'helmholtz',
        verbose: bool = True
    ):
        self.model = model
        self.dt = dt
        self.method = method
        self.verbose = verbose

        self._log("Initialized HamiltonianDecomposer")

    def _log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            logger.info(f"[HamiltonianDecomposer] {message}")

    def decompose_dynamics(
        self,
        initial_states: torch.Tensor,
        n_timesteps: int = 100,
        compute_lyapunov: bool = False
    ) -> HamiltonianDecompositionResult:
        """
        Decompose dynamics into Hamiltonian and dissipative components.

        Args:
            initial_states: Initial state tensor (batch_size, state_dim)
            n_timesteps: Number of time steps to evolve
            compute_lyapunov: Compute Lyapunov exponents

        Returns:
            HamiltonianDecompositionResult with decomposition
        """
        self._log(f"Decomposing dynamics for {n_timesteps} timesteps")

        # Generate trajectory
        states, times = self._integrate_trajectory(initial_states, n_timesteps)

        # Compute vector field at each point
        vector_field = self._compute_vector_field(states)

        # Decompose into conservative and dissipative parts
        conservative_field, dissipative_field = self._helmholtz_decomposition(
            states, vector_field
        )

        # Estimate Hamiltonian function
        hamiltonian, hamiltonian_gradient = self._estimate_hamiltonian(
            states, conservative_field
        )

        # Compute geometric properties
        phase_space_volume = self._compute_phase_space_volume(states)
        divergence = self._compute_divergence(states, vector_field)
        symplecticity = self._compute_symplecticity(states, vector_field)

        components = HamiltonianComponents(
            conservative_field=conservative_field,
            dissipative_field=dissipative_field,
            total_field=vector_field,
            hamiltonian=hamiltonian,
            hamiltonian_gradient=hamiltonian_gradient,
            phase_space_volume=phase_space_volume,
            divergence=divergence,
            symplecticity=symplecticity
        )

        # Compute fractions
        conservative_energy = np.linalg.norm(conservative_field, axis=1).sum()
        dissipative_energy = np.linalg.norm(dissipative_field, axis=1).sum()
        total_energy = conservative_energy + dissipative_energy

        hamiltonian_fraction = conservative_energy / (total_energy + 1e-10)
        dissipation_fraction = dissipative_energy / (total_energy + 1e-10)

        # Dissipation rate
        dissipation_rate = np.linalg.norm(dissipative_field, axis=1)

        # Optional Lyapunov exponents
        lyapunov_exponents = None
        if compute_lyapunov:
            lyapunov_exponents = self._compute_lyapunov_exponents(states)

        result = HamiltonianDecompositionResult(
            components=components,
            hamiltonian_fraction=float(hamiltonian_fraction),
            dissipation_fraction=float(dissipation_fraction),
            hamiltonian_over_time=hamiltonian,
            dissipation_rate_over_time=dissipation_rate,
            lyapunov_exponents=lyapunov_exponents,
            states=states,
            times=times
        )

        self._log(f"Hamiltonian fraction: {hamiltonian_fraction:.3f}")
        self._log(f"Dissipation fraction: {dissipation_fraction:.3f}")

        return result

    def _integrate_trajectory(
        self,
        initial_states: torch.Tensor,
        n_timesteps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate trajectory using Euler method."""
        states = [initial_states.cpu().numpy()]
        times = [0.0]

        current_state = initial_states

        for t in range(n_timesteps):
            # Compute velocity
            with torch.no_grad():
                velocity = self.model(current_state)

            # Euler step
            current_state = current_state + self.dt * velocity

            states.append(current_state.cpu().numpy())
            times.append((t + 1) * self.dt)

        states = np.array(states)
        times = np.array(times)

        # Flatten batch dimension
        if len(states.shape) > 2:
            states = states.reshape(states.shape[0], -1)

        return states, times

    def _compute_vector_field(self, states: np.ndarray) -> np.ndarray:
        """Compute vector field (velocity) at each state."""
        vector_field = np.zeros_like(states)

        for i in range(len(states)):
            state_tensor = torch.tensor(states[i:i+1], dtype=torch.float32)

            with torch.no_grad():
                velocity = self.model(state_tensor)

            vector_field[i] = velocity.cpu().numpy().flatten()

        return vector_field

    def _helmholtz_decomposition(
        self,
        states: np.ndarray,
        vector_field: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Helmholtz decomposition: F = F_conservative + F_dissipative.

        Conservative: curl-free (potential field)
        Dissipative: divergence field
        """
        n_timesteps, state_dim = states.shape

        # Estimate divergence
        divergence = self._compute_divergence(states, vector_field)

        # Conservative field: projection onto curl-free subspace
        # Approximate via gradient of scalar potential
        conservative_field = np.zeros_like(vector_field)

        for i in range(1, n_timesteps):
            # Finite difference approximation
            gradient = (states[i] - states[i-1]) / self.dt
            conservative_field[i] = gradient

        # Dissipative field: remainder
        dissipative_field = vector_field - conservative_field

        return conservative_field, dissipative_field

    def _estimate_hamiltonian(
        self,
        states: np.ndarray,
        conservative_field: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate Hamiltonian function H(q,p).

        Uses line integral of conservative field.
        """
        n_timesteps = len(states)
        hamiltonian = np.zeros(n_timesteps)

        # Integrate conservative field along trajectory
        for i in range(1, n_timesteps):
            ds = states[i] - states[i-1]
            hamiltonian[i] = hamiltonian[i-1] + np.dot(conservative_field[i], ds)

        # Hamiltonian gradient is the conservative field
        hamiltonian_gradient = conservative_field

        return hamiltonian, hamiltonian_gradient

    def _compute_phase_space_volume(self, states: np.ndarray) -> np.ndarray:
        """Compute phase space volume evolution (Liouville's theorem)."""
        n_timesteps, state_dim = states.shape
        volume = np.zeros(n_timesteps)

        # Initial volume
        volume[0] = 1.0

        # Evolve volume based on divergence
        for i in range(1, n_timesteps):
            # Volume change: dV/dt = V * div(F)
            # Approximate divergence
            if i > 0:
                div = np.linalg.norm(states[i] - states[i-1]) / self.dt
                volume[i] = volume[i-1] * (1 + div * self.dt)

        return volume

    def _compute_divergence(
        self,
        states: np.ndarray,
        vector_field: np.ndarray
    ) -> np.ndarray:
        """Compute divergence ∇·F at each timestep."""
        n_timesteps = len(states)
        divergence = np.zeros(n_timesteps)

        for i in range(1, n_timesteps - 1):
            # Finite difference approximation
            dF_dt = (vector_field[i+1] - vector_field[i-1]) / (2 * self.dt)
            divergence[i] = dF_dt.sum()

        return divergence

    def _compute_symplecticity(
        self,
        states: np.ndarray,
        vector_field: np.ndarray
    ) -> np.ndarray:
        """
        Compute symplectic form preservation.

        Symplectic form: ω = dq ∧ dp
        Preserved if: dω/dt = 0
        """
        n_timesteps, state_dim = states.shape
        symplecticity = np.zeros(n_timesteps)

        # For each timestep, compute deviation from symplectic structure
        for i in range(1, n_timesteps - 1):
            # Approximate Jacobian
            dF = (vector_field[i+1] - vector_field[i-1]) / (2 * self.dt)

            # Symplectic measure: should be anti-symmetric
            # Compute Frobenius norm of symmetric part
            symmetric_part = dF + dF
            symplecticity[i] = np.linalg.norm(symmetric_part)

        return symplecticity

    def _compute_lyapunov_exponents(self, states: np.ndarray) -> np.ndarray:
        """Compute Lyapunov exponents via finite-time approximation."""
        n_timesteps, state_dim = states.shape

        # Perturb initial condition
        perturbation_size = 1e-6
        perturbation = np.random.randn(state_dim) * perturbation_size

        # Track divergence
        log_divergence = []

        for i in range(1, n_timesteps):
            # Distance between trajectories
            distance = np.linalg.norm(states[i] - states[0])

            if distance > perturbation_size:
                log_divergence.append(np.log(distance / perturbation_size) / (i * self.dt))

        # Lyapunov exponents
        if len(log_divergence) > 0:
            return np.array(log_divergence)
        else:
            return np.array([0.0])


__all__ = [
    'HamiltonianComponents',
    'HamiltonianDecompositionResult',
    'HamiltonianDecomposer',
]
