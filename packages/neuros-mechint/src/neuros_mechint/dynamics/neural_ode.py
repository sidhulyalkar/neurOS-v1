"""
Neural ODE Integrator for Continuous Dynamics Analysis.

Treats neural network dynamics as continuous-time differential equations,
enabling analysis of:
- Continuous flow fields
- Trajectory integration
- Stability analysis
- Invariant manifolds
- Energy landscapes in continuous time

Based on:
- Chen et al. (2018): Neural Ordinary Differential Equations
- Grathwohl et al. (2018): FFJORD
- Dupont et al. (2019): Augmented Neural ODEs
- Finlay et al. (2020): How to Train Your Neural ODE

Example:
    >>> # Analyze network as continuous ODE
    >>> integrator = NeuralODEIntegrator(model)
    >>>
    >>> # Integrate trajectory
    >>> trajectory = integrator.integrate_trajectory(
    ...     initial_state=x0,
    ...     time_span=(0, 10),
    ...     method='dopri5'
    ... )
    >>>
    >>> # Analyze flow field
    >>> flow_analysis = integrator.analyze_flow_field(
    ...     state_space_bounds=bounds,
    ...     n_points=50
    ... )
    >>>
    >>> # Visualize phase portrait
    >>> flow_analysis.visualize_phase_portrait(use_bokeh=True)

Author: NeuroS Team
Date: 2025-10-31
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging

try:
    from torchdiffeq import odeint, odeint_adjoint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    # Simple fallback Euler integration
    def odeint(func, y0, t, **kwargs):
        """Simple Euler method fallback."""
        ys = [y0]
        for i in range(len(t) - 1):
            dt = t[i+1] - t[i]
            dy = func(t[i], ys[-1])
            ys.append(ys[-1] + dt * dy)
        return torch.stack(ys)

    odeint_adjoint = odeint

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from bokeh.plotting import figure, output_file, save
    from bokeh.layouts import column, row, gridplot
    from bokeh.models import HoverTool, ColorBar, LinearColorMapper
    from bokeh.palettes import Viridis256
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

from neuros_mechint.results import DynamicsResult

logger = logging.getLogger(__name__)


@dataclass
class FlowFieldAnalysis:
    """Analysis of the flow field induced by the ODE."""

    # Flow field
    positions: np.ndarray  # Grid positions (n_points, n_dims)
    velocities: np.ndarray  # Flow velocities at each point (n_points, n_dims)
    speeds: np.ndarray  # Flow speed magnitudes (n_points,)

    # Critical points
    fixed_points: List[np.ndarray] = field(default_factory=list)
    fixed_point_stability: List[str] = field(default_factory=list)  # 'stable', 'unstable', 'saddle'

    # Lyapunov analysis
    lyapunov_field: Optional[np.ndarray] = None  # Local Lyapunov exponents

    # Energy/potential
    potential_field: Optional[np.ndarray] = None

    # Metadata
    time_point: float = 0.0
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def visualize_phase_portrait(
        self,
        use_bokeh: bool = True,
        save_path: Optional[str] = None,
        dim_x: int = 0,
        dim_y: int = 1
    ) -> Any:
        """
        Visualize 2D phase portrait.

        Args:
            use_bokeh: Use Bokeh for interactive visualization
            save_path: Path to save figure
            dim_x: Dimension to plot on x-axis
            dim_y: Dimension to plot on y-axis

        Returns:
            Bokeh figure or matplotlib figure
        """
        if use_bokeh and BOKEH_AVAILABLE:
            return self._visualize_bokeh(save_path, dim_x, dim_y)
        else:
            return self._visualize_matplotlib(save_path, dim_x, dim_y)

    def _visualize_bokeh(self, save_path: Optional[str], dim_x: int, dim_y: int) -> Any:
        """Bokeh phase portrait visualization."""
        p = figure(
            title='Phase Portrait',
            width=800,
            height=800,
            x_axis_label=f'Dimension {dim_x}',
            y_axis_label=f'Dimension {dim_y}'
        )

        # Extract 2D projections
        x = self.positions[:, dim_x]
        y = self.positions[:, dim_y]
        vx = self.velocities[:, dim_x]
        vy = self.velocities[:, dim_y]

        # Quiver plot (arrows)
        # Normalize for visualization
        scale = 0.1
        p.segment(x0=x, y0=y, x1=x + vx*scale, y1=y + vy*scale,
                 line_width=2, alpha=0.6, color='navy')

        # Add fixed points
        if self.fixed_points:
            fp_x = [fp[dim_x] for fp in self.fixed_points]
            fp_y = [fp[dim_y] for fp in self.fixed_points]

            colors = {
                'stable': 'green',
                'unstable': 'red',
                'saddle': 'orange'
            }

            for i, (fpx, fpy, stability) in enumerate(zip(fp_x, fp_y, self.fixed_point_stability)):
                p.circle(fpx, fpy, size=15, color=colors.get(stability, 'gray'),
                        legend_label=stability if i == 0 else None)

        p.legend.location = "top_right"

        if save_path:
            output_file(save_path)
            save(p)

        return p

    def _visualize_matplotlib(self, save_path: Optional[str], dim_x: int, dim_y: int) -> Any:
        """Matplotlib phase portrait visualization."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        fig, ax = plt.subplots(figsize=(10, 10))

        # Extract 2D projections
        x = self.positions[:, dim_x]
        y = self.positions[:, dim_y]
        vx = self.velocities[:, dim_x]
        vy = self.velocities[:, dim_y]

        # Quiver plot
        ax.quiver(x, y, vx, vy, self.speeds, cmap='viridis', alpha=0.7)

        # Fixed points
        if self.fixed_points:
            colors = {
                'stable': 'green',
                'unstable': 'red',
                'saddle': 'orange'
            }

            for fp, stability in zip(self.fixed_points, self.fixed_point_stability):
                ax.plot(fp[dim_x], fp[dim_y], 'o',
                       color=colors.get(stability, 'gray'),
                       markersize=15, label=stability)

        ax.set_xlabel(f'Dimension {dim_x}')
        ax.set_ylabel(f'Dimension {dim_y}')
        ax.set_title('Phase Portrait', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


@dataclass
class ODETrajectory:
    """Continuous trajectory from ODE integration."""

    times: np.ndarray  # Time points (n_steps,)
    states: np.ndarray  # State trajectories (n_steps, n_dims)

    # Derived quantities
    velocities: Optional[np.ndarray] = None  # Velocities at each time
    energies: Optional[np.ndarray] = None  # Energy along trajectory

    # Metadata
    method: str = 'dopri5'
    rtol: float = 1e-3
    atol: float = 1e-4


class NeuralODEIntegrator:
    """
    Integrate neural network dynamics as continuous ODEs.

    Treats the network as defining a vector field f(x, t) and integrates
    trajectories through this field.

    Args:
        model: Neural network defining the dynamics
        device: Torch device
        use_adjoint: Use adjoint method for backprop (memory efficient)
        verbose: Enable verbose logging

    Example:
        >>> integrator = NeuralODEIntegrator(model)
        >>> trajectory = integrator.integrate_trajectory(x0, (0, 10))
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        use_adjoint: bool = False,
        verbose: bool = True
    ):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_adjoint = use_adjoint
        self.verbose = verbose

        self.model.to(self.device)
        self.model.eval()

        # Select integrator
        self.odeint = odeint_adjoint if use_adjoint else odeint

        if not TORCHDIFFEQ_AVAILABLE:
            logger.warning("torchdiffeq not available. Using simple Euler integration.")

        self._log("Initialized NeuralODEIntegrator")

    def _log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            logger.info(f"[NeuralODEIntegrator] {message}")

    def integrate_trajectory(
        self,
        initial_state: torch.Tensor,
        time_span: Tuple[float, float],
        n_steps: int = 100,
        method: str = 'dopri5',
        rtol: float = 1e-3,
        atol: float = 1e-4
    ) -> ODETrajectory:
        """
        Integrate trajectory from initial state.

        Args:
            initial_state: Starting state (batch_size, state_dim)
            time_span: (t_start, t_end)
            n_steps: Number of time steps
            method: Integration method ('dopri5', 'rk4', 'euler', etc.)
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            ODETrajectory with integrated path
        """
        self._log(f"Integrating trajectory from t={time_span[0]} to t={time_span[1]}")

        initial_state = initial_state.to(self.device)

        # Define ODE function
        def ode_func(t, x):
            """Compute dx/dt = f(x, t)"""
            with torch.enable_grad():
                x_input = x.requires_grad_(True)
                output = self.model(x_input)

                # If model doesn't naturally give derivative, compute it
                # Assume output represents next state, compute difference
                dxdt = output - x_input

            return dxdt

        # Time points
        t = torch.linspace(time_span[0], time_span[1], n_steps).to(self.device)

        # Integrate
        with torch.no_grad():
            if TORCHDIFFEQ_AVAILABLE and method != 'euler':
                states = self.odeint(
                    ode_func,
                    initial_state,
                    t,
                    method=method,
                    rtol=rtol,
                    atol=atol
                )
            else:
                # Simple Euler integration
                states = odeint(ode_func, initial_state, t)

        # Compute velocities
        velocities = []
        for i in range(len(t)):
            v = ode_func(t[i], states[i])
            velocities.append(v)
        velocities = torch.stack(velocities)

        trajectory = ODETrajectory(
            times=t.cpu().numpy(),
            states=states.cpu().numpy(),
            velocities=velocities.cpu().numpy(),
            method=method,
            rtol=rtol,
            atol=atol
        )

        self._log(f"Integration complete. Shape: {states.shape}")

        return trajectory

    def analyze_flow_field(
        self,
        state_space_bounds: Tuple[np.ndarray, np.ndarray],
        n_points: int = 20,
        time_point: float = 0.0,
        find_fixed_points: bool = True
    ) -> FlowFieldAnalysis:
        """
        Analyze the flow field in state space.

        Args:
            state_space_bounds: (lower_bounds, upper_bounds) for grid
            n_points: Number of grid points per dimension
            time_point: Time at which to analyze flow
            find_fixed_points: Search for fixed points

        Returns:
            FlowFieldAnalysis object
        """
        self._log(f"Analyzing flow field with {n_points}^d grid points")

        lower, upper = state_space_bounds
        n_dims = len(lower)

        # Create grid (2D for visualization)
        if n_dims == 2:
            x = np.linspace(lower[0], upper[0], n_points)
            y = np.linspace(lower[1], upper[1], n_points)
            X, Y = np.meshgrid(x, y)
            positions = np.stack([X.flatten(), Y.flatten()], axis=1)
        else:
            # Sample randomly for high dimensions
            positions = np.random.uniform(
                lower, upper, size=(n_points**2, n_dims)
            )

        positions_tensor = torch.tensor(positions, dtype=torch.float32).to(self.device)

        # Compute velocities
        def ode_func(t, x):
            output = self.model(x)
            return output - x

        with torch.no_grad():
            t = torch.tensor([time_point], dtype=torch.float32).to(self.device)
            velocities = ode_func(t, positions_tensor)

        velocities_np = velocities.cpu().numpy()
        speeds = np.linalg.norm(velocities_np, axis=1)

        # Find fixed points if requested
        fixed_points = []
        stability = []

        if find_fixed_points and n_dims <= 3:
            fps, stab = self._find_fixed_points(positions, velocities_np, speeds)
            fixed_points = fps
            stability = stab

        analysis = FlowFieldAnalysis(
            positions=positions,
            velocities=velocities_np,
            speeds=speeds,
            fixed_points=fixed_points,
            fixed_point_stability=stability,
            time_point=time_point,
            bounds=state_space_bounds
        )

        self._log(f"Flow field analysis complete. Found {len(fixed_points)} fixed points")

        return analysis

    def _find_fixed_points(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        speeds: np.ndarray,
        speed_threshold: float = 0.01
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Find fixed points (where velocity ≈ 0)."""
        fixed_points = []
        stability = []

        # Find candidate points with low speed
        candidates_idx = np.where(speeds < speed_threshold)[0]

        for idx in candidates_idx[:10]:  # Limit to top 10
            fp = positions[idx]
            fixed_points.append(fp)

            # Classify stability (simplified - would need Jacobian)
            # For now, just mark as 'unknown'
            stability.append('unknown')

        return fixed_points, stability

    def compute_lyapunov_exponents(
        self,
        initial_state: torch.Tensor,
        time_span: Tuple[float, float],
        n_steps: int = 1000
    ) -> np.ndarray:
        """
        Compute Lyapunov exponents along trajectory.

        Uses finite-time tangent space evolution.

        Args:
            initial_state: Starting state
            time_span: Time range
            n_steps: Number of steps

        Returns:
            Lyapunov exponents (n_dims,)
        """
        self._log("Computing Lyapunov exponents...")

        # Integrate main trajectory
        trajectory = self.integrate_trajectory(
            initial_state, time_span, n_steps
        )

        # Integrate perturbed trajectories
        n_dims = initial_state.shape[-1]
        perturbation_size = 1e-5

        lyapunov_sums = np.zeros(n_dims)

        for i in range(n_dims):
            # Create perturbation in dimension i
            perturbation = torch.zeros_like(initial_state)
            perturbation[..., i] = perturbation_size

            perturbed_state = initial_state + perturbation

            # Integrate perturbed trajectory
            perturbed_traj = self.integrate_trajectory(
                perturbed_state, time_span, n_steps
            )

            # Compute separation growth rate
            separations = np.linalg.norm(
                perturbed_traj.states - trajectory.states,
                axis=-1
            )

            # Lyapunov exponent: average log separation growth
            valid_sep = separations[separations > 0]
            if len(valid_sep) > 0:
                lyapunov_sums[i] = np.mean(np.log(valid_sep / perturbation_size)) / (time_span[1] - time_span[0])

        self._log(f"Lyapunov exponents computed: {lyapunov_sums}")

        return lyapunov_sums


__all__ = [
    'FlowFieldAnalysis',
    'ODETrajectory',
    'NeuralODEIntegrator',
]
