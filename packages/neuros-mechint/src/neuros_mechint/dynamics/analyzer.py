"""
Unified Dynamics Analyzer

This module provides a unified interface to all dynamical systems analysis tools.

The DynamicsAnalyzer class combines all operators into a single, easy-to-use interface
for comprehensive analysis of dynamical systems.
"""

from typing import Optional, Dict, Any
import numpy as np
import logging

# Import all operators
from .koopman import KoopmanOperator, KoopmanResult
from .lyapunov import LyapunovAnalyzer, LyapunovResult
from .fixed_points import FixedPointFinder, FixedPointResult
from .manifold import ManifoldAnalyzer, ManifoldResult
from .phase_space import PhaseSpaceAnalyzer, PhaseSpaceResult
from .granger import GrangerCausality, GrangerResult
from .bifurcation import BifurcationDetector, BifurcationResult
from .perturbation import PerturbationAnalyzer, PerturbationResponse, SensitivityResult, RobustnessResult
from .recurrence import RecurrenceAnalyzer, RecurrenceResult
from .transfer_operator import TransferOperator, TransferOperatorResult
from .synchronization import SynchronizationAnalyzer, SynchronizationResult
from .information import InformationAnalyzer, InformationResult
from .optimal_transport import OptimalTransport, OptimalTransportResult
from .spectral import SpectralAnalyzer, SpectralResult
from .reservoir import ReservoirComputing, ReservoirResult
from .neural_ode import NeuralODEIntegrator
from .slow_features import SlowFeatureAnalyzer

logger = logging.getLogger(__name__)


class DynamicsAnalyzer:
    """
    Unified interface for comprehensive dynamical systems analysis.

    This class provides access to all analysis methods through a single,
    coherent API. It combines spectral, geometric, topological, and
    information-theoretic approaches.

    Example:
        ```python
        from neuros_mechint.dynamics import DynamicsAnalyzer

        analyzer = DynamicsAnalyzer(dt=0.01)

        # Run individual analyses
        koopman = analyzer.estimate_koopman_operator(trajectories)
        lyapunov = analyzer.compute_lyapunov_exponents(trajectories)
        manifold = analyzer.analyze_manifold(trajectories)

        # Or run all analyses at once
        results = analyzer.run_all_analyses(trajectories)
        ```
    """

    def __init__(
        self,
        dt: float = 0.01,
        device: str = "cpu",
        verbose: bool = True
    ):
        """
        Initialize unified dynamics analyzer.

        Args:
            dt: Time step between observations
            device: Device for torch computations ("cpu" or "cuda")
            verbose: Whether to log information
        """
        self.dt = dt
        self.device = device
        self.verbose = verbose

        # Initialize all operators
        self._init_operators()

    def _init_operators(self):
        """Initialize all analysis operators."""
        # Core operators
        self.koopman = KoopmanOperator(dt=self.dt, device=self.device, verbose=self.verbose)
        self.lyapunov = LyapunovAnalyzer(dt=self.dt, device=self.device, verbose=self.verbose)
        self.fixed_points = FixedPointFinder(dt=self.dt, device=self.device, verbose=self.verbose)
        self.manifold = ManifoldAnalyzer(dt=self.dt, device=self.device, verbose=self.verbose)
        self.phase_space = PhaseSpaceAnalyzer(dt=self.dt, device=self.device, verbose=self.verbose)

        # Advanced operators
        self.granger = GrangerCausality(dt=self.dt, verbose=self.verbose)
        self.bifurcation = BifurcationDetector(dt=self.dt, verbose=self.verbose)
        self.perturbation = PerturbationAnalyzer(dt=self.dt, device=self.device, verbose=self.verbose)

        # Novel methods
        self.recurrence = RecurrenceAnalyzer(dt=self.dt, verbose=self.verbose)
        self.transfer_operator = TransferOperator(dt=self.dt, verbose=self.verbose)
        self.synchronization = SynchronizationAnalyzer(dt=self.dt, device=self.device, verbose=self.verbose)
        self.information = InformationAnalyzer(dt=self.dt, verbose=self.verbose)

        # Additional methods
        self.optimal_transport = OptimalTransport(verbose=self.verbose)
        self.spectral = SpectralAnalyzer(dt=self.dt, verbose=self.verbose)
        self.reservoir = None  # Initialized on demand

        # Existing modules
        self.neural_ode = None  # Initialized on demand
        self.slow_features = SlowFeatureAnalyzer()

    # ==================== Core Operators ====================

    def estimate_koopman_operator(
        self,
        trajectories: np.ndarray,
        method: str = "standard",
        rank: Optional[int] = None,
        **kwargs
    ) -> KoopmanResult:
        """
        Estimate Koopman operator using Dynamic Mode Decomposition.

        Args:
            trajectories: Trajectory data (n_timesteps, n_features)
            method: DMD variant ("standard", "extended", "kernel", "hankel", "optimal", "sparse")
            rank: Truncation rank
            **kwargs: Additional method-specific arguments

        Returns:
            KoopmanResult with operator and modes
        """
        return self.koopman.fit(trajectories, method=method, rank=rank, **kwargs)

    def compute_lyapunov_exponents(
        self,
        trajectories: np.ndarray,
        method: str = "orthogonalization",
        **kwargs
    ) -> LyapunovResult:
        """
        Compute Lyapunov exponents for chaos quantification.

        Args:
            trajectories: Trajectory data (n_timesteps, n_features)
            method: Computation method ("orthogonalization", "jacobian", "divergence", "ftle")
            **kwargs: Method-specific parameters

        Returns:
            LyapunovResult with exponents and characterization
        """
        return self.lyapunov.compute_exponents(trajectories, method=method, **kwargs)

    def find_fixed_points(
        self,
        trajectories: np.ndarray,
        method: str = "velocity",
        **kwargs
    ) -> FixedPointResult:
        """
        Detect fixed points and equilibria.

        Args:
            trajectories: Trajectory data (n_timesteps, n_features)
            method: Detection method ("velocity", "optimization", "recurrence")
            **kwargs: Method-specific parameters

        Returns:
            FixedPointResult with detected fixed points
        """
        return self.fixed_points.find_fixed_points(trajectories, method=method, **kwargs)

    def analyze_manifold(
        self,
        trajectories: np.ndarray,
        compute_curvature: bool = True,
        compute_tangent_spaces: bool = True,
        compute_geodesics: bool = False,
        **kwargs
    ) -> ManifoldResult:
        """
        Analyze manifold structure of state space.

        Args:
            trajectories: Trajectory data (n_timesteps, n_features)
            compute_curvature: Whether to compute curvature
            compute_tangent_spaces: Whether to compute tangent spaces
            compute_geodesics: Whether to compute geodesic distances
            **kwargs: Additional parameters

        Returns:
            ManifoldResult with manifold properties
        """
        return self.manifold.analyze(
            trajectories,
            compute_curvature=compute_curvature,
            compute_tangent_spaces=compute_tangent_spaces,
            compute_geodesics=compute_geodesics,
            **kwargs
        )

    def analyze_phase_space(
        self,
        trajectories: np.ndarray,
        detect_attractors: bool = True,
        compute_basins: bool = True,
        compute_poincare: bool = False,
        **kwargs
    ) -> PhaseSpaceResult:
        """
        Analyze phase space structure.

        Args:
            trajectories: Trajectory data (n_timesteps, n_features)
            detect_attractors: Whether to detect attractors
            compute_basins: Whether to compute basins of attraction
            compute_poincare: Whether to compute Poincaré sections
            **kwargs: Additional parameters

        Returns:
            PhaseSpaceResult with phase space characterization
        """
        return self.phase_space.analyze(
            trajectories,
            detect_attractors=detect_attractors,
            compute_basins=compute_basins,
            compute_poincare=compute_poincare,
            **kwargs
        )

    # ==================== Advanced Operators ====================

    def granger_causality(
        self,
        data: np.ndarray,
        lag_order: Optional[int] = None,
        method: str = "pairwise"
    ) -> GrangerResult:
        """
        Analyze Granger causality relationships.

        Args:
            data: Multivariate time series (n_timesteps, n_features)
            lag_order: Lag order (auto-detected if None)
            method: Analysis method ("pairwise", "conditional", "multivariate")

        Returns:
            GrangerResult with causality matrix
        """
        return self.granger.analyze(data, lag_order=lag_order, method=method)

    def detect_bifurcations(
        self,
        trajectories: np.ndarray,
        parameter_values: Optional[np.ndarray] = None,
        compute_early_warnings: bool = True
    ) -> BifurcationResult:
        """
        Detect bifurcations and critical transitions.

        Args:
            trajectories: Trajectory data (n_timesteps, n_features)
            parameter_values: Optional parameter values at each time step
            compute_early_warnings: Whether to compute early warning signals

        Returns:
            BifurcationResult with detected bifurcations
        """
        return self.bifurcation.detect(
            trajectories,
            parameter_values=parameter_values,
            compute_early_warnings=compute_early_warnings
        )

    def analyze_perturbation_response(
        self,
        system_function,
        initial_state: np.ndarray,
        perturbation_direction: np.ndarray,
        perturbation_magnitude: float,
        duration: float = 10.0
    ) -> PerturbationResponse:
        """
        Analyze system response to perturbations.

        Args:
            system_function: Function f(x, t) computing dx/dt
            initial_state: Initial state
            perturbation_direction: Direction of perturbation
            perturbation_magnitude: Magnitude of perturbation
            duration: Simulation duration

        Returns:
            PerturbationResponse object
        """
        return self.perturbation.analyze_response(
            system_function,
            initial_state,
            perturbation_direction,
            perturbation_magnitude,
            duration=duration
        )

    # ==================== Novel Methods ====================

    def recurrence_analysis(
        self,
        trajectories: np.ndarray,
        compute_network: bool = False
    ) -> RecurrenceResult:
        """
        Perform recurrence plot analysis.

        Args:
            trajectories: Trajectory data (n_timesteps, n_features)
            compute_network: Whether to compute recurrence network properties

        Returns:
            RecurrenceResult with RQA metrics
        """
        return self.recurrence.analyze(trajectories, compute_network=compute_network)

    def transfer_operator_analysis(
        self,
        trajectories: np.ndarray,
        method: str = "ulam",
        n_eigenpairs: int = 10
    ) -> TransferOperatorResult:
        """
        Estimate transfer operator for stochastic dynamics.

        Args:
            trajectories: Trajectory data (n_timesteps, n_features)
            method: Estimation method ("ulam", "galerkin", "edmd")
            n_eigenpairs: Number of eigenpairs to compute

        Returns:
            TransferOperatorResult
        """
        return self.transfer_operator.estimate(
            trajectories,
            method=method,
            n_eigenpairs=n_eigenpairs
        )

    def synchronization_analysis(
        self,
        trajectories: np.ndarray,
        compute_phases: bool = True,
        detect_clusters: bool = True
    ) -> SynchronizationResult:
        """
        Analyze synchronization in coupled systems.

        Args:
            trajectories: Trajectory data (n_timesteps, n_oscillators) or
                         (n_timesteps, n_oscillators, n_features)
            compute_phases: Whether to compute phase synchronization
            detect_clusters: Whether to detect cluster synchronization

        Returns:
            SynchronizationResult
        """
        return self.synchronization.analyze(
            trajectories,
            compute_phases=compute_phases,
            detect_clusters=detect_clusters
        )

    def information_analysis(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        compute_transfer_entropy: bool = True,
        compute_complexity: bool = True
    ) -> InformationResult:
        """
        Information-theoretic analysis.

        Args:
            X: First time series (n_timesteps,) or (n_timesteps, n_features)
            Y: Second time series (optional)
            compute_transfer_entropy: Whether to compute transfer entropy
            compute_complexity: Whether to compute complexity measures

        Returns:
            InformationResult
        """
        return self.information.analyze(
            X,
            Y=Y,
            compute_transfer_entropy=compute_transfer_entropy,
            compute_complexity=compute_complexity
        )

    # ==================== Additional Methods ====================

    def wasserstein_distance(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        p: int = 1
    ) -> float:
        """
        Compute Wasserstein distance between distributions.

        Args:
            X: First distribution (n_samples, n_features)
            Y: Second distribution (m_samples, n_features)
            p: Order of Wasserstein distance (1 or 2)

        Returns:
            Wasserstein distance
        """
        result = self.optimal_transport.wasserstein_distance(X, Y, p=p)
        return result.distance

    def spectral_analysis(
        self,
        time_series: np.ndarray,
        compute_wavelets: bool = False,
        compute_embedding: bool = False
    ) -> SpectralResult:
        """
        Spectral analysis of time series.

        Args:
            time_series: Time series data (n_timesteps,) or (n_timesteps, n_features)
            compute_wavelets: Whether to compute wavelet transform
            compute_embedding: Whether to compute spectral embedding

        Returns:
            SpectralResult
        """
        return self.spectral.analyze(
            time_series,
            compute_wavelets=compute_wavelets,
            compute_embedding=compute_embedding
        )

    def slow_feature_analysis(
        self,
        timeseries: np.ndarray,
        n_features: int = 5,
        expansion_degree: int = 2
    ):
        """
        Extract slowly varying features from time series.

        Args:
            timeseries: Time series data (n_timesteps, n_channels)
            n_features: Number of slow features to extract
            expansion_degree: Degree of polynomial expansion

        Returns:
            SlowFeatureResult
        """
        return self.slow_features.analyze_timeseries(
            timeseries,
            n_features=n_features,
            expansion_degree=expansion_degree
        )

    # ==================== Unified Analysis ====================

    def run_all_analyses(
        self,
        trajectories: np.ndarray,
        include_expensive: bool = False
    ) -> Dict[str, Any]:
        """
        Run all applicable analyses on trajectory data.

        This method runs a comprehensive suite of analyses and returns
        all results in a dictionary.

        Args:
            trajectories: Trajectory data (n_timesteps, n_features)
            include_expensive: Whether to include computationally expensive analyses

        Returns:
            Dictionary containing all analysis results
        """
        results = {}

        if self.verbose:
            logger.info("Running comprehensive dynamics analysis...")

        # Core operators
        try:
            results['koopman'] = self.estimate_koopman_operator(trajectories)
            if self.verbose:
                logger.info("✓ Koopman operator analysis complete")
        except Exception as e:
            logger.warning(f"Koopman analysis failed: {e}")
            results['koopman'] = None

        try:
            results['lyapunov'] = self.compute_lyapunov_exponents(trajectories)
            if self.verbose:
                logger.info("✓ Lyapunov exponent analysis complete")
        except Exception as e:
            logger.warning(f"Lyapunov analysis failed: {e}")
            results['lyapunov'] = None

        try:
            results['fixed_points'] = self.find_fixed_points(trajectories)
            if self.verbose:
                logger.info("✓ Fixed point analysis complete")
        except Exception as e:
            logger.warning(f"Fixed point analysis failed: {e}")
            results['fixed_points'] = None

        try:
            results['manifold'] = self.analyze_manifold(
                trajectories,
                compute_geodesics=include_expensive
            )
            if self.verbose:
                logger.info("✓ Manifold analysis complete")
        except Exception as e:
            logger.warning(f"Manifold analysis failed: {e}")
            results['manifold'] = None

        try:
            results['phase_space'] = self.analyze_phase_space(trajectories)
            if self.verbose:
                logger.info("✓ Phase space analysis complete")
        except Exception as e:
            logger.warning(f"Phase space analysis failed: {e}")
            results['phase_space'] = None

        # Advanced operators
        if trajectories.shape[1] >= 2:  # Need multiple variables for Granger
            try:
                results['granger'] = self.granger_causality(trajectories)
                if self.verbose:
                    logger.info("✓ Granger causality analysis complete")
            except Exception as e:
                logger.warning(f"Granger analysis failed: {e}")
                results['granger'] = None

        try:
            results['bifurcation'] = self.detect_bifurcations(trajectories)
            if self.verbose:
                logger.info("✓ Bifurcation analysis complete")
        except Exception as e:
            logger.warning(f"Bifurcation analysis failed: {e}")
            results['bifurcation'] = None

        # Novel methods
        try:
            results['recurrence'] = self.recurrence_analysis(trajectories)
            if self.verbose:
                logger.info("✓ Recurrence analysis complete")
        except Exception as e:
            logger.warning(f"Recurrence analysis failed: {e}")
            results['recurrence'] = None

        if include_expensive:
            try:
                results['transfer_operator'] = self.transfer_operator_analysis(trajectories)
                if self.verbose:
                    logger.info("✓ Transfer operator analysis complete")
            except Exception as e:
                logger.warning(f"Transfer operator analysis failed: {e}")
                results['transfer_operator'] = None

        try:
            results['spectral'] = self.spectral_analysis(trajectories)
            if self.verbose:
                logger.info("✓ Spectral analysis complete")
        except Exception as e:
            logger.warning(f"Spectral analysis failed: {e}")
            results['spectral'] = None

        try:
            results['slow_features'] = self.slow_feature_analysis(trajectories)
            if self.verbose:
                logger.info("✓ Slow feature analysis complete")
        except Exception as e:
            logger.warning(f"Slow feature analysis failed: {e}")
            results['slow_features'] = None

        if self.verbose:
            logger.info("Comprehensive analysis complete!")

        return results

    def summary_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a text summary of analysis results.

        Args:
            results: Dictionary of analysis results from run_all_analyses

        Returns:
            Formatted text summary
        """
        lines = ["=" * 70]
        lines.append("DYNAMICAL SYSTEMS ANALYSIS SUMMARY")
        lines.append("=" * 70)

        # Lyapunov
        if results.get('lyapunov'):
            lyap = results['lyapunov']
            lines.append("\nLYAPUNOV ANALYSIS:")
            lines.append(f"  Max exponent: {lyap.max_exponent:.4f}")
            lines.append(f"  Is chaotic: {lyap.is_chaotic}")
            lines.append(f"  Lyapunov dimension: {lyap.lyapunov_dimension:.2f}")

        # Fixed points
        if results.get('fixed_points'):
            fp = results['fixed_points']
            lines.append("\nFIXED POINT ANALYSIS:")
            lines.append(f"  Total fixed points: {len(fp.fixed_points)}")
            lines.append(f"  Stable: {fp.n_stable}, Unstable: {fp.n_unstable}, Saddles: {fp.n_saddles}")

        # Manifold
        if results.get('manifold'):
            manifold = results['manifold']
            lines.append("\nMANIFOLD ANALYSIS:")
            lines.append(f"  Intrinsic dimension: {manifold.intrinsic_dimension}")
            lines.append(f"  Participation ratio: {manifold.participation_ratio:.2f}")

        # Phase space
        if results.get('phase_space'):
            phase = results['phase_space']
            lines.append("\nPHASE SPACE ANALYSIS:")
            lines.append(f"  Number of attractors: {phase.n_attractors}")
            lines.append(f"  Topology: {phase.topology_type}")

        # Spectral
        if results.get('spectral'):
            spec = results['spectral']
            lines.append("\nSPECTRAL ANALYSIS:")
            if len(spec.dominant_frequencies) > 0:
                lines.append(f"  Dominant frequencies: {spec.dominant_frequencies[:3]}")
            lines.append(f"  Spectral entropy: {spec.spectral_entropy:.2f}")

        lines.append("=" * 70)

        return "\n".join(lines)
