"""
Fluctuation Theorems for Neural Network Thermodynamics.

Implements fluctuation theorems from non-equilibrium statistical mechanics
to analyze neural networks as thermodynamic systems.

Key Theorems:
- **Crooks Fluctuation Theorem**: P(σ=A)/P(σ=-A) = exp(A)
- **Jarzynski Equality**: <exp(-W)> = exp(-ΔF)
- **Gallavotti-Cohen Theorem**: lim_{t→∞} (1/t) ln[P(σ)/P(-σ)] = σ
- **Hatano-Sasa Relation**: Generalization to driven steady states

These theorems provide fundamental constraints on entropy production
and dissipation in non-equilibrium systems.

Based on:
- Crooks (1999): Entropy production fluctuation theorem
- Jarzynski (1997): Nonequilibrium equality for free energy differences
- Gallavotti & Cohen (1995): Dynamical ensembles in stationary states
- Hatano & Sasa (2001): Steady-state thermodynamics

Example:
    >>> # Test Crooks theorem on model
    >>> analyzer = FluctuationTheoremAnalyzer(model)
    >>>
    >>> # Run forward and reverse processes
    >>> result = analyzer.test_crooks_theorem(
    ...     forward_data=forward_inputs,
    ...     reverse_data=reverse_inputs
    ... )
    >>>
    >>> # Check if theorem holds
    >>> print(f"Crooks ratio validity: {result.crooks_validity:.3f}")
    >>> print(f"Jarzynski free energy: {result.jarzynski_free_energy:.4f}")
    >>>
    >>> # Visualize distributions
    >>> result.visualize_fluctuations(use_bokeh=True)

Author: NeuroS Team
Date: 2025-10-31
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
from scipy import stats

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from bokeh.plotting import figure, output_file, save
    from bokeh.layouts import column, row, gridplot
    from bokeh.models import HoverTool, Span, Label
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

from neuros_mechint.results import MechIntResult

logger = logging.getLogger(__name__)


# Physical constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
ROOM_TEMPERATURE = 300  # K
BETA = 1.0 / (BOLTZMANN_CONSTANT * ROOM_TEMPERATURE)


@dataclass
class FluctuationTheoremResult:
    """Results from fluctuation theorem analysis."""

    # Entropy production distribution
    entropy_production: np.ndarray  # Observed entropy production values
    entropy_histogram: Tuple[np.ndarray, np.ndarray]  # (counts, bins)

    # Crooks theorem test
    crooks_validity: float  # How well Crooks theorem holds (0-1)
    forward_work: np.ndarray  # Work in forward process
    reverse_work: np.ndarray  # Work in reverse process

    # Jarzynski equality
    jarzynski_free_energy: float  # Free energy from Jarzynski
    jarzynski_convergence: float  # Convergence quality

    # Gallavotti-Cohen coefficient
    gc_coefficient: float  # Should equal 1 if theorem holds
    gc_validity: float  # Quality of linear fit (R²)

    # Hatano-Sasa relation (if in steady state)
    hs_housekeeping_heat: float = 0.0
    hs_excess_heat: float = 0.0

    # Statistical properties
    mean_entropy_production: float = 0.0
    std_entropy_production: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0

    # Metadata
    n_samples: int = 0
    temperature: float = ROOM_TEMPERATURE

    def visualize_fluctuations(
        self,
        use_bokeh: bool = True,
        save_path: Optional[str] = None
    ) -> Any:
        """
        Visualize fluctuation theorem results.

        Args:
            use_bokeh: Use Bokeh for interactive plots
            save_path: Path to save visualization

        Returns:
            Bokeh layout or matplotlib figure
        """
        if use_bokeh and BOKEH_AVAILABLE:
            return self._visualize_bokeh(save_path)
        else:
            return self._visualize_matplotlib(save_path)

    def _visualize_bokeh(self, save_path: Optional[str]) -> Any:
        """Create interactive Bokeh visualization."""

        plots = []

        # 1. Entropy production distribution
        counts, bins = self.entropy_histogram
        bin_centers = (bins[:-1] + bins[1:]) / 2

        p1 = figure(
            title='Entropy Production Distribution',
            x_axis_label='Entropy Production σ',
            y_axis_label='P(σ)',
            width=600,
            height=400
        )

        p1.quad(
            top=counts, bottom=0,
            left=bins[:-1], right=bins[1:],
            fill_alpha=0.6, fill_color='navy'
        )

        # Add theoretical Gaussian for comparison
        x_theory = np.linspace(bins[0], bins[-1], 200)
        y_theory = stats.norm.pdf(
            x_theory,
            self.mean_entropy_production,
            self.std_entropy_production
        ) * (counts.sum() * (bins[1] - bins[0]))
        p1.line(x_theory, y_theory, line_width=2, color='red',
               legend_label='Gaussian fit', line_dash='dashed')

        p1.legend.location = "top_right"
        plots.append(p1)

        # 2. Crooks theorem: log ratio plot
        # P(σ=A)/P(σ=-A) vs A
        if len(self.forward_work) > 0 and len(self.reverse_work) > 0:
            # Compute log ratio at different σ values
            sigma_range = np.linspace(
                self.entropy_production.min(),
                self.entropy_production.max(),
                50
            )

            log_ratios = []
            sigma_vals = []

            for sigma in sigma_range:
                # Count samples near +sigma and -sigma
                tol = 0.1
                count_plus = ((self.forward_work >= sigma - tol) &
                            (self.forward_work <= sigma + tol)).sum()
                count_minus = ((self.reverse_work >= -sigma - tol) &
                             (self.reverse_work <= -sigma + tol)).sum()

                if count_plus > 0 and count_minus > 0:
                    log_ratio = np.log(count_plus / count_minus)
                    log_ratios.append(log_ratio)
                    sigma_vals.append(sigma)

            if len(sigma_vals) > 0:
                p2 = figure(
                    title='Crooks Theorem Test: ln[P(σ)/P(-σ)] vs σ',
                    x_axis_label='σ',
                    y_axis_label='ln[P(σ)/P(-σ)]',
                    width=600,
                    height=400
                )

                p2.circle(sigma_vals, log_ratios, size=8, alpha=0.6, color='blue')

                # Theoretical line: should equal σ
                p2.line(sigma_vals, sigma_vals, line_width=2, color='red',
                       line_dash='dashed', legend_label='Crooks prediction (y=x)')

                p2.legend.location = "top_left"
                plots.append(p2)

        # 3. Gallavotti-Cohen test
        # ln[P(σ)/P(-σ)] / t vs σ should be linear with slope 1
        if len(self.entropy_production) > 10:
            hist_pos, bins_pos = np.histogram(
                self.entropy_production[self.entropy_production > 0],
                bins=30
            )
            hist_neg, bins_neg = np.histogram(
                -self.entropy_production[self.entropy_production < 0],
                bins=30
            )

            # Match bins
            bin_centers_pos = (bins_pos[:-1] + bins_pos[1:]) / 2
            bin_centers_neg = (bins_neg[:-1] + bins_neg[1:]) / 2

            p3 = figure(
                title='Gallavotti-Cohen Theorem',
                x_axis_label='σ',
                y_axis_label='ln[P(σ)/P(-σ)]',
                width=600,
                height=400
            )

            # Plot ratio where both have counts
            ratios = []
            sigmas = []
            for i, (c_pos, sigma_pos) in enumerate(zip(hist_pos, bin_centers_pos)):
                # Find corresponding negative bin
                idx_neg = np.argmin(np.abs(bin_centers_neg - sigma_pos))
                c_neg = hist_neg[idx_neg]

                if c_pos > 0 and c_neg > 0:
                    ratio = np.log(c_pos / c_neg)
                    ratios.append(ratio)
                    sigmas.append(sigma_pos)

            if len(sigmas) > 0:
                p3.circle(sigmas, ratios, size=8, alpha=0.6, color='green')

                # Theoretical line
                sigma_range = np.array(sigmas)
                p3.line(sigma_range, sigma_range, line_width=2, color='red',
                       line_dash='dashed', legend_label='GC prediction (slope=1)')

                p3.legend.location = "top_left"
                plots.append(p3)

        # Create grid layout
        grid = gridplot([plots[i:i+2] for i in range(0, len(plots), 2)])

        if save_path:
            output_file(save_path)
            save(grid)

        return grid

    def _visualize_matplotlib(self, save_path: Optional[str]) -> Any:
        """Create matplotlib visualization."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available")
            return None

        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig)

        # 1. Entropy production distribution
        ax1 = fig.add_subplot(gs[0, 0])
        counts, bins = self.entropy_histogram
        ax1.hist(self.entropy_production, bins=50, alpha=0.7, density=True, color='navy')

        # Gaussian fit
        x = np.linspace(self.entropy_production.min(), self.entropy_production.max(), 200)
        y = stats.norm.pdf(x, self.mean_entropy_production, self.std_entropy_production)
        ax1.plot(x, y, 'r--', linewidth=2, label='Gaussian fit')

        ax1.set_xlabel('Entropy Production σ')
        ax1.set_ylabel('P(σ)')
        ax1.set_title('Entropy Production Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Crooks theorem test
        ax2 = fig.add_subplot(gs[0, 1])
        if len(self.forward_work) > 0 and len(self.reverse_work) > 0:
            sigma_range = np.linspace(
                self.entropy_production.min(),
                self.entropy_production.max(),
                50
            )

            log_ratios = []
            sigma_vals = []

            for sigma in sigma_range:
                tol = 0.1
                count_plus = ((self.forward_work >= sigma - tol) &
                            (self.forward_work <= sigma + tol)).sum()
                count_minus = ((self.reverse_work >= -sigma - tol) &
                             (self.reverse_work <= -sigma + tol)).sum()

                if count_plus > 0 and count_minus > 0:
                    log_ratios.append(np.log(count_plus / count_minus))
                    sigma_vals.append(sigma)

            if len(sigma_vals) > 0:
                ax2.scatter(sigma_vals, log_ratios, alpha=0.6, color='blue', s=50)
                ax2.plot(sigma_vals, sigma_vals, 'r--', linewidth=2, label='Crooks (y=x)')

        ax2.set_xlabel('σ')
        ax2.set_ylabel('ln[P(σ)/P(-σ)]')
        ax2.set_title('Crooks Fluctuation Theorem', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Work distributions
        ax3 = fig.add_subplot(gs[1, 0])
        if len(self.forward_work) > 0:
            ax3.hist(self.forward_work, bins=40, alpha=0.6, label='Forward', color='blue')
        if len(self.reverse_work) > 0:
            ax3.hist(-self.reverse_work, bins=40, alpha=0.6, label='Reverse (negated)', color='red')
        ax3.set_xlabel('Work W')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Forward vs Reverse Work Distributions', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Summary statistics
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')

        summary_text = f"""
Fluctuation Theorem Results
{'=' * 40}

Entropy Production:
  Mean:     {self.mean_entropy_production:.4f}
  Std:      {self.std_entropy_production:.4f}
  Skewness: {self.skewness:.4f}
  Kurtosis: {self.kurtosis:.4f}

Theorem Tests:
  Crooks Validity:     {self.crooks_validity:.3f}
  GC Coefficient:      {self.gc_coefficient:.3f}
  GC R² (validity):    {self.gc_validity:.3f}

Jarzynski:
  Free Energy ΔF:      {self.jarzynski_free_energy:.4f}
  Convergence:         {self.jarzynski_convergence:.3f}

Samples: {self.n_samples}
Temperature: {self.temperature:.1f} K
        """

        ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class FluctuationTheoremAnalyzer:
    """
    Analyzer for testing fluctuation theorems on neural networks.

    Tests fundamental thermodynamic theorems on network activations
    to understand non-equilibrium behavior and dissipation.

    Args:
        model: Neural network to analyze
        device: Torch device
        temperature: System temperature in Kelvin
        verbose: Enable verbose logging

    Example:
        >>> analyzer = FluctuationTheoremAnalyzer(model)
        >>> result = analyzer.test_crooks_theorem(forward_data, reverse_data)
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        temperature: float = ROOM_TEMPERATURE,
        verbose: bool = True
    ):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.temperature = temperature
        self.beta = 1.0 / (BOLTZMANN_CONSTANT * temperature)
        self.verbose = verbose

        self.model.to(self.device)
        self.model.eval()

        self._log("Initialized FluctuationTheoremAnalyzer")

    def _log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            logger.info(f"[FluctuationTheoremAnalyzer] {message}")

    def test_crooks_theorem(
        self,
        forward_data: torch.Tensor,
        reverse_data: torch.Tensor,
        n_samples: int = 1000
    ) -> FluctuationTheoremResult:
        """
        Test Crooks fluctuation theorem.

        Crooks: P_F(W)/P_R(-W) = exp(β(W - ΔF))

        Args:
            forward_data: Data for forward process
            reverse_data: Data for reverse process
            n_samples: Number of samples to use

        Returns:
            FluctuationTheoremResult
        """
        self._log("Testing Crooks fluctuation theorem...")

        # Compute work in forward and reverse processes
        forward_work = self._compute_work(forward_data, n_samples)
        reverse_work = self._compute_work(reverse_data, n_samples)

        # Compute entropy production
        entropy_production = forward_work + reverse_work  # Simplification

        # Test Crooks theorem: check if log ratio matches expectation
        crooks_validity = self._test_crooks_ratio(forward_work, reverse_work)

        # Test Jarzynski equality
        jarzynski_free_energy, jarzynski_convergence = self._test_jarzynski(forward_work)

        # Test Gallavotti-Cohen theorem
        gc_coefficient, gc_validity = self._test_gallavotti_cohen(entropy_production)

        # Compute statistics
        mean_entropy = float(np.mean(entropy_production))
        std_entropy = float(np.std(entropy_production))
        skewness = float(stats.skew(entropy_production))
        kurtosis = float(stats.kurtosis(entropy_production))

        # Histogram
        counts, bins = np.histogram(entropy_production, bins=50, density=True)

        result = FluctuationTheoremResult(
            entropy_production=entropy_production,
            entropy_histogram=(counts, bins),
            crooks_validity=crooks_validity,
            forward_work=forward_work,
            reverse_work=reverse_work,
            jarzynski_free_energy=jarzynski_free_energy,
            jarzynski_convergence=jarzynski_convergence,
            gc_coefficient=gc_coefficient,
            gc_validity=gc_validity,
            mean_entropy_production=mean_entropy,
            std_entropy_production=std_entropy,
            skewness=skewness,
            kurtosis=kurtosis,
            n_samples=n_samples,
            temperature=self.temperature
        )

        self._log(f"Crooks validity: {crooks_validity:.3f}")
        self._log(f"Jarzynski ΔF: {jarzynski_free_energy:.4f}")
        self._log(f"GC coefficient: {gc_coefficient:.3f}")

        return result

    def _compute_work(self, data: torch.Tensor, n_samples: int) -> np.ndarray:
        """
        Compute work done during forward pass.

        Simplified: Work ≈ change in "energy" (output norm)
        """
        data = data[:n_samples].to(self.device)
        work_values = []

        with torch.no_grad():
            for i in range(len(data)):
                input_i = data[i:i+1]

                # Initial "energy"
                E_initial = input_i.norm().item()

                # Final energy (output)
                output = self.model(input_i)
                E_final = output.norm().item()

                # Work = ΔE
                work = E_final - E_initial
                work_values.append(work)

        return np.array(work_values)

    def _test_crooks_ratio(
        self,
        forward_work: np.ndarray,
        reverse_work: np.ndarray
    ) -> float:
        """
        Test if Crooks ratio holds.

        Returns validity score (0-1).
        """
        # Sample work values
        work_vals = np.linspace(
            min(forward_work.min(), -reverse_work.max()),
            max(forward_work.max(), -reverse_work.min()),
            20
        )

        deviations = []
        tol = 0.2

        for w in work_vals:
            # Count samples near w
            count_fwd = ((forward_work >= w - tol) & (forward_work <= w + tol)).sum()
            count_rev = ((reverse_work >= -w - tol) & (reverse_work <= -w + tol)).sum()

            if count_fwd > 0 and count_rev > 0:
                # Log ratio
                log_ratio = np.log(count_fwd / count_rev)

                # Should equal β*w (assuming ΔF ≈ 0 for simplicity)
                expected = self.beta * w

                # Normalized deviation
                deviation = abs(log_ratio - expected) / (abs(expected) + 1)
                deviations.append(deviation)

        if not deviations:
            return 0.0

        # Validity: inverse of mean deviation
        mean_deviation = np.mean(deviations)
        validity = np.exp(-mean_deviation)

        return float(validity)

    def _test_jarzynski(self, work: np.ndarray) -> Tuple[float, float]:
        """
        Test Jarzynski equality: <exp(-βW)> = exp(-βΔF)

        Returns: (ΔF, convergence_quality)
        """
        # Compute expectation
        exp_work = np.exp(-self.beta * work)
        mean_exp_work = np.mean(exp_work)

        # Free energy
        if mean_exp_work > 0:
            delta_F = -np.log(mean_exp_work) / self.beta
        else:
            delta_F = 0.0

        # Convergence: check if estimate is stable
        # Use bootstrap to estimate variance
        n_boot = 100
        boot_estimates = []

        for _ in range(n_boot):
            boot_sample = np.random.choice(work, size=len(work), replace=True)
            boot_exp = np.exp(-self.beta * boot_sample).mean()
            if boot_exp > 0:
                boot_estimates.append(-np.log(boot_exp) / self.beta)

        if boot_estimates:
            convergence = 1.0 / (1.0 + np.std(boot_estimates))
        else:
            convergence = 0.0

        return float(delta_F), float(convergence)

    def _test_gallavotti_cohen(self, entropy_production: np.ndarray) -> Tuple[float, float]:
        """
        Test Gallavotti-Cohen theorem: ln[P(σ)/P(-σ)] / t → σ

        Returns: (coefficient, R²)
        """
        # Separate positive and negative entropy production
        sigma_pos = entropy_production[entropy_production > 0]
        sigma_neg = -entropy_production[entropy_production < 0]

        if len(sigma_pos) < 10 or len(sigma_neg) < 10:
            return 0.0, 0.0

        # Histogram
        bins = np.linspace(0, max(sigma_pos.max(), sigma_neg.max()), 30)
        hist_pos, _ = np.histogram(sigma_pos, bins=bins)
        hist_neg, _ = np.histogram(sigma_neg, bins=bins)

        # Compute log ratios where both have counts
        bin_centers = (bins[:-1] + bins[1:]) / 2
        valid_mask = (hist_pos > 0) & (hist_neg > 0)

        if valid_mask.sum() < 3:
            return 0.0, 0.0

        sigma_vals = bin_centers[valid_mask]
        log_ratios = np.log(hist_pos[valid_mask] / hist_neg[valid_mask])

        # Fit line: log_ratio = coefficient * sigma
        # Should have coefficient ≈ 1
        slope, intercept, r_value, _, _ = stats.linregress(sigma_vals, log_ratios)

        return float(slope), float(r_value**2)


__all__ = [
    'FluctuationTheoremResult',
    'FluctuationTheoremAnalyzer',
]
