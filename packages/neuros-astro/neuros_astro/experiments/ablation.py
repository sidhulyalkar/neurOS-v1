"""
Ablation study framework for comparing models with/without astrocyte modality.

Provides structured comparison of neural-only vs neural+astro models
to quantify the contribution of astrocyte signals.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from neuros_astro.experiments.tracker import ExperimentConfig, ExperimentResult
from neuros_astro.analysis.statistics import permutation_test, effect_size_cohens_d


@dataclass
class AblationCondition:
    """Configuration for one condition in an ablation study."""

    condition_name: str
    description: str
    modalities: List[str]  # e.g., ['neural'] or ['neural', 'astro']
    config: ExperimentConfig
    result: Optional[ExperimentResult] = None


@dataclass
class AblationComparison:
    """Result of comparing two ablation conditions."""

    baseline_name: str
    test_name: str
    metric_name: str

    baseline_value: float
    test_value: float
    difference: float
    percent_change: float

    effect_size: float
    p_value: float
    significant: bool

    interpretation: str  # "improvement", "degradation", "no_change"


class AblationStudy:
    """
    Framework for systematic ablation experiments.

    Compares performance of models with different modality combinations
    to quantify the contribution of astrocyte signals.
    """

    def __init__(
        self,
        study_name: str,
        output_dir: str | Path = "./ablation_results",
    ):
        """
        Initialize ablation study.

        Args:
            study_name: Name of the study
            output_dir: Directory to save results
        """
        self.study_name = study_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.conditions: Dict[str, AblationCondition] = {}
        self.comparisons: List[AblationComparison] = []

    def add_condition(
        self,
        condition_name: str,
        description: str,
        modalities: List[str],
        config: ExperimentConfig,
    ) -> None:
        """
        Add an ablation condition.

        Args:
            condition_name: Name of condition (e.g., 'baseline', 'with_astro')
            description: Human-readable description
            modalities: List of modalities enabled
            config: Experiment configuration
        """
        condition = AblationCondition(
            condition_name=condition_name,
            description=description,
            modalities=modalities,
            config=config,
        )

        self.conditions[condition_name] = condition

        print(f"✓ Added condition: {condition_name}")
        print(f"  Modalities: {', '.join(modalities)}")

    def set_result(
        self,
        condition_name: str,
        result: ExperimentResult,
    ) -> None:
        """
        Set result for a condition.

        Args:
            condition_name: Name of condition
            result: Experiment result
        """
        if condition_name not in self.conditions:
            raise ValueError(f"Condition not found: {condition_name}")

        self.conditions[condition_name].result = result

        print(f"✓ Set result for condition: {condition_name}")

    def compare_conditions(
        self,
        baseline_name: str,
        test_name: str,
        metric: str = "prediction_loss",
        alpha: float = 0.05,
    ) -> AblationComparison:
        """
        Compare two conditions on a metric.

        Args:
            baseline_name: Name of baseline condition
            test_name: Name of test condition
            metric: Metric to compare
            alpha: Significance level

        Returns:
            AblationComparison object
        """
        if baseline_name not in self.conditions:
            raise ValueError(f"Baseline condition not found: {baseline_name}")

        if test_name not in self.conditions:
            raise ValueError(f"Test condition not found: {test_name}")

        baseline = self.conditions[baseline_name]
        test = self.conditions[test_name]

        if baseline.result is None or test.result is None:
            raise ValueError("Both conditions must have results set")

        # Get metric values
        baseline_value = self._get_metric_value(baseline.result, metric)
        test_value = self._get_metric_value(test.result, metric)

        # Compute difference and percent change
        difference = test_value - baseline_value
        if baseline_value != 0:
            percent_change = (difference / abs(baseline_value)) * 100
        else:
            percent_change = 0.0

        # For statistical testing, we need multiple samples
        # If only single values, set p_value to NaN
        p_value = np.nan
        effect_size = 0.0

        # Interpret result
        if abs(percent_change) < 1.0:
            interpretation = "no_change"
        elif difference < 0:
            # Lower is better for loss metrics
            interpretation = "improvement" if "loss" in metric.lower() or "error" in metric.lower() else "degradation"
        else:
            # Higher is better for accuracy/score metrics
            interpretation = "improvement" if "accuracy" in metric.lower() or "score" in metric.lower() else "degradation"

        significant = abs(percent_change) > 5.0  # Heuristic: >5% change is significant

        comparison = AblationComparison(
            baseline_name=baseline_name,
            test_name=test_name,
            metric_name=metric,
            baseline_value=baseline_value,
            test_value=test_value,
            difference=difference,
            percent_change=percent_change,
            effect_size=effect_size,
            p_value=p_value,
            significant=significant,
            interpretation=interpretation,
        )

        self.comparisons.append(comparison)

        return comparison

    def _get_metric_value(
        self,
        result: ExperimentResult,
        metric: str,
    ) -> float:
        """Get metric value from result."""
        # Try result attributes first
        if hasattr(result, metric):
            return float(getattr(result, metric))

        # Try model_metrics dict
        if metric in result.model_metrics:
            return float(result.model_metrics[metric])

        raise ValueError(f"Metric not found in result: {metric}")

    def generate_comparison_table(self) -> str:
        """
        Generate formatted comparison table.

        Returns:
            Formatted string table
        """
        if not self.comparisons:
            return "No comparisons performed yet."

        lines = [
            "=" * 100,
            "ABLATION STUDY COMPARISON RESULTS",
            "=" * 100,
            f"Study: {self.study_name}",
            "",
            f"{'Baseline':<20} {'Test':<20} {'Metric':<25} {'Baseline':<12} {'Test':<12} {'Change':<12} {'Result':<15}",
            "-" * 100,
        ]

        for comp in self.comparisons:
            change_str = f"{comp.percent_change:+.2f}%"
            result_str = f"{comp.interpretation} {'*' if comp.significant else ''}"

            lines.append(
                f"{comp.baseline_name:<20} "
                f"{comp.test_name:<20} "
                f"{comp.metric_name:<25} "
                f"{comp.baseline_value:<12.4f} "
                f"{comp.test_value:<12.4f} "
                f"{change_str:<12} "
                f"{result_str:<15}"
            )

        lines.extend([
            "=" * 100,
            "",
            "* = Significant change (>5%)",
            "Result interpretation:",
            "  - improvement: Test condition performs better",
            "  - degradation: Test condition performs worse",
            "  - no_change: Minimal difference (<1%)",
        ])

        return "\n".join(lines)

    def save_summary(
        self,
        path: Optional[str | Path] = None,
    ) -> None:
        """
        Save study summary to file.

        Args:
            path: Optional path (if None, uses default in output_dir)
        """
        if path is None:
            path = self.output_dir / f"{self.study_name}_summary.txt"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        summary = self.generate_comparison_table()

        with open(path, 'w') as f:
            f.write(summary)

        print(f"✓ Saved ablation summary to {path}")


def compare_ablation_results(
    baseline_results: List[float],
    test_results: List[float],
    metric_name: str = "performance",
    alpha: float = 0.05,
) -> Dict:
    """
    Compare ablation results with statistical testing.

    Use this when you have multiple trials/folds for each condition.

    Args:
        baseline_results: List of metric values for baseline condition
        test_results: List of metric values for test condition
        metric_name: Name of metric
        alpha: Significance level

    Returns:
        Dict with comparison statistics

    Example:
        >>> baseline = [0.75, 0.77, 0.76, 0.74, 0.75]  # 5 folds
        >>> with_astro = [0.82, 0.83, 0.81, 0.84, 0.82]  # 5 folds
        >>> stats = compare_ablation_results(baseline, with_astro, metric_name='accuracy')
        >>> print(f"p-value: {stats['p_value']:.4f}")
        >>> print(f"Effect size: {stats['effect_size']:.2f}")
    """
    baseline = np.array(baseline_results)
    test = np.array(test_results)

    # Compute statistics
    baseline_mean = float(np.mean(baseline))
    baseline_std = float(np.std(baseline, ddof=1))

    test_mean = float(np.mean(test))
    test_std = float(np.std(test, ddof=1))

    difference = test_mean - baseline_mean
    percent_change = (difference / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0.0

    # Statistical tests
    p_value = permutation_test(test, baseline, n_permutations=10000)
    effect_size = effect_size_cohens_d(test, baseline)

    # Interpretation
    significant = p_value < alpha

    if "loss" in metric_name.lower() or "error" in metric_name.lower():
        # Lower is better
        improved = difference < 0
    else:
        # Higher is better
        improved = difference > 0

    return {
        'metric_name': metric_name,
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'test_mean': test_mean,
        'test_std': test_std,
        'difference': difference,
        'percent_change': percent_change,
        'p_value': p_value,
        'effect_size': effect_size,
        'significant': significant,
        'improved': improved,
        'interpretation': 'significant_improvement' if (significant and improved) else
                         'significant_degradation' if (significant and not improved) else
                         'no_significant_difference',
    }
