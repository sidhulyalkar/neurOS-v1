"""
Model-to-brain alignment tools for NeuroFMX.

This module provides comprehensive methods for aligning neural network
representations with brain recordings, including:

- Canonical Correlation Analysis (CCA): Linear alignment methods
- Representational Similarity Analysis (RSA): Geometry-based comparisons
- Partial Least Squares (PLS): Predictive alignment with latent variables
- Evaluation Metrics: Noise ceiling, bootstrap CIs, permutation tests

Examples:
    CCA alignment:
        >>> from neuros_mechint.alignment import CCA
        >>> cca = CCA(n_components=10)
        >>> cca.fit(model_activations, brain_recordings)
        >>> model_canonical, brain_canonical = cca.transform(model_activations, brain_recordings)

    RSA comparison:
        >>> from neuros_mechint.alignment import RSA
        >>> rsa = RSA(metric='correlation', comparison='spearman')
        >>> similarity = rsa.compare(model_activations, brain_recordings)

    PLS prediction:
        >>> from neuros_mechint.alignment import PLS
        >>> pls = PLS(n_components=20)
        >>> pls.fit(model_activations, brain_recordings)
        >>> predicted_brain = pls.predict(new_model_activations)

    Noise ceiling:
        >>> from neuros_mechint.alignment import NoiseCeiling
        >>> nc = NoiseCeiling(method='split-half')
        >>> ceiling = nc.estimate(brain_recordings_with_repetitions)
"""

# CCA imports
from .cca import (
    CCA,
    RegularizedCCA,
    KernelCCA,
    TimeVaryingCCA,
    select_cca_dimensions
)

# RSA imports
from .rsa import (
    RepresentationalDissimilarityMatrix,
    RSA,
    HierarchicalRSA,
    MDSVisualization,
    compare_multiple_rdms
)

# PLS imports
from .pls import (
    PLS,
    CrossValidatedPLS,
    PLSVisualization
)

# Metrics imports
from .metrics import (
    NoiseCeiling,
    BootstrapCI,
    PermutationTest,
    NormalizedScore,
    CrossValidatedMetric
)

__all__ = [
    # CCA
    'CCA',
    'RegularizedCCA',
    'KernelCCA',
    'TimeVaryingCCA',
    'select_cca_dimensions',

    # RSA
    'RepresentationalDissimilarityMatrix',
    'RSA',
    'HierarchicalRSA',
    'MDSVisualization',
    'compare_multiple_rdms',

    # PLS
    'PLS',
    'CrossValidatedPLS',
    'PLSVisualization',

    # Metrics
    'NoiseCeiling',
    'BootstrapCI',
    'PermutationTest',
    'NormalizedScore',
    'CrossValidatedMetric',
]


# Module metadata
__version__ = '1.0.0'
__author__ = 'NeuroFMX Team'
__description__ = 'Model-to-brain alignment tools for neural foundation models'


def get_alignment_summary():
    """
    Get a summary of available alignment methods.

    Returns:
        Dictionary with method descriptions and use cases
    """
    return {
        'CCA': {
            'description': 'Canonical Correlation Analysis - finds linear projections that maximize correlation',
            'variants': ['Standard CCA', 'Regularized CCA', 'Kernel CCA', 'Time-Varying CCA'],
            'use_cases': [
                'Linear alignment between model and brain',
                'Finding shared representational spaces',
                'Multi-timepoint alignment with Time-Varying CCA'
            ],
            'output': 'Canonical correlations and transformed representations'
        },
        'RSA': {
            'description': 'Representational Similarity Analysis - compares representational geometries',
            'variants': ['Standard RSA', 'Hierarchical RSA', 'MDS Visualization'],
            'use_cases': [
                'Comparing model and brain representational structures',
                'Testing if geometries are similar without alignment',
                'Visualizing representational spaces'
            ],
            'output': 'Similarity scores between RDMs and visualizations'
        },
        'PLS': {
            'description': 'Partial Least Squares - predictive alignment with latent variables',
            'variants': ['Standard PLS', 'Cross-Validated PLS'],
            'use_cases': [
                'Predicting brain activity from model activations',
                'Finding latent variables that explain both spaces',
                'Analyzing component contributions'
            ],
            'output': 'Predictions and latent variable scores'
        },
        'Metrics': {
            'description': 'Evaluation metrics for assessing alignment quality',
            'methods': [
                'Noise Ceiling - maximum achievable accuracy',
                'Bootstrap CI - confidence intervals',
                'Permutation Test - statistical significance',
                'Normalized Score - fraction of explainable variance'
            ],
            'use_cases': [
                'Estimating reliability of brain measurements',
                'Computing confidence intervals for scores',
                'Testing statistical significance',
                'Normalizing scores by noise ceiling'
            ],
            'output': 'Statistical measures and significance tests'
        }
    }


def print_alignment_summary():
    """Print a formatted summary of alignment methods."""
    summary = get_alignment_summary()

    print("=" * 80)
    print("NeuroFMX Model-to-Brain Alignment Tools")
    print("=" * 80)
    print()

    for method, info in summary.items():
        print(f"\n{method}")
        print("-" * 40)
        print(f"Description: {info['description']}")
        print()

        if 'variants' in info:
            print("Variants:")
            for variant in info['variants']:
                print(f"  - {variant}")
            print()

        if 'methods' in info:
            print("Methods:")
            for m in info['methods']:
                print(f"  - {m}")
            print()

        print("Use Cases:")
        for use_case in info['use_cases']:
            print(f"  - {use_case}")
        print()

        print(f"Output: {info['output']}")

    print("\n" + "=" * 80)
    print("For detailed examples, see individual module docstrings")
    print("=" * 80)


# Quick reference guide
QUICK_REFERENCE = """
Quick Reference: Model-to-Brain Alignment
==========================================

1. LINEAR ALIGNMENT (CCA)
   Use when: You want to find linear transformations that align representations

   from neuros_mechint.alignment import CCA
   cca = CCA(n_components=10)
   cca.fit(model_acts, brain_acts)
   model_canonical, brain_canonical = cca.transform(model_acts, brain_acts)

2. GEOMETRY COMPARISON (RSA)
   Use when: You want to compare representational structures without alignment

   from neuros_mechint.alignment import RSA
   rsa = RSA(metric='correlation', comparison='spearman')
   similarity = rsa.compare(model_acts, brain_acts)

3. PREDICTIVE ALIGNMENT (PLS)
   Use when: You want to predict brain activity from model activations

   from neuros_mechint.alignment import PLS
   pls = PLS(n_components=20)
   pls.fit(model_acts, brain_acts)
   brain_pred = pls.predict(new_model_acts)

4. STATISTICAL EVALUATION
   Use when: You need to assess reliability and significance

   from neuros_mechint.alignment import NoiseCeiling, PermutationTest

   # Noise ceiling
   nc = NoiseCeiling(method='split-half')
   ceiling = nc.estimate(brain_with_repetitions)

   # Permutation test
   perm = PermutationTest(n_permutations=1000)
   result = perm.test(metric_fn, model_acts, brain_acts)

For more examples, use: help(alignment_method)
"""


if __name__ == "__main__":
    print_alignment_summary()
    print("\n")
    print(QUICK_REFERENCE)
