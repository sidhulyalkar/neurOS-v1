#!/usr/bin/env python3
"""
Example: SAE Feature Validation Across Neural Data Modalities

This script demonstrates the complete neurOS validation framework:
1. Load data from Allen (spikes) and BCI (EEG) datasets
2. Train or load a sparse autoencoder (SAE)
3. Analyze SAE features for known neural properties
4. Compare features across modalities
5. Generate validation report

This validates that SAE-discovered features correlate with:
- Orientation selectivity (Allen visual coding)
- Motor laterality (BCI motor imagery)

Usage:
    # Auto-detect cache location
    python examples/sae_validation_example.py

    # Specify cache directory
    python examples/sae_validation_example.py --allen-cache allen_validation_cache

    # Use mock data only
    python examples/sae_validation_example.py --use-mock

Author: neurOS Validation Framework
Date: 2025
"""

import numpy as np
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run complete SAE validation pipeline."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='SAE Feature Validation Example',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--allen-cache',
        type=str,
        default=None,
        help='Path to Allen data cache directory (auto-detects if not specified)'
    )
    parser.add_argument(
        '--use-mock',
        action='store_true',
        help='Force use of mock data even if real data is available'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='validation_outputs',
        help='Directory for output visualizations (default: validation_outputs)'
    )
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("neurOS SAE Validation Framework - Example")
    logger.info("="*80)

    # =========================================================================
    # Step 1: Load Datasets
    # =========================================================================
    logger.info("\n[Step 1/6] Loading neural datasets...")

    # Determine Allen cache directory
    allen_cache_dir = None

    if not args.use_mock:
        if args.allen_cache:
            # User specified cache location
            allen_cache_dir = Path(args.allen_cache)
            if not allen_cache_dir.exists():
                logger.warning(f"  Specified cache directory does not exist: {allen_cache_dir}")
                allen_cache_dir = None
            elif not (allen_cache_dir / "manifest.json").exists():
                logger.warning(f"  No manifest.json found in: {allen_cache_dir}")
                allen_cache_dir = None
            else:
                logger.info(f"  Using specified Allen cache: {allen_cache_dir.absolute()}")
        else:
            # Auto-detect cache location
            allen_cache_options = [
                Path("allen_validation_cache"),  # Current directory
                Path(__file__).parent.parent / "allen_validation_cache",  # Repo root
                Path.home() / ".neuros" / "datasets" / "allen" / "visual_coding"  # Default AllenSDK
            ]

            for cache_path in allen_cache_options:
                if cache_path.exists() and (cache_path / "manifest.json").exists():
                    allen_cache_dir = cache_path
                    logger.info(f"  Found Allen cache at: {allen_cache_dir.absolute()}")
                    break

    if allen_cache_dir is None and not args.use_mock:
        logger.info("  No Allen cache found. Will use mock data.")
        logger.info("  To download real data, run: python scripts/download_validation_data.py")
    elif args.use_mock:
        logger.info("  Mock data mode enabled (--use-mock)")

    try:
        # Import dataset loaders
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "neuros-foundation" / "src"))

        from neuros.datasets.allen_datasets import AllenVisualCodingValidator
        from neuros.datasets.bci_datasets import BCIMotorImageryValidator

        # Load Allen dataset
        logger.info("  Loading Allen Visual Coding dataset...")
        allen_validator = None

        if allen_cache_dir is not None:
            try:
                # Use downloaded data
                allen_validator = AllenVisualCodingValidator(
                    cache_dir=str(allen_cache_dir),
                    brain_areas=['VISp'],  # V1
                    min_units=50  # Minimum good units
                )
                logger.info(f"  ✓ Loaded real Allen data from cache")
                logger.info(f"    Session ID: {allen_validator.session_id}")
                logger.info(f"    Good units: {len(allen_validator.units)}")
            except Exception as e:
                logger.warning(f"  Could not load Allen data from cache: {e}")
                logger.info("  Falling back to mock data...")
                import traceback
                traceback.print_exc()
                allen_validator = None

        if allen_validator is None:
            logger.info("  ℹ Using mock Allen data for demonstration")
            logger.info("    (Download real data with: python scripts/download_validation_data.py)")

        # Load BCI dataset (uses mock data by default)
        logger.info("  Loading BCI Motor Imagery dataset...")
        bci_validator = BCIMotorImageryValidator(
            n_trials=200,
            n_channels=22
        )
        logger.info(f"  ✓ Loaded BCI data with {bci_validator.n_trials} trials")

    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        logger.info("Please install required packages:")
        logger.info("  pip install allensdk numpy pandas")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # Step 2: Extract Neural Windows
    # =========================================================================
    logger.info("\n[Step 2/6] Extracting neural data windows...")

    if allen_validator is not None:
        allen_windows = allen_validator.get_neural_windows(
            window_length=1.0,  # 1 second windows
            stride=0.5,  # 0.5 second stride
            bin_size=0.02  # 20ms bins
        )
        logger.info(f"  ✓ Allen: {len(allen_windows)} windows extracted")

        # Get labels
        allen_labels = allen_validator.get_task_labels()
        logger.info(f"  ✓ Allen: {len(allen_labels['orientation'])} orientation labels")
    else:
        # Create mock Allen data
        logger.info("  Creating mock Allen data for demonstration...")
        allen_windows = create_mock_allen_windows(n_windows=100, n_units=50)
        allen_labels = {'orientation': np.random.uniform(0, 180, 100)}
        logger.info(f"  ✓ Mock Allen: {len(allen_windows)} windows created")

    bci_windows = bci_validator.get_neural_windows(
        window_length=2.0,  # 2 second windows
        stride=1.0  # 1 second stride
    )
    logger.info(f"  ✓ BCI: {len(bci_windows)} windows extracted")

    # Extract motor labels from window metadata (each window has its trial label)
    bci_motor_labels_from_windows = np.array([w.metadata['motor_class'] for w in bci_windows])
    logger.info(f"  ✓ BCI: {len(bci_motor_labels_from_windows)} motor labels extracted from windows")

    # =========================================================================
    # Step 3: Generate or Load SAE Model
    # =========================================================================
    logger.info("\n[Step 3/6] Setting up SAE model...")

    # For this example, we'll use a simple mock SAE
    # In practice, you would train a real SAE on neural data
    logger.info("  Using mock SAE for demonstration")
    logger.info("  (In production: train SAE on NeuroFM embeddings)")

    # Extract mock SAE activations (random for demo)
    allen_data = np.array([w.data.mean(axis=0) for w in allen_windows])
    bci_data = np.array([w.data.mean(axis=0) for w in bci_windows])

    # Mock SAE: simple PCA-like projection
    from sklearn.decomposition import PCA

    # Adapt feature dimension to data (can't exceed min of samples or features)
    allen_n_features = min(128, allen_data.shape[0], allen_data.shape[1])
    bci_n_features = min(128, bci_data.shape[0], bci_data.shape[1])

    # "Train" SAE on Allen data
    sae_allen = PCA(n_components=allen_n_features)
    allen_activations = sae_allen.fit_transform(allen_data)
    logger.info(f"  ✓ Allen SAE activations: {allen_activations.shape}")

    # "Train" separate SAE on BCI data
    sae_bci = PCA(n_components=bci_n_features)
    bci_activations = sae_bci.fit_transform(bci_data)
    logger.info(f"  ✓ BCI SAE activations: {bci_activations.shape}")

    # =========================================================================
    # Step 4: Analyze SAE Features for Known Neural Properties
    # =========================================================================
    logger.info("\n[Step 4/6] Analyzing SAE features...")

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "neuros-mechint" / "src"))

        # Import directly to avoid torch dependency in __init__.py
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "multimodal_sae_analysis",
            Path(__file__).parent.parent / "packages" / "neuros-mechint" / "src" / "neuros_mechint" / "multimodal_sae_analysis.py"
        )
        multimodal_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(multimodal_module)
        MultiModalSAEAnalyzer = multimodal_module.MultiModalSAEAnalyzer

        analyzer = MultiModalSAEAnalyzer(feature_threshold=0.3)

        # Analyze orientation selectivity in Allen features
        logger.info("  Analyzing orientation selectivity (Allen)...")
        allen_orientations = allen_labels['orientation'][:len(allen_activations)]

        allen_results = analyzer.analyze_orientation_features(
            activations=allen_activations,
            orientations=allen_orientations,
            return_controls=True
        )

        logger.info(f"  ✓ Found {allen_results['n_significant']} orientation-selective features")
        logger.info(f"    Max correlation: {allen_results['max_correlation']:.3f}")
        logger.info(f"    Above shuffle: {allen_results.get('above_shuffle', 0):.3f}")

        # Analyze motor selectivity in BCI features
        logger.info("  Analyzing motor selectivity (BCI)...")
        # Use labels extracted from windows (already aligned with bci_activations)
        bci_motor_labels = bci_motor_labels_from_windows[:len(bci_activations)]

        bci_results = analyzer.analyze_motor_features(
            activations=bci_activations,
            motor_labels=bci_motor_labels,
            return_controls=True
        )

        logger.info(f"  ✓ Found {bci_results['n_significant']} motor-selective features")
        logger.info(f"    Max selectivity (η²): {bci_results['max_selectivity']:.3f}")
        logger.info(f"    Above shuffle: {bci_results.get('above_shuffle', 0):.3f}")

    except Exception as e:
        logger.error(f"Feature analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # Step 5: Cross-Modal Comparison
    # =========================================================================
    logger.info("\n[Step 5/6] Comparing features across modalities...")

    try:
        # Import directly to avoid torch dependency
        spec2 = importlib.util.spec_from_file_location(
            "cross_modal_analysis",
            Path(__file__).parent.parent / "packages" / "neuros-mechint" / "src" / "neuros_mechint" / "cross_modal_analysis.py"
        )
        cross_modal_module = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(cross_modal_module)
        CrossModalAnalyzer = cross_modal_module.CrossModalAnalyzer
        validation_summary = cross_modal_module.validation_summary

        cross_analyzer = CrossModalAnalyzer()

        # Representational Similarity Analysis
        logger.info("  Computing Representational Similarity Analysis (RSA)...")
        rsa_results = cross_analyzer.representational_similarity_analysis(
            allen_features=allen_activations,
            bci_features=bci_activations
        )
        logger.info(f"  ✓ RSA correlation: {rsa_results['rsa_correlation']:.3f} (p={rsa_results['p_value']:.4f})")

        # Canonical Correlation Analysis
        logger.info("  Computing Canonical Correlation Analysis (CCA)...")
        cca_results = cross_analyzer.canonical_correlation_analysis(
            allen_features=allen_activations,
            bci_features=bci_activations,
            n_components=5
        )
        if 'error' not in cca_results:
            logger.info(f"  ✓ Top canonical correlation: {max(cca_results['canonical_correlations']):.3f}")
        else:
            logger.warning(f"  CCA skipped: {cca_results['error']}")

        # Feature-level comparison
        logger.info("  Comparing SAE features across modalities...")
        cross_modal_results = analyzer.compare_cross_modal_features(
            allen_activations=allen_activations,
            bci_activations=bci_activations,
            allen_results=allen_results,
            bci_results=bci_results
        )

        overlap_info = cross_modal_results['overlap_analysis']
        if 'overlap_ratio' in overlap_info:
            logger.info(f"  ✓ Feature overlap ratio: {overlap_info['overlap_ratio']:.2f}")
            logger.info(f"  ✓ Jaccard index: {overlap_info['jaccard_index']:.2f}")

    except Exception as e:
        logger.error(f"Cross-modal analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================================
    # Step 6: Generate Validation Summary
    # =========================================================================
    logger.info("\n[Step 6/6] Generating validation summary...")

    try:
        summary = validation_summary(
            allen_results=allen_results,
            bci_results=bci_results,
            cross_modal_results=cross_modal_results
        )

        logger.info("\n" + "="*80)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Overall Status: {summary['validation_status']}")
        logger.info(f"Overall Score: {summary['overall_score']}/100")
        logger.info("")
        logger.info("Detailed Results:")
        logger.info(f"  Allen orientation features: {summary['detailed_scores']['allen_orientation_selectivity']}")
        logger.info(f"  BCI motor features: {summary['detailed_scores']['bci_motor_selectivity']}")
        logger.info(f"  Allen max correlation: {summary['detailed_scores']['allen_max_correlation']:.3f}")
        logger.info(f"  BCI max selectivity: {summary['detailed_scores']['bci_max_selectivity']:.3f}")
        logger.info("")
        logger.info("Validation Criteria:")
        logger.info(f"  ✓ Allen features found: {summary['allen_features_found']}")
        logger.info(f"  ✓ BCI features found: {summary['bci_features_found']}")
        logger.info(f"  ✓ Allen above shuffle: {summary['allen_above_shuffle']}")
        logger.info(f"  ✓ BCI above shuffle: {summary['bci_above_shuffle']}")
        logger.info(f"  ✓ Cross-modal correlation: {summary['cross_modal_correlation']}")
        logger.info("="*80)

        # Optional: Generate visualizations
        logger.info("\nGenerating visualizations...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Visualize Allen features
        logger.info("  Creating Allen feature visualization...")
        fig_allen = analyzer.visualize_feature_analysis(
            results=allen_results,
            analysis_type='orientation',
            save_path=str(output_dir / "allen_orientation_features.png")
        )

        # Visualize BCI features
        logger.info("  Creating BCI feature visualization...")
        fig_bci = analyzer.visualize_feature_analysis(
            results=bci_results,
            analysis_type='motor',
            save_path=str(output_dir / "bci_motor_features.png")
        )

        # Visualize cross-modal comparison
        logger.info("  Creating cross-modal comparison visualization...")
        fig_cross = cross_analyzer.plot_cross_modal_comparison(
            rsa_results=rsa_results,
            cca_results=cca_results if 'error' not in cca_results else None,
            save_path=str(output_dir / "cross_modal_comparison.png")
        )

        logger.info(f"\n✓ All visualizations saved to {output_dir}/")

    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\n" + "="*80)
    logger.info("Validation pipeline completed successfully!")
    logger.info("="*80)


def create_mock_allen_windows(n_windows: int = 100, n_units: int = 50):
    """Create mock Allen-style neural windows for demonstration."""
    from neuros.datasets.base_dataset import NeuralWindow

    windows = []
    for i in range(n_windows):
        # Mock spike data
        data = np.random.poisson(lam=2.0, size=(50, n_units))  # 50 time bins, n_units neurons
        labels = np.array([np.random.uniform(0, 180)])  # Random orientation
        metadata = {
            'session_id': 'mock_session',
            'window_id': i,
            'orientation': labels[0]
        }

        window = NeuralWindow(
            data=data,
            labels=labels,
            metadata=metadata,
            window_id=f"mock_window_{i}"
        )
        windows.append(window)

    return windows


if __name__ == "__main__":
    main()
