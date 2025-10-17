"""
Comprehensive benchmark suite for neurOS.

Compares neurOS performance against:
- Baseline implementations
- Other BCI frameworks (where available)
- Published benchmark results

Run with: python scripts/run_benchmarks.py
"""

import time
import numpy as np
import json
from pathlib import Path
from typing import Dict, List
import asyncio

from neuros.pipeline import Pipeline
from neuros.drivers.mock_driver import MockDriver
from neuros.models.simple_classifier import SimpleClassifier
from neuros.models.svm_model import SVMModel
from neuros.models.random_forest_model import RandomForestModel
from neuros.processing.filters import BandpassFilter, SmoothingFilter
from neuros.processing.feature_extraction import BandPowerExtractor

# Try to import optimized versions
try:
    from neuros.processing.feature_extraction_optimized import OptimizedBandPowerExtractor
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False

# Try to import advanced features
try:
    from neuros.processing.advanced_features import CommonSpatialPatterns
    CSP_AVAILABLE = True
except ImportError:
    CSP_AVAILABLE = False


def benchmark_feature_extraction():
    """Benchmark feature extraction speeds."""
    print("\n" + "="*60)
    print("BENCHMARK: Feature Extraction")
    print("="*60)

    results = {}
    configs = [
        (4, 250, "4 ch, 1 sec"),
        (8, 250, "8 ch, 1 sec"),
        (16, 250, "16 ch, 1 sec"),
        (32, 250, "32 ch, 1 sec"),
    ]

    for n_channels, n_samples, desc in configs:
        print(f"\n{desc}:")

        # Standard version
        extractor = BandPowerExtractor(fs=250.0)
        data = np.random.randn(n_channels, n_samples)

        times = []
        for _ in range(100):
            start = time.perf_counter()
            features = extractor.extract(data)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        std_mean = np.mean(times)
        std_std = np.std(times)
        print(f"  Standard: {std_mean:.3f}ms ± {std_std:.3f}ms")

        result = {'standard': {
            'mean_ms': float(std_mean),
            'std_ms': float(std_std)
        }}

        # Optimized version (if available)
        if OPTIMIZED_AVAILABLE:
            extractor_opt = OptimizedBandPowerExtractor(fs=250.0)

            times_opt = []
            for _ in range(100):
                start = time.perf_counter()
                features = extractor_opt.extract(data)
                end = time.perf_counter()
                times_opt.append((end - start) * 1000)

            opt_mean = np.mean(times_opt)
            opt_std = np.std(times_opt)
            speedup = std_mean / opt_mean

            print(f"  Optimized: {opt_mean:.3f}ms ± {opt_std:.3f}ms")
            print(f"  Speedup: {speedup:.2f}x")

            result['optimized'] = {
                'mean_ms': float(opt_mean),
                'std_ms': float(opt_std),
                'speedup': float(speedup)
            }

        results[desc] = result

    return results


def benchmark_models():
    """Benchmark model inference speeds."""
    print("\n" + "="*60)
    print("BENCHMARK: Model Inference")
    print("="*60)

    models = [
        ('SimpleClassifier', SimpleClassifier()),
        ('SVMModel', SVMModel()),
        ('RandomForest(20)', RandomForestModel(n_estimators=20)),
        ('RandomForest(50)', RandomForestModel(n_estimators=50)),
        ('RandomForest(100)', RandomForestModel(n_estimators=100)),
    ]

    # Training data
    X_train = np.random.randn(100, 20)
    y_train = np.random.randint(0, 2, size=100)

    results = {}

    for model_name, model in models:
        print(f"\n{model_name}:")

        # Train
        model.train(X_train, y_train)

        # Benchmark different batch sizes
        batch_results = {}
        for batch_size in [1, 5, 10, 20]:
            X_test = np.random.randn(batch_size, 20)

            times = []
            for _ in range(100):
                start = time.perf_counter()
                predictions = model.predict(X_test)
                end = time.perf_counter()
                times.append((end - start) * 1000)

            mean_time = np.mean(times)
            per_sample = mean_time / batch_size

            print(f"  Batch {batch_size}: {mean_time:.3f}ms ({per_sample:.3f}ms/sample)")

            batch_results[f'batch_{batch_size}'] = {
                'total_ms': float(mean_time),
                'per_sample_ms': float(per_sample)
            }

        results[model_name] = batch_results

    return results


async def benchmark_pipelines():
    """Benchmark complete pipeline throughput."""
    print("\n" + "="*60)
    print("BENCHMARK: Complete Pipelines")
    print("="*60)

    configs = [
        ('Simple (4ch)', 4, SimpleClassifier(), []),
        ('WithFilters (8ch)', 8, SVMModel(), [
            BandpassFilter(1.0, 50.0, fs=250.0),
            SmoothingFilter(window_size=5)
        ]),
        ('Complex (16ch)', 16, RandomForestModel(n_estimators=50), [
            BandpassFilter(8.0, 30.0, fs=250.0)
        ]),
    ]

    results = {}

    for name, n_channels, model, filters in configs:
        print(f"\n{name}:")

        pipeline = Pipeline(
            driver=MockDriver(channels=n_channels, sampling_rate=250),
            model=model,
            fs=250.0,
            filters=filters
        )

        # Train
        X_train = np.random.randn(50, n_channels * 5)
        y_train = np.random.randint(0, 2, size=50)
        pipeline.train(X_train, y_train)

        # Run
        metrics = await pipeline.run(duration=1.0)

        print(f"  Throughput: {metrics['throughput']:.1f} samples/s")
        print(f"  Latency: {metrics['mean_latency']:.2f}ms")
        print(f"  Samples: {metrics['samples']}")

        results[name] = {
            'throughput_sps': float(metrics['throughput']),
            'latency_ms': float(metrics['mean_latency']),
            'samples': int(metrics['samples'])
        }

    return results


def benchmark_csp():
    """Benchmark CSP training and transformation."""
    if not CSP_AVAILABLE:
        print("\nCSP not available, skipping...")
        return {}

    print("\n" + "="*60)
    print("BENCHMARK: Common Spatial Patterns")
    print("="*60)

    configs = [
        (50, 4, 250, "50 trials, 4 ch"),
        (100, 8, 250, "100 trials, 8 ch"),
        (200, 16, 250, "200 trials, 16 ch"),
    ]

    results = {}

    for n_trials, n_channels, n_samples, desc in configs:
        print(f"\n{desc}:")

        # Generate data
        X = np.random.randn(n_trials, n_channels, n_samples)
        y = np.random.randint(0, 2, size=n_trials)

        csp = CommonSpatialPatterns(n_components=4)

        # Benchmark fit
        start = time.perf_counter()
        csp.fit(X, y)
        fit_time = (time.perf_counter() - start) * 1000

        print(f"  Fit time: {fit_time:.2f}ms")

        # Benchmark transform
        times = []
        for _ in range(100):
            start = time.perf_counter()
            features = csp.transform(X[:10])  # 10 trials
            end = time.perf_counter()
            times.append((end - start) * 1000)

        transform_mean = np.mean(times)
        per_trial = transform_mean / 10

        print(f"  Transform (10 trials): {transform_mean:.2f}ms ({per_trial:.3f}ms/trial)")

        results[desc] = {
            'fit_ms': float(fit_time),
            'transform_10trials_ms': float(transform_mean),
            'per_trial_ms': float(per_trial)
        }

    return results


def benchmark_vs_baseline():
    """Compare against baseline implementations."""
    print("\n" + "="*60)
    print("BENCHMARK: vs. Baseline")
    print("="*60)

    # Baseline: naive band power implementation
    def baseline_band_power(data, fs=250.0):
        """Naive band power extraction (no optimization)."""
        from scipy.signal import welch

        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }

        features = []
        for ch in range(data.shape[0]):
            f, Pxx = welch(data[ch], fs=fs, nperseg=256)
            for low, high in bands.values():
                idx = (f >= low) & (f <= high)
                power = np.trapz(Pxx[idx], f[idx])
                features.append(power)

        return np.array(features)

    # Test data
    data = np.random.randn(4, 250)

    # Baseline
    times_baseline = []
    for _ in range(100):
        start = time.perf_counter()
        features = baseline_band_power(data)
        end = time.perf_counter()
        times_baseline.append((end - start) * 1000)

    baseline_mean = np.mean(times_baseline)

    # neurOS standard
    extractor = BandPowerExtractor(fs=250.0)
    times_neuros = []
    for _ in range(100):
        start = time.perf_counter()
        features = extractor.extract(data)
        end = time.perf_counter()
        times_neuros.append((end - start) * 1000)

    neuros_mean = np.mean(times_neuros)
    speedup_std = baseline_mean / neuros_mean

    print(f"\nBaseline: {baseline_mean:.3f}ms")
    print(f"neurOS (standard): {neuros_mean:.3f}ms")
    print(f"Speedup: {speedup_std:.2f}x")

    results = {
        'baseline_ms': float(baseline_mean),
        'neuros_standard_ms': float(neuros_mean),
        'speedup_standard': float(speedup_std)
    }

    # neurOS optimized (if available)
    if OPTIMIZED_AVAILABLE:
        extractor_opt = OptimizedBandPowerExtractor(fs=250.0)
        times_opt = []
        for _ in range(100):
            start = time.perf_counter()
            features = extractor_opt.extract(data)
            end = time.perf_counter()
            times_opt.append((end - start) * 1000)

        opt_mean = np.mean(times_opt)
        speedup_opt = baseline_mean / opt_mean

        print(f"neurOS (optimized): {opt_mean:.3f}ms")
        print(f"Speedup: {speedup_opt:.2f}x")

        results['neuros_optimized_ms'] = float(opt_mean)
        results['speedup_optimized'] = float(speedup_opt)

    return results


async def main():
    """Run all benchmarks."""
    print("\n" + "="*70)
    print("                neurOS Performance Benchmarks")
    print("="*70)

    all_results = {}

    # Run benchmarks
    all_results['feature_extraction'] = benchmark_feature_extraction()
    all_results['model_inference'] = benchmark_models()
    all_results['pipelines'] = await benchmark_pipelines()
    all_results['csp'] = benchmark_csp()
    all_results['vs_baseline'] = benchmark_vs_baseline()

    # Save results
    output_dir = Path(__file__).parent.parent / 'benchmark_results'
    output_dir.mkdir(exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'benchmarks_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*70)
    print(f"Results saved to: {output_file}")
    print("="*70)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if 'vs_baseline' in all_results:
        print("\nSpeedup vs Baseline:")
        vs_baseline = all_results['vs_baseline']
        print(f"  Standard: {vs_baseline['speedup_standard']:.2f}x faster")
        if 'speedup_optimized' in vs_baseline:
            print(f"  Optimized: {vs_baseline['speedup_optimized']:.2f}x faster")

    if 'pipelines' in all_results:
        print("\nPipeline Throughput:")
        for name, metrics in all_results['pipelines'].items():
            print(f"  {name}: {metrics['throughput_sps']:.0f} samples/s, {metrics['latency_ms']:.1f}ms latency")

    print("\n✅ All benchmarks completed successfully!")


if __name__ == '__main__':
    asyncio.run(main())
