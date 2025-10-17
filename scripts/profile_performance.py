"""
Performance profiling script for neurOS pipelines.

This script profiles key components and identifies bottlenecks.
"""

import cProfile
import pstats
import io
import time
import asyncio
import numpy as np
from pathlib import Path
import json

from neuros.pipeline import Pipeline, MultiModalPipeline
from neuros.drivers.mock_driver import MockDriver
from neuros.models.simple_classifier import SimpleClassifier
from neuros.models.svm_model import SVMModel
from neuros.models.random_forest_model import RandomForestModel
from neuros.processing.filters import BandpassFilter, SmoothingFilter
from neuros.processing.feature_extraction import BandPowerExtractor


def profile_function(func, *args, **kwargs):
    """Profile a function and return stats."""
    profiler = cProfile.Profile()
    profiler.enable()

    result = func(*args, **kwargs)

    profiler.disable()

    # Get stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

    return result, stream.getvalue()


async def profile_pipeline(pipeline_name, pipeline, duration=1.0):
    """Profile a pipeline run."""
    print(f"\n{'='*60}")
    print(f"Profiling: {pipeline_name}")
    print(f"{'='*60}")

    start_time = time.perf_counter()
    metrics = await pipeline.run(duration=duration)
    end_time = time.perf_counter()

    total_time = end_time - start_time

    print(f"\nMetrics:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Samples processed: {metrics['samples']}")
    print(f"  Throughput: {metrics['throughput']:.1f} samples/s")
    print(f"  Mean latency: {metrics['mean_latency']:.2f}ms")

    return {
        'name': pipeline_name,
        'total_time': total_time,
        'metrics': metrics
    }


def profile_feature_extraction():
    """Profile feature extraction performance."""
    print(f"\n{'='*60}")
    print("Profiling: Feature Extraction")
    print(f"{'='*60}")

    extractor = BandPowerExtractor(fs=250.0)

    # Test different data sizes
    sizes = [
        (4, 250),     # 4 channels, 1 second
        (8, 250),     # 8 channels, 1 second
        (16, 250),    # 16 channels, 1 second
        (4, 1000),    # 4 channels, 4 seconds
        (32, 250),    # 32 channels, 1 second
    ]

    results = []
    for n_channels, n_samples in sizes:
        data = np.random.randn(n_channels, n_samples)

        # Time the extraction
        times = []
        for _ in range(100):
            start = time.perf_counter()
            features = extractor.extract(data)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        avg_time = np.mean(times)
        std_time = np.std(times)

        result = {
            'channels': n_channels,
            'samples': n_samples,
            'avg_time_ms': float(avg_time),
            'std_time_ms': float(std_time),
            'features_extracted': len(features)
        }
        results.append(result)

        print(f"\nChannels: {n_channels}, Samples: {n_samples}")
        print(f"  Avg time: {avg_time:.3f}ms ± {std_time:.3f}ms")
        print(f"  Features: {len(features)}")

    return results


def profile_filtering():
    """Profile filtering performance."""
    print(f"\n{'='*60}")
    print("Profiling: Filtering")
    print(f"{'='*60}")

    # Test bandpass filter
    bp_filter = BandpassFilter(lowcut=1.0, highcut=50.0, fs=250.0, order=4)
    smooth_filter = SmoothingFilter(window_size=5)

    sizes = [
        (4, 250),     # 4 channels, 1 second
        (8, 250),     # 8 channels, 1 second
        (16, 250),    # 16 channels, 1 second
        (32, 250),    # 32 channels, 1 second
    ]

    results = []
    for n_channels, n_samples in sizes:
        data = np.random.randn(n_channels, n_samples)

        # Bandpass filter timing
        bp_times = []
        for _ in range(50):
            start = time.perf_counter()
            filtered = bp_filter.apply(data)
            end = time.perf_counter()
            bp_times.append((end - start) * 1000)

        # Smoothing filter timing
        smooth_times = []
        for _ in range(50):
            start = time.perf_counter()
            smoothed = smooth_filter.apply(data)
            end = time.perf_counter()
            smooth_times.append((end - start) * 1000)

        result = {
            'channels': n_channels,
            'samples': n_samples,
            'bandpass_ms': float(np.mean(bp_times)),
            'smoothing_ms': float(np.mean(smooth_times)),
            'total_ms': float(np.mean(bp_times) + np.mean(smooth_times))
        }
        results.append(result)

        print(f"\nChannels: {n_channels}, Samples: {n_samples}")
        print(f"  Bandpass: {np.mean(bp_times):.3f}ms")
        print(f"  Smoothing: {np.mean(smooth_times):.3f}ms")
        print(f"  Total: {result['total_ms']:.3f}ms")

    return results


def profile_model_prediction():
    """Profile model prediction performance."""
    print(f"\n{'='*60}")
    print("Profiling: Model Prediction")
    print(f"{'='*60}")

    models = [
        ('SimpleClassifier', SimpleClassifier()),
        ('SVMModel', SVMModel(C=1.0, gamma='scale')),
        ('RandomForest', RandomForestModel(n_estimators=50)),
    ]

    # Train models
    X_train = np.random.randn(100, 20)
    y_train = np.random.randint(0, 2, size=100)

    results = []
    for model_name, model in models:
        model.train(X_train, y_train)

        # Test prediction times with different batch sizes
        batch_sizes = [1, 5, 10, 20, 50]

        model_results = {'name': model_name, 'batch_results': []}

        for batch_size in batch_sizes:
            X_test = np.random.randn(batch_size, 20)

            times = []
            for _ in range(100):
                start = time.perf_counter()
                predictions = model.predict(X_test)
                end = time.perf_counter()
                times.append((end - start) * 1000)

            avg_time = np.mean(times)
            time_per_sample = avg_time / batch_size

            batch_result = {
                'batch_size': batch_size,
                'avg_time_ms': float(avg_time),
                'time_per_sample_ms': float(time_per_sample)
            }
            model_results['batch_results'].append(batch_result)

            print(f"\n{model_name} - Batch size: {batch_size}")
            print(f"  Total time: {avg_time:.3f}ms")
            print(f"  Per sample: {time_per_sample:.3f}ms")

        results.append(model_results)

    return results


async def profile_complete_pipelines():
    """Profile complete pipeline workflows."""
    print(f"\n{'='*60}")
    print("Profiling: Complete Pipelines")
    print(f"{'='*60}")

    # Simple pipeline (4 channels * 5 bands = 20 features)
    pipeline1 = Pipeline(
        driver=MockDriver(channels=4, sampling_rate=250),
        model=SimpleClassifier(),
        fs=250.0
    )
    X_train = np.random.randn(50, 4 * 5)  # Match expected features
    y_train = np.random.randint(0, 2, size=50)
    pipeline1.train(X_train, y_train)

    # Pipeline with filters
    pipeline2 = Pipeline(
        driver=MockDriver(channels=8, sampling_rate=250),
        model=SVMModel(C=1.0, gamma='scale'),
        fs=250.0,
        filters=[
            BandpassFilter(1.0, 50.0, fs=250.0),
            SmoothingFilter(window_size=5)
        ]
    )
    X_train2 = np.random.randn(50, 8 * 5)  # Match expected features
    pipeline2.train(X_train2, y_train)

    # Complex pipeline (16 channels * 5 bands = 80 features)
    pipeline3 = Pipeline(
        driver=MockDriver(channels=16, sampling_rate=250),
        model=RandomForestModel(n_estimators=50),
        fs=250.0,
        filters=[BandpassFilter(8.0, 30.0, fs=250.0)]
    )
    X_train3 = np.random.randn(50, 16 * 5)  # Match expected features
    pipeline3.train(X_train3, y_train)

    results = []
    results.append(await profile_pipeline("Simple Pipeline", pipeline1, duration=1.0))
    results.append(await profile_pipeline("Pipeline with Filters", pipeline2, duration=1.0))
    results.append(await profile_pipeline("Complex Pipeline", pipeline3, duration=1.0))

    return results


async def main():
    """Run all profiling tasks."""
    print("\n" + "="*60)
    print("neurOS Performance Profiling")
    print("="*60)

    # Profile individual components
    feature_results = profile_feature_extraction()
    filter_results = profile_filtering()
    model_results = profile_model_prediction()

    # Profile complete pipelines
    pipeline_results = await profile_complete_pipelines()

    # Compile all results
    all_results = {
        'feature_extraction': feature_results,
        'filtering': filter_results,
        'model_prediction': model_results,
        'complete_pipelines': pipeline_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # Save results
    output_dir = Path(__file__).parent.parent / 'profiling_results'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / 'performance_profile.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)

    print("\nFeature Extraction (4 channels, 250 samples):")
    print(f"  {feature_results[0]['avg_time_ms']:.3f}ms")

    print("\nFiltering (4 channels, 250 samples):")
    print(f"  {filter_results[0]['total_ms']:.3f}ms")

    print("\nModel Prediction (single sample):")
    for model_result in model_results:
        single_sample = model_result['batch_results'][0]
        print(f"  {model_result['name']}: {single_sample['avg_time_ms']:.3f}ms")

    print("\nComplete Pipelines (1 second):")
    for pipeline_result in pipeline_results:
        print(f"  {pipeline_result['name']}:")
        print(f"    Throughput: {pipeline_result['metrics']['throughput']:.1f} samples/s")
        print(f"    Latency: {pipeline_result['metrics']['mean_latency']:.2f}ms")


if __name__ == '__main__':
    asyncio.run(main())
