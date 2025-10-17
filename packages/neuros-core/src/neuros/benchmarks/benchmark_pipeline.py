"""
Pipeline benchmark for neurOS.

This module defines a simple benchmark that trains a baseline model on
synthetic data, runs the pipeline with a mock driver and reports
performance metrics.  These include throughput (samples per second), mean
latency (seconds) and classification accuracy on a held‑out test set.
"""

from __future__ import annotations

import asyncio
import time
from typing import Dict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from neuros.drivers.mock_driver import MockDriver
from neuros.models.simple_classifier import SimpleClassifier
from neuros.pipeline import Pipeline
from neuros.processing.feature_extraction import BandPowerExtractor
from neuros.processing.filters import SmoothingFilter


async def run_benchmark(duration: float = 10.0) -> Dict[str, float]:
    """Run a benchmark and return aggregated metrics.

    Parameters
    ----------
    duration : float
        Number of seconds to stream data during the benchmark.

    Returns
    -------
    dict
        Dictionary containing throughput, mean latency and accuracy.
    """
    # generate synthetic dataset for training and testing
    n_samples = 1000
    n_channels = 8
    fs = 250.0
    # synthetic raw signals: sinusoid plus noise
    t = np.linspace(0, 1, int(fs), endpoint=False)
    signals = []
    labels = []
    for i in range(n_samples):
        label = np.random.randint(0, 2)
        freq = 10.0 if label == 0 else 20.0
        sig = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
        # stack channels with small phase shifts
        trial = np.vstack([
            np.roll(sig, shift) for shift in range(n_channels)
        ])
        signals.append(trial)
        labels.append(label)
    signals = np.array(signals)  # shape (n_samples, channels, samples)
    labels = np.array(labels)
    # extract features using band power extractor
    extractor = BandPowerExtractor(fs)
    feature_list = []
    for trial in signals:
        feature_list.append(extractor.extract(trial))
    features = np.array(feature_list)
    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    # train model
    model = SimpleClassifier(max_iter=200)
    model.train(X_train, y_train)
    # compute offline accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # create pipeline with trained model and mock driver
    driver = MockDriver(sampling_rate=fs, channels=n_channels)
    pipeline = Pipeline(driver=driver, model=model, fs=fs, adaptation=True)
    # run pipeline for specified duration
    metrics = await pipeline.run(duration=duration)
    # include offline accuracy in metrics
    metrics["accuracy"] = float(accuracy)
    return metrics