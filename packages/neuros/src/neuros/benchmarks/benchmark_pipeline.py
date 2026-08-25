"""Composed pipeline benchmark for the user-facing neurOS SDK.

This benchmark intentionally instantiates concrete driver/model packages, so it
lives in the ``neuros`` meta-distribution rather than the stable kernel.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from neuros.drivers.mock_driver import MockDriver
from neuros.models.simple_classifier import SimpleClassifier
from neuros.pipeline import Pipeline
from neuros.processing.feature_extraction import BandPowerExtractor


async def run_benchmark(duration: float = 10.0) -> Dict[str, float]:
    """Train a deterministic synthetic baseline and benchmark the runtime path."""

    if duration <= 0:
        raise ValueError("duration must be positive")

    rng = np.random.default_rng(42)
    n_samples = 1000
    n_channels = 8
    fs = 250.0
    t = np.linspace(0, 1, int(fs), endpoint=False)

    signals = []
    labels = []
    for _ in range(n_samples):
        label = int(rng.integers(0, 2))
        freq = 10.0 if label == 0 else 20.0
        sig = np.sin(2 * np.pi * freq * t) + 0.1 * rng.standard_normal(len(t))
        signals.append(np.vstack([np.roll(sig, shift) for shift in range(n_channels)]))
        labels.append(label)

    extractor = BandPowerExtractor(fs)
    features = np.asarray([extractor.extract(trial) for trial in signals])
    labels_array = np.asarray(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels_array,
        test_size=0.2,
        random_state=42,
        stratify=labels_array,
    )

    model = SimpleClassifier(max_iter=200)
    model.train(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    driver = MockDriver(sampling_rate=fs, channels=n_channels)
    pipeline = Pipeline(driver=driver, model=model, fs=fs, adaptation=True)
    metrics = await pipeline.run(duration=duration)
    metrics["accuracy"] = float(accuracy)
    return metrics
